import os
import re
import json
from dotenv import load_dotenv
from flask import Flask, request
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated, List, Dict, Optional
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
import datetime
import uuid
from collections import defaultdict
from spm_upload import SignavioImporter
from bpmn_generator import BPMNLayoutGenerator

# --- 1. Konfiguration und Initialisierung ---
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(slack_app)

importer = SignavioImporter(
    mail=os.environ.get("USER_MAIL"),
    password=os.environ.get("USER_PASSWORD"),
    workspace=os.environ.get("WORKSPACE_ID"),
    host=os.environ.get("HOST_URL")
)


# --- 2. Definition der graphen-basierten Wissensstruktur ---

class ProcessEdge(TypedDict):
    target_id: str
    label: Optional[str]

class ProcessNode(TypedDict):
    id: str
    type: str
    label: str
    lane: str
    next_nodes: List[ProcessEdge]

class ProcessKnowledge(TypedDict):
    prozessname: str
    prozessziel: str
    akteure: List[str]
    nodes: List[ProcessNode]

extraction_llm = llm.with_structured_output(ProcessKnowledge)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    process_knowledge: ProcessKnowledge
    confirmation_pending: Optional[str]
    signavio_model_id: Optional[str]

# --- 2.1 Logik-Funktionen für die Werkzeuge ---
def _enforce_merging_gateways(knowledge: ProcessKnowledge) -> ProcessKnowledge:
    """
    Analysiert den Graphen und fügt fehlende schließende Gateways programmatisch hinzu.
    """
    print("Führe Qualitätssicherung für Gateways aus...")
    nodes_dict = {node['id']: node for node in knowledge['nodes']}
    in_degree = defaultdict(int)
    predecessors = defaultdict(list)

    for node_id, node in nodes_dict.items():
        for edge in node.get('next_nodes', []):
            target_id = edge['target_id']
            if target_id in nodes_dict:
                in_degree[target_id] += 1
                predecessors[target_id].append(node_id)

    nodes_to_add = []
    reroute_map = {}

    for target_id, count in in_degree.items():
        if count > 1 and "Gateway" not in nodes_dict[target_id]['type']:
            print(f"Merge Point bei '{target_id}' erkannt. Füge schließendes Gateway hinzu.")
            
            new_gateway_id = f"merge_gateway_{uuid.uuid4().hex[:6]}"
            new_gateway = {
                "id": new_gateway_id,
                "type": "exclusiveGateway",
                "label": "",
                "lane": nodes_dict[target_id]['lane'],
                "next_nodes": [{"target_id": target_id, "label": None}]
            }
            nodes_to_add.append(new_gateway)
            reroute_map[target_id] = new_gateway_id

    if not nodes_to_add:
        print("Keine fehlenden schließenden Gateways gefunden.")
        return knowledge

    new_nodes_list = knowledge['nodes']
    for node in list(new_nodes_list):
        for edge in node.get('next_nodes', []):
            if edge['target_id'] in reroute_map:
                if node['id'] in predecessors[edge['target_id']]:
                    print(f"Leite Kante von '{node['id']}' zu '{edge['target_id']}' um auf -> '{reroute_map[edge['target_id']]}'")
                    edge['target_id'] = reroute_map[edge['target_id']]
    
    new_nodes_list.extend(nodes_to_add)
    knowledge['nodes'] = new_nodes_list
    print("Gateway-Qualitätssicherung abgeschlossen.")
    return knowledge


def _extract_and_update_knowledge_logic(konversationshistorie: List[BaseMessage], bisheriges_wissen: dict):
    """Extrahiert Informationen und baut den Prozessgraphen auf."""
    history_for_prompt = [msg.model_dump() for msg in konversationshistorie]
    
    extraction_prompt = f"""
    Du bist ein Experte für Prozessmodellierung nach BPMN 2.0. Deine Aufgabe ist es, eine Konversation in eine BPMN-konforme, graphen-basierte JSON-Struktur zu übersetzen.
    **KERN-ANWEISUNG:**
    Aktualisiere das `bisheriges_wissen`-Objekt basierend auf der letzten Nutzernachricht. Wenn keine neuen Prozessinformationen vorhanden sind, gib das Objekt **exakt unverändert** zurück.
    **REGELN ZUR MODELLIERUNG:**
    1.  **Knoten & Kanten:** Jeder Prozessschritt ist ein Knoten. Die Verbindungen (`next_nodes`) sind eine Liste von Objekten, die eine `target_id` und ein `label` enthalten.
    2.  **Kanten-Beschriftung (SEHR WICHTIG):** Bei einem `exclusiveGateway` MUSST du das `label` der ausgehenden Kanten mit der Bedingung des Pfades füllen (z.B. "Ja", "Nein", "< 100€"). In allen anderen Fällen dürfen Kanten keine Labels haben, nur wenn sie auf ein exklusives Gateway folgen.
    3.  **Lanes & Akteure:** Ordne jedem Knoten eine `lane` zu und halte die globale `akteure`-Liste synchron.
    4.  **PROZESSENDE (KRITISCH):** Wenn der Nutzer explizit sagt, dass der Prozess oder ein Pfad "abgeschlossen", "beendet" oder "fertig" ist, MUSST du einen neuen Knoten vom Typ `endEvent` erstellen. Der letzte Task in diesem Pfad muss auf dieses `endEvent` verweisen. Gib dem `endEvent` eine passende Beschreibung im `label`-Feld (z.B. "Prozess beendet" oder "Material bereitgestellt"). Ein Prozess ist erst vollständig, wenn ALLE offenen Pfade in einem `endEvent` münden.
    5.  **Gateways:** Ein 'exclusiveGateway' muss mit einer prägnanten Frage beschriftet sein. Leite diese aus den Aussagen des Nutzers ab, sodass eine kurze und prägnante Frage für die Bedingungen der ausgehenden Pfade entsteht.
    **DEINE AKTUELLE AUFGABE:**
    Analysiere die Konversation und das bisherige Wissen. Führe deine Kern-Anweisung aus und gib das VOLLSTÄNDIGE und AKTUALISIERTE (oder unveränderte) Prozesswissen-Objekt zurück.
    Bisheriges Wissen:
    {json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)}
    Konversationshistorie:
    {json.dumps(history_for_prompt, indent=2, ensure_ascii=False)}
    """
    print("\n--- Extraktions-Logik wird aufgerufen ---")
    updated_knowledge_raw = extraction_llm.invoke(extraction_prompt)
    
    updated_knowledge_clean = _enforce_merging_gateways(updated_knowledge_raw)

    print("--- Extraktion abgeschlossen, neues Wissen:", json.dumps(updated_knowledge_clean, indent=2, ensure_ascii=False))
    return updated_knowledge_clean

def _generate_interim_summary_logic(bisheriges_wissen: dict):
    """Erstellt eine prägnante Zwischenzusammenfassung für den Nutzer."""
    summary_prompt = f"""
    Du bist ein Prozessanalyst. Deine Aufgabe ist es, eine prägnante Zwischenbilanz des Prozesses zu geben.
    Beschreibe den Ablauf und erwähne explizit die Bedingungen auf den Pfaden nach einer Entscheidung (z.B. "Wenn die Bedingung 'Ja' erfüllt ist, ...").
    WICHTIG: Gib nur reinen Text ohne Markdown-Formatierungen zurück.
    Prozess-Struktur als Grundlage: {json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)}
    """
    print("\n--- Zwischenzusammenfassungs-Logik wird aufgerufen ---")
    prose_summary = llm.invoke(summary_prompt).content
    return {"prose_summary": prose_summary, "json_data": bisheriges_wissen}

def _generate_final_summary_logic(bisheriges_wissen: dict):
    """Erstellt eine formale, finale Prozessbeschreibung."""
    summary_prompt = f"""
    Du bist ein Principal Process Consultant. Deine Aufgabe ist es, eine formale, finale Prozessbeschreibung zu erstellen.
    Beschreibe den Ablauf schrittweise und erwähne explizit die Bedingungen auf den Pfaden nach einer Entscheidung (z.B. "Im Fall 'Ja (< 100€)' wird...").
    WICHTIG: Gib nur reinen Text ohne Markdown-Formatierungen zurück.
    Prozess-Struktur als Grundlage: {json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)}
    """
    print("\n--- Finale Zusammenfassungs-Logik wird aufgerufen ---")
    prose_summary = llm.invoke(summary_prompt).content
    final_prose = f"Vielen Dank für die Bestätigung. Hier ist die finale Zusammenfassung des Prozesses '{bisheriges_wissen.get('prozessname', '')}':\n\n{prose_summary}"
    return {"prose_summary": final_prose, "json_data": bisheriges_wissen}

# --- 2.2 Definition der Agenten-Werkzeuge ---
@tool
def update_wissensbasis():
    """Aktualisiert die Wissensdatenbank mit den neuesten Informationen aus dem Gespräch."""
    return "Wissensbasis wird aktualisiert."

@tool
def provide_interim_summary():
    """Erstellt eine prägnante Zwischenzusammenfassung des bisher erfassten Prozesses."""
    return "Zwischenzusammenfassung wird erstellt."

@tool
def propose_reset():
    """Schlägt dem Nutzer vor, die Konversation zurückzusetzen und wartet auf Bestätigung."""
    return "Bestätigung für Reset wird angefordert."

@tool
def create_final_summary():
    """Erstellt die finale, formale Prozessbeschreibung am Ende des Interviews."""
    return "Finale Zusammenfassung wird erstellt."


# --- 3. LangGraph-Setup mit erweiterter Tool-Logik ---
tools = [update_wissensbasis, provide_interim_summary, propose_reset, create_final_summary]
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState):
    print("\n--- Agenten-Knoten ---")
    print("Aktuelles Prozesswissen:", json.dumps(state['process_knowledge'], indent=2, ensure_ascii=False))
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}

def custom_tool_node(state: AgentState):
    print("\n--- Custom Tool-Knoten ---")
    tool_call = state['messages'][-1].tool_calls[0]
    tool_name = tool_call.get("name")
    tool_call_id = tool_call['id']

    if tool_name == "update_wissensbasis":
        updated_knowledge = _extract_and_update_knowledge_logic(state['messages'], state['process_knowledge'])
        tool_message = ToolMessage(content="Wissen erfolgreich aktualisiert.", tool_call_id=tool_call_id)
        return {"messages": [tool_message], "process_knowledge": updated_knowledge}
        
    elif tool_name == "propose_reset":
        confirmation_question = "Ich habe verstanden, dass Sie neu starten möchten. Soll ich den aktuellen Fortschritt wirklich verwerfen? Bitte antworten Sie mit 'Ja' oder 'Nein'."
        tool_message = ToolMessage(content=confirmation_question, tool_call_id=tool_call_id)
        return {"messages": [tool_message], "confirmation_pending": "reset"}
    
    elif tool_name in ["provide_interim_summary", "create_final_summary"]:
        knowledge_to_process = None 

        if tool_name == "provide_interim_summary":
            knowledge_to_process = state['process_knowledge']
            summary_data = _generate_interim_summary_logic(knowledge_to_process)
        else:
            print("--- Führe letztes Wissens-Update vor finaler Zusammenfassung aus ---")
            knowledge_to_process = _extract_and_update_knowledge_logic(
                state['messages'], 
                state['process_knowledge']
            )
            summary_data = _generate_final_summary_logic(knowledge_to_process)

        json_data = summary_data.get("json_data")
        upload_messages = []
        new_model_id = state.get("signavio_model_id")

        if importer and json_data and json_data.get("nodes"):
            try:
                print("Generiere BPMN XML aus JSON-Wissen...")
                generator = BPMNLayoutGenerator(json_data)
                bpmn_xml_string = generator.generate_bpmn_xml()
                with open("output.bpmn", "w", encoding="utf-8") as f:
                    f.write(bpmn_xml_string)
                print("BPMN XML erfolgreich in 'output.bpmn' gespeichert.")

                current_model_id = state.get("signavio_model_id")
                if current_model_id:
                    print(f"Versuche, altes Signavio-Modell zu löschen: {current_model_id}")
                    if importer.delete_model(current_model_id):
                        upload_messages.append(f"Alte Version des BPMN-Modells (ID: `{current_model_id}`) wurde in Signavio gelöscht.")
                    else:
                        upload_messages.append(f":warning: Konnte das alte BPMN-Modell (ID: `{current_model_id}`) nicht löschen.")

                diagram_name = json_data.get("prozessname", "Unbenannter Prozess")
                directory_id = "570c56290f95468c9fde64b84c79298b"
                print(f"Lade neues Modell '{diagram_name}' hoch...")
                upload_response = importer.import_bpmn_xml_from_string(bpmn_xml_string, directory_id, diagram_name)

                if upload_response and upload_response.get("createdIds"):
                    new_model_id = upload_response["createdIds"][0]
                    print(f"Upload erfolgreich. Neue Modell-ID: {new_model_id}")
                    upload_messages.append(f"Prozess wurde erfolgreich nach Signavio hochgeladen. Neue Modell-ID: `{new_model_id}`")
                else:
                    error_details = str(upload_response) if upload_response else "Keine Antwort vom Server."
                    print(f"Signavio-Upload fehlgeschlagen. Antwort: {error_details}")
                    upload_messages.append(f":x: Der Upload nach Signavio ist fehlgeschlagen. Details: `{error_details}`")

            except Exception as e:
                print(f"Ein schwerwiegender Fehler ist bei der BPMN-Verarbeitung aufgetreten: {e}")
                import traceback
                traceback.print_exc()
                upload_messages.append(f":x: Fehler bei BPMN-Erstellung/Upload: `{e}`")

        upload_messages.append("Hier ist der Prozess: https://editor.signavio.com/p/hub-preview/de_de/model/" + new_model_id)

        output_content = {
            "prose_summary": summary_data.get("prose_summary"),
            "json_data": json_data,
            "upload_messages": upload_messages
        }
        tool_message = ToolMessage(content=json.dumps(output_content), tool_call_id=tool_call_id)

        return {
            "messages": [tool_message],
            "process_knowledge": knowledge_to_process,
            "signavio_model_id": new_model_id
        }

def initial_router(state: AgentState) -> str:
    if state['messages'][-1].tool_calls:
        return "tools"
    return "end"

def after_tool_router(state: AgentState) -> str:
    last_ai_message = next(m for m in reversed(state['messages']) if isinstance(m, AIMessage))
    tool_name = last_ai_message.tool_calls[0].get("name")
    if tool_name in ["propose_reset", "create_final_summary", "provide_interim_summary"]:
        return "end"
    return "continue"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", custom_tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", initial_router, {"tools": "tools", "end": END})
workflow.add_conditional_edges("tools", after_tool_router, {"continue": "agent", "end": END})
agent_graph = workflow.compile()

# --- 4. System-Prompt und Konversations-Management ---
conversations = {}

SYSTEM_PROMPT = """
# 1. ROLLE & MISSION
Du bist 'Prozess-Bot', ein KI-gestützter Principal Process Consultant. Deine Antworten sollten klar, präzise und ohne Markdown-Formatierung sein.

# 2. GESPRÄCHSSTART (ONBOARDING)
Bei Beginn einer neuen Konversation, befolge exakt diese Schritte:
1.  **Vorstellung:** Stelle dich vor und erkläre kurz deine Aufgabe.
2.  **Reset-Funktion erklären:** Informiere den Nutzer, dass er jederzeit mit Stichworten wie "reset" oder "neu starten" die aktuelle Erfassung verwerfen und von vorne beginnen kann.
3.  **Initial-Fragen stellen:** Frage nach genau zwei Dingen, um den Prozess zu initialisieren:
    a) dem Titel des Prozesses.
    b) dem auslösenden Ereignis (dem Start-Event).
4.  **Auf Antwort warten:** Warte die Antwort des Nutzers auf diese beiden Fragen ab, bevor du mit der detaillierten Schritt-für-Schritt-Erfassung beginnst. Rufe `update_wissensbasis` auf, um diese initialen Informationen zu speichern.

# 3. METHODIK & DIALOGFÜHRUNG
- **Dialogsteuerung:** Führe den Nutzer aktiv durch das Interview. Deine Hauptaufgabe ist es, den Prozess lückenlos und ohne Annahmen zu erfassen.

- **SEQUENZIELLE PFAD-BEARBEITUNG (WICHTIGSTE REGEL):** Diese Regel hat Vorrang vor allen anderen und definiert, wie du den Prozess erforschst.
    - **Bei einfachen Schritten:** Frage nach jedem einzelnen Task oder Event immer "Was passiert danach?", es sei denn, der Nutzer hat das Ende des Prozesses explizit erwähnt.
    - **Bei Entscheidungen (Gateways):** Wenn eine Entscheidung mit mehreren Pfaden aufkommt, musst du die Pfade **strikt nacheinander** abarbeiten. Gehe wie folgt vor:
        1. Nimm den ersten Pfad, den der Nutzer beschreibt (z.B. "Fall A: < 100€").
        2. Verfolge **ausschließlich diesen einen Pfad**. Frage so lange "Was passiert danach?", bis dieser Pfad explizit in einem End-Event mündet.
        3. **Erst wenn dieser erste Pfad vollständig abgeschlossen ist**, sagst du: "Verstanden, der Pfad für 'Fall A' ist damit abgeschlossen. Kehren wir nun zur Entscheidung zurück: Was passiert im 'Fall B'?"
        4. Beginne dann die Erfassung des zweiten (und aller folgenden) Pfade auf die gleiche Weise.

- **Passive Formulierungen auflösen:** Wenn eine Aktion passiv beschrieben wird (z.B. "die Prüfung wird durchgeführt"), frage aktiv nach dem verantwortlichen Akteur ("Wer führt die Prüfung durch?").

- **Lane-Zuordnung:** Wenn du dir unsicher bist, in welcher Lane ein Element zu verordnen ist, dann frage den Nutzer explizit (vor allem bei Start-Events). Gehe nur mit dem Gespräch weiter, wenn du dir sicher bist, dass alle Elemente eindeutig zugeordnet werden können.

- **Intelligente Klärung von Verantwortlichkeiten:** Frage nur dann nach dem verantwortlichen Akteur für eine Lane, wenn diese Information aus der Antwort des Nutzers nicht klar hervorgeht.

- **Wissens-Aktualisierung:** Rufe nach jeder relevanten Nutzer-Antwort `update_wissensbasis()` auf.

- **Zusammenfassung:** Wenn der Nutzer nach einer Zusammenfassung fragt, rufe `provide_interim_summary()` auf.

- **Abschluss einleiten:** Wenn du glaubst, dass alle Pfade logisch in einem End-Event münden, frage den Nutzer proaktiv, ob der Prozess vollständig ist.

- **Finale Zusammenfassung:** Wenn der Nutzer bestätigt, dass der Prozess vollständig ist, rufe `create_final_summary()` auf.

- **Reset-Erkennung:** Wenn der Nutzer das Gespräch abbrechen oder neu starten möchte, rufe `propose_reset()` auf.
"""

# --- 5. Slack- und Flask-Routen für die Interaktion ---
def get_initial_state() -> AgentState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "process_knowledge": {"prozessname": "", "prozessziel": "", "akteure": [], "nodes": []},
        "confirmation_pending": None
    }

def _handle_confirmation_logic(current_state: AgentState, user_text_normalized: str, user_id: str, say) -> bool:
    """Behandelt ausstehende Bestätigungen. Gibt True zurück, wenn die Nachricht verarbeitet wurde."""
    if current_state.get("confirmation_pending") == "reset":
        if user_text_normalized == 'ja':
            print(f"Reset bestätigt von Nutzer {user_id}.")
            conversations[user_id] = get_initial_state()
            say("Verstanden. Die Konversation wurde zurückgesetzt.")
            return True
        elif user_text_normalized == 'nein':
            print(f"Reset abgelehnt von Nutzer {user_id}.")
            current_state["confirmation_pending"] = None
            current_state["messages"] = [m for m in current_state["messages"] if not isinstance(m, ToolMessage)]
            current_state["messages"].pop()
            say("Okay, wir machen an der alten Stelle weiter.")
            return True
        else:
            print("Reset-Bestätigung ignoriert. Fahre normal fort.")
            current_state["confirmation_pending"] = None
            current_state["messages"] = [m for m in current_state["messages"] if not isinstance(m, ToolMessage)]
            return False
    return False

def _handle_tool_output(last_message: BaseMessage, channel_id: str, say):
    """Prüft den Inhalt einer Tool-Nachricht und gibt ihn formatiert in Slack aus."""
    if not isinstance(last_message, ToolMessage):
        return False

    try:
        content = json.loads(last_message.content)
    except (json.JSONDecodeError, TypeError):
        say(last_message.content)
        return True

    if "prose_summary" in content and "json_data" in content:
        prose_summary = content.get("prose_summary")
        json_data = content.get("json_data")
        upload_messages = content.get("upload_messages", [])

        if prose_summary:
            say(prose_summary)

        if json_data:
            try:
                file_name = f"{json_data.get('prozessname', 'prozess')}_wissensstand.json".replace(" ", "_")
                slack_app.client.files_upload_v2(
                    channel=channel_id,
                    content=json.dumps(json_data, indent=2, ensure_ascii=False),
                    filename=file_name,
                    title="Interner Wissensspeicher (JSON)",
                    initial_comment="Hier ist die maschinenlesbare Repräsentation des erfassten Prozesses:"
                )
            except Exception as e:
                print(f"Fehler beim JSON-Datei-Upload: {e}")
                say("Ich konnte die JSON-Datei leider nicht hochladen.")
        
        if upload_messages:
            for msg in upload_messages:
                say(msg)

        return True
    return False

@slack_app.event("message")
def handle_all_messages(body, say):
    """Haupt-Event-Handler: Orchestriert die Nachrichtenverarbeitung."""
    try:
        user_id = body["event"]["user"]
        user_text_raw = body["event"]["text"]
        channel_id = body["event"]["channel"]
        
        if "bot_id" in body["event"]:
            return

        if user_id not in conversations:
            conversations[user_id] = get_initial_state()

        current_state = conversations[user_id]
        
        if _handle_confirmation_logic(current_state, user_text_raw.strip().lower(), user_id, say):
            return

        current_state["messages"].append(HumanMessage(content=user_text_raw))
        
        final_state = agent_graph.invoke(current_state)
        
        last_message = final_state['messages'][-1]

        if not _handle_tool_output(last_message, channel_id, say):
            say(last_message.content)
        
        conversations[user_id] = final_state
        
    except Exception as e:
        print(f"Fehler in handle_all_messages: {e}")
        say(f"Entschuldigung, ein interner Fehler ist aufgetreten: {e}")


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    print("Flask-Server mit finaler Logik wird gestartet...")
    flask_app.run(host='0.0.0.0', port=5001)