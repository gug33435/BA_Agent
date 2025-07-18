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
# HINWEIS: Dieser Code erfordert aktuelle Versionen der Bibliotheken.
# Führen Sie 'pip install --upgrade langchain langgraph langchain-core langchain-google-genai' aus.
from langgraph.graph import StateGraph, END
import datetime

# --- 1. Konfiguration und Initialisierung ---
load_dotenv()
SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN")
SLACK_SIGNING_SECRET = os.environ.get("SLACK_SIGNING_SECRET")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=GOOGLE_API_KEY)

slack_app = App(token=SLACK_BOT_TOKEN, signing_secret=SLACK_SIGNING_SECRET)
flask_app = Flask(__name__)
handler = SlackRequestHandler(slack_app)


# --- 2. Definition der neuen, graphen-basierten Wissensstruktur ---

class ProcessNode(TypedDict):
    id: str
    type: str
    label: str
    next_nodes: List[str]

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

# --- 2.1 Logik-Funktionen für die Werkzeuge ---

def _extract_and_update_knowledge_logic(konversationshistorie: List[BaseMessage], bisheriges_wissen: dict):
    """Extrahiert Informationen und baut den Prozessgraphen auf."""
    history_for_prompt = [msg.model_dump() for msg in konversationshistorie]
    
    extraction_prompt = f"""
    Du bist ein Experte für Prozessmodellierung und Datenextraktion. Deine Aufgabe ist es, eine Konversation in eine BPMN-konforme, graphen-basierte JSON-Struktur zu übersetzen.

    **KERN-ANWEISUNG:**
    Deine Aufgabe ist es, das `bisheriges_wissen`-Objekt zu nehmen und es basierend auf der letzten Nutzernachricht zu **ergänzen oder zu modifizieren**.
    Wenn die letzte Nachricht des Nutzers keine neuen, relevanten Prozessinformationen enthält (z.B. nur "ja", "danke", "gerne"), dann gib das `bisheriges_wissen`-Objekt **exakt unverändert** zurück. Lösche niemals vorhandenes Wissen.

    **REGELN ZUR MODELLIERUNG:**
    1.  **Atomarität:** Zerlege Sätze in einzelne, atomare Prozessschritte. "Rechnung prüfen und freigeben" sind ZWEI getrennte Knoten.
    2.  **BPMN-Wording:**
        * **Aktivitäten (type: 'task'):** Formuliere als "Substantiv + Verb (Grundform)". Bsp: "Rechnung prüfen".
        * **Ereignisse (type: 'startEvent', 'endEvent'):** Formuliere als "Substantiv + Partizip Perfekt". Bsp: "Rechnung eingegangen".
        * **Gateways (type: 'exclusiveGateway'):** Formuliere als prägnante Frage. Bsp: "Rechnungssumme > 1000€?".
    3.  **Graphen-Logik:**
        * Jedes Element ist ein Knoten in der `nodes`-Liste mit einer neuen, eindeutigen `id`.
        * Verbinde die Knoten logisch über die `next_nodes`-Liste.

    **BEISPIEL FÜR EIN GATEWAY:**
    * **Letzte Nutzernachricht:** "Danach prüft die Buchhaltung die Abrechnung. Wenn die Summe unter 100 Euro liegt, wird sie sofort zur Zahlung freigegeben. Ansonsten muss sie zusätzlich vom Teamleiter genehmigt werden."
    * **Korrekte Modellierung:** Du musst einen 'task' ("Abrechnung prüfen") erstellen, gefolgt von einem 'exclusiveGateway' ("Summe < 100€?"). Dieses Gateway hat dann ZWEI Einträge in `next_nodes`, die auf die beiden unterschiedlichen Pfade verweisen.

    **DEINE AKTUELLE AUFGABE:**
    Analysiere die Konversationshistorie und das bisherige Wissen. Führe deine Kern-Anweisung aus und gib das VOLLSTÄNDIGE und AKTUALISIERTE (oder unveränderte) Prozesswissen-Objekt zurück.

    Bisheriges Wissen:
    {json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)}

    Konversationshistorie:
    {json.dumps(history_for_prompt, indent=2, ensure_ascii=False)}
    """
    print("\n--- Extraktions-Logik wird aufgerufen ---")
    updated_knowledge = extraction_llm.invoke(extraction_prompt)
    print("--- Extraktion abgeschlossen, neues Wissen:", json.dumps(updated_knowledge, indent=2, ensure_ascii=False))
    return updated_knowledge

# KORREKTUR: Die Funktion gibt jetzt einen kombinierten String zurück
def _generate_summary_logic(bisheriges_wissen: dict):
    """Erstellt eine textuelle Zusammenfassung UND einen formatierten JSON-Block."""
    summary_prompt = f"""
    Du bist ein Prozessanalyst. Deine Aufgabe ist es, aus der folgenden JSON-Graphenstruktur eine klare und verständliche Zusammenfassung des Geschäftsprozesses in Prosa zu schreiben.
    Gehe den Graphen schrittweise vom Start- bis zum Endknoten durch und beschreibe den Ablauf in chronologischer Reihenfolge.
    Erwähne auch die beteiligten Akteure und das übergeordnete Prozessziel.
    WICHTIG: Gib nur reinen Text ohne Markdown-Formatierungen (wie ** oder *) zurück.

    Prozess-Struktur:
    {json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)}
    """
    print("\n--- Zusammenfassungs-Logik wird aufgerufen ---")
    prose_summary = llm.invoke(summary_prompt).content
    print("--- Prosa-Zusammenfassung erstellt:", prose_summary)

    # Erstelle den formatierten JSON-Block für die Slack-Ausgabe
    json_string = json.dumps(bisheriges_wissen, indent=2, ensure_ascii=False)
    formatted_json_block = f"```json\n{json_string}\n```"

    # Kombiniere beide Teile für die finale Ausgabe
    final_output = (
        f"{prose_summary}\n\n"
        "--- Interner Wissensspeicher (JSON-Repräsentation) ---\n"
        f"{formatted_json_block}"
    )
    return final_output

# --- 2.2 Definition der Agenten-Werkzeuge ---

@tool
def update_wissensbasis():
    """Aktualisiert die Wissensdatenbank mit den neuesten Informationen aus dem Gespräch."""
    return "Wissensbasis wird aktualisiert."

@tool
def provide_interim_summary():
    """Erstellt eine textuelle Zusammenfassung des bisher erfassten Prozesses inklusive der internen JSON-Struktur."""
    return "Zusammenfassung wird erstellt."

@tool
def propose_reset():
    """Schlägt dem Nutzer vor, die Konversation zurückzusetzen und wartet auf Bestätigung."""
    return "Bestätigung für Reset wird angefordert."


# --- 3. LangGraph-Setup mit erweiterter Tool-Logik ---

tools = [update_wissensbasis, provide_interim_summary, propose_reset]
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
        print("Werkzeug 'update_wissensbasis' wird ausgeführt.")
        updated_knowledge = _extract_and_update_knowledge_logic(state['messages'], state['process_knowledge'])
        tool_message = ToolMessage(content="Wissen erfolgreich aktualisiert.", tool_call_id=tool_call_id)
        return {"messages": [tool_message], "process_knowledge": updated_knowledge}
    
    elif tool_name == "provide_interim_summary":
        print("Werkzeug 'provide_interim_summary' wird ausgeführt.")
        summary_text = _generate_summary_logic(state['process_knowledge'])
        tool_message = ToolMessage(content=summary_text, tool_call_id=tool_call_id)
        return {"messages": [tool_message]}
        
    elif tool_name == "propose_reset":
        print("Werkzeug 'propose_reset' wird ausgeführt.")
        confirmation_question = "Ich habe verstanden, dass Sie neu starten möchten. Soll ich den aktuellen Fortschritt wirklich verwerfen? Bitte antworten Sie mit 'Ja' oder 'Nein'."
        tool_message = ToolMessage(content=confirmation_question, tool_call_id=tool_call_id)
        return {
            "messages": [tool_message],
            "confirmation_pending": "reset"
        }

def router(state: AgentState) -> str:
    if state['messages'][-1].tool_calls:
        return "tools"
    return "end"

workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", custom_tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
workflow.add_edge('tools', 'agent')
agent_graph = workflow.compile()

# --- 4. System-Prompt und Konversations-Management ---
conversations = {}

SYSTEM_PROMPT = """
# 1. ROLLE & MISSION
Du bist 'Prozess-Bot', ein KI-gestützter Principal Process Consultant. Deine Antworten sollten klar, präzise und ohne Markdown-Formatierung (wie ** oder *) sein.

# 2. ZU ERHEBENDE KERN-INFORMATIONEN
Dein Ziel ist es, Informationen zu Prozessname, Ziel, Akteuren, Start, Aktivitäten, Entscheidungen und Ende zu sammeln.

# 3. METHODIK & GESPRÄCHSFÜHRUNG
- **Wissens-Aktualisierung:** Nach relevanten Nutzer-Antworten, rufe `update_wissensbasis()` auf.
- **Zusammenfassung:** Wenn der Nutzer nach einer Zusammenfassung fragt, rufe `provide_interim_summary()` auf.
- **Reset-Erkennung:** Wenn der Nutzer das Gespräch abbrechen oder neu starten möchte (z.B. "lass uns neu anfangen", "von vorne", "neuer Prozess"), rufe das Werkzeug `propose_reset()` auf, um eine Bestätigung anzufordern.
- **Kritisches Nachfragen:** Reagiere auf unklare oder unvollständige Antworten mit gezielten Nachfragen.
"""

# --- 5. Slack- und Flask-Routen für die Interaktion ---

def get_initial_state() -> AgentState:
    return {
        "messages": [SystemMessage(content=SYSTEM_PROMPT)],
        "process_knowledge": {"prozessname": "", "prozessziel": "", "akteure": [], "nodes": []},
        "confirmation_pending": None
    }

@slack_app.event("message")
def handle_all_messages(body, say):
    try:
        user_id = body["event"]["user"]
        user_text_raw = body["event"]["text"]
        
        if "bot_id" in body["event"]:
            return

        if user_id not in conversations:
            conversations[user_id] = get_initial_state()

        current_state = conversations[user_id]
        user_text_normalized = user_text_raw.strip().lower()

        if current_state.get("confirmation_pending") == "reset":
            if user_text_normalized == 'ja':
                print(f"Reset bestätigt von Nutzer {user_id}.")
                conversations[user_id] = get_initial_state()
                say("Verstanden. Die Konversation wurde zurückgesetzt. Senden Sie eine Nachricht, um ein neues Interview zu beginnen.")
                return
            elif user_text_normalized == 'nein':
                print(f"Reset abgelehnt von Nutzer {user_id}.")
                current_state["confirmation_pending"] = None
                current_state["messages"] = [m for m in current_state["messages"] if not isinstance(m, ToolMessage)]
                current_state["messages"].pop() 
                say("Okay, wir machen an der alten Stelle weiter.")
                return
            else:
                say("Ich habe das nicht verstanden. Bitte antworten Sie mit 'Ja' oder 'Nein'.")
                return

        current_state["messages"].append(HumanMessage(content=user_text_raw))
        
        final_state = agent_graph.invoke(current_state)
        
        ai_response = final_state['messages'][-1].content
        say(ai_response)
        
        conversations[user_id] = final_state
        
    except Exception as e:
        print(f"Fehler in handle_all_messages: {e}")
        say(f"Entschuldigung, ein interner Fehler ist aufgetreten: {e}")


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)

if __name__ == "__main__":
    print("Flask-Server mit robuster Extraktions-Logik wird gestartet...")
    flask_app.run(host='0.0.0.0', port=5001)