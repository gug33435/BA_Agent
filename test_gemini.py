import os
import google.generativeai as genai
from dotenv import load_dotenv

# Lädt die Umgebungsvariablen aus der .env-Datei
load_dotenv()

# Liest den Google API Schlüssel aus
google_api_key = os.getenv("GOOGLE_API_KEY")

# Überprüft, ob der Schlüssel gefunden wurde
if not google_api_key:
    print("FEHLER: GOOGLE_API_KEY nicht in der .env-Datei gefunden!")
else:
    print("API-Schlüssel gefunden. Konfiguriere Gemini...")
    
    try:
        # Konfiguriert die API mit Ihrem Schlüssel
        genai.configure(api_key=google_api_key)
        
        # Initialisiert das Gemini Pro Modell
        model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Stellt eine einfache Test-Frage
        prompt = "Was ist der Hauptzweck von Geschäftsprozessmanagement in einem Satz?"
        print(f"\nSende Test-Frage an Gemini: '{prompt}'")
        
        # Sendet den Prompt an das Modell und erhält die Antwort
        response = model.generate_content(prompt)
        
        # Gibt die Antwort der KI aus
        print("\nAntwort von Gemini:")
        print(response.text)
        print("\nVerbindung zur Gemini API erfolgreich!")

    except Exception as e:
        print(f"\nEin Fehler ist aufgetreten: {e}")