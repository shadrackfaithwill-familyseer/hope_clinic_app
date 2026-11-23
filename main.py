
import json
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"

SESSIONS_DIR = BASE_DIR / "sessions"

CACHE_DIR = BASE_DIR / "cache"
CACHE_FILE = CACHE_DIR / "response_cache.json"

CACHE_TTL = 24 * 60 * 60  # 24 hours

MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

# Automatically create required directories.
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. RESPONSE CONFIGURATION
# =========================================================

VALID_CATEGORIES = {
    "General",
    "Appointment",
    "Medical",
    "Emergency",
    "Other"
}

DEFAULT_RESPONSE = {
    "response": (
        "I'm sorry, I'm currently unable to process your "
        "request. Please contact clinic staff for assistance."
    ),
    "category": "Other",
    "requires_human": True
}

DEFAULT_SUMMARY = {
    "session_summary": "Summary generation failed.",
    "questions_and_answers": [],
    "key_topics": [],
    "safety_notes": []
}

EMERGENCY_KEYWORDS = [
    "emergency",
    "urgent",
    "unconscious",
    "not breathing",
    "difficulty breathing",
    "severe bleeding",
    "heavy bleeding",
    "chest pain",
    "seizure",
    "stroke",
    "overdose",
    "suicidal",
    "kill myself",
    "kujinyonga",
    "kupumua kwa shida",
    "kutokwa damu nyingi",
    "maumivu makali ya kifua"
]


# =========================================================
# 3. SYSTEM PROMPTS
# =========================================================

def build_system_prompt(knowledge_text):

    return f"""
You are the Hope Health Clinic AI Smart Assistant.

Your role is to assist patients and visitors with general
clinic information and basic healthcare guidance.

CLINIC INFORMATION:

{knowledge_text}

IMPORTANT RULES:

1. Be polite, professional, empathetic and helpful.
2. Never diagnose medical conditions.
3. Never prescribe medication or recommend specific dosages.
4. Do not replace professional medical advice.
5. Never invent clinic services, prices or opening hours.
6. If information is unavailable, say so honestly.
7. Encourage patients to consult qualified healthcare professionals.
8. For emergencies, advise immediate medical attention.
9. In Kenya, emergency numbers include 999 and 112.
10. Respond in the language used by the patient where possible.
11. You may communicate in English, Kiswahili and Sheng.
12. Do not claim that an appointment has been booked unless
    an authorized clinic booking system confirms it.
13. Escalate requests requiring staff intervention.

EMERGENCY SAFETY:

When a patient describes a possible life-threatening emergency,
prioritize immediate emergency assistance. Do not delay urgent
advice by asking unnecessary questions.

RESPONSE FORMAT:

Return valid JSON only.

Use exactly these fields:

{{
    "response": "Your helpful response",
    "category": "General, Appointment, Medical, Emergency or Other",
    "requires_human": false
}}

Set requires_human to true when the patient needs assistance
from a clinic staff member.
"""


SUMMARY_SYSTEM_PROMPT = """
You are a healthcare conversation summarization assistant.

Review the supplied patient-assistant conversation.

Create a short, factual summary.

IMPORTANT:
1. Do not invent information.
2. Do not introduce medical diagnoses.
3. Preserve important safety information.
4. Do not unnecessarily repeat personal information.
5. Distinguish patient statements from assistant guidance.

Return valid JSON only, using these fields:

{
    "session_summary": "Brief summary of the conversation",
    "questions_and_answers": [
        {
            "question": "Patient's question",
            "answer": "Assistant's answer"
        }
    ],
    "key_topics": ["Topic 1", "Topic 2"],
    "safety_notes": ["Important safety information"]
}
"""


# =========================================================
# 4. LOAD CLINIC KNOWLEDGE
# =========================================================

def load_knowledge():

    if not KNOWLEDGE_FILE.exists():

        print("\n[WARNING] knowledge.txt was not found.")

        return (
            "No additional clinic information is currently available."
        )

    try:

        knowledge = KNOWLEDGE_FILE.read_text(
            encoding="utf-8"
        ).strip()

        if not knowledge:

            print("\n[WARNING] knowledge.txt is empty.")

            return "No additional clinic information is available."

        return knowledge

    except OSError as error:

        print(f"\n[ERROR] Could not read clinic knowledge: {error}")

        return "No additional clinic information is available."


# =========================================================
# 5. INITIALIZE GEMINI CLIENT
# =========================================================

def create_gemini_client():

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:

        raise ValueError(
            "GEMINI_API_KEY is missing from your .env file."
        )

    return genai.Client(api_key=api_key)


# =========================================================
# 6. CONVERSATION CONTENT
# =========================================================

def make_content(role, message):

    return types.Content(
        role=role,
        parts=[
            types.Part.from_text(text=message)
        ]
    )


# =========================================================
# 7. JSON EXTRACTION
# =========================================================

def extract_json(response_text):

    if not response_text:

        raise ValueError("Empty JSON response received.")

    response_text = response_text.strip()

    # Remove Markdown code fences if present.
    if response_text.startswith("```"):

        lines = response_text.splitlines()

        if lines[0].startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        response_text = "\n".join(lines).strip()

    return json.loads(response_text)


# =========================================================
# 8. RESPONSE CACHE
# =========================================================

def load_cache():

    if not CACHE_FILE.exists():
        return {}

    try:

        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

        if isinstance(data, dict):
            return data

        return {}

    except (json.JSONDecodeError, OSError) as error:

        print(f"[CACHE WARNING] Could not read cache: {error}")

        return {}


def save_cache(cache_data):

    temp_file = CACHE_DIR / "response_cache.tmp"

    try:

        with open(
            temp_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                cache_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        os.replace(temp_file, CACHE_FILE)

    except OSError as error:

        print(f"[CACHE WARNING] Could not save cache: {error}")


def is_emergency_question(question):

    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in EMERGENCY_KEYWORDS
    )


def is_cacheable_question(question):

    question_lower = question.lower()

    # Never cache emergency or potentially patient-specific requests.
    if is_emergency_question(question):
        return False

    restricted_topics = [
        "my health",
        "my condition",
        "my treatment",
        "my appointment",
        "my name",
        "my symptoms",
        "my medication",
        "my results",
        "diagnosis",
        "prescription",
        "dosage",
        "medicine",
        "medication",
        "pain",
        "bleeding",
        "symptom",
        "pregnant",
        "pregnancy",
        "sick",
        "fever"
    ]

    if any(
        topic in question_lower
        for topic in restricted_topics
    ):
        return False

    # Only cache general clinic information.
    allowed_topics = [
        "opening hours",
        "working hours",
        "clinic hours",
        "what services",
        "services offered",
        "clinic location",
        "clinic address",
        "clinic phone",
        "contact details",
        "contact number",
        "clinic email",
        "where is the clinic",
        "where is your clinic",
        "what time do you open",
        "what time do you close"
    ]

    return any(
        topic in question_lower
        for topic in allowed_topics
    )


def get_cache_key(question, system_prompt):

    content = (
        MODEL_NAME
        + system_prompt
        + question.strip().lower()
    )

    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()


def get_cached_response(question, system_prompt):

    if not is_cacheable_question(question):
        return None

    cache_data = load_cache()

    cache_key = get_cache_key(
        question,
        system_prompt
    )

    cached_item = cache_data.get(cache_key)

    if not isinstance(cached_item, dict):
        return None

    saved_time = cached_item.get("timestamp")

    if not isinstance(saved_time, (int, float)):
        return None

    if time.time() - saved_time > CACHE_TTL:

        del cache_data[cache_key]

        save_cache(cache_data)

        print("[CACHE] Cached response expired.")

        return None

    cached_response = cached_item.get("response")

    if not validate_ai_response(cached_response):
        return None

    print("[CACHE] Cache HIT!")

    return cached_response


def store_cached_response(
    question,
    system_prompt,
    response
):

    if not is_cacheable_question(question):
        return

    if not validate_ai_response(response):
        return

    cache_data = load_cache()

    cache_key = get_cache_key(
        question,
        system_prompt
    )

    cache_data[cache_key] = {
        "timestamp": time.time(),
        "response": response
    }

    save_cache(cache_data)

    print("[CACHE] Response successfully stored.")


# =========================================================
# 9. GEMINI REQUEST
# =========================================================

def generate_ai_response(
    client,
    contents,
    system_instruction
):

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.4
            )
        )

        if not response.text:

            raise ValueError(
                "Gemini returned an empty response."
            )

        return extract_json(response.text)

    except Exception as error:

        print("\n[ERROR] Gemini request failed.")
        print(f"Details: {error}")

        return None


# =========================================================
# 10. VALIDATE AI RESPONSE
# =========================================================

def validate_ai_response(data):

    if not isinstance(data, dict):
        return False

    if not isinstance(data.get("response"), str):
        return False

    if not data["response"].strip():
        return False

    if data.get("category") not in VALID_CATEGORIES:
        return False

    if not isinstance(data.get("requires_human"), bool):
        return False

    return True


def validate_summary(summary):

    if not isinstance(summary, dict):
        return False

    if not isinstance(
        summary.get("session_summary"), str
    ):
        return False

    if not isinstance(
        summary.get("questions_and_answers"), list
    ):
        return False

    if not isinstance(summary.get("key_topics"), list):
        return False

    if not isinstance(summary.get("safety_notes"), list):
        return False

    return True


# =========================================================
# 11. SAVE PATIENT SESSION
# =========================================================

def save_session(history, summary):

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    filename = f"session_{timestamp}.json"

    filepath = SESSIONS_DIR / filename

    session_data = {
        "session_id": filename,
        "created_at": datetime.now().isoformat(),
        "conversation": [
            {
                "role": item.role,
                "message": "".join(
                    part.text or ""
                    for part in (item.parts or [])
                )
            }
            for item in history
        ],
        "summary": summary
    }

    try:

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                session_data,
                file,
                indent=4,
                ensure_ascii=False
            )

        print("\n[SUCCESS] Session saved successfully.")
        print(f"Saved as: {filename}")

    except OSError as error:

        print(
            f"\n[ERROR] Could not save patient session: {error}"
        )


# =========================================================
# 12. GENERATE SESSION SUMMARY
# =========================================================

def generate_session_summary(client, history):

    print("\n[INFO] Generating session summary...")

    summary_history = list(history)

    summary_history.append(
        make_content(
            "user",
            SUMMARY_SYSTEM_PROMPT
        )
    )

    summary = generate_ai_response(
        client,
        summary_history,
        (
            "You are a healthcare conversation summarizer. "
            "Return only the JSON summary requested by the user."
        )
    )

    if not validate_summary(summary):

        print("[WARNING] Summary could not be generated.")

        return DEFAULT_SUMMARY.copy()

    return summary


# =========================================================
# 13. NEW PATIENT SESSION
# =========================================================

def run_new_session(client, system_prompt):

    print("\n======================================")
    print("       NEW PATIENT SESSION")
    print("======================================")

    print("Type 'exit' to end the conversation.")
    print("Type 'cancel' to return to the main menu.\n")

    history = []

    while True:

        try:

            user_message = input("Patient: ").strip()

        except (KeyboardInterrupt, EOFError):

            print("\n[INFO] Session interrupted.")

            break

        if not user_message:

            print("[INFO] Please enter a message.")

            continue

        command = user_message.lower()

        if command in ("cancel", "exit"):

            print("\n[INFO] Closing patient session.")

            break

        # Add patient message to conversation history.
        history.append(
            make_content("user", user_message)
        )

        print("\n[INFO] Processing patient request...")

        # -------------------------------------------------
        # FIRST: CHECK FOR AN EMERGENCY
        # -------------------------------------------------

        if is_emergency_question(user_message):

            ai_data = {
                "response": (
                    "Your safety is important. If you or someone "
                    "near you may be experiencing a medical "
                    "emergency, seek immediate medical attention "
                    "or go to the nearest emergency department. "
                    "In Kenya, call 999 or 112 for emergency "
                    "assistance. Do not wait for an online response."
                ),
                "category": "Emergency",
                "requires_human": True
            }

        else:

            # -------------------------------------------------
            # SECOND: CHECK RESPONSE CACHE
            # -------------------------------------------------

            ai_data = get_cached_response(
                user_message,
                system_prompt
            )

            # -------------------------------------------------
            # THIRD: CONTACT GEMINI IF CACHE MISSES
            # -------------------------------------------------

            if ai_data is None:

                print("[INFO] Cache MISS. Contacting Gemini...")

                ai_data = generate_ai_response(
                    client,
                    history,
                    system_prompt
                )

                if validate_ai_response(ai_data):

                    store_cached_response(
                        user_message,
                        system_prompt,
                        ai_data
                    )

            else:

                print(
                    "[SUCCESS] Response retrieved from cache."
                )

        # -------------------------------------------------
        # FOURTH: VALIDATE RESPONSE
        # -------------------------------------------------

        if not validate_ai_response(ai_data):

            print("\nAssistant:")
            print(DEFAULT_RESPONSE["response"])

            print("[INFO] Human staff assistance recommended.")

            # Remove failed patient message from API history.
            history.pop()

            continue

        # -------------------------------------------------
        # FIFTH: DISPLAY RESPONSE
        # -------------------------------------------------

        assistant_message = ai_data["response"]

        print("\nAssistant:")
        print(assistant_message)

        print(f"\nCategory: {ai_data['category']}")

        if ai_data["requires_human"]:

            print(
                "[INFO] Human staff assistance is recommended."
            )

        # -------------------------------------------------
        # SIXTH: SAVE ASSISTANT RESPONSE TO HISTORY
        # -------------------------------------------------

        history.append(
            make_content(
                "model",
                json.dumps(
                    ai_data,
                    ensure_ascii=False
                )
            )
        )

    # -----------------------------------------------------
    # SEVENTH: SAVE SESSION AND GENERATE SUMMARY
    # -----------------------------------------------------

    if history:

        summary = generate_session_summary(
            client,
            history
        )

        save_session(
            history,
            summary
        )

    print("\n[INFO] Patient session closed.")


# =========================================================
# 14. OPEN PAST PATIENT SESSION
# =========================================================

def open_past_session():

    print("\n======================================")
    print("        PAST PATIENT SESSIONS")
    print("======================================")

    session_files = sorted(
        SESSIONS_DIR.glob("session_*.json"),
        reverse=True
    )

    if not session_files:

        print("\n[INFO] No saved sessions found.")

        return

    for index, filepath in enumerate(
        session_files,
        start=1
    ):

        print(f"{index}. {filepath.name}")

    print("\n0. Return to main menu")

    choice = input("\nSelect a session: ").strip()

    if choice == "0":
        return

    if not choice.isdigit():

        print("\n[ERROR] Please enter a valid number.")

        return

    selection = int(choice)

    if not 1 <= selection <= len(session_files):

        print("\n[ERROR] Invalid session selection.")

        return

    selected_file = session_files[selection - 1]

    try:

        with open(
            selected_file,
            "r",
            encoding="utf-8"
        ) as file:

            session_data = json.load(file)

        print("\n======================================")
        print("           SESSION DETAILS")
        print("======================================")

        print(f"Session: {selected_file.name}")

        print(
            f"Date: {session_data.get('created_at', 'Unknown')}"
        )

        print("\n--- CONVERSATION ---\n")

        conversation = session_data.get(
            "conversation",
            []
        )

        for message in conversation:

            if not isinstance(message, dict):
                continue

            role = message.get("role", "unknown")

            content = message.get("message", "")

            if role == "user":

                print(f"Patient: {content}\n")

            elif role == "model":

                try:

                    parsed = json.loads(content)

                    if isinstance(parsed, dict):

                        content = parsed.get(
                            "response",
                            content
                        )

                except (json.JSONDecodeError, TypeError):

                    pass

                print(f"Assistant: {content}\n")

        # Display session summary.
        print("\n--- SESSION SUMMARY ---\n")

        summary = session_data.get("summary", {})

        if isinstance(summary, dict):

            print(
                summary.get(
                    "session_summary",
                    "No summary available."
                )
            )

            print("\nKey topics:")

            for topic in summary.get("key_topics", []):

                print(f"- {topic}")

            print("\nSafety notes:")

            for note in summary.get("safety_notes", []):

                print(f"- {note}")

        input(
            "\nPress Enter to return to the main menu..."
        )

    except (OSError, json.JSONDecodeError) as error:

        print(
            f"\n[ERROR] Could not open session: {error}"
        )


# =========================================================
# 15. MAIN MENU
# =========================================================

def main():

    print("\n======================================")
    print(" HOPE HEALTH CLINIC AI SMART ASSISTANT")
    print("======================================")

    knowledge_text = load_knowledge()

    system_prompt = build_system_prompt(
        knowledge_text
    )

    client = None

    while True:

        print("\n1. Start a new session")
        print("2. Open a past session")
        print("3. Exit")

        try:

            choice = input(
                "\nChoose an option (1-3): "
            ).strip()

        except (KeyboardInterrupt, EOFError):

            print("\n[INFO] Application stopped.")

            break

        if choice == "1":

            print("\n[INFO] Preparing a new session...")

            try:

                if client is None:

                    print(
                        "[INFO] Initializing Gemini client..."
                    )

                    client = create_gemini_client()

                    print(
                        "[SUCCESS] Gemini client initialized."
                    )

                run_new_session(
                    client,
                    system_prompt
                )

            except Exception as error:

                print(
                    "\n[ERROR] Could not start the session."
                )

                print(f"Details: {error}")

        elif choice == "2":

            print("\n[INFO] Opening saved sessions...")

            open_past_session()

        elif choice == "3":

            print(
                "\nThank you for using Hope Health Clinic AI Assistant."
            )

            break

        else:

            print(
                "\n[ERROR] Invalid option. Select 1, 2 or 3."
            )


# =========================================================
# 16. PROGRAM ENTRY POINT
# =========================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\n\n[INFO] Application stopped by user.")

    except Exception as error:

        print(f"\n[FATAL ERROR] {error}")