
import requests
import sqlite3
from datetime import datetime

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
DB_FILE = "ragnarok_memory.db"

def connect_db():
    return sqlite3.connect(DB_FILE)

def setup_database():
    with connect_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)

def save_message(role, content):
    with connect_db() as db:
        db.execute(
            "INSERT INTO conversations (time, role, content) VALUES (?, ?, ?)",
            (datetime.now().isoformat(timespec="seconds"), role, content)
        )

def remember_fact(fact):
    with connect_db() as db:
        db.execute(
            "INSERT OR IGNORE INTO facts (fact, created_at) VALUES (?, ?)",
            (fact, datetime.now().isoformat(timespec="seconds"))
        )

def get_facts():
    with connect_db() as db:
        return db.execute(
            "SELECT id, fact, created_at FROM facts ORDER BY id"
        ).fetchall()

def show_history(limit=10):
    with connect_db() as db:
        rows = db.execute(
            "SELECT time, role, content FROM conversations "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    if not rows:
        print("Ragnarok: No saved conversations yet.")
        return

    print("\n--- Recent conversation history ---")
    for time, role, content in reversed(rows):
        print(f"[{time}] {role}: {content}")
    print()

def show_facts():
    facts = get_facts()
    if not facts:
        print("Ragnarok: Abhi koi saved fact nahi hai.")
        return

    print("\n--- Ragnarok's saved facts ---")
    for fact_id, fact, created_at in facts:
        print(f"{fact_id}. {fact} (saved {created_at})")
    print()

def delete_fact(fact_id):
    with connect_db() as db:
        cursor = db.execute(
            "DELETE FROM facts WHERE id = ?", (fact_id,)
        )
    if cursor.rowcount:
        print("Ragnarok: Fact deleted.")
    else:
        print("Ragnarok: Is ID ka koi fact nahi mila.")

def get_recent_messages(limit=12):
    with connect_db() as db:
        rows = db.execute(
            "SELECT role, content FROM conversations "
            "ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    rows.reverse()
    return [{"role": role, "content": content} for role, content in rows]

setup_database()

system_message = {
    "role": "system",
    "content": (
        "You are Ragnarok, a helpful local personal AI assistant. "
        "Use simple language. Saved facts are provided as context, "
        "but do not claim to remember anything that is not provided."
    )
}

print("=" * 45)
print(" RAGNAROK AI - SMART MEMORY")
print("=" * 45)
print("Model:", MODEL)
print("Commands:")
print("/remember <fact>  - save an important fact")
print("/memory           - view saved facts")
print("/forget <id>      - delete a saved fact by ID")
print("/history          - view recent conversations")
print("/exit             - quit")
print()

while True:
    try:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        lower = user_input.lower()

        if lower in ["/exit", "exit", "quit"]:
            print("Ragnarok: Goodbye!")
            break

        if lower == "/history":
            show_history()
            continue

        if lower == "/memory":
            show_facts()
            continue

        if lower.startswith("/remember "):
            fact = user_input[len("/remember "):].strip()
            if fact:
                remember_fact(fact)
                print("Ragnarok: Fact saved in local memory.\n")
            else:
                print("Usage: /remember <fact>\n")
            continue

        if lower.startswith("/forget "):
            try:
                fact_id = int(user_input.split(maxsplit=1)[1])
                delete_fact(fact_id)
            except (ValueError, IndexError):
                print("Usage: /forget <id>")
            print()
            continue

        save_message("user", user_input)

        facts = get_facts()
        facts_text = "\n".join(
            f"- {fact}" for _, fact, _ in facts
        ) or "(No saved facts yet)"

        messages = [
            system_message,
            {
                "role": "system",
                "content": "Saved user facts:\n" + facts_text
            }
        ] + get_recent_messages()

        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        answer = response.json()["message"]["content"]

        print(f"\nRagnarok: {answer}\n")
        save_message("assistant", answer)

    except requests.exceptions.RequestException as error:
        print("Ragnarok connection error:", error, "\n")
    except KeyboardInterrupt:
        print("\nRagnarok shutting down.")
        break
