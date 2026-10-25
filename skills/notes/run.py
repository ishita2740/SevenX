import json, os, sqlite3, sys
from datetime import datetime

data = json.loads(sys.stdin.read())
p = data["params"]
action = p.get("action", "").lower()
text = p.get("text", "")
note_id = p.get("note_id")
tag = p.get("tag", "general")

DB = os.path.join(os.environ.get("RAGNAROK_SKILL_DIR", "."), "..", "..", "ragnarok_memory.db")

def ok(d): print(json.dumps({"success": True, "data": d}))
def fail(e): print(json.dumps({"success": False, "error": e}))

try:
    with sqlite3.connect(DB) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT NOT NULL,
            tag TEXT DEFAULT 'general',
            created_at TEXT NOT NULL
        )""")

        if action == "add":
            if not text: fail("text required")
            else:
                db.execute("INSERT INTO notes (text, tag, created_at) VALUES (?, ?, ?)",
                           (text, tag, datetime.now().isoformat(timespec="seconds")))
                ok({"saved": text, "tag": tag})

        elif action == "list":
            rows = db.execute(
                "SELECT id, text, tag, created_at FROM notes ORDER BY id DESC LIMIT 50"
            ).fetchall()
            ok({"notes": [{"id": r[0], "text": r[1], "tag": r[2], "created": r[3]} for r in rows],
                "count": len(rows)})

        elif action == "search":
            if not text: fail("text (search query) required")
            else:
                rows = db.execute(
                    "SELECT id, text, tag, created_at FROM notes WHERE text LIKE ? ORDER BY id DESC LIMIT 20",
                    (f"%{text}%",)
                ).fetchall()
                ok({"query": text, "results": [{"id": r[0], "text": r[1], "tag": r[2]} for r in rows]})

        elif action == "delete":
            if note_id is None: fail("note_id required")
            else:
                db.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
                ok({"deleted": note_id})

        elif action == "export":
            rows = db.execute("SELECT text, tag, created_at FROM notes ORDER BY id").fetchall()
            lines = [f"[{r[2]}] [{r[1]}] {r[0]}" for r in rows]
            ok({"exported": "\n".join(lines), "count": len(rows)})

        else:
            fail(f"Unknown action: {action}. Use: add, list, search, delete, export")

except Exception as e:
    fail(str(e))
