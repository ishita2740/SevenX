import json, os, shutil, sys
from datetime import datetime

data = json.loads(sys.stdin.read())
p = data["params"]
action = p.get("action", "").lower()
path = p.get("path", "")
content = p.get("content", "")
dest = p.get("destination", "")

def ok(d): print(json.dumps({"success": True, "data": d}))
def fail(e): print(json.dumps({"success": False, "error": e}))

try:
    if action == "read":
        if not os.path.isfile(path):
            fail(f"File not found: {path}")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read(50000)  # cap at 50KB
            ok({"content": text, "size": os.path.getsize(path), "lines": text.count("\n") + 1})

    elif action == "write":
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        ok({"written": path, "bytes": len(content.encode("utf-8"))})

    elif action == "append":
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)
        ok({"appended": path, "bytes": len(content.encode("utf-8"))})

    elif action == "copy":
        if not dest: fail("destination required for copy")
        elif os.path.isdir(path):
            shutil.copytree(path, dest)
            ok({"copied_dir": path, "to": dest})
        else:
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            shutil.copy2(path, dest)
            ok({"copied": path, "to": dest})

    elif action == "move":
        if not dest: fail("destination required for move")
        else:
            shutil.move(path, dest)
            ok({"moved": path, "to": dest})

    elif action == "delete":
        if os.path.isdir(path):
            shutil.rmtree(path)
            ok({"deleted_dir": path})
        elif os.path.isfile(path):
            os.remove(path)
            ok({"deleted": path})
        else:
            fail(f"Path not found: {path}")

    elif action == "info":
        if not os.path.exists(path):
            fail(f"Path not found: {path}")
        else:
            st = os.stat(path)
            ok({
                "path": os.path.abspath(path),
                "is_file": os.path.isfile(path),
                "is_dir": os.path.isdir(path),
                "size_bytes": st.st_size,
                "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "created": datetime.fromtimestamp(st.st_ctime).isoformat(timespec="seconds"),
            })

    elif action == "search":
        if not os.path.isdir(path):
            fail(f"Not a directory: {path}")
        elif not content:
            fail("content (search query) required")
        else:
            matches = []
            for root, _, files in os.walk(path):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                            for i, line in enumerate(fh, 1):
                                if content.lower() in line.lower():
                                    matches.append({"file": fp, "line": i, "text": line.strip()[:200]})
                                    if len(matches) >= 50:
                                        break
                    except (OSError, UnicodeDecodeError):
                        pass
                    if len(matches) >= 50:
                        break
            ok({"query": content, "matches": matches, "count": len(matches)})

    elif action == "tree":
        if not os.path.isdir(path):
            fail(f"Not a directory: {path}")
        else:
            entries = []
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and d != ".venv" and d != "node_modules"]
                depth = root.replace(path, "").count(os.sep)
                if depth > 3: continue
                for f in files:
                    entries.append(os.path.relpath(os.path.join(root, f), path))
                    if len(entries) >= 200: break
                if len(entries) >= 200: break
            ok({"root": path, "files": entries, "count": len(entries)})

    else:
        fail(f"Unknown action: {action}. Use: read, write, append, copy, move, delete, search, info, tree")

except Exception as e:
    fail(str(e))
