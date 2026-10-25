import json, subprocess, sys

data = json.loads(sys.stdin.read())
p = data["params"]
action = p.get("action", "read").lower()

try:
    if action == "read":
        r = subprocess.run(
            ["powershell", "-Command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=5
        )
        text = r.stdout.strip()
        print(json.dumps({"success": True, "data": {
            "clipboard": text,
            "length": len(text),
            "lines": text.count("\n") + 1 if text else 0,
        }}))
    elif action == "write":
        text = p.get("text", "")
        subprocess.run(
            ["powershell", "-Command", f"Set-Clipboard -Value '{text}'"],
            capture_output=True, text=True, timeout=5
        )
        print(json.dumps({"success": True, "data": {"written": len(text), "preview": text[:100]}}))
    else:
        print(json.dumps({"success": False, "error": f"Unknown action: {action}. Use: read, write"}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
