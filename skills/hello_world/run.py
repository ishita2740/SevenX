import json, sys
data = json.loads(sys.stdin.read())
name = data["params"].get("name", "Ragnarok User")
print(json.dumps({
    "success": True,
    "data": {"greeting": f"Namaste {name}! Ragnarok skill framework active hai."}
}))
