import json, os, sys
data = json.loads(sys.stdin.read())
params = data["params"]
directory = params.get("directory", ".")
ext = params.get("extension")
try:
    if not os.path.isdir(directory):
        print(json.dumps({"success": False, "error": f"Directory not found: {directory}"}))
        sys.exit(0)
    entries = []
    for name in sorted(os.listdir(directory)):
        full = os.path.join(directory, name)
        if ext and not name.endswith(ext):
            continue
        info = {"name": name, "is_dir": os.path.isdir(full)}
        if not info["is_dir"]:
            info["size_bytes"] = os.path.getsize(full)
        entries.append(info)
    print(json.dumps({"success": True, "data": {"directory": os.path.abspath(directory), "count": len(entries), "files": entries}}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
