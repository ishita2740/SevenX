import json, os, sys, subprocess

data = json.loads(sys.stdin.read())
p = data["params"]
action = p.get("action", "top").lower()
name = p.get("name", "")
limit = min(int(p.get("limit", 10)), 50)

def ps_cmd(cmd):
    r = subprocess.run(
        ["powershell", "-Command", cmd],
        capture_output=True, text=True, timeout=10
    )
    return r.stdout.strip()

try:
    if action == "top":
        out = ps_cmd(
            f"Get-Process | Sort-Object -Property WorkingSet64 -Descending | "
            f"Select-Object -First {limit} Name, Id, "
            f"@{{N='MemMB';E={{[math]::Round(.WorkingSet64/1MB,1)}}}}, "
            f"@{{N='CPU_s';E={{[math]::Round(.CPU,1)}}}} | "
            f"ConvertTo-Json"
        )
        procs = json.loads(out) if out else []
        if isinstance(procs, dict): procs = [procs]
        print(json.dumps({"success": True, "data": {"processes": procs, "count": len(procs)}}))

    elif action == "find":
        if not name:
            print(json.dumps({"success": False, "error": "name required for find"}))
        else:
            out = ps_cmd(
                f"Get-Process -Name '*{name}*' -ErrorAction SilentlyContinue | "
                f"Select-Object Name, Id, "
                f"@{{N='MemMB';E={{[math]::Round(.WorkingSet64/1MB,1)}}}}, "
                f"@{{N='CPU_s';E={{[math]::Round(.CPU,1)}}}} | "
                f"ConvertTo-Json"
            )
            procs = json.loads(out) if out else []
            if isinstance(procs, dict): procs = [procs]
            print(json.dumps({"success": True, "data": {"name": name, "processes": procs, "found": len(procs)}}))

    elif action == "count":
        out = ps_cmd("(Get-Process).Count")
        print(json.dumps({"success": True, "data": {"total_processes": int(out)}}))

    else:
        print(json.dumps({"success": False, "error": f"Unknown action: {action}"}))

except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
