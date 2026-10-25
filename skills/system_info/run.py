import json, os, platform, sys
try:
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
        "cwd": os.getcwd(),
        "pid": os.getpid(),
    }
    # Disk usage for current drive
    try:
        import shutil
        total, used, free = shutil.disk_usage(os.path.splitdrive(os.getcwd())[0] or "/")
        info["disk_total_gb"] = round(total / (1024**3), 1)
        info["disk_used_gb"] = round(used / (1024**3), 1)
        info["disk_free_gb"] = round(free / (1024**3), 1)
    except Exception:
        pass
    print(json.dumps({"success": True, "data": info}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
