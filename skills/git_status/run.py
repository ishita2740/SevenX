import json, os, subprocess, sys

data = json.loads(sys.stdin.read())
params = data["params"]
repo = params.get("repo_path", ".")
log_n = params.get("log_count", 5)

def git(args, cwd):
    try:
        r = subprocess.run(
            ["git"] + args, capture_output=True, text=True,
            timeout=10, cwd=cwd
        )
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return str(e), 1

try:
    if not os.path.isdir(repo):
        print(json.dumps({"success": False, "error": f"Not a directory: {repo}"}))
        sys.exit(0)

    branch, rc = git(["branch", "--show-current"], repo)
    if rc != 0:
        print(json.dumps({"success": False, "error": "Not a git repository"}))
        sys.exit(0)

    status, _ = git(["status", "--porcelain"], repo)
    log_out, _ = git(["log", f"--oneline", f"-{log_n}"], repo)
    diff_stat, _ = git(["diff", "--stat", "--cached"], repo)

    changes = []
    for line in status.splitlines():
        if line.strip():
            changes.append({"status": line[:2].strip(), "file": line[3:]})

    commits = []
    for line in log_out.splitlines():
        if line.strip():
            parts = line.split(" ", 1)
            commits.append({"hash": parts[0], "message": parts[1] if len(parts) > 1 else ""})

    print(json.dumps({"success": True, "data": {
        "branch": branch,
        "changes": changes,
        "change_count": len(changes),
        "recent_commits": commits,
        "staged_diff": diff_stat or "(nothing staged)",
    }}))
except Exception as e:
    print(json.dumps({"success": False, "error": str(e)}))
