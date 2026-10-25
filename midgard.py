"""
midgard.py - Ragnarok Dashboard (Flask API + Web UI)

Serves a premium dark-theme web dashboard for Ragnarok with:
    - Real-time chat with the Ollama model
    - Memory explorer (episodic + semantic)
    - Skill registry viewer and executor
    - Task graph visualization
    - System stats

All API endpoints return JSON. The UI is a single-page app
served from dashboard/static/.
"""

import json
import os
import sys
import traceback
from datetime import datetime

from flask import Flask, jsonify, request, send_from_directory

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(__file__))

from episodic_memory import EpisodicMemory, normalize_text
from semantic_memory import SemanticMemory
from forge import Forge
from odin import Odin, TaskGraph, TaskNode, execute_graph
from huginn import classify_rules
from heimdall import Heimdall

import requests as http_requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DB_FILE = os.path.join(os.path.dirname(__file__), "ragnarok_memory.db")
MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
STATIC_DIR = os.path.join(os.path.dirname(__file__), "dashboard", "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# Global instances (initialized on first request)
_memory = None
_semantic = None
_forge = None
_odin = None
_heimdall = None


def _init():
    global _memory, _semantic, _forge, _odin
    if _memory is None:
        _memory = EpisodicMemory(DB_FILE)
        _semantic = SemanticMemory(DB_FILE)
        _forge = Forge()
        _odin = Odin(_forge, DB_FILE)
        global _heimdall
        _heimdall = Heimdall(DB_FILE, _forge)
        _heimdall.start()


# ---------------------------------------------------------------------------
# UI routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------

@app.route("/api/chat", methods=["POST"])
def api_chat():
    _init()
    data = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        # Classify intent
        intent, confidence = classify_rules(user_input)

        # Try episodic recall first
        answer = _memory.answer_recall_question(user_input)
        source = "episodic"

        # Then semantic
        if answer is None:
            answer = _semantic.answer_question(user_input)
            source = "semantic"

        # Then LLM
        if answer is None:
            facts = []
            try:
                facts_raw = _semantic.all_facts()
                facts_text = "\n".join(f"- {f.text}" for f in facts_raw[:20])
            except Exception:
                facts_text = ""

            episodes_text = _memory.context_for_prompt(user_input)
            semantic_text = _semantic.context_for_prompt(user_input)

            system_content = (
                "You are Ragnarok, a personal AI assistant. "
                "Reply in natural, simple Hinglish written in Roman script. "
                "Use Hindi sentence structure and common English technical words."
            )
            if facts_text:
                system_content += f"\n\nSAVED USER FACTS:\n{facts_text}"
            if episodes_text:
                system_content += f"\n\nRECENT CONTEXT:\n{episodes_text}"
            if semantic_text:
                system_content += f"\n\nSEMANTIC MEMORY:\n{semantic_text}"

            messages = [
                {"role": "system", "content": system_content},
            ] + _memory.chat_context(8) + [
                {"role": "user", "content": user_input},
            ]

            resp = http_requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            source = "llm"

        # Log to episodic memory
        _memory.log_exchange(user_input, answer)

        # Auto-extract facts
        extracted = []
        try:
            for cand in _semantic.extract(user_input):
                if not _semantic.find_conflicts(cand):
                    _semantic.add(cand, replace_conflicts=False)
                    extracted.append(cand.text)
        except Exception:
            pass

        return jsonify({
            "answer": answer,
            "source": source,
            "intent": intent,
            "confidence": round(confidence, 2),
            "extracted_facts": extracted,
        })

    except http_requests.exceptions.ConnectionError:
        return jsonify({
            "answer": "Ollama se connect nahi ho paya. Check karo ki Ollama chal raha hai.",
            "source": "error",
            "intent": "error",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Memory APIs
# ---------------------------------------------------------------------------

@app.route("/api/memory/episodic")
def api_episodic():
    _init()
    n = request.args.get("n", 20, type=int)
    kind = request.args.get("kind")
    kinds = (kind,) if kind else None
    episodes = _memory.recent(min(n, 200), kinds)
    return jsonify([{
        "id": e.id,
        "kind": e.kind,
        "role": e.role,
        "text": e.content[:500],
        "timestamp": e.time,
    } for e in episodes])


@app.route("/api/memory/episodic/search")
def api_episodic_search():
    _init()
    q = request.args.get("q", "")
    if not q:
        return jsonify([])
    results = _memory.search(q)
    return jsonify([{
        "id": e.id,
        "kind": e.kind,
        "role": e.role,
        "text": e.content[:500],
        "timestamp": e.time,
    } for e in results[:50]])


@app.route("/api/memory/semantic")
def api_semantic():
    _init()
    facts = _semantic.all_facts()
    return jsonify([{
        "id": f.id,
        "kind": f.kind,
        "text": f.text,
        "subject": f.subject,
        "relation": f.relation,
        "object": f.obj,
        "importance": f.importance,
        "source": f.source,
    } for f in facts])


@app.route("/api/memory/semantic/search")
def api_semantic_search():
    _init()
    q = request.args.get("q", "")
    if not q:
        return jsonify([])
    hits = _semantic.retrieve(q)
    return jsonify([{
        "id": h.fact.id,
        "text": h.fact.text,
        "score": round(h.score, 3),
        "kind": h.fact.kind,
    } for h in hits])


@app.route("/api/memory/remember", methods=["POST"])
def api_remember():
    _init()
    data = request.get_json(force=True)
    fact = (data.get("fact") or "").strip()
    if not fact:
        return jsonify({"error": "Empty fact"}), 400
    results = _semantic.remember(fact)
    saved = any(r.status != "duplicate" for r in results)
    if saved:
        _memory.log_event(f"Fact saved: {fact}", role="system")
    return jsonify({"saved": saved, "fact": fact})


@app.route("/api/memory/forget", methods=["POST"])
def api_forget():
    _init()
    data = request.get_json(force=True)
    fact_id = data.get("id")
    if fact_id is None:
        return jsonify({"error": "No id"}), 400
    deleted = _semantic.delete(int(fact_id))
    return jsonify({"deleted": deleted is not None})


@app.route("/api/memory/stats")
def api_memory_stats():
    _init()
    ep = _memory.stats()
    sem = _semantic.stats()
    return jsonify({"episodic": ep, "semantic": sem})


# ---------------------------------------------------------------------------
# Skills API
# ---------------------------------------------------------------------------

@app.route("/api/skills")
def api_skills():
    _init()
    return jsonify([s.to_dict() for s in _forge.list_skills()])


@app.route("/api/skills/<name>")
def api_skill_info(name):
    _init()
    skill = _forge.get(name)
    if not skill:
        return jsonify({"error": f"Skill '{name}' not found"}), 404
    return jsonify(skill.to_dict())


@app.route("/api/skills/<name>/run", methods=["POST"])
def api_skill_run(name):
    _init()
    data = request.get_json(force=True)
    params = data.get("params", {})
    result = _forge.run(name, params)
    _memory.log_event(
        f"Skill executed via dashboard: {result.summary()}", role="system"
    )
    return jsonify(result.to_dict())


@app.route("/api/skills/log")
def api_skill_log():
    _init()
    n = request.args.get("n", 10, type=int)
    return jsonify([r.to_dict() for r in _forge.recent_executions(n)])


# ---------------------------------------------------------------------------
# Orchestration API
# ---------------------------------------------------------------------------

@app.route("/api/orchestrate", methods=["POST"])
def api_orchestrate():
    _init()
    data = request.get_json(force=True)
    goal = (data.get("goal") or "").strip()
    if not goal:
        return jsonify({"error": "Empty goal"}), 400

    graph = _odin.orchestrate(goal)
    _memory.log_event(
        f"Dashboard orchestration: {graph.summary()}", role="system"
    )
    return jsonify(graph.to_dict())


@app.route("/api/graphs")
def api_graphs():
    _init()
    n = request.args.get("n", 10, type=int)
    rows = _odin.recent_tasks(n)
    return jsonify([{
        "goal": r[0], "status": r[1], "created_at": r[2], "id": r[3]
    } for r in rows])



# ---------------------------------------------------------------------------
# Heimdall API
# ---------------------------------------------------------------------------

@app.route("/api/heimdall/status")
def api_heimdall_status():
    _init()
    return jsonify(_heimdall.status())


@app.route("/api/heimdall/events")
def api_heimdall_events():
    _init()
    n = request.args.get("n", 20, type=int)
    return jsonify(_heimdall.recent(n))


@app.route("/api/heimdall/watch", methods=["POST"])
def api_heimdall_watch():
    _init()
    data = request.get_json(force=True)
    name = data.get("name", "")
    path = data.get("path", "")
    patterns = data.get("patterns", ["*"])
    recursive = data.get("recursive", True)
    interval = data.get("interval", 5.0)
    if not name or not path:
        return jsonify({"error": "name and path required"}), 400
    _heimdall.watch(name, path, patterns, recursive, interval)
    return jsonify({"status": "watching", "name": name, "path": path})


@app.route("/api/heimdall/unwatch", methods=["POST"])
def api_heimdall_unwatch():
    _init()
    data = request.get_json(force=True)
    name = data.get("name", "")
    removed = _heimdall.unwatch(name)
    return jsonify({"removed": removed})


@app.route("/api/heimdall/schedule", methods=["POST"])
def api_heimdall_schedule():
    _init()
    data = request.get_json(force=True)
    name = data.get("name", "")
    schedule = data.get("schedule", "every 5m")
    skill = data.get("skill", "")
    params = data.get("params", {})
    if not name or not skill:
        return jsonify({"error": "name and skill required"}), 400
    _heimdall.schedule(name, schedule, skill_name=skill, skill_params=params)
    return jsonify({"status": "scheduled", "name": name})


@app.route("/api/heimdall/unschedule", methods=["POST"])
def api_heimdall_unschedule():
    _init()
    data = request.get_json(force=True)
    name = data.get("name", "")
    removed = _heimdall.unschedule(name)
    return jsonify({"removed": removed})

# ---------------------------------------------------------------------------
# System API
# ---------------------------------------------------------------------------

@app.route("/api/system")
def api_system():
    _init()
    import platform, shutil
    total, used, free = shutil.disk_usage(os.path.splitdrive(os.getcwd())[0] or "/")
    return jsonify({
        "platform": platform.system(),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "model": MODEL,
        "db_file": DB_FILE,
        "skills": _forge.skill_count(),
        "disk_free_gb": round(free / (1024**3), 1),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _init()
    print("=" * 50)
    print(" RAGNAROK MIDGARD DASHBOARD")
    print("=" * 50)
    print(f"  URL: http://localhost:5000")
    print(f"  Model: {MODEL}")
    print(f"  Skills: {_forge.skill_count()}")
    print("  Heimdall: Background agent active")
    print()
    app.run(host="0.0.0.0", port=5000, debug=False)


