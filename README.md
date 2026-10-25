# Ragnarok - Sovereign Personal AI Agent

A fully local, privacy-preserving AI assistant running on Ollama + SQLite. Zero cloud dependency.

## Architecture

| Module | Purpose |
|--------|---------|
| `ragnarok.py` | Main CLI (FR1+FR2+FR3+FR4) |
| `episodic_memory.py` | FR1.1 - Chronological memory log |
| `semantic_memory.py` | FR1.2 - Facts, embeddings, knowledge graph |
| `huginn.py` | FR2.2 - Intent classifier |
| `forge.py` | FR2.1 - Skill registry |
| `shield_wall.py` | FR2.3 - Sandboxed skill executor |
| `skill_schema.py` | FR2.1 - Skill data models |
| `odin.py` | FR3 - DAG orchestrator |
| `heimdall.py` | FR4 - File watchers, cron scheduler |
| `midgard.py` | Dashboard - Flask API + Web UI |

## Quick Start

`ash
# Prerequisites: Python 3.10+, Ollama with qwen2.5:3b
ollama pull qwen2.5:3b

# CLI mode
python ragnarok.py

# Dashboard mode (http://localhost:5000)
python midgard.py
`

## Skills (10)

calculator, clipboard, file_ops, git_status, hello_world,
list_files, notes, processes, summarize_text, system_info

## Stats

- **6,878 lines** of Python + HTML/CSS
- **10 modules** + **10 skills**
- **19 REST API endpoints**
- **241 tests passing**
- **0 bytes** sent to cloud
