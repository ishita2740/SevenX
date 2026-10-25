"""
huginn.py - Ragnarok FR2.2 (Intent Classification)

FR2.2: Given a user prompt, the system must analyze the intent, select
the appropriate combination of registered skills, and generate the
necessary parameters for execution.

Huginn classifies user input into one of:
    QUERY    - simple Q&A, routed to RAG / model
    TASK     - actionable, routed to the Orchestrator (Odin)
    MEMORY   - store/recall/forget, routed to Muninn
    SYSTEM   - settings, skill management
    CHAT     - casual conversation

Classification uses the lightweight qwen2.5:3b model with structured
JSON output.  Falls back to rule-based if Ollama is unavailable.
"""

import json
import re
from typing import Optional

import requests

from episodic_memory import normalize_text


# ---------------------------------------------------------------------------
# Intent classes
# ---------------------------------------------------------------------------

INTENTS = ("QUERY", "TASK", "MEMORY", "SYSTEM", "CHAT")

TASK_INDICATORS = {
    # English
    "run", "execute", "do", "create", "make", "build", "generate",
    "send", "email", "schedule", "find", "list", "summarize",
    "delete", "remove", "move", "copy", "rename", "open", "close",
    "download", "upload", "install", "convert", "calculate",
    "check", "scan", "analyze", "compare", "merge", "sort",
    "show me", "get me", "fetch",
    # Hinglish
    "karo", "banao", "bhejo", "nikalo", "hatao", "kholo", "band",
    "dhundho", "dikha", "gino", "jodo", "todo", "mitao",
}

QUERY_INDICATORS = {
    "what", "why", "how", "explain", "kya", "kyun", "kaise",
    "define", "describe", "difference", "between", "matlab",
    "samjhao", "batao", "meaning",
}

MEMORY_INDICATORS = {
    "remember", "recall", "forget", "yaad", "save",
    "maine", "mene", "i did", "what did i",
}

SYSTEM_INDICATORS = {
    "setting", "settings", "config", "configure", "model",
    "skill", "skills", "help",
}

CHAT_INDICATORS = {
    "hello", "hi", "hey", "namaste", "thanks", "thank",
    "shukriya", "bye", "good morning", "good night",
    "haha", "lol", "ok", "okay", "theek", "accha",
    "how are you", "kaise ho", "sup",
}


# ---------------------------------------------------------------------------
# Rule-based classifier (fast, offline fallback)
# ---------------------------------------------------------------------------

def classify_rules(text):
    """Rule-based intent classification. Returns (intent, confidence)."""
    low = normalize_text(text)
    words = set(low.split())

    # Commands are SYSTEM
    if text.strip().startswith("/"):
        return "SYSTEM", 0.95

    # Check for task indicators (action words)
    task_score = sum(1 for w in TASK_INDICATORS if w in words or w in low)
    query_score = sum(1 for w in QUERY_INDICATORS if w in words)
    memory_score = sum(1 for w in MEMORY_INDICATORS if w in words or w in low)
    system_score = sum(1 for w in SYSTEM_INDICATORS if w in words)
    chat_score = sum(1 for w in CHAT_INDICATORS if w in words or w in low)

    scores = {
        "TASK": task_score * 2.0,
        "QUERY": query_score * 1.5,
        "MEMORY": memory_score * 2.0,
        "SYSTEM": system_score * 1.5,
        "CHAT": chat_score * 1.0,
    }

    # Multi-word phrase boosts
    if any(p in low for p in ("show me", "get me", "find me", "mujhe dikha")):
        scores["TASK"] += 3
    if any(p in low for p in ("maine kya kiya", "what did i do", "i did")):
        scores["MEMORY"] += 4
    if "?" in text and task_score == 0:
        scores["QUERY"] += 2

    best = max(scores, key=scores.get)
    total = sum(scores.values()) or 1
    confidence = scores[best] / total if total > 0 else 0.3

    # Default to CHAT if nothing matched well
    if scores[best] < 1:
        return "CHAT", 0.4

    return best, min(confidence, 0.95)


# ---------------------------------------------------------------------------
# LLM-based classifier (uses qwen2.5:3b for higher accuracy)
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """You are an intent classifier for a personal AI assistant.

Classify the user message into exactly ONE of these intents:
- TASK: The user wants something done (run a skill, create, list, send, etc.)
- QUERY: The user is asking a factual/knowledge question
- MEMORY: The user is asking about past events or wants to save/recall info
- SYSTEM: The user is asking about settings, skills, or system config
- CHAT: Casual conversation, greetings, small talk

Output ONLY a JSON object: {"intent": "...", "confidence": 0.0-1.0, "reason": "..."}
Do not output anything else."""


def classify_llm(text, model="qwen2.5:3b",
                 base_url="http://localhost:11434"):
    """LLM-based classification using local Ollama."""
    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": CLASSIFY_PROMPT},
                    {"role": "user", "content": text},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "num_predict": 80},
            },
            timeout=15,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
        data = json.loads(raw)
        intent = data.get("intent", "CHAT").upper()
        if intent not in INTENTS:
            intent = "CHAT"
        confidence = max(0, min(1, float(data.get("confidence", 0.5))))
        return intent, confidence
    except Exception:
        return None, 0


# ---------------------------------------------------------------------------
# Hybrid classifier
# ---------------------------------------------------------------------------

def classify(text, use_llm=False, model="qwen2.5:3b"):
    """Classify user intent. Returns (intent, confidence).

    If use_llm is True, tries LLM first and falls back to rules.
    Rules are always available (no network needed).
    """
    if use_llm:
        intent, confidence = classify_llm(text, model=model)
        if intent is not None and confidence >= 0.5:
            return intent, confidence

    return classify_rules(text)
