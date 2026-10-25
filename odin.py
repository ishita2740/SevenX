"""
odin.py - Ragnarok FR3 (Autonomous Workflow Orchestration)

FR3.1: Decompose high-level goals into actionable sub-tasks (DAG).
FR3.2: Interact with user-approved applications via skills.
FR3.3: Monitor output, evaluate, and retry/adjust on failure.

Odin:
    - Takes a user goal and available skills
    - Asks the LLM to produce a step-by-step plan (JSON)
    - Builds a TaskGraph (DAG of TaskNodes)
    - Executes each node via the Forge
    - Reflects on results and can retry failed steps
    - Checkpoints state for crash recovery (NFR3.2)
"""

import json
import re
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from forge import Forge
from skill_schema import SkillResult


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class TaskNode:
    """One step in an execution graph."""

    def __init__(self, id, skill_name, params=None, depends_on=None,
                 description=""):
        self.id = id
        self.skill_name = skill_name
        self.params = params or {}
        self.depends_on = depends_on or []
        self.description = description
        self.status = "pending"   # pending | running | success | failed | skipped
        self.result = None        # SkillResult
        self.retries = 0
        self.max_retries = 2
        self.error = None

    def to_dict(self):
        return {
            "id": self.id,
            "skill": self.skill_name,
            "params": self.params,
            "depends_on": self.depends_on,
            "description": self.description,
            "status": self.status,
            "retries": self.retries,
            "error": self.error,
        }


class TaskGraph:
    """A DAG of TaskNodes representing an execution plan."""

    def __init__(self, goal, nodes=None):
        self.id = uuid.uuid4().hex[:12]
        self.goal = goal
        self.nodes = {n.id: n for n in (nodes or [])}
        self.status = "planning"  # planning | executing | completed | failed
        self.created_at = datetime.now().isoformat(timespec="seconds")
        self.completed_at = None
        self.reflection = None

    def add_node(self, node):
        self.nodes[node.id] = node

    def ready_nodes(self):
        """Nodes whose dependencies are all satisfied (success)."""
        ready = []
        for node in self.nodes.values():
            if node.status != "pending":
                continue
            deps_met = all(
                self.nodes[d].status == "success"
                for d in node.depends_on
                if d in self.nodes
            )
            if deps_met:
                ready.append(node)
        return ready

    @property
    def is_complete(self):
        return all(n.status in ("success", "skipped") for n in self.nodes.values())

    @property
    def has_failure(self):
        return any(n.status == "failed" for n in self.nodes.values())

    def summary(self):
        total = len(self.nodes)
        done = sum(1 for n in self.nodes.values() if n.status == "success")
        failed = sum(1 for n in self.nodes.values() if n.status == "failed")
        return (
            f"Goal: {self.goal}\n"
            f"Status: {self.status} | Steps: {done}/{total} done"
            + (f", {failed} failed" if failed else "")
        )

    def format_plan(self):
        lines = [f"\n--- Execution Plan: {self.goal} ---"]
        lines.append(f"Graph ID: {self.id}")
        lines.append(f"Status: {self.status}\n")
        for node in self.nodes.values():
            status_icon = {
                "pending": "[ ]", "running": "[~]", "success": "[+]",
                "failed": "[X]", "skipped": "[-]",
            }.get(node.status, "[?]")
            deps = f" (after: {', '.join(node.depends_on)})" if node.depends_on else ""
            lines.append(
                f"  {status_icon} {node.id}: {node.skill_name}"
                f"{deps}"
            )
            if node.description:
                lines.append(f"        {node.description}")
            if node.error:
                lines.append(f"        Error: {node.error}")
        return "\n".join(lines)

    def to_dict(self):
        return {
            "id": self.id,
            "goal": self.goal,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "reflection": self.reflection,
        }


# ---------------------------------------------------------------------------
# Planner: LLM produces a step-by-step plan
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """You are a task planner for Ragnarok, a local AI assistant.

Given a user goal and a list of available skills, create a step-by-step plan.

AVAILABLE SKILLS:
{skills_text}

RULES:
1. Each step must use exactly one skill from the list above.
2. Steps can depend on previous steps (use the step id).
3. Use realistic parameter values based on the user's goal.
4. If a step needs output from a previous step, set the parameter value to "{{step_id.field}}" as a placeholder.
5. Keep plans minimal - use the fewest steps necessary.
6. If the goal cannot be accomplished with available skills, return an empty plan.

Output ONLY valid JSON:
{{
  "steps": [
    {{
      "id": "step_1",
      "skill": "skill_name",
      "params": {{"key": "value"}},
      "depends_on": [],
      "description": "What this step does"
    }}
  ],
  "feasible": true,
  "explanation": "Brief explanation"
}}"""


def _skills_description(forge):
    """Build a text description of available skills for the planner."""
    lines = []
    for skill in forge.list_skills():
        params = []
        for p in skill.parameters:
            req = "required" if p.required else f"optional, default={p.default}"
            params.append(f"    {p.name} ({p.type}, {req}): {p.description}")
        param_text = "\n".join(params) if params else "    (no parameters)"
        lines.append(
            f"- {skill.name}: {skill.description}\n"
            f"  Parameters:\n{param_text}"
        )
    return "\n".join(lines) or "(No skills available)"


def plan(goal, forge, model="qwen2.5:7b",
         base_url="http://localhost:11434"):
    """Ask the LLM to plan a sequence of skill calls for the goal.

    Returns a TaskGraph. Falls back to a simple single-step plan
    if the LLM fails or produces invalid JSON.
    """
    skills_text = _skills_description(forge)
    system = PLAN_SYSTEM.replace("{skills_text}", skills_text)

    try:
        response = requests.post(
            f"{base_url}/api/chat",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"Goal: {goal}"},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 500},
            },
            timeout=60,
        )
        response.raise_for_status()
        raw = response.json()["message"]["content"].strip()
        return _parse_plan(goal, raw, forge)
    except Exception as e:
        # Fallback: try to match a single skill from the goal
        return _fallback_plan(goal, forge, str(e))


def _parse_plan(goal, raw_json, forge):
    """Parse the LLM's JSON plan into a TaskGraph."""
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown fences
        match = re.search(r"\{.*\}", raw_json, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return _fallback_plan(goal, forge, "Invalid JSON from planner")
        else:
            return _fallback_plan(goal, forge, "No JSON in planner output")

    if not data.get("feasible", True):
        graph = TaskGraph(goal)
        graph.status = "failed"
        graph.reflection = data.get("explanation", "Goal not feasible with available skills.")
        return graph

    steps = data.get("steps", [])
    if not steps:
        return _fallback_plan(goal, forge, "Planner returned empty steps")

    graph = TaskGraph(goal)
    valid_ids = set()

    for step in steps[:10]:  # Cap at 10 steps
        step_id = str(step.get("id", f"step_{len(valid_ids)+1}"))
        skill_name = step.get("skill", "")
        params = step.get("params", {})
        depends = step.get("depends_on", [])
        desc = step.get("description", "")

        # Validate skill exists
        if not forge.get(skill_name):
            continue

        # Validate dependencies exist
        depends = [d for d in depends if d in valid_ids]

        node = TaskNode(
            id=step_id,
            skill_name=skill_name,
            params=params,
            depends_on=depends,
            description=desc,
        )
        graph.add_node(node)
        valid_ids.add(step_id)

    if not graph.nodes:
        return _fallback_plan(goal, forge, "No valid steps after validation")

    graph.status = "ready"
    return graph


def _fallback_plan(goal, forge, reason=""):
    """Try to match a single skill from the goal text."""
    goal_lower = goal.lower()
    graph = TaskGraph(goal)

    for skill in forge.list_skills():
        if skill.name in goal_lower or any(
            w in goal_lower for w in skill.description.lower().split()
            if len(w) > 4
        ):
            node = TaskNode(
                id="step_1",
                skill_name=skill.name,
                params={},
                description=f"Auto-matched from goal (fallback: {reason})",
            )
            graph.add_node(node)
            graph.status = "ready"
            return graph

    graph.status = "failed"
    graph.reflection = (
        f"Could not create a plan: {reason}. "
        "Available skills may not match the goal."
    )
    return graph


# ---------------------------------------------------------------------------
# Executor: runs the DAG
# ---------------------------------------------------------------------------

def _resolve_refs(params, results):
    """Replace {{step_id.field}} references with actual values."""
    resolved = {}
    ref_pattern = re.compile(r"\{\{(\w+)\.(\w+)\}\}")

    for key, value in params.items():
        if isinstance(value, str):
            match = ref_pattern.match(value)
            if match:
                step_id, field = match.groups()
                prev = results.get(step_id)
                if prev and prev.success and isinstance(prev.data, dict):
                    resolved[key] = prev.data.get(field, value)
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        else:
            resolved[key] = value
    return resolved


def execute_graph(graph, forge, approval_callback=None,
                  on_step=None):
    """Execute all nodes in the TaskGraph in dependency order.

    Args:
        graph: TaskGraph to execute.
        forge: Forge instance for skill execution.
        approval_callback: Optional approval function for risky skills.
        on_step: Optional callback(node, result) called after each step.

    Returns:
        The completed TaskGraph.
    """
    if graph.status == "failed":
        return graph

    graph.status = "executing"
    results = {}  # step_id -> SkillResult
    max_iterations = len(graph.nodes) * 3  # prevent infinite loops
    iteration = 0

    while not graph.is_complete and iteration < max_iterations:
        iteration += 1
        ready = graph.ready_nodes()

        if not ready:
            # No nodes ready but not complete = deadlock or all failed
            if graph.has_failure:
                # Skip nodes that depend on failed steps
                for node in graph.nodes.values():
                    if node.status == "pending":
                        for dep in node.depends_on:
                            if dep in graph.nodes and graph.nodes[dep].status == "failed":
                                node.status = "skipped"
                                node.error = f"Skipped: dependency '{dep}' failed."
                                break
                continue
            break

        for node in ready:
            node.status = "running"

            # Resolve references to previous step outputs
            resolved_params = _resolve_refs(node.params, results)

            # Execute
            result = forge.run(
                node.skill_name,
                resolved_params,
                approval_callback=approval_callback,
            )
            results[node.id] = result

            if result.success:
                node.status = "success"
                node.result = result
            elif node.retries < node.max_retries:
                # Retry
                node.retries += 1
                node.status = "pending"
                node.error = f"Retry {node.retries}: {result.error}"
            else:
                node.status = "failed"
                node.error = result.error
                node.result = result

            if on_step:
                on_step(node, result)

    # Final status
    if graph.is_complete:
        graph.status = "completed"
    else:
        graph.status = "failed"

    graph.completed_at = datetime.now().isoformat(timespec="seconds")
    return graph


# ---------------------------------------------------------------------------
# Checkpoint / recovery (NFR3.2)
# ---------------------------------------------------------------------------

def checkpoint(graph, db_file):
    """Save the execution graph state to SQLite for crash recovery."""
    with sqlite3.connect(db_file) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS task_graphs (
                id TEXT PRIMARY KEY,
                goal TEXT NOT NULL,
                graph_json TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT,
                completed_at TEXT
            )
        """)
        db.execute("""
            INSERT OR REPLACE INTO task_graphs
                (id, goal, graph_json, status, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            graph.id,
            graph.goal,
            json.dumps(graph.to_dict()),
            graph.status,
            graph.created_at,
            graph.completed_at,
        ))


def load_checkpoint(graph_id, db_file):
    """Load a saved graph from SQLite. Returns dict or None."""
    try:
        with sqlite3.connect(db_file) as db:
            row = db.execute(
                "SELECT graph_json FROM task_graphs WHERE id = ?",
                (graph_id,)
            ).fetchone()
        if row:
            return json.loads(row[0])
    except (sqlite3.Error, json.JSONDecodeError):
        pass
    return None


def recent_graphs(db_file, n=5):
    """Load the last N task graphs."""
    try:
        with sqlite3.connect(db_file) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS task_graphs (
                    id TEXT PRIMARY KEY,
                    goal TEXT NOT NULL,
                    graph_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT,
                    completed_at TEXT
                )
            """)
            rows = db.execute(
                "SELECT goal, status, created_at, id FROM task_graphs "
                "ORDER BY created_at DESC LIMIT ?",
                (n,)
            ).fetchall()
        return rows
    except sqlite3.Error:
        return []


# ---------------------------------------------------------------------------
# High-level API for ragnarok.py
# ---------------------------------------------------------------------------

class Odin:
    """The Orchestrator - plans and executes multi-step tasks."""

    def __init__(self, forge, db_file="ragnarok_memory.db",
                 model="qwen2.5:7b", base_url="http://localhost:11434"):
        self.forge = forge
        self.db_file = db_file
        self.model = model
        self.base_url = base_url
        self.current_graph = None

    def orchestrate(self, goal, approval_callback=None,
                    on_step=None):
        """Plan and execute a goal. Returns the completed TaskGraph."""
        # Plan
        graph = plan(
            goal, self.forge,
            model=self.model, base_url=self.base_url,
        )

        if graph.status == "failed":
            self.current_graph = graph
            return graph

        # Show plan to user
        print(graph.format_plan())

        # Checkpoint before execution
        checkpoint(graph, self.db_file)

        # Execute
        def step_callback(node, result):
            checkpoint(graph, self.db_file)
            if on_step:
                on_step(node, result)
            # Print progress
            status = "+" if result.success else "X"
            print(f"  [{status}] {node.id}: {result.summary()}")

        graph = execute_graph(
            graph, self.forge,
            approval_callback=approval_callback,
            on_step=step_callback,
        )

        # Final checkpoint
        checkpoint(graph, self.db_file)
        self.current_graph = graph
        return graph

    def recent_tasks(self, n=5):
        return recent_graphs(self.db_file, n)
