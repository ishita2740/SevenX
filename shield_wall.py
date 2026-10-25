"""
shield_wall.py - Ragnarok FR2.3 (Sandboxed Execution)

FR2.3: The system must execute skills in isolated, sandboxed environments
to prevent malicious or erroneous code from compromising the host system.

Skills communicate via JSON:
    stdin  <- {"params": {...}}
    stdout -> {"success": true, "data": ...} or {"success": false, "error": ...}

The sandbox enforces:
    - Subprocess isolation (separate process)
    - Timeout enforcement
    - Working directory restricted to skill's folder
    - Minimal environment variables
    - stdout/stderr capture
"""

import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, Optional

from skill_schema import SkillDefinition, SkillResult


# ---------------------------------------------------------------------------
# Sandbox executor
# ---------------------------------------------------------------------------

def _safe_env(skill):
    """Build a minimal environment for the sandboxed process."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "SYSTEMROOT": os.environ.get("SYSTEMROOT", r"C:\Windows"),
        "TEMP": os.environ.get("TEMP", ""),
        "TMP": os.environ.get("TMP", ""),
        "PYTHONIOENCODING": "utf-8",
        "RAGNAROK_SKILL": skill.name,
        "RAGNAROK_SKILL_DIR": skill.skill_dir,
    }
    # On Windows, some Python operations need USERPROFILE
    if "USERPROFILE" in os.environ:
        env["USERPROFILE"] = os.environ["USERPROFILE"]
    if "APPDATA" in os.environ:
        env["APPDATA"] = os.environ["APPDATA"]
    if "LOCALAPPDATA" in os.environ:
        env["LOCALAPPDATA"] = os.environ["LOCALAPPDATA"]
    return env


def _parse_output(stdout_text):
    """Parse JSON from the skill's stdout. Returns (data_dict, error)."""
    text = stdout_text.strip()
    if not text:
        return None, "Skill produced no output."

    # Find the last JSON object in stdout (skills may print debug before it)
    last_brace = text.rfind("}")
    if last_brace == -1:
        return {"raw_output": text}, None

    # Walk backwards to find the matching opening brace
    depth = 0
    start = last_brace
    for i in range(last_brace, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
        if depth == 0:
            start = i
            break

    try:
        data = json.loads(text[start:last_brace + 1])
        return data, None
    except json.JSONDecodeError:
        return {"raw_output": text}, None


def execute_skill(
    skill,
    params,
    python_exe=None,
    approval_callback=None,
):
    """Execute a skill in a sandboxed subprocess.

    Args:
        skill: SkillDefinition to execute.
        params: Dict of parameter values.
        python_exe: Path to Python interpreter (default: sys.executable).
        approval_callback: Optional callable(skill) -> bool for approval.
                          If None and skill requires approval, it is denied.

    Returns:
        SkillResult with the outcome.
    """
    # --- NFR4.3: Permission check ---
    if skill.requires_approval:
        if approval_callback is None:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error=f"Skill '{skill.name}' requires approval "
                      f"(risk: {skill.risk_level}) but no approval handler.",
                approved=False,
            )
        if not approval_callback(skill):
            return SkillResult(
                skill_name=skill.name,
                success=False,
                error="User denied execution.",
                approved=False,
            )

    # --- Validate parameters ---
    try:
        validated = skill.validate_params(params)
    except ValueError as e:
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=str(e),
        )

    # --- Prepare stdin payload ---
    payload = json.dumps({"params": validated}, ensure_ascii=False)
    python = python_exe or sys.executable

    # --- Execute in subprocess ---
    start = time.perf_counter()
    try:
        result = subprocess.run(
            [python, skill.entrypoint_path],
            input=payload,
            capture_output=True,
            text=True,
            timeout=skill.timeout_seconds,
            cwd=skill.skill_dir,
            env=_safe_env(skill),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse output
        data, parse_error = _parse_output(result.stdout)

        if result.returncode != 0:
            error_msg = result.stderr.strip() or parse_error or "Non-zero exit code."
            return SkillResult(
                skill_name=skill.name,
                success=False,
                data=data,
                error=error_msg,
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=elapsed_ms,
            )

        # Check if the skill's own JSON reports failure
        if isinstance(data, dict) and data.get("success") is False:
            return SkillResult(
                skill_name=skill.name,
                success=False,
                data=data.get("data"),
                error=data.get("error", "Skill reported failure."),
                exit_code=0,
                stdout=result.stdout,
                stderr=result.stderr,
                duration_ms=elapsed_ms,
            )

        # Success
        actual_data = data.get("data", data) if isinstance(data, dict) else data
        return SkillResult(
            skill_name=skill.name,
            success=True,
            data=actual_data,
            exit_code=0,
            stdout=result.stdout,
            stderr=result.stderr,
            duration_ms=elapsed_ms,
        )

    except subprocess.TimeoutExpired:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=f"Timed out after {skill.timeout_seconds}s.",
            duration_ms=elapsed_ms,
        )
    except FileNotFoundError:
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=f"Entrypoint not found: {skill.entrypoint_path}",
        )
    except OSError as e:
        return SkillResult(
            skill_name=skill.name,
            success=False,
            error=f"OS error: {e}",
        )
