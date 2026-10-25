"""
skill_schema.py - Ragnarok FR2.1 (Skill Registration)

FR2.1: The system must allow users to define, register, and manage
reusable "skills" via a standardized interface.

Every skill lives in its own folder under skills/ with:
    skill.yaml   - metadata, parameters, risk level
    run.py       - entrypoint (receives JSON on stdin, writes JSON to stdout)

This module provides:
    SkillParam       parameter definition
    SkillDefinition  parsed + validated skill.yaml
    SkillResult      execution outcome
    load_skill()     read and validate a skill directory
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class SkillParam:
    """One parameter in a skill's interface."""

    __slots__ = ("name", "type", "required", "default", "description")
    TYPES = ("string", "int", "float", "bool", "path", "json")

    def __init__(self, name, type_="string", required=True,
                 default=None, description=""):
        if not re.match(r"^[a-z][a-z0-9_]{0,39}$", name):
            raise ValueError(
                f"Parameter name '{name}' invalid - "
                "lowercase letters, digits, underscores, max 40 chars."
            )
        type_ = type_.lower()
        if type_ not in self.TYPES:
            raise ValueError(
                f"Parameter type '{type_}' unknown - "
                f"allowed: {', '.join(self.TYPES)}"
            )
        self.name = name
        self.type = type_
        self.required = required
        self.default = default
        self.description = description

    def validate(self, value):
        if value is None:
            if self.required and self.default is None:
                raise ValueError(f"Parameter '{self.name}' is required.")
            return self.default
        try:
            if self.type == "string":
                return str(value)
            elif self.type == "int":
                return int(value)
            elif self.type == "float":
                return float(value)
            elif self.type == "bool":
                if isinstance(value, str):
                    return value.lower() in ("true", "1", "yes", "haan")
                return bool(value)
            elif self.type == "path":
                path = str(value)
                if not os.path.exists(path):
                    raise ValueError(f"Path does not exist: {path}")
                return os.path.abspath(path)
            elif self.type == "json":
                return value
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Parameter '{self.name}' ({self.type}): {e}"
            ) from None
        return value

    def to_dict(self):
        d = {"name": self.name, "type": self.type, "required": self.required}
        if self.description:
            d["description"] = self.description
        if self.default is not None:
            d["default"] = self.default
        return d


RISK_LEVELS = ("low", "medium", "high", "critical")


class SkillDefinition:
    """Parsed and validated skill.yaml."""

    def __init__(self, name, version="1.0", description="", author="user",
                 risk_level="low", timeout_seconds=30, parameters=None,
                 entrypoint="run.py", dependencies=None, skill_dir="",
                 allowed_paths=None, network_access=False):
        if not re.match(r"^[a-z][a-z0-9_]{1,49}$", name):
            raise ValueError(
                f"Skill name '{name}' invalid - lowercase, "
                "2-50 chars, letters/digits/underscores."
            )
        if risk_level not in RISK_LEVELS:
            raise ValueError(
                f"Risk level '{risk_level}' unknown - "
                f"allowed: {', '.join(RISK_LEVELS)}"
            )
        if timeout_seconds < 1 or timeout_seconds > 600:
            raise ValueError("Timeout must be 1-600 seconds.")

        self.name = name
        self.version = str(version)
        self.description = description
        self.author = author
        self.risk_level = risk_level
        self.timeout_seconds = timeout_seconds
        self.parameters = parameters or []
        self.entrypoint = entrypoint
        self.dependencies = dependencies or []
        self.skill_dir = skill_dir
        self.allowed_paths = allowed_paths or []
        self.network_access = network_access
        self.registered_at = None

    @property
    def entrypoint_path(self):
        return os.path.join(self.skill_dir, self.entrypoint)

    @property
    def requires_approval(self):
        return self.risk_level in ("medium", "high", "critical")

    def validate_params(self, params):
        validated = {}
        known_names = {p.name for p in self.parameters}
        for key in params:
            if key not in known_names:
                raise ValueError(
                    f"Unknown parameter '{key}' for skill '{self.name}'."
                )
        for param in self.parameters:
            value = params.get(param.name)
            validated[param.name] = param.validate(value)
        return validated

    def to_dict(self):
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "risk_level": self.risk_level,
            "timeout_seconds": self.timeout_seconds,
            "parameters": [p.to_dict() for p in self.parameters],
            "entrypoint": self.entrypoint,
            "dependencies": self.dependencies,
            "allowed_paths": self.allowed_paths,
            "network_access": self.network_access,
            "registered_at": self.registered_at,
        }


class SkillResult:
    """Outcome of a skill execution."""

    def __init__(self, skill_name, success, data=None, error=None,
                 exit_code=0, stdout="", stderr="", duration_ms=0,
                 approved=True):
        self.skill_name = skill_name
        self.success = success
        self.data = data
        self.error = error
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = duration_ms
        self.approved = approved
        self.timestamp = datetime.now().isoformat(timespec="seconds")

    def to_dict(self):
        return {
            "skill": self.skill_name,
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 1),
            "timestamp": self.timestamp,
        }

    def summary(self):
        if not self.approved:
            return f"[DENIED] {self.skill_name}: User denied approval."
        status = "OK" if self.success else "FAIL"
        time_str = f"{self.duration_ms:.0f}ms"
        if self.error:
            return f"[{status}] {self.skill_name} ({time_str}): {self.error}"
        if self.data:
            preview = str(self.data)
            if len(preview) > 200:
                preview = preview[:197] + "..."
            return f"[{status}] {self.skill_name} ({time_str}): {preview}"
        return f"[{status}] {self.skill_name} ({time_str})"


# ---------------------------------------------------------------------------
# YAML loading
# ---------------------------------------------------------------------------

def _parse_yaml_text(text):
    """Minimal YAML-like parser when pyyaml is unavailable."""
    result = {}
    current_key = None
    current_list = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("- ") and current_key is not None:
            value = stripped[2:].strip().strip("'\"")
            if current_list is None:
                current_list = []
                result[current_key] = current_list
            current_list.append(value)
            continue
        if ":" in stripped:
            current_list = None
            key, _, value = stripped.partition(":")
            key = key.strip()
            value = value.strip().strip("'\"")
            current_key = key
            if value:
                if value.lower() in ("true", "yes"):
                    result[key] = True
                elif value.lower() in ("false", "no"):
                    result[key] = False
                elif value.isdigit():
                    result[key] = int(value)
                else:
                    try:
                        result[key] = float(value)
                    except ValueError:
                        result[key] = value
    return result


def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if HAS_YAML:
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"skill.yaml must be a mapping, got {type(data).__name__}"
            )
        return data
    return _parse_yaml_text(text)


def load_skill(skill_dir):
    """Load and validate a skill from its directory."""
    skill_dir = os.path.abspath(skill_dir)
    yaml_path = os.path.join(skill_dir, "skill.yaml")

    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"skill.yaml not found in {skill_dir}")

    data = _load_yaml(yaml_path)

    params = []
    raw_params = data.get("parameters", [])
    if isinstance(raw_params, list):
        for item in raw_params:
            if isinstance(item, dict):
                params.append(SkillParam(
                    name=item["name"],
                    type_=item.get("type", "string"),
                    required=item.get("required", True),
                    default=item.get("default"),
                    description=item.get("description", ""),
                ))
            elif isinstance(item, str):
                params.append(SkillParam(name=item))

    entrypoint = data.get("entrypoint", "run.py")
    entrypoint_path = os.path.join(skill_dir, entrypoint)
    if not os.path.isfile(entrypoint_path):
        raise FileNotFoundError(
            f"Entrypoint '{entrypoint}' not found in {skill_dir}"
        )

    skill = SkillDefinition(
        name=data.get("name", os.path.basename(skill_dir)),
        version=str(data.get("version", "1.0")),
        description=data.get("description", ""),
        author=data.get("author", "user"),
        risk_level=data.get("risk_level", "low"),
        timeout_seconds=int(data.get("timeout_seconds", 30)),
        parameters=params,
        entrypoint=entrypoint,
        dependencies=data.get("dependencies", []),
        skill_dir=skill_dir,
        allowed_paths=data.get("allowed_paths", []),
        network_access=bool(data.get("network_access", False)),
    )
    skill.registered_at = datetime.now().isoformat(timespec="seconds")
    return skill
