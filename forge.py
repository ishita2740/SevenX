"""
forge.py - Ragnarok FR2.1 + FR2.2 (Skill Registry & Selection)

FR2.1: Define, register, and manage reusable skills.
FR2.2: Select appropriate skills and generate parameters.

The Forge:
    - Auto-discovers skills from the skills/ directory
    - Maintains a registry of loaded SkillDefinitions
    - Executes skills via the Shield Wall (sandboxed)
    - Provides CLI commands: /skills, /run, /skillinfo
"""

import os
from typing import Callable, Dict, List, Optional

from skill_schema import SkillDefinition, SkillResult, load_skill
from shield_wall import execute_skill


DEFAULT_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "skills")


class Forge:
    """The Dwarven Forge - skill registry and executor."""

    def __init__(self, skills_dir=None, python_exe=None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self.python_exe = python_exe
        self._registry = {}  # name -> SkillDefinition
        self._execution_log = []  # List of SkillResult
        self.auto_discover()

    # -- Discovery and registration ----------------------------------------

    def auto_discover(self):
        """Scan skills_dir for valid skill directories."""
        if not os.path.isdir(self.skills_dir):
            return

        for entry in sorted(os.listdir(self.skills_dir)):
            skill_path = os.path.join(self.skills_dir, entry)
            yaml_path = os.path.join(skill_path, "skill.yaml")
            if os.path.isdir(skill_path) and os.path.isfile(yaml_path):
                try:
                    self.register(skill_path)
                except (ValueError, FileNotFoundError, OSError) as e:
                    print(f"  Warning: skill '{entry}' load failed: {e}")

    def register(self, skill_dir):
        """Load and register a skill from a directory.
        Returns the SkillDefinition."""
        skill = load_skill(skill_dir)
        self._registry[skill.name] = skill
        return skill

    def unregister(self, name):
        """Remove a skill from the registry. Returns True if found."""
        return self._registry.pop(name, None) is not None

    # -- Queries -----------------------------------------------------------

    def get(self, name):
        """Get a skill by name, or None."""
        return self._registry.get(name)

    def list_skills(self):
        """All registered skills, sorted by name."""
        return sorted(self._registry.values(), key=lambda s: s.name)

    @property
    def skill_names(self):
        return sorted(self._registry.keys())

    def skill_count(self):
        return len(self._registry)

    # -- Execution ---------------------------------------------------------

    def run(self, name, params=None, approval_callback=None):
        """Execute a registered skill by name.

        Args:
            name: Skill name.
            params: Dict of parameter values.
            approval_callback: Optional callable(SkillDefinition) -> bool.

        Returns:
            SkillResult.
        """
        skill = self.get(name)
        if not skill:
            return SkillResult(
                skill_name=name,
                success=False,
                error=f"Skill '{name}' not found in registry.",
            )

        result = execute_skill(
            skill,
            params or {},
            python_exe=self.python_exe,
            approval_callback=approval_callback,
        )
        self._execution_log.append(result)
        return result

    def recent_executions(self, n=10):
        """Last N execution results."""
        return self._execution_log[-n:]

    # -- Display helpers ---------------------------------------------------

    def format_skill_list(self):
        skills = self.list_skills()
        if not skills:
            return "No skills registered. Add skills to the skills/ folder."

        lines = ["\n--- Dwarven Forge: Registered Skills ---"]
        for skill in skills:
            risk = skill.risk_level.upper()
            params = ", ".join(p.name for p in skill.parameters) or "none"
            lines.append(
                f"  {skill.name} (v{skill.version}) [{risk}]\n"
                f"    {skill.description or 'No description'}\n"
                f"    Parameters: {params}"
            )
        lines.append(f"\nTotal: {len(skills)} skill(s)")
        return "\n".join(lines)

    def format_skill_info(self, name):
        skill = self.get(name)
        if not skill:
            return f"Skill '{name}' not found."

        lines = [
            f"\n--- Skill: {skill.name} ---",
            f"  Version     : {skill.version}",
            f"  Author      : {skill.author}",
            f"  Risk Level  : {skill.risk_level}",
            f"  Timeout     : {skill.timeout_seconds}s",
            f"  Entrypoint  : {skill.entrypoint}",
            f"  Description : {skill.description or 'None'}",
            f"  Network     : {'yes' if skill.network_access else 'no'}",
            f"  Registered  : {skill.registered_at or 'n/a'}",
            f"  Directory   : {skill.skill_dir}",
        ]
        if skill.parameters:
            lines.append("  Parameters:")
            for p in skill.parameters:
                req = "required" if p.required else f"default={p.default}"
                lines.append(f"    {p.name} ({p.type}, {req})")
                if p.description:
                    lines.append(f"      {p.description}")
        if skill.dependencies:
            lines.append(f"  Dependencies: {', '.join(skill.dependencies)}")
        return "\n".join(lines)

    def format_execution_log(self, n=5):
        recent = self.recent_executions(n)
        if not recent:
            return "No skill executions yet."
        lines = ["\n--- Recent Skill Executions ---"]
        for r in recent:
            lines.append(f"  {r.summary()}")
        return "\n".join(lines)
