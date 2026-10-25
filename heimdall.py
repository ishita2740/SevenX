"""
heimdall.py - Ragnarok FR4 (Always-On Background Agent)

FR4.1: The system must continuously monitor for triggers.
FR4.2: The system must be operational on system startup.

Heimdall provides:
    - File system watchers (new/modified/deleted files)
    - Cron-like scheduled tasks
    - Event bus for inter-module communication
    - Background thread management
    - Auto-trigger skill execution on events

All watchers run in background threads, zero network required.
"""

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Event Bus
# ---------------------------------------------------------------------------

class Event:
    """An event in the Heimdall event bus."""

    __slots__ = ("type", "data", "timestamp", "source")

    def __init__(self, type_, data=None, source="heimdall"):
        self.type = type_
        self.data = data or {}
        self.timestamp = datetime.now().isoformat(timespec="seconds")
        self.source = source

    def to_dict(self):
        return {
            "type": self.type,
            "data": self.data,
            "timestamp": self.timestamp,
            "source": self.source,
        }

    def __repr__(self):
        return f"Event({self.type}, {self.source}, {self.timestamp})"


class EventBus:
    """Simple publish-subscribe event bus (thread-safe)."""

    def __init__(self):
        self._subscribers = {}  # event_type -> [callback, ...]
        self._lock = threading.Lock()
        self._history = []  # last N events
        self._max_history = 100

    def subscribe(self, event_type, callback):
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type, callback):
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type] = [
                    cb for cb in self._subscribers[event_type]
                    if cb != callback
                ]

    def publish(self, event):
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
            callbacks = list(self._subscribers.get(event.type, []))
            # Also notify wildcard subscribers
            callbacks += list(self._subscribers.get("*", []))

        for cb in callbacks:
            try:
                cb(event)
            except Exception as e:
                print(f"Heimdall: Event handler error ({event.type}): {e}")

    def recent(self, n=20):
        with self._lock:
            return list(self._history[-n:])


# ---------------------------------------------------------------------------
# File Watcher
# ---------------------------------------------------------------------------

class FileWatcher:
    """Watch a directory for file changes."""

    def __init__(self, path, patterns=None, recursive=False,
                 interval=5.0, event_bus=None):
        self.path = os.path.abspath(path)
        self.patterns = patterns or ["*"]
        self.recursive = recursive
        self.interval = max(1.0, interval)
        self.event_bus = event_bus
        self._snapshot = {}  # path -> (mtime, size, hash)
        self._running = False
        self._thread = None

    def _matches(self, filename):
        if "*" in self.patterns:
            return True
        for pattern in self.patterns:
            if pattern.startswith("."):
                if filename.endswith(pattern):
                    return True
            elif pattern in filename:
                return True
        return False

    def _scan(self):
        """Scan directory and return {path: (mtime, size)}."""
        snapshot = {}
        if not os.path.isdir(self.path):
            return snapshot

        if self.recursive:
            for root, dirs, files in os.walk(self.path):
                # Skip hidden and __pycache__ dirs
                dirs[:] = [d for d in dirs
                           if not d.startswith(".") and d != "__pycache__"]
                for f in files:
                    if self._matches(f):
                        full = os.path.join(root, f)
                        try:
                            st = os.stat(full)
                            snapshot[full] = (st.st_mtime, st.st_size)
                        except OSError:
                            pass
        else:
            for f in os.listdir(self.path):
                full = os.path.join(self.path, f)
                if os.path.isfile(full) and self._matches(f):
                    try:
                        st = os.stat(full)
                        snapshot[full] = (st.st_mtime, st.st_size)
                    except OSError:
                        pass
        return snapshot

    def _diff(self, old, new):
        """Compare snapshots and return (created, modified, deleted) paths."""
        created = [p for p in new if p not in old]
        deleted = [p for p in old if p not in new]
        modified = [
            p for p in new
            if p in old and new[p] != old[p]
        ]
        return created, modified, deleted

    def _loop(self):
        """Background polling loop."""
        self._snapshot = self._scan()
        while self._running:
            time.sleep(self.interval)
            if not self._running:
                break

            new_snapshot = self._scan()
            created, modified, deleted = self._diff(self._snapshot, new_snapshot)
            self._snapshot = new_snapshot

            if created and self.event_bus:
                self.event_bus.publish(Event("file.created", {
                    "files": created,
                    "watch_path": self.path,
                }))

            if modified and self.event_bus:
                self.event_bus.publish(Event("file.modified", {
                    "files": modified,
                    "watch_path": self.path,
                }))

            if deleted and self.event_bus:
                self.event_bus.publish(Event("file.deleted", {
                    "files": deleted,
                    "watch_path": self.path,
                }))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name=f"heimdall-watch-{os.path.basename(self.path)}"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=self.interval + 1)

    @property
    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()

    def to_dict(self):
        return {
            "path": self.path,
            "patterns": self.patterns,
            "recursive": self.recursive,
            "interval": self.interval,
            "running": self.is_running,
            "files_tracked": len(self._snapshot),
        }


# ---------------------------------------------------------------------------
# Cron Scheduler
# ---------------------------------------------------------------------------

class CronJob:
    """A scheduled recurring task."""

    def __init__(self, name, schedule, action, description=""):
        self.name = name
        self.schedule = schedule  # e.g. "*/5 * * * *" or "every 5m"
        self.action = action     # callable() or skill_name
        self.description = description
        self.last_run = None
        self.next_run = None
        self.run_count = 0
        self.enabled = True
        self._interval = self._parse_interval(schedule)

    @staticmethod
    def _parse_interval(schedule):
        """Parse simple interval strings like 'every 5m', 'every 1h'."""
        match = re.match(r"every\s+(\d+)\s*(s|m|h|min|sec|hr|hour)", schedule, re.I)
        if match:
            n = int(match.group(1))
            unit = match.group(2).lower()
            if unit in ("s", "sec"):
                return n
            elif unit in ("m", "min"):
                return n * 60
            elif unit in ("h", "hr", "hour"):
                return n * 3600
        # Default: try to interpret as seconds
        try:
            return int(schedule)
        except ValueError:
            return 300  # default 5 min

    def is_due(self):
        if not self.enabled:
            return False
        if self.last_run is None:
            return True
        elapsed = time.time() - self.last_run
        return elapsed >= self._interval

    def mark_run(self):
        self.last_run = time.time()
        self.run_count += 1
        self.next_run = datetime.fromtimestamp(
            self.last_run + self._interval
        ).isoformat(timespec="seconds")

    def to_dict(self):
        return {
            "name": self.name,
            "schedule": self.schedule,
            "description": self.description,
            "enabled": self.enabled,
            "run_count": self.run_count,
            "last_run": datetime.fromtimestamp(self.last_run).isoformat(
                timespec="seconds"
            ) if self.last_run else None,
            "next_run": self.next_run,
            "interval_seconds": self._interval,
        }


class Scheduler:
    """Background cron scheduler."""

    def __init__(self, event_bus=None):
        self.jobs = {}  # name -> CronJob
        self.event_bus = event_bus
        self._running = False
        self._thread = None
        self._check_interval = 5  # seconds

    def add(self, job):
        self.jobs[job.name] = job
        return job

    def remove(self, name):
        return self.jobs.pop(name, None)

    def _loop(self):
        while self._running:
            time.sleep(self._check_interval)
            if not self._running:
                break

            for job in list(self.jobs.values()):
                if job.is_due():
                    job.mark_run()
                    # Execute action
                    if callable(job.action):
                        try:
                            job.action()
                        except Exception as e:
                            print(f"Heimdall: Cron '{job.name}' error: {e}")
                    # Publish event
                    if self.event_bus:
                        self.event_bus.publish(Event("cron.executed", {
                            "job": job.name,
                            "run_count": job.run_count,
                        }))

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True,
            name="heimdall-scheduler"
        )
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._check_interval + 1)

    @property
    def is_running(self):
        return self._running and self._thread and self._thread.is_alive()


# ---------------------------------------------------------------------------
# Event Log (persistent)
# ---------------------------------------------------------------------------

def _setup_event_table(db_file):
    with sqlite3.connect(db_file) as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS heimdall_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                data TEXT,
                source TEXT,
                timestamp TEXT NOT NULL
            )
        """)


def log_event(db_file, event):
    try:
        with sqlite3.connect(db_file) as db:
            db.execute(
                "INSERT INTO heimdall_events (type, data, source, timestamp) "
                "VALUES (?, ?, ?, ?)",
                (event.type,
                 json.dumps(event.data, default=str),
                 event.source,
                 event.timestamp),
            )
    except sqlite3.Error:
        pass


def recent_events(db_file, n=20, event_type=None):
    try:
        with sqlite3.connect(db_file) as db:
            _setup_event_table(db_file)
            if event_type:
                rows = db.execute(
                    "SELECT type, data, source, timestamp FROM heimdall_events "
                    "WHERE type = ? ORDER BY id DESC LIMIT ?",
                    (event_type, n),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT type, data, source, timestamp FROM heimdall_events "
                    "ORDER BY id DESC LIMIT ?",
                    (n,),
                ).fetchall()
        return [
            {"type": r[0], "data": json.loads(r[1] or "{}"),
             "source": r[2], "timestamp": r[3]}
            for r in rows
        ]
    except (sqlite3.Error, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Heimdall (main controller)
# ---------------------------------------------------------------------------

class Heimdall:
    """The All-Seeing Guardian - background monitoring and scheduling."""

    def __init__(self, db_file="ragnarok_memory.db", forge=None):
        self.db_file = db_file
        self.forge = forge
        self.event_bus = EventBus()
        self.scheduler = Scheduler(event_bus=self.event_bus)
        self.watchers = {}  # name -> FileWatcher
        self._running = False

        # Setup persistence
        _setup_event_table(db_file)

        # Auto-log all events to SQLite
        self.event_bus.subscribe("*", lambda e: log_event(db_file, e))

    def start(self):
        """Start all background services."""
        if self._running:
            return
        self._running = True
        self.scheduler.start()
        for w in self.watchers.values():
            w.start()
        self.event_bus.publish(Event("heimdall.started", source="system"))

    def stop(self):
        """Stop all background services."""
        self._running = False
        self.scheduler.stop()
        for w in self.watchers.values():
            w.stop()
        self.event_bus.publish(Event("heimdall.stopped", source="system"))

    # -- File watchers -----------------------------------------------------

    def watch(self, name, path, patterns=None, recursive=False, interval=5.0):
        """Add and start a file watcher."""
        if name in self.watchers:
            self.watchers[name].stop()

        watcher = FileWatcher(
            path, patterns, recursive, interval, self.event_bus
        )
        self.watchers[name] = watcher

        if self._running:
            watcher.start()

        self.event_bus.publish(Event("watcher.added", {
            "name": name, "path": path, "patterns": patterns,
        }))
        return watcher

    def unwatch(self, name):
        """Stop and remove a file watcher."""
        watcher = self.watchers.pop(name, None)
        if watcher:
            watcher.stop()
            self.event_bus.publish(Event("watcher.removed", {"name": name}))
            return True
        return False

    # -- Cron jobs ---------------------------------------------------------

    def schedule(self, name, schedule, action=None, skill_name=None,
                 skill_params=None, description=""):
        """Add a scheduled task."""
        if skill_name and self.forge:
            def skill_action():
                result = self.forge.run(skill_name, skill_params or {})
                self.event_bus.publish(Event("skill.auto_executed", {
                    "skill": skill_name,
                    "success": result.success,
                    "summary": result.summary(),
                }))
            action = skill_action
            description = description or f"Run skill: {skill_name}"

        job = CronJob(name, schedule, action or (lambda: None), description)
        self.scheduler.add(job)

        self.event_bus.publish(Event("cron.added", {
            "name": name, "schedule": schedule,
        }))
        return job

    def unschedule(self, name):
        """Remove a scheduled task."""
        removed = self.scheduler.remove(name)
        if removed:
            self.event_bus.publish(Event("cron.removed", {"name": name}))
        return removed is not None

    # -- Event triggers ----------------------------------------------------

    def on(self, event_type, callback):
        """Subscribe to an event type."""
        self.event_bus.subscribe(event_type, callback)

    def on_file_change(self, callback):
        """Subscribe to all file change events."""
        for et in ("file.created", "file.modified", "file.deleted"):
            self.event_bus.subscribe(et, callback)

    # -- Queries -----------------------------------------------------------

    def status(self):
        return {
            "running": self._running,
            "watchers": {
                name: w.to_dict() for name, w in self.watchers.items()
            },
            "cron_jobs": {
                name: j.to_dict() for name, j in self.scheduler.jobs.items()
            },
            "recent_events": self.event_bus.recent(10),
            "scheduler_running": self.scheduler.is_running,
        }

    def recent(self, n=20, event_type=None):
        return recent_events(self.db_file, n, event_type)

    def format_status(self):
        s = self.status()
        lines = ["\n--- Heimdall Status ---"]
        lines.append(f"  Running: {s['running']}")
        lines.append(f"  Scheduler: {s['scheduler_running']}")

        lines.append("\n  File Watchers:")
        if s["watchers"]:
            for name, w in s["watchers"].items():
                lines.append(
                    f"    {name}: {w['path']} "
                    f"({w['files_tracked']} files, "
                    f"{'active' if w['running'] else 'stopped'})"
                )
        else:
            lines.append("    (none)")

        lines.append("\n  Scheduled Jobs:")
        if s["cron_jobs"]:
            for name, j in s["cron_jobs"].items():
                status = "enabled" if j["enabled"] else "disabled"
                lines.append(
                    f"    {name}: {j['schedule']} ({status}, "
                    f"ran {j['run_count']}x)"
                )
                if j["description"]:
                    lines.append(f"      {j['description']}")
        else:
            lines.append("    (none)")

        return "\n".join(lines)
