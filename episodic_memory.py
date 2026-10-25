"""
episodic_memory.py - Ragnarok FR1.1 (Episodic Memory)

FR1.1: The system must maintain a chronological log of user interactions,
completed tasks, and contextual events, allowing it to recall specific
past occurrences.

What this module provides
-------------------------
* One chronological log (SQLite table `conversations`, extended in place,
  so old chat history is preserved) holding three kinds of episodes:
      chat   - user / assistant messages
      event  - contextual events ("Hackathon jeeta")
      task   - completed tasks   ("README likha")
* Logging API       : log_chat, log_exchange, log_event, log_task
* Recall API        : search by keywords, by time ("kal", "last week",
                      "2026-10-20", "pichhle 7 din"), by kind (tasks/events)
* Natural language  : answer_recall_question("Maine kal kya kiya?")
* Prompt context    : context_for_prompt() gives the LLM relevant dated
                      events/tasks so it can recall past occurrences
* Management        : get, recent, stats, delete_episode, clear

Run `python episodic_memory.py --selftest` to verify everything offline.
"""

import re
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple

DEFAULT_DB = "ragnarok_memory.db"

KINDS = ("chat", "event", "task")
CHAT_ROLES = ("user", "assistant")
LOG_ROLES = ("user", "system")


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Episode(NamedTuple):
    id: int
    time: str            # ISO 8601, e.g. 2026-10-25T12:00:00
    kind: str            # chat | event | task
    role: str            # user | assistant | system
    content: str
    session_id: Optional[str]


class TimeRange(NamedTuple):
    start: str           # inclusive, ISO 8601
    end: str             # exclusive, ISO 8601
    label: str


class QueryResult(NamedTuple):
    episodes: List[Episode]
    total: int
    time_range: Optional[TimeRange]
    keywords: Tuple[str, ...]
    kinds: Optional[Tuple[str, ...]]


class RecallAnswer(NamedTuple):
    text: str
    found: bool
    has_time_range: bool


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def normalize_text(text: str) -> str:
    """Lowercase, unify spellings, strip punctuation (Unicode aware, so
    Devanagari text is kept)."""
    text = text.lower().replace("\u2019", "'")
    text = text.replace("colour", "color").replace("favourite", "favorite")
    text = re.sub(r"[^\w\s]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


STOP_WORDS = {
    # English
    "what", "which", "who", "how", "when", "where", "did", "do", "does",
    "is", "are", "was", "were", "am", "the", "a", "an", "my", "me", "i",
    "we", "you", "it", "this", "that", "to", "of", "for", "in", "on", "at",
    "about", "tell", "please", "can", "could", "know", "and", "or", "with",
    "have", "had", "has", "show", "find", "search", "remember",
    # Hindi / Hinglish
    "kya", "kab", "kahan", "kaun", "kaunsa", "kaunsi", "kaunse", "kitna",
    "kitni", "kitne", "hai", "hain", "tha", "thi", "hua", "hui", "huye",
    "ka", "ki", "ke", "ko", "se", "mein", "par", "pe", "batao", "btao",
    "bataiye", "mujhe", "maine", "mene", "humne", "hamne", "mera", "meri",
    "mere", "hamara", "tum", "tumhe", "aap", "yaad", "ya", "aur", "kiya",
    "kiye", "baare", "bare", "baat",
    # time words (handled by extract_time_range)
    "today", "aaj", "yesterday", "kal", "parson", "parso", "day", "days",
    "din", "dino", "week", "weeks", "hafte", "hafta", "month", "months",
    "mahine", "mahina", "last", "past", "pichhle", "pichle", "previous",
    "before",
    # kind words (handled by infer_kinds)
    "task", "tasks", "event", "events", "chat", "chats", "kaam",
    "completed", "complete", "conversation", "conversations",
}

_KIND_WORDS = {
    "task": "task", "tasks": "task", "completed": "task",
    "complete": "task", "kaam": "task",
    "event": "event", "events": "event",
    "chat": "chat", "chats": "chat",
    "conversation": "chat", "conversations": "chat",
}


def keyword_list(text: str) -> List[str]:
    """Meaningful search words, in order, without duplicates."""
    seen, out = set(), []
    for word in normalize_text(text).split():
        if word in STOP_WORDS:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def infer_kinds(words: Iterable[str]) -> Optional[Tuple[str, ...]]:
    """'tasks' / 'events' / 'chat' words select the kind of episode."""
    kinds: List[str] = []
    for word in words:
        kind = _KIND_WORDS.get(word)
        if kind and kind not in kinds:
            kinds.append(kind)
    return tuple(kinds) or None


# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DMY_RE = re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b")
_LAST_N_DAYS_RE = re.compile(
    r"\b(?:last|past|pichhle|pichle)\s+(\d{1,3})\s+(?:days?|din|dino)\b")
_DAY_BEFORE_RE = re.compile(
    r"\bday\s+before\s+yesterday\b|\bparsons?\b|\bparso\b")
_YESTERDAY_RE = re.compile(r"\byesterday\b|\bkal\b")
_TODAY_RE = re.compile(r"\btoday\b|\baaj\b")
_LAST_WEEK_RE = re.compile(
    r"\b(?:last|previous|pichhle|pichle)\s+(?:week|hafte|hafta)\b")
_THIS_WEEK_RE = re.compile(r"\b(?:this|is)\s+(?:week|hafte|hafta)\b")
_LAST_MONTH_RE = re.compile(
    r"\b(?:last|previous|pichhle|pichle)\s+(?:month|mahine|mahina)\b")
_THIS_MONTH_RE = re.compile(r"\b(?:this|is)\s+(?:month|mahine|mahina)\b")

_LEAD_ISO_RE = re.compile(
    r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})(?:[ T]+(\d{1,2}):(\d{2}))?(?=\s|$)")
_LEAD_DMY_RE = re.compile(
    r"^\s*(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})(?:[ T]+(\d{1,2}):(\d{2}))?(?=\s|$)")


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def extract_time_range(
    text: str, now: Optional[datetime] = None
) -> Tuple[Optional[TimeRange], str]:
    """Find a time phrase in `text`.

    Returns (TimeRange or None, text with the phrase removed).
    Supported: today/aaj, yesterday/kal, day before yesterday/parson,
    this/last week (is/pichhle hafte), this/last month (is/pichhle mahine),
    last N days (pichhle N din), YYYY-MM-DD and DD-MM-YYYY (also / and .).
    "kal" is read as *yesterday* because episodic memory is about the past.
    """
    now = now or datetime.now()
    raw = text.lower()
    today = datetime(now.year, now.month, now.day)
    tomorrow = today + timedelta(days=1)

    def build(start, end, label):
        return TimeRange(_iso(start), _iso(end), label)

    def cut(match):
        return raw[:match.start()] + " " + raw[match.end():]

    # Explicit dates first.
    for regex, order in ((_ISO_RE, "ymd"), (_DMY_RE, "dmy")):
        for match in regex.finditer(raw):
            first, second, third = (int(g) for g in match.groups())
            year, month, day = (
                (first, second, third) if order == "ymd"
                else (third, second, first)
            )
            try:
                moment = datetime(year, month, day)
            except ValueError:
                continue
            return (
                build(moment, moment + timedelta(days=1),
                      moment.strftime("%Y-%m-%d")),
                cut(match),
            )

    match = _LAST_N_DAYS_RE.search(raw)
    if match:
        n = max(1, int(match.group(1)))
        return (
            build(today - timedelta(days=n - 1), tomorrow, f"last {n} days"),
            cut(match),
        )

    this_monday = today - timedelta(days=today.weekday())
    first_of_month = today.replace(day=1)
    prev_month_first = (first_of_month - timedelta(days=1)).replace(day=1)
    one_day = timedelta(days=1)

    rules = (
        (_DAY_BEFORE_RE, today - 2 * one_day, today - one_day,
         "day before yesterday"),
        (_YESTERDAY_RE, today - one_day, today, "yesterday"),
        (_TODAY_RE, today, tomorrow, "today"),
        (_LAST_WEEK_RE, this_monday - timedelta(days=7), this_monday,
         "last week"),
        (_THIS_WEEK_RE, this_monday, tomorrow, "this week"),
        (_LAST_MONTH_RE, prev_month_first, first_of_month, "last month"),
        (_THIS_MONTH_RE, first_of_month, tomorrow, "this month"),
    )
    for regex, start, end, label in rules:
        match = regex.search(raw)
        if match:
            return build(start, end, label), cut(match)

    return None, text


def parse_leading_date(
    text: str, now: Optional[datetime] = None
) -> Tuple[Optional[datetime], str]:
    """For '/event 2026-10-20 14:30 Hackathon jeeta'.

    Returns (datetime or None, remaining text). A date without a time is
    stored as 00:00. Raises ValueError for invalid or future dates.
    """
    now = now or datetime.now()
    for regex, order in ((_LEAD_ISO_RE, "ymd"), (_LEAD_DMY_RE, "dmy")):
        match = regex.match(text)
        if not match:
            continue
        groups = match.groups()
        first, second, third = int(groups[0]), int(groups[1]), int(groups[2])
        year, month, day = (
            (first, second, third) if order == "ymd"
            else (third, second, first)
        )
        hour = int(groups[3]) if groups[3] else 0
        minute = int(groups[4]) if groups[4] else 0
        try:
            when = datetime(year, month, day, hour, minute)
        except ValueError:
            raise ValueError("Date ya time valid nahi hai.") from None
        if when > now:
            raise ValueError(
                "Future ki date allowed nahi hai; episodic memory sirf "
                "past occurrences ke liye hai."
            )
        return when, text[match.end():].strip()
    return None, text.strip()


# ---------------------------------------------------------------------------
# Ranking and formatting
# ---------------------------------------------------------------------------

def _token_match(word: str, tokens: set) -> bool:
    if word in tokens:
        return True
    if len(word) >= 4:
        for token in tokens:
            if len(token) >= 4 and (
                token.startswith(word) or word.startswith(token)
            ):
                return True
    return False


def rank_episodes(
    pool: Sequence[Episode],
    kws: Sequence[str],
    limit: int = 10,
    min_coverage: float = 0.0,
) -> Tuple[List[Episode], int]:
    """Pick the best `limit` episodes for the keywords; return them in
    chronological order together with the total number of matches."""
    if not kws:
        return [], 0

    phrase = " ".join(kws)
    scored = []
    for episode in pool:
        normalized = normalize_text(episode.content)
        tokens = set(normalized.split())
        matched = sum(1 for word in kws if _token_match(word, tokens))
        if not matched:
            continue
        coverage = matched / len(kws)
        if coverage < min_coverage:
            continue
        score = matched * 2
        if len(kws) > 1 and phrase in normalized:
            score += 3
        scored.append((score, coverage, episode.time, episode.id, episode))

    total = len(scored)
    scored.sort(key=lambda item: item[:4], reverse=True)
    best = [item[4] for item in scored[:max(1, limit)]]
    best.sort(key=lambda e: (e.time, e.id))
    return best, total


def format_episode(episode: Episode, width: int = 300) -> str:
    stamp = episode.time.replace("T", " ")
    label = episode.role if episode.kind == "chat" else episode.kind.upper()
    text = " ".join(episode.content.split())
    if width and len(text) > width:
        text = text[:width - 3].rstrip() + "..."
    return f"#{episode.id} [{stamp}] {label}: {text}"


def format_episodes(episodes: Iterable[Episode], width: int = 300) -> str:
    return "\n".join(format_episode(e, width) for e in episodes)


_SELF_PAST_RE = re.compile(
    r"\b(?:maine|mene|humne|hamne)\b"
    r"|\b(?:what|when|which|how many)\s+did\s+(?:i|we)\b"
    r"|\bdid\s+(?:i|we)\b"
    r"|\bwhat\s+(?:i|we)\s+did\b"
    r"|\bremember\s+(?:when|what)\s+(?:i|we)\b"
)
_QUESTION_RE = re.compile(
    r"\b(?:kya|kab|kaun|kaunsa|kaunsi|kaunse|kitna|kitni|kitne|kahan|"
    r"what|when|which|did|how)\b"
)


def is_recall_question(text: str) -> bool:
    """True only for short questions about what the user did in the past,
    e.g. 'Maine kal kya kiya?' or 'When did I finish the README?'."""
    normalized = normalize_text(text)
    words = normalized.split()
    if not words or len(words) > 14:
        return False
    return bool(
        _SELF_PAST_RE.search(normalized) and _QUESTION_RE.search(normalized)
    )


# ---------------------------------------------------------------------------
# The episodic memory store
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Chronological log of chats, events and completed tasks."""

    def __init__(
        self,
        db_file: str = DEFAULT_DB,
        clock: Callable[[], datetime] = datetime.now,
    ):
        self.db_file = db_file
        self._clock = clock
        self.session_id = uuid.uuid4().hex[:8]
        self.setup()

    # -- infrastructure ----------------------------------------------------

    def _now(self) -> datetime:
        return self._clock().replace(microsecond=0)

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.db_file)
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def setup(self) -> None:
        """Create the table, add new columns to an existing one, and convert
        records written by older Ragnarok versions."""
        with self._db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    time TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            """)
            columns = {
                row[1]
                for row in db.execute("PRAGMA table_info(conversations)")
            }
            if "kind" not in columns:
                db.execute(
                    "ALTER TABLE conversations "
                    "ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat'"
                )
            if "session_id" not in columns:
                db.execute(
                    "ALTER TABLE conversations ADD COLUMN session_id TEXT"
                )
            # Older versions stored events/tasks as role='event'/'task'.
            db.execute(
                "UPDATE conversations SET kind = role, role = 'user' "
                "WHERE role IN ('event', 'task')"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_time "
                "ON conversations(time)"
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_kind "
                "ON conversations(kind)"
            )

    def new_session(self) -> str:
        self.session_id = uuid.uuid4().hex[:8]
        return self.session_id

    # -- logging -----------------------------------------------------------

    def _check(self, kind: str, role: str, content: str) -> str:
        if kind not in KINDS:
            raise ValueError(f"Unknown episode kind: {kind!r}")
        content = (content or "").strip()
        if not content:
            raise ValueError("Episode content khaali nahi ho sakta.")
        allowed = CHAT_ROLES if kind == "chat" else LOG_ROLES
        if role not in allowed:
            raise ValueError(
                f"Role {role!r} {kind} ke liye valid nahi hai "
                f"(allowed: {', '.join(allowed)})."
            )
        return content

    def _stamp(self, when: Optional[datetime]) -> str:
        current = self._now()
        if when is None:
            return _iso(current)
        if when > current + timedelta(minutes=1):
            raise ValueError("Episode future mein log nahi ho sakta.")
        return _iso(when.replace(microsecond=0))

    def log(
        self,
        kind: str,
        content: str,
        role: Optional[str] = None,
        when: Optional[datetime] = None,
    ) -> int:
        """Append one episode and return its id."""
        role = role or "user"
        content = self._check(kind, role, content)
        stamp = self._stamp(when)
        with self._db() as db:
            cursor = db.execute(
                "INSERT INTO conversations "
                "(time, role, content, kind, session_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (stamp, role, content, kind, self.session_id),
            )
            return cursor.lastrowid

    def log_chat(self, role: str, content: str) -> int:
        return self.log("chat", content, role=role)

    def log_event(
        self,
        content: str,
        when: Optional[datetime] = None,
        role: str = "user",
    ) -> int:
        return self.log("event", content, role=role, when=when)

    def log_task(
        self,
        content: str,
        when: Optional[datetime] = None,
        role: str = "user",
    ) -> int:
        """Record a COMPLETED task."""
        return self.log("task", content, role=role, when=when)

    def log_exchange(self, user_text: str, assistant_text: str) -> Tuple[int, int]:
        """Save a user message and its reply together (both or neither)."""
        user_text = self._check("chat", "user", user_text)
        assistant_text = self._check("chat", "assistant", assistant_text)
        stamp = _iso(self._now())
        with self._db() as db:
            first = db.execute(
                "INSERT INTO conversations "
                "(time, role, content, kind, session_id) "
                "VALUES (?, 'user', ?, 'chat', ?)",
                (stamp, user_text, self.session_id),
            ).lastrowid
            second = db.execute(
                "INSERT INTO conversations "
                "(time, role, content, kind, session_id) "
                "VALUES (?, 'assistant', ?, 'chat', ?)",
                (stamp, assistant_text, self.session_id),
            ).lastrowid
        return first, second

    # -- reading -----------------------------------------------------------

    def _fetch(
        self,
        kinds: Optional[Sequence[str]] = None,
        time_range: Optional[TimeRange] = None,
        exclude_roles: Optional[Sequence[str]] = None,
        role: Optional[str] = None,
    ) -> List[Episode]:
        sql = ("SELECT id, time, kind, role, content, session_id "
               "FROM conversations WHERE 1=1")
        params: List[str] = []
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        if time_range:
            sql += " AND time >= ? AND time < ?"
            params += [time_range.start, time_range.end]
        if exclude_roles:
            sql += f" AND role NOT IN ({','.join('?' * len(exclude_roles))})"
            params += list(exclude_roles)
        if role:
            sql += " AND role = ?"
            params.append(role)
        sql += " ORDER BY time ASC, id ASC"
        with self._db() as db:
            rows = db.execute(sql, params).fetchall()
        return [Episode(*row) for row in rows]

    def get(self, episode_id: int) -> Optional[Episode]:
        with self._db() as db:
            row = db.execute(
                "SELECT id, time, kind, role, content, session_id "
                "FROM conversations WHERE id = ?",
                (episode_id,),
            ).fetchone()
        return Episode(*row) if row else None

    def recent(
        self, limit: int = 10, kinds: Optional[Sequence[str]] = None
    ) -> List[Episode]:
        """Latest `limit` episodes, shown oldest to newest."""
        sql = ("SELECT id, time, kind, role, content, session_id "
               "FROM conversations")
        params: List = []
        if kinds:
            sql += f" WHERE kind IN ({','.join('?' * len(kinds))})"
            params += list(kinds)
        sql += " ORDER BY time DESC, id DESC LIMIT ?"
        params.append(max(1, limit))
        with self._db() as db:
            rows = db.execute(sql, params).fetchall()
        rows.reverse()
        return [Episode(*row) for row in rows]

    def chat_context(self, limit: int = 6) -> List[dict]:
        """Recent chat messages in the format Ollama expects. Events and
        tasks are not chat, so they are never included here."""
        with self._db() as db:
            rows = db.execute(
                "SELECT role, content FROM conversations "
                "WHERE kind = 'chat' AND role IN ('user', 'assistant') "
                "ORDER BY time DESC, id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
        rows.reverse()
        return [{"role": role, "content": content} for role, content in rows]

    def count(
        self,
        kinds: Optional[Sequence[str]] = None,
        time_range: Optional[TimeRange] = None,
        role: Optional[str] = None,
    ) -> int:
        return len(self._fetch(kinds=kinds, time_range=time_range, role=role))

    def stats(self) -> dict:
        with self._db() as db:
            per_kind = dict(db.execute(
                "SELECT kind, COUNT(*) FROM conversations GROUP BY kind"
            ).fetchall())
            first, last = db.execute(
                "SELECT MIN(time), MAX(time) FROM conversations"
            ).fetchone()
        counts = {kind: per_kind.get(kind, 0) for kind in KINDS}
        return {
            "total": sum(counts.values()),
            "by_kind": counts,
            "first": first,
            "last": last,
        }

    # -- searching and recall ----------------------------------------------

    def query(
        self,
        text: str,
        limit: int = 10,
        kinds: Optional[Sequence[str]] = None,
        default_kinds: Optional[Sequence[str]] = None,
        exclude_roles: Optional[Sequence[str]] = None,
        min_coverage: float = 0.0,
    ) -> QueryResult:
        """Understand a free-text request: time phrase, kind words and
        keywords, then return the matching episodes (chronological).

        `kinds` forces the kinds. Otherwise kind words in the text decide;
        `default_kinds` applies only to time-only requests like 'kal'.
        """
        time_range, remaining = extract_time_range(text, self._now())
        words = normalize_text(remaining).split()
        kinds = tuple(kinds) if kinds else infer_kinds(words)
        kws = keyword_list(remaining)

        if kws:
            pool = self._fetch(
                kinds=kinds, time_range=time_range,
                exclude_roles=exclude_roles,
            )
            episodes, total = rank_episodes(pool, kws, limit, min_coverage)
        elif time_range or kinds:
            pool = self._fetch(
                kinds=kinds or default_kinds, time_range=time_range,
                exclude_roles=exclude_roles,
            )
            total = len(pool)
            episodes = pool[-max(1, limit):]
        else:
            episodes, total = [], 0

        return QueryResult(episodes, total, time_range, tuple(kws), kinds)

    def search(self, text: str, limit: int = 10) -> QueryResult:
        return self.query(text, limit=limit)

    def timeline(self, text: str = "today", limit: int = 50) -> QueryResult:
        return self.query(text or "today", limit=limit)

    def _build_recall(self, res: QueryResult) -> RecallAnswer:
        time_range = res.time_range
        scope = f" ({time_range.label})" if time_range else ""

        if not res.keywords and not time_range and not res.kinds:
            return RecallAnswer(
                "Kuch specific batao, jaise: /recall maine kal kya kiya",
                False, False,
            )

        if res.episodes:
            lines = [
                f"Episodic memory ke hisaab se{scope}:",
                format_episodes(res.episodes),
            ]
            if res.total > len(res.episodes):
                lines.append(
                    f"({res.total} records mein se {len(res.episodes)} "
                    "dikha raha hoon.)"
                )
        else:
            what = "event ya task" if not res.keywords else "matching record"
            lines = [f"Episodic memory mein{scope} koi {what} nahi mila."]

        if time_range and not res.keywords:
            chats = self.count(("chat",), time_range, role="user")
            if chats:
                lines.append(
                    f"Is dauraan {chats} chat message(s) bhi hue; dekhne "
                    f"ke liye: /timeline {time_range.label}"
                )
        return RecallAnswer(
            "\n".join(lines), bool(res.episodes), time_range is not None
        )

    def recall(self, text: str) -> RecallAnswer:
        """Explicit recall (used by /recall). Always returns text."""
        res = self.query(text, limit=10, default_kinds=("event", "task"))
        return self._build_recall(res)

    def answer_recall_question(self, text: str) -> Optional[str]:
        """Answer 'Maine kal kya kiya?'-style questions straight from the
        log. Returns None when the text is not such a question (or nothing
        relevant exists), so the caller can use the normal model instead."""
        if not is_recall_question(text):
            return None

        res = self.query(text, limit=10, default_kinds=("event", "task"))
        if res.time_range or res.kinds:
            return self._build_recall(res).text

        if res.keywords:
            res = self.query(
                text, limit=5, kinds=("event", "task"), min_coverage=0.5
            )
            if res.episodes:
                return self._build_recall(res).text
        return None

    def context_for_prompt(
        self, text: str, limit: int = 3, min_coverage: float = 0.5
    ) -> str:
        """Dated events/tasks relevant to `text`, formatted for the LLM
        prompt. Empty string when nothing relevant exists."""
        res = self.query(
            text, limit=limit, kinds=("event", "task"),
            exclude_roles=("system",), min_coverage=min_coverage,
        )
        if not res.keywords or not res.episodes:
            return ""
        lines = []
        for episode in res.episodes:
            day = episode.time.replace("T", " ")[:16]
            content = " ".join(episode.content.split())[:300]
            lines.append(f"- [{day}] {episode.kind.upper()}: {content}")
        return "\n".join(lines)

    # -- management --------------------------------------------------------

    def delete_episode(self, episode_id: int) -> bool:
        with self._db() as db:
            cursor = db.execute(
                "DELETE FROM conversations WHERE id = ?", (episode_id,)
            )
            return cursor.rowcount > 0

    def clear(self, kinds: Optional[Sequence[str]] = None) -> int:
        """Delete all episodes (or only the given kinds). Returns count."""
        with self._db() as db:
            if kinds:
                cursor = db.execute(
                    "DELETE FROM conversations WHERE kind IN "
                    f"({','.join('?' * len(kinds))})",
                    list(kinds),
                )
            else:
                cursor = db.execute("DELETE FROM conversations")
            return cursor.rowcount


# ---------------------------------------------------------------------------
# Self-test:  python episodic_memory.py --selftest
# ---------------------------------------------------------------------------

def _selftest() -> int:
    import os
    import tempfile

    checks = 0

    def check(condition, message):
        nonlocal checks
        if not condition:
            raise AssertionError("FAILED: " + message)
        checks += 1

    def raises(function, *args, **kwargs):
        try:
            function(*args, **kwargs)
        except ValueError:
            return True
        return False

    fixed_now = datetime(2026, 10, 25, 12, 0, 0)   # a Sunday
    folder = tempfile.mkdtemp()

    # ---- legacy database migration ------------------------------------
    legacy_db = os.path.join(folder, "legacy.db")
    connection = sqlite3.connect(legacy_db)
    connection.execute(
        "CREATE TABLE conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "time TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL)"
    )
    connection.executemany(
        "INSERT INTO conversations (time, role, content) VALUES (?, ?, ?)",
        [
            ("2026-10-01T10:00:00", "user", "hello"),
            ("2026-10-01T10:00:05", "assistant", "hi there"),
            ("2026-10-02T09:00:00", "event", "Old event row"),
            ("2026-10-03T09:00:00", "task", "Old task row"),
        ],
    )
    connection.commit()
    connection.close()
    legacy = EpisodicMemory(legacy_db, clock=lambda: fixed_now)
    stats = legacy.stats()
    check(stats["by_kind"] == {"chat": 2, "event": 1, "task": 1},
          "legacy rows migrated to kinds")
    check(legacy.get(3).role == "user" and legacy.get(3).kind == "event",
          "legacy event row converted")
    EpisodicMemory(legacy_db, clock=lambda: fixed_now)   # idempotent
    check(legacy.stats()["by_kind"] == stats["by_kind"],
          "migration is idempotent")

    # ---- logging -----------------------------------------------------------
    mem = EpisodicMemory(os.path.join(folder, "t.db"), clock=lambda: fixed_now)
    ev_hack = mem.log_event("Hackathon jeeta", datetime(2026, 10, 24, 15, 30))
    t_readme = mem.log_task("Ragnarok ka README likha",
                            datetime(2026, 10, 24, 18, 0))
    t_calc = mem.log_task("Calculator skill test kiya",
                          datetime(2026, 10, 20, 10, 0))
    mem.log_event("College seminar attend kiya",
                  datetime(2026, 10, 14, 11, 0))
    mem.log_event("Pehla prototype demo", datetime(2026, 9, 30, 9, 0))
    mem.log_event("Fact saved: Mera project Ragnarok hai.", role="system")
    mem.log_exchange("Recursion kya hota hai?", "Recursion ek technique hai.")

    check(mem.stats()["by_kind"] == {"chat": 2, "event": 4, "task": 2},
          "counts by kind")
    check(raises(mem.log_event, "x", datetime(2026, 12, 1)),
          "future episode rejected")
    check(raises(mem.log, "bogus", "x"), "unknown kind rejected")
    check(raises(mem.log_event, "   "), "empty content rejected")
    check(raises(mem.log_chat, "event", "x"), "bad chat role rejected")
    check(raises(mem.log_exchange, "hi", ""), "empty reply rejected")

    # ---- chronological order, backdated records ---------------------------
    times = [e.time for e in mem.recent(100)]
    check(times == sorted(times), "recent() is chronological")
    check(mem.recent(2)[-1].role == "assistant", "latest record is the reply")
    chat = mem.chat_context(10)
    check([m["role"] for m in chat] == ["user", "assistant"],
          "chat_context has only chat")

    # ---- time parsing -----------------------------------------------------
    rng, rest = extract_time_range("Maine kal kya kiya?", fixed_now)
    check(rng.label == "yesterday" and rng.start == "2026-10-24T00:00:00"
          and rng.end == "2026-10-25T00:00:00", "kal -> yesterday")
    rng, _ = extract_time_range("last 3 days", fixed_now)
    check(rng.start == "2026-10-23T00:00:00"
          and rng.end == "2026-10-26T00:00:00", "last 3 days")
    rng, rest = extract_time_range("hackathon pichhle 10 din", fixed_now)
    check("10" not in rest and rng.label == "last 10 days",
          "N-days phrase removed from text")
    rng, _ = extract_time_range("last week", fixed_now)
    check(rng.start == "2026-10-12T00:00:00"
          and rng.end == "2026-10-19T00:00:00", "last week (Mon-Mon)")
    rng, _ = extract_time_range("this week", fixed_now)
    check(rng.start == "2026-10-19T00:00:00", "this week starts Monday")
    rng, _ = extract_time_range("last month", fixed_now)
    check(rng.start == "2026-09-01T00:00:00"
          and rng.end == "2026-10-01T00:00:00", "last month")
    rng, _ = extract_time_range("parson", fixed_now)
    check(rng.start == "2026-10-23T00:00:00", "parson = day before yesterday")
    rng, _ = extract_time_range("day before yesterday", fixed_now)
    check(rng.label == "day before yesterday", "full phrase")
    rng, _ = extract_time_range("20-10-2026", fixed_now)
    check(rng.label == "2026-10-20", "DD-MM-YYYY")
    rng, _ = extract_time_range("2026-13-45 hello", fixed_now)
    check(rng is None, "invalid date ignored")
    check(extract_time_range("recursion kya hai", fixed_now)[0] is None,
          "no time phrase")

    when, rest = parse_leading_date("2026-10-20 14:30 Meeting hui", fixed_now)
    check(when == datetime(2026, 10, 20, 14, 30) and rest == "Meeting hui",
          "leading date with time")
    when, rest = parse_leading_date("20/10/2026 Demo", fixed_now)
    check(when == datetime(2026, 10, 20) and rest == "Demo", "leading DMY")
    when, rest = parse_leading_date("Sirf text", fixed_now)
    check(when is None and rest == "Sirf text", "no leading date")
    check(raises(parse_leading_date, "2030-01-01 Future", fixed_now),
          "future leading date rejected")
    check(raises(parse_leading_date, "2026-02-30 X", fixed_now),
          "invalid leading date rejected")

    # ---- searching --------------------------------------------------------
    res = mem.search("hackathon")
    check([e.id for e in res.episodes] == [ev_hack], "keyword search")
    res = mem.search("tasks")
    check({e.kind for e in res.episodes} == {"task"} and res.total == 2,
          "kind word selects tasks")
    res = mem.search("task kal")
    check([e.id for e in res.episodes] == [t_readme], "kind + time")
    res = mem.search("calculator 2026-10-20")
    check([e.id for e in res.episodes] == [t_calc], "keyword + date")
    res = mem.search("recursion")
    check(len(res.episodes) == 2, "search finds chat too")
    check(mem.search("xyznotthere").episodes == [], "no match")
    res = mem.timeline("kal")
    check(res.total == 2, "timeline yesterday")
    res = mem.timeline()
    check(res.total == 3 and res.time_range.label == "today",
          "timeline defaults to today")
    check(mem.search("").episodes == [], "empty search")

    # ---- recall in natural language ----------------------------------------
    answer = mem.answer_recall_question("Maine kal kya kiya?")
    check("Hackathon" in answer and "README" in answer
          and "Calculator" not in answer, "recall: kal")
    check("chat message" not in answer, "no chats yesterday, no hint")
    answer_today = mem.answer_recall_question("Maine aaj kya kiya?")
    check("Fact saved" in answer_today and "chat message" in answer_today,
          "recall: aaj lists the event and mentions today's chats")
    answer = mem.answer_recall_question("Maine pichhle hafte kya kiya?")
    check("seminar" in answer, "recall: last week")
    answer = mem.answer_recall_question("Maine pichhle mahine kya kiya?")
    check("prototype" in answer, "recall: last month")
    answer = mem.answer_recall_question("Maine 2026-10-20 ko kya kiya?")
    check("Calculator" in answer, "recall: date")
    answer = mem.answer_recall_question("What did I do yesterday?")
    check("Hackathon" in answer, "recall: English")
    answer = mem.answer_recall_question("Maine kaunse tasks complete kiye?")
    check("README" in answer and "Hackathon" not in answer,
          "recall: only tasks")
    answer = mem.answer_recall_question("Maine hackathon kab jeeta?")
    check(answer and "Hackathon jeeta" in answer, "recall: keywords")
    answer = mem.answer_recall_question("Maine parson kya kiya?")
    check("koi event ya task nahi mila" in answer, "recall: empty day")
    check(mem.answer_recall_question("Recursion kya hota hai?") is None,
          "normal question not hijacked")
    check(mem.answer_recall_question("What do I like?") is None,
          "'what do I' is not recall")
    check(mem.answer_recall_question("Mera project kya hai?") is None,
          "fact question not hijacked")
    check(mem.answer_recall_question("Maine code likha, kya error hai?")
          is None, "no matching episode -> falls through")
    check(is_recall_question("When did I finish the README?"),
          "is_recall_question English")
    check(not is_recall_question("hello"), "hello is not recall")
    check("nahi mila" in mem.recall("xyznotthere").text, "recall no match")
    check("Kuch specific" in mem.recall("kya").text, "recall needs a filter")

    # ---- prompt context -----------------------------------------------------
    context = mem.context_for_prompt("hackathon ke baare mein batao")
    check("EVENT" in context and "Hackathon" in context, "context found")
    check(mem.context_for_prompt("weather kaisa hai") == "",
          "no context when unrelated")
    check("Fact saved" not in mem.context_for_prompt("ragnarok project"),
          "system events excluded from prompt context")

    # ---- management -----------------------------------------------------------
    check(mem.delete_episode(t_calc) and mem.get(t_calc) is None,
          "delete_episode")
    check(not mem.delete_episode(99999), "delete missing id")
    check(mem.clear(("chat",)) == 2 and mem.stats()["by_kind"]["chat"] == 0,
          "clear chat only")
    check(mem.stats()["by_kind"]["event"] == 4, "events survive clear(chat)")
    check(mem.clear() > 0 and mem.stats()["total"] == 0, "clear all")

    print(f"All {checks} checks passed.")
    return 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("episodic_memory.py is a module used by ragnarok.py.")
    print("Run  python episodic_memory.py --selftest  to verify it.")