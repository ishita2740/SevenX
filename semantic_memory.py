"""
semantic_memory.py - Ragnarok FR1.2 (Semantic Memory)

FR1.2: The system must extract, store, and retrieve generalized knowledge,
user preferences, and factual data in a structured format (e.g. vector
embeddings or knowledge graphs).

What this module provides
-------------------------
STORE (structured)   Every memory is one row of the `facts` table (the old
                     table is extended in place, nothing is lost) with:
                       kind        fact | preference | knowledge
                       subject / relation / obj
                                   a knowledge-graph edge, e.g.
                                   (user) -[likes]-> (Python)
                       embedding   a vector (Ollama `nomic-embed-text`, or
                                   the built-in local hash embedder)
                       importance, confidence, source, timestamps,
                       access_count / last_accessed

EXTRACT              parse_statement()    rule-based, Hinglish + English
                     SemanticMemory.extract()  rules + optional local LLM,
                                   every LLM item must be grounded in the
                                   user's own words (anti-hallucination)

RETRIEVE             retrieve()           hybrid: vector + keyword + graph
                     context_for_prompt() the relevant facts for the LLM
                     answer_question()    "Mera project kya hai?" answered
                                   from the knowledge graph
                     triples / neighbors / entities  graph queries

EVOLVE               a new single-valued fact (age, name, favourite ...)
                     replaces the old one; "likes X" vs "dislikes X" too.

Everything is local (NFR1.1, NFR1.3): the only network calls go to the
local Ollama server. Run `python semantic_memory.py --selftest` to verify.
"""

import json
import math
import re
import sqlite3
import struct
import sys
import time
import zlib
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import requests

from episodic_memory import normalize_text

DEFAULT_DB = "ragnarok_memory.db"

USER = "user"
KINDS = ("fact", "preference", "knowledge")

DEFAULT_MIN_SCORE = 0.15
W_SEMANTIC, W_LEXICAL, W_GRAPH = 0.55, 0.35, 0.10

NOT_SAVED_MESSAGE = (
    "Ye information abhi meri saved memory mein nahi hai. "
    "Main guess nahi karunga."
)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class Fact(NamedTuple):
    id: int
    text: str
    kind: str
    subject: Optional[str]
    relation: Optional[str]
    obj: Optional[str]
    category: str
    importance: int
    confidence: float
    source: str
    created_at: str
    updated_at: str
    last_confirmed: Optional[str]
    access_count: int
    last_accessed: Optional[str]


class Candidate(NamedTuple):
    """A memory that has been extracted but not necessarily saved yet."""
    kind: str
    text: str
    subject: Optional[str]
    relation: Optional[str]
    obj: Optional[str]
    importance: int = 3
    confidence: float = 1.0
    source: str = "manual"


class AddResult(NamedTuple):
    status: str                 # added | updated | duplicate
    fact: Fact
    replaced: List[Fact]        # old versions replaced by an update


class ScoredFact(NamedTuple):
    fact: Fact
    score: float
    semantic: float
    lexical: float
    graph: float


class QuestionSpec(NamedTuple):
    mode: str                   # relation | favorite | preference
    relation: Optional[str]
    noun: Optional[str]


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

SEM_STOP_WORDS = {
    # English
    "what", "which", "who", "how", "when", "where", "why", "is", "the", "my",
    "tell", "me", "about", "please", "do", "you", "know", "does", "can",
    "could", "i", "am", "are", "was", "were", "it", "this", "that", "a", "an",
    "for", "to", "of", "in", "on", "at", "and", "or", "with", "have", "has",
    "had", "did", "we", "show", "find", "remember",
    # Hindi / Hinglish
    "kya", "hai", "hain", "ka", "ki", "ke", "ko", "se", "mein", "me", "par",
    "pe", "batao", "btao", "bataiye", "mujhe", "mera", "meri", "mere", "ya",
    "aur", "kis", "kitna", "kitni", "kitne", "kaun", "kaunsa", "kaunsi",
    "kaunse", "kab", "kahan", "kyun", "kaise", "hota", "hoti", "hote", "ho",
    "hu", "hun", "h", "tha", "thi", "hua", "hui", "tum", "tumhe", "aap",
    "yaad", "kuch", "bhi", "to", "ye", "yeh", "wo", "woh",
}

PERSONAL_WORDS = {"my", "mera", "meri", "mere", "mujhe", "i"}

QUESTION_WORDS = {
    "kya", "kaun", "kaunsa", "kaunsi", "kaunse", "kis", "kitna", "kitni",
    "kitne", "kahan", "kab", "what", "which", "who", "how", "where", "when",
    "whats",
}

ATTR_ALIASES = {
    "naam": "name", "name": "name",
    "umar": "age", "umr": "age", "age": "age", "old": "age",
    "birthday": "birthday", "janamdin": "birthday", "dob": "birthday",
    "project": "project", "projects": "project",
    "college": "studies_at", "university": "studies_at",
    "institute": "studies_at",
    "team": "team",
    "hobby": "hobby", "hobbies": "hobby",
    "city": "lives_in", "shehar": "lives_in", "live": "lives_in",
}

SINGLE_VALUED = {"name", "age", "birthday", "lives_in", "studies_at",
                 "works_at"}

OPPOSITE = {
    "likes": ("dislikes",),
    "dislikes": ("likes", "prefers"),
    "prefers": ("dislikes",),
}

PREFERENCE_RELATIONS = ("likes", "prefers", "dislikes")
PREFERENCE_WORDS = {"pasand", "prefer", "preference", "preferences",
                    "language", "like", "likes"}
LANGUAGE_WORDS = ("hinglish", "hindi", "english")
NEGATIONS = {"nahi", "nahin", "not", "dont", "don", "hate", "dislike",
             "never"}

COPULAS = {"hai", "hain", "h", "hoon", "hu", "hun", "he", "hoga"}
FAV_WORDS = {"favorite", "favourite", "fav"}
SKIP_AFTER_ATTR = {"ka", "ki", "ke", "name", "naam"}
INTENSIFIERS = {"bahut", "bohot", "bhot", "kaafi", "zyada", "sabse",
                "really", "very"}
USER_ALIASES = {"user", "i", "me", "my", "main", "mera", "meri", "mere",
                "mujhe", "myself", "the user"}


def is_single_valued(relation: Optional[str]) -> bool:
    return bool(relation) and (
        relation in SINGLE_VALUED or relation.startswith("favorite_")
    )


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", normalize_text(text)).strip("_")
    return slug[:40]


def keyword_tokens(text: str) -> List[str]:
    seen, out = set(), []
    for word in normalize_text(text).split():
        if word in SEM_STOP_WORDS:
            continue
        if len(word) < 2 and not word.isdigit():
            continue
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def _match(word: str, tokens: set) -> bool:
    if word in tokens:
        return True
    if len(word) >= 4:
        for token in tokens:
            if len(token) >= 4 and (
                token.startswith(word) or word.startswith(token)
            ):
                return True
    return False


# ---------------------------------------------------------------------------
# Embedders (NFR1.3: local only)
# ---------------------------------------------------------------------------

class EmbeddingError(Exception):
    pass


def _normalize_vec(vec: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return list(vec)
    return [x / norm for x in vec]


class HashEmbedder:
    """Offline fallback: hashed word + character-trigram features.

    It needs no model and no network and is deterministic. It captures
    lexical similarity (shared words, spelling variants, plurals) but not
    true meaning; the Ollama embedder does that.
    """

    name = "local-hash-v1"
    baseline = 0.0
    dim = 512

    def baseline_for(self, name: str) -> float:
        return self.baseline

    def _add(self, vec, feature, weight):
        h = zlib.crc32(feature.encode("utf-8"))
        sign = 1.0 if (h >> 16) & 1 else -1.0
        vec[h % self.dim] += sign * weight

    def _vector(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        for token in normalize_text(text).split():
            if token in SEM_STOP_WORDS:
                continue
            if len(token) < 2 and not token.isdigit():
                continue
            # Normalize simple English plurals for offline similarity.
            if len(token) > 4 and token.endswith("ies"):
                token = token[:-3] + "y"
            elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
                token = token[:-1]
            self._add(vec, "w:" + token, 1.0)
            padded = f"^{token}$"
            for i in range(len(padded) - 2):
                self._add(vec, "c:" + padded[i:i + 3], 0.4)
        return _normalize_vec(vec)

    def embed(self, texts: Sequence[str], kind: str = "document"):
        return [self._vector(t) for t in texts], self.name


class OllamaEmbedder:
    """Embeddings from the local Ollama server (default nomic-embed-text).

    Setup once:  ollama pull nomic-embed-text
    """

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "nomic-embed-text", timeout: float = 30.0,
                 baseline: float = 0.35):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.baseline = baseline
        self.name = f"ollama:{model}"

    def baseline_for(self, name: str) -> float:
        return self.baseline

    def _prefix(self, kind: str) -> str:
        # nomic-embed-text is trained with task prefixes.
        if self.model.startswith("nomic-embed-text"):
            return "search_query: " if kind == "query" else "search_document: "
        return ""

    def embed(self, texts: Sequence[str], kind: str = "document"):
        prefix = self._prefix(kind)
        inputs = [prefix + t for t in texts]
        try:
            response = requests.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": inputs},
                timeout=self.timeout,
            )
            if response.status_code == 404:
                vectors = self._embed_legacy(inputs)
            else:
                response.raise_for_status()
                vectors = response.json()["embeddings"]
            return self._validate(vectors, len(inputs)), self.name
        except EmbeddingError:
            raise
        except (requests.exceptions.RequestException, KeyError, ValueError,
                TypeError) as error:
            raise EmbeddingError(
                f"Ollama embedding failed ({self.model}): {error}"
            ) from error

    def _embed_legacy(self, inputs):
        vectors = []
        for text in inputs:
            response = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=self.timeout,
            )
            response.raise_for_status()
            vectors.append(response.json()["embedding"])
        return vectors

    @staticmethod
    def _validate(vectors, expected):
        if not isinstance(vectors, list) or len(vectors) != expected:
            raise EmbeddingError("Embedding response has wrong length.")
        dim = None
        clean = []
        for vec in vectors:
            if not isinstance(vec, list) or not vec:
                raise EmbeddingError("Embedding vector is empty.")
            vec = [float(x) for x in vec]
            dim = dim or len(vec)
            if len(vec) != dim:
                raise EmbeddingError("Embedding dimensions differ.")
            clean.append(_normalize_vec(vec))
        return clean


class HybridEmbedder:
    """Use Ollama embeddings; if that fails, switch (for the rest of the
    session) to the local hash embedder so Ragnarok never breaks."""

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback
        self.using_fallback = False
        self.last_error: Optional[str] = None

    @property
    def active(self):
        return self.fallback if self.using_fallback else self.primary

    @property
    def name(self) -> str:
        return self.active.name

    def baseline_for(self, name: str) -> float:
        if name == self.primary.name:
            return self.primary.baseline_for(name)
        return self.fallback.baseline_for(name)

    def embed(self, texts: Sequence[str], kind: str = "document"):
        if not self.using_fallback:
            try:
                return self.primary.embed(texts, kind)
            except EmbeddingError as error:
                self.last_error = str(error)
                self.using_fallback = True
        return self.fallback.embed(texts, kind)

    def probe(self) -> str:
        """Try the primary embedder once; returns the active model name."""
        self.using_fallback = False
        self.last_error = None
        try:
            self.embed(["ping"], "query")
        except EmbeddingError as error:
            self.last_error = str(error)
            self.using_fallback = True
        return self.name


def _encode(vec: Sequence[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _decode(blob) -> Optional[List[float]]:
    if not blob or len(blob) % 4:
        return None
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


# ---------------------------------------------------------------------------
# Rule-based extraction (Hinglish + English)
# ---------------------------------------------------------------------------

_QUESTION_TOKENS_HI = {"kya", "kyun", "kyu", "kaise", "kab", "kahan", "kaun",
                       "kaunsa", "kaunsi", "kaunse", "kitna", "kitni",
                       "kitne"}
_QUESTION_STARTERS_EN = {"what", "why", "how", "when", "where", "who",
                         "which", "do", "does", "did", "is", "are", "can",
                         "could", "will", "would", "should"}


def is_question(text: str) -> bool:
    if "?" in text:
        return True
    words = normalize_text(text).split()
    if not words:
        return False
    if words[0] in _QUESTION_STARTERS_EN:
        return True
    return any(w in _QUESTION_TOKENS_HI for w in words)


def split_clauses(text: str) -> List[str]:
    """Split on sentence ends and on 'aur/and' before a new first-person
    statement. Decimals and versions (v2.0) are not split."""
    clauses = []
    for part in re.split(r"[.!]+(?=\s|$)|[;\n]+", text):
        pieces = re.split(
            r"\s+(?:aur|and)\s+(?=(?:mera|meri|mere|mujhe|my|i)\b)",
            part, flags=re.IGNORECASE,
        )
        for piece in pieces:
            piece = piece.strip(" ,")
            if piece:
                clauses.append(piece)
    return clauses


def _sentence(clause: str) -> str:
    clause = " ".join(clause.split()).strip()
    return clause if clause.endswith(("?", ".", "!")) else clause + "."


def _clean_value(value: str, max_len: int = 120) -> Optional[str]:
    value = " ".join(value.split()).strip(" ,;:\"'")
    if "," in value:
        head = value.split(",")[0].strip()
        if head:
            value = head
    # Remove trailing Hinglish copulas after trimming explanatory clauses.
    value = re.sub(
        r"\s+(?:hai|hain|hoon|hu|hun)$", "", value,
        flags=re.IGNORECASE
    ).strip()
    if not value or len(value) > max_len:
        return None
    if any(w in _QUESTION_TOKENS_HI for w in normalize_text(value).split()):
        return None
    return value


def _trim_intensifiers(tokens: List[str]) -> List[str]:
    while tokens and tokens[0].lower() in INTENSIFIERS:
        tokens = tokens[1:]
    while tokens and tokens[-1].lower() in INTENSIFIERS:
        tokens = tokens[:-1]
    return tokens


def _has_language_word(text: str) -> bool:
    tokens = set(normalize_text(text).split())
    return bool(tokens & set(LANGUAGE_WORDS)) or "explanations" in tokens \
        or "explanation" in tokens


def _importance_for(relation: Optional[str], obj: Optional[str]) -> int:
    if relation == "name":
        return 5
    if relation == "project":
        return 4
    if relation in ("likes", "prefers") and obj and _has_language_word(obj):
        return 5
    return 3


def _kind_for(relation: Optional[str]) -> str:
    if relation in PREFERENCE_RELATIONS or (
        relation and relation.startswith("favorite_")
    ):
        return "preference"
    if relation in ("is_a", "means"):
        return "knowledge"
    return "fact"


def _make(clause, relation, obj, subject=USER, source="manual") -> Optional[Candidate]:
    obj = _clean_value(obj or "")
    if not obj or not relation:
        return None
    return Candidate(
        kind=_kind_for(relation),
        text=_sentence(clause),
        subject=subject,
        relation=relation,
        obj=obj,
        importance=_importance_for(relation, obj),
        confidence=1.0,
        source=source,
    )


def _parse_attr(tokens: List[str], low: List[str]):
    """tokens exclude the leading mera/my. Returns (relation, value)."""
    while low and low[-1] in COPULAS:
        low, tokens = low[:-1], tokens[:-1]
    if len(low) < 2:
        return None
    if low[0] in FAV_WORDS:
        if len(low) < 3:
            return None
        noun = slugify(low[1])
        if noun == "colour":
            noun = "color"
        return ("favorite_" + noun, " ".join(tokens[2:])) if noun else None
    relation = ATTR_ALIASES.get(low[0])
    if not relation or low[0] == "old":
        return None
    i = 1
    while i < len(low) - 1 and low[i] in SKIP_AFTER_ATTR:
        i += 1
    return relation, " ".join(tokens[i:])


_EN_DISLIKE = re.compile(
    r"^i\s+(?:(?:do\s*not|don'?t|dont)\s+like|dislike|hate)\s+(?P<o>.+)$",
    re.I)
_EN_LIKE = re.compile(
    r"^i\s+(?:really\s+|also\s+)?(?:like|love|enjoy)\s+(?P<o>.+)$", re.I)
_EN_PREFER = re.compile(r"^i\s+(?:really\s+)?prefer\s+(?P<o>.+)$", re.I)
_EN_LIVE = re.compile(r"^i\s+live\s+in\s+(?P<o>.+)$", re.I)
_EN_STUDY = re.compile(
    r"^i\s+(?:study|am\s+studying)\s+(?:at|in)\s+(?P<o>.+)$", re.I)
_EN_WORK = re.compile(
    r"^i\s+(?:work|am\s+working)\s+(?:at|for|in)\s+(?P<o>.+)$", re.I)
_HI_PLACE = re.compile(
    r"^main\s+(?P<o>.+?)\s+(?:mein|me)\s+"
    r"(?:(?P<live>rehta|rehti|rahta|rahti)|(?P<study>padhta|padhti)|"
    r"(?P<work>kaam\s+karta|kaam\s+karti))(?:\s+(?:hoon|hu|hun|hai))?$",
    re.I)

_KN_EK = re.compile(
    r"^(?P<s>.+?)\s+ek\s+(?P<o>.+?)\s+(?:hai|hain|hota\s+hai|hoti\s+hai)$",
    re.I)
_KN_IS = re.compile(r"^(?P<s>.+?)\s+(?:is|are)\s+(?:a|an|the)\s+(?P<o>.+)$",
                    re.I)
_KN_MEANS_HI = re.compile(
    r"^(?P<s>.+?)\s+ka\s+matlab\s+(?P<o>.+?)\s+(?:hai|hota\s+hai)$", re.I)
_KN_MEANS_EN = re.compile(r"^(?P<s>.+?)\s+means\s+(?P<o>.+)$", re.I)


def _parse_knowledge(clause: str) -> Optional[Candidate]:
    for regex, relation in ((_KN_MEANS_HI, "means"), (_KN_EK, "is_a"),
                            (_KN_IS, "is_a"), (_KN_MEANS_EN, "means")):
        match = regex.match(clause.strip(" ,"))
        if not match:
            continue
        subject = " ".join(match.group("s").split())
        subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.I)
        if (not subject or len(subject.split()) > 6
                or subject.lower() in USER_ALIASES):
            continue
        obj = _clean_value(match.group("o"))
        if not obj or len(obj.split()) > 14:
            continue
        return Candidate("knowledge", _sentence(clause), subject, relation,
                         obj, 3, 1.0, "manual")
    return None


def _parse_clause(clause: str, mode: str) -> Optional[Candidate]:
    clause = clause.strip(" ,:")
    tokens = clause.split()
    if len(tokens) < 2:
        return None
    if mode == "knowledge":
        return _parse_knowledge(clause)

    low = [t.strip(",").lower() for t in tokens]
    first = low[0]

    # Hinglish: mera/meri/mere <attribute> <value> hai
    if first in ("mera", "meri", "mere"):
        parsed = _parse_attr(tokens[1:], low[1:])
        return _make(clause, *parsed) if parsed else None

    # Hinglish: mujhe <X> pasand hai / mujhe <X> pasand nahi hai
    if first == "mujhe" and "pasand" in low:
        idx = low.index("pasand")
        before, after = list(low[1:idx]), low[idx + 1:]
        allowed = COPULAS | {"nahi", "nahin", "aata", "aati", "aate"}
        if not set(after) <= allowed:
            return None
        negative = bool(set(after) & {"nahi", "nahin"})
        obj_tokens = tokens[1:idx]
        if before and before[-1] in ("nahi", "nahin"):
            negative = True
            obj_tokens = obj_tokens[:-1]
        obj_tokens = _trim_intensifiers(obj_tokens)
        if not obj_tokens:
            return None
        return _make(clause, "dislikes" if negative else "likes",
                     " ".join(obj_tokens))

    # English: my <attribute> is <value>
    if first == "my":
        rest, rest_low = tokens[1:], low[1:]
        if "is" in rest_low:
            idx = rest_low.index("is")
            rest = rest[:idx] + rest[idx + 1:]
            rest_low = rest_low[:idx] + rest_low[idx + 1:]
            parsed = _parse_attr(rest, rest_low)
            return _make(clause, *parsed) if parsed else None
        return None

    # English first-person statements
    for regex, relation in ((_EN_DISLIKE, "dislikes"), (_EN_LIKE, "likes"),
                            (_EN_PREFER, "prefers"), (_EN_LIVE, "lives_in"),
                            (_EN_STUDY, "studies_at"), (_EN_WORK, "works_at")):
        match = regex.match(clause)
        if match:
            obj = match.group("o")
            if relation == "prefers":
                obj = re.split(r"\s+(?:over|rather\s+than|instead\s+of)\s+",
                               obj, flags=re.I)[0]
            obj_tokens = _trim_intensifiers(obj.split())
            return _make(clause, relation, " ".join(obj_tokens))

    # Hinglish: main <place> mein rehta/padhta/kaam karta hoon
    match = _HI_PLACE.match(clause)
    if match:
        relation = ("lives_in" if match.group("live")
                    else "studies_at" if match.group("study") else "works_at")
        return _make(clause, relation, match.group("o"))

    return None


def parse_statement(text: str, mode: str = "user") -> List[Candidate]:
    """Rule-based extraction. mode='user' finds facts/preferences about the
    user; mode='knowledge' finds general knowledge ('X ek Y hai')."""
    out, seen = [], set()
    for clause in split_clauses(text):
        if is_question(clause):
            continue
        cand = _parse_clause(clause, mode)
        if not cand:
            continue
        key = (normalize_text(cand.subject or ""), cand.relation,
               normalize_text(cand.obj or ""))
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


_PREFERENCE_TEXT_WORDS = {"pasand", "prefer", "prefers", "favorite", "like",
                          "likes", "love", "hate", "dislike", "enjoy"}


def classify_free_text(clause: str, kind: Optional[str] = None,
                       source: str = "manual") -> Candidate:
    """A memory that does not fit a known pattern is still stored as text."""
    text = _sentence(clause)
    tokens = set(normalize_text(text).split())
    kind = kind or ("preference" if tokens & _PREFERENCE_TEXT_WORDS
                    else "fact")
    importance = 5 if kind == "preference" and _has_language_word(text) else 3
    return Candidate(kind, text, None, None, None, importance, 1.0, source)


_CATEGORY_RULES = {
    "project": ("project", "ragnarok", "jalnetra", "gassentinel",
                "infradrishti"),
    "preference": ("prefer", "favorite", "favourite", "pasand", "language"),
    "education": ("college", "university", "student", "department", "degree"),
    "achievement": ("hackathon", "winner", "prize", "award", "rank"),
}


def guess_category(text: str) -> str:
    lowered = text.lower()
    for label, terms in _CATEGORY_RULES.items():
        if any(term in lowered for term in terms):
            return label
    return "general"


# ---------------------------------------------------------------------------
# LLM extraction helpers
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM = (
    "You extract memorable information from a user's message for a personal "
    "assistant. Output ONLY valid JSON of the form {\"facts\": [...]}. "
    "Each item has the keys: kind (\"fact\", \"preference\" or "
    "\"knowledge\"), subject (use \"user\" for facts about the user), "
    "relation (short snake_case such as name, age, project, likes, "
    "dislikes, prefers, lives_in, studies_at, works_at, favorite_color, "
    "is_a), object (a short value copied from the message), text (one clear "
    "sentence in the user's own language), importance (1 to 5). Include only "
    "information that is explicitly stated. Never guess. If nothing is "
    "worth remembering output {\"facts\": []}."
)


def _clean_text(value, max_len: int) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(value or ""))
    return " ".join(text.split())[:max_len].strip()


def _grounded(cand: Candidate, source_text: str) -> bool:
    """The extracted value must really appear in what the user wrote."""
    source_tokens = set(normalize_text(source_text).split())
    target = cand.obj or cand.text
    words = [w for w in normalize_text(target).split()
             if w not in SEM_STOP_WORDS and len(w) > 1 or w.isdigit()]
    if not words:
        return False
    hits = sum(1 for w in words if _match(w, source_tokens))
    return hits / len(words) >= 0.6


def parse_llm_output(raw: str, source_text: str) -> List[Candidate]:
    """Turn the model's JSON into validated, grounded candidates. Anything
    malformed or ungrounded is silently dropped."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()
    data = None
    try:
        data = json.loads(raw)
    except ValueError:
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = raw.find(opener), raw.rfind(closer)
            if 0 <= start < end:
                try:
                    data = json.loads(raw[start:end + 1])
                    break
                except ValueError:
                    continue
    if isinstance(data, dict):
        items = data.get("facts") or data.get("items") or data.get("memories")
        if not isinstance(items, list):
            items = [data] if ("text" in data or "object" in data) else []
    elif isinstance(data, list):
        items = data
    else:
        items = []

    out = []
    for item in items[:5]:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind", "fact")).lower().strip()
        if kind not in KINDS:
            kind = "fact"
        subject = _clean_text(item.get("subject"), 60) or USER
        if subject.lower() in USER_ALIASES:
            subject = USER
        relation = slugify(_clean_text(item.get("relation"), 40)) or None
        obj = _clean_text(item.get("object", item.get("obj")), 120) or None
        text = _clean_text(item.get("text"), 240)
        if not text:
            if not (relation and obj):
                continue
            text = f"{subject} {relation.replace('_', ' ')} {obj}."
        elif not obj and not relation:
            subject = relation = None
        try:
            importance = max(1, min(5, int(item.get("importance", 3))))
        except (TypeError, ValueError):
            importance = 3
        if relation and obj:
            kind = kind if kind != "fact" else _kind_for(relation)
        cand = Candidate(kind, text, subject if relation else None,
                         relation, obj if relation else None,
                         importance, 0.6, "extracted")
        if _grounded(cand, source_text):
            out.append(cand)
    return out


# ---------------------------------------------------------------------------
# Question understanding
# ---------------------------------------------------------------------------

def question_spec(question: str) -> Optional[QuestionSpec]:
    """Understand short personal questions such as 'Mera project kya hai?'.
    Returns None for everything else, so the model answers normally."""
    words = normalize_text(question).split()
    if not words or len(words) > 10:
        return None
    word_set = set(words)
    if not (word_set & PERSONAL_WORDS and word_set & QUESTION_WORDS):
        return None

    for i, word in enumerate(words):
        if word in FAV_WORDS:
            noun = next(
                (w for w in words[i + 1:]
                 if w not in SEM_STOP_WORDS and w not in QUESTION_WORDS
                 and w not in COPULAS),
                None,
            )
            return QuestionSpec("favorite", None, noun)

    for word in words:
        relation = ATTR_ALIASES.get(word)
        if relation:
            return QuestionSpec("relation", relation, None)

    if word_set & PREFERENCE_WORDS:
        return QuestionSpec("preference", None, None)
    return None


# ---------------------------------------------------------------------------
# The semantic memory store
# ---------------------------------------------------------------------------

_COLUMNS = ("id, fact, kind, subject, relation, obj, category, importance, "
            "confidence, source, created_at, updated_at, last_confirmed, "
            "access_count, last_accessed")


def _row_to_fact(row) -> Fact:
    (fact_id, text, kind, subject, relation, obj, category, importance,
     confidence, source, created_at, updated_at, last_confirmed,
     access_count, last_accessed) = row
    return Fact(
        id=fact_id, text=text, kind=kind or "fact", subject=subject,
        relation=relation, obj=obj, category=category or "general",
        importance=importance or 3,
        confidence=1.0 if confidence is None else confidence,
        source=source or "manual", created_at=created_at,
        updated_at=updated_at or created_at, last_confirmed=last_confirmed,
        access_count=access_count or 0, last_accessed=last_accessed,
    )


class SemanticMemory:
    """Structured long-term memory: facts, preferences, knowledge."""

    def __init__(self, db_file: str = DEFAULT_DB, embedder=None,
                 clock: Callable[[], datetime] = datetime.now):
        self.db_file = db_file
        self.embedder = embedder or HashEmbedder()
        self._clock = clock
        self.setup()

    # -- infrastructure ----------------------------------------------------

    def _now(self) -> str:
        return self._clock().isoformat(timespec="seconds")

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
        """Create/extend the facts table and upgrade legacy rows."""
        with self._db() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fact TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(facts)")}
            additions = {
                "category": "TEXT DEFAULT 'general'",
                "importance": "INTEGER DEFAULT 3",
                "last_confirmed": "TEXT",
                "source": "TEXT DEFAULT 'manual'",
                "kind": "TEXT",
                "subject": "TEXT",
                "relation": "TEXT",
                "obj": "TEXT",
                "confidence": "REAL DEFAULT 1.0",
                "updated_at": "TEXT",
                "access_count": "INTEGER DEFAULT 0",
                "last_accessed": "TEXT",
                "embedding": "BLOB",
                "embed_model": "TEXT",
            }
            for name, definition in additions.items():
                if name not in columns:
                    db.execute(
                        f"ALTER TABLE facts ADD COLUMN {name} {definition}"
                    )
            db.execute("CREATE INDEX IF NOT EXISTS idx_facts_relation "
                       "ON facts(relation)")

            # Legacy rows (kind IS NULL) get structure from the parser.
            legacy = db.execute(
                "SELECT id, fact, category, importance, created_at "
                "FROM facts WHERE kind IS NULL"
            ).fetchall()
            for fact_id, text, category, importance, created_at in legacy:
                cands = parse_statement(text) or parse_statement(
                    text, "knowledge")
                if len(cands) == 1:
                    c = cands[0]
                    kind, subject, relation, obj = (
                        c.kind, c.subject, c.relation, c.obj)
                    new_importance = c.importance
                else:
                    c = classify_free_text(text)
                    kind, subject, relation, obj = c.kind, None, None, None
                    new_importance = c.importance
                if not category or category == "general":
                    category = guess_category(text)
                db.execute(
                    "UPDATE facts SET kind=?, subject=?, relation=?, obj=?, "
                    "category=?, importance=?, updated_at=?, "
                    "confidence=COALESCE(confidence, 1.0) WHERE id=?",
                    (kind, subject, relation, obj, category,
                     max(importance or 3, new_importance),
                     created_at, fact_id),
                )

    # -- reading -----------------------------------------------------------

    def _load(self):
        with self._db() as db:
            rows = db.execute(
                f"SELECT {_COLUMNS}, embedding, embed_model FROM facts "
                "ORDER BY id"
            ).fetchall()
        return [(_row_to_fact(r[:15]), r[15], r[16]) for r in rows]

    def all_facts(self, kinds: Optional[Sequence[str]] = None) -> List[Fact]:
        facts = [item[0] for item in self._load()]
        if kinds:
            facts = [f for f in facts if f.kind in kinds]
        return facts

    def get(self, fact_id: int) -> Optional[Fact]:
        with self._db() as db:
            row = db.execute(
                f"SELECT {_COLUMNS} FROM facts WHERE id = ?", (fact_id,)
            ).fetchone()
        return _row_to_fact(row) if row else None

    def count(self) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def stats(self) -> dict:
        items = self._load()
        by_kind = Counter(f.kind for f, _, _ in items)
        models = Counter(m for _, blob, m in items if blob and m)
        return {
            "total": len(items),
            "by_kind": {k: by_kind.get(k, 0) for k in KINDS},
            "triples": sum(1 for f, _, _ in items
                           if f.subject and f.relation and f.obj),
            "embedded": sum(1 for _, blob, _ in items if blob),
            "models": dict(models),
            "embedder": getattr(self.embedder, "name", "?"),
        }

    # -- writing -------------------------------------------------------------

    def _embed_one(self, text: str):
        try:
            vectors, name = self.embedder.embed([text], "document")
            return _encode(vectors[0]), name
        except EmbeddingError:
            return None, None

    def _duplicate_of(self, cand: Candidate) -> Optional[Fact]:
        norm = normalize_text(cand.text)
        triple = None
        if cand.subject and cand.relation and cand.obj:
            triple = (normalize_text(cand.subject), cand.relation,
                      normalize_text(cand.obj))
        for fact in self.all_facts():
            if normalize_text(fact.text) == norm:
                return fact
            if triple and fact.subject and fact.relation and fact.obj and (
                (normalize_text(fact.subject), fact.relation,
                 normalize_text(fact.obj)) == triple
            ):
                return fact
        return None

    def find_conflicts(self, cand: Candidate) -> List[Fact]:
        """Existing facts the new one would replace (age changed, 'likes X'
        vs 'dislikes X', a new favourite colour ...)."""
        if not (cand.subject and cand.relation and cand.obj):
            return []
        subject = normalize_text(cand.subject)
        obj = normalize_text(cand.obj)
        out = []
        for fact in self.all_facts():
            if not (fact.subject and fact.relation and fact.obj):
                continue
            if normalize_text(fact.subject) != subject:
                continue
            fact_obj = normalize_text(fact.obj)
            if (fact.relation == cand.relation and fact_obj != obj
                    and is_single_valued(cand.relation)):
                out.append(fact)
            elif fact.relation in OPPOSITE.get(cand.relation, ()) \
                    and fact_obj == obj:
                out.append(fact)
        return out

    def add(self, cand: Candidate, replace_conflicts: bool = True) -> AddResult:
        text = _clean_text(cand.text, 1000)
        if not text:
            raise ValueError("Fact khaali nahi ho sakta.")
        if cand.kind not in KINDS:
            raise ValueError(f"Unknown kind: {cand.kind!r}")
        cand = cand._replace(
            text=text,
            importance=max(1, min(5, int(cand.importance))),
            confidence=max(0.0, min(1.0, float(cand.confidence))),
        )

        existing = self._duplicate_of(cand)
        if existing:
            return AddResult("duplicate", existing, [])

        conflicts = self.find_conflicts(cand) if replace_conflicts else []
        blob, model = self._embed_one(text)
        now = self._now()
        category = guess_category(text)

        with self._db() as db:
            if conflicts:
                keep = conflicts[0]
                db.execute(
                    "UPDATE facts SET fact=?, kind=?, subject=?, relation=?, "
                    "obj=?, category=?, importance=?, confidence=?, source=?, "
                    "updated_at=?, last_confirmed=?, embedding=?, "
                    "embed_model=? WHERE id=?",
                    (text, cand.kind, cand.subject, cand.relation, cand.obj,
                     category, cand.importance, cand.confidence, cand.source,
                     now, now, blob, model, keep.id),
                )
                for old in conflicts[1:]:
                    db.execute("DELETE FROM facts WHERE id = ?", (old.id,))
                fact_id, status = keep.id, "updated"
            else:
                cursor = db.execute(
                    "INSERT INTO facts (fact, created_at, category, "
                    "importance, last_confirmed, source, kind, subject, "
                    "relation, obj, confidence, updated_at, access_count, "
                    "embedding, embed_model) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
                    (text, now, category, cand.importance, now, cand.source,
                     cand.kind, cand.subject, cand.relation, cand.obj,
                     cand.confidence, now, blob, model),
                )
                fact_id, status = cursor.lastrowid, "added"
        return AddResult(status, self.get(fact_id), conflicts)

    def remember(self, text: str, source: str = "manual") -> List[AddResult]:
        """Save what the user typed. Known patterns become structured
        triples; anything else is stored as plain text."""
        clauses = split_clauses(text)
        if not clauses:
            raise ValueError("Fact khaali nahi ho sakta.")
        results = []
        for clause in clauses:
            cand = _parse_clause(clause, "user") or classify_free_text(clause)
            results.append(self.add(cand._replace(source=source)))
        return results

    def learn(self, text: str) -> List[AddResult]:
        """Save general knowledge ('Python ek programming language hai')."""
        clauses = split_clauses(text)
        if not clauses:
            raise ValueError("Knowledge khaali nahi ho sakta.")
        results = []
        for clause in clauses:
            cand = _parse_clause(clause, "knowledge") or classify_free_text(
                clause, kind="knowledge")
            results.append(self.add(cand._replace(source="learned")))
        return results

    def delete(self, fact_id: int) -> Optional[Fact]:
        fact = self.get(fact_id)
        if not fact:
            return None
        with self._db() as db:
            db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return fact

    def _touch(self, ids: Iterable[int]) -> None:
        ids = list(ids)
        if not ids:
            return
        with self._db() as db:
            db.execute(
                "UPDATE facts SET access_count = COALESCE(access_count, 0) + 1, "
                f"last_accessed = ? WHERE id IN ({','.join('?' * len(ids))})",
                [self._now()] + ids,
            )

    # -- embeddings ------------------------------------------------------------

    def _ensure_embeddings(self, model_name: str) -> Optional[str]:
        """Embed every fact not yet embedded in `model_name`'s space.
        Returns the model actually used, or None if nothing was needed or
        the embedder failed."""
        stale = [(f.id, f.text) for f, blob, model in self._load()
                 if not blob or model != model_name]
        if not stale:
            return None
        try:
            vectors, used = self.embedder.embed([t for _, t in stale],
                                                "document")
        except EmbeddingError:
            return None
        with self._db() as db:
            for (fact_id, _), vec in zip(stale, vectors):
                db.execute(
                    "UPDATE facts SET embedding=?, embed_model=? WHERE id=?",
                    (_encode(vec), used, fact_id),
                )
        return used

    def reindex(self) -> Tuple[int, str]:
        """Re-embed every fact with the current embedder."""
        items = self._load()
        if not items:
            return 0, getattr(self.embedder, "name", "?")
        texts = [f.text for f, _, _ in items]
        vectors, used = [], None
        for start in range(0, len(texts), 32):
            batch, used = self.embedder.embed(texts[start:start + 32],
                                              "document")
            vectors.extend(batch)
        with self._db() as db:
            for (fact, _, _), vec in zip(items, vectors):
                db.execute(
                    "UPDATE facts SET embedding=?, embed_model=? WHERE id=?",
                    (_encode(vec), used, fact.id),
                )
        return len(items), used

    # -- retrieval ------------------------------------------------------------

    def retrieve(self, query: str, k: int = 5,
                 kinds: Optional[Sequence[str]] = None,
                 min_score: float = DEFAULT_MIN_SCORE,
                 touch: bool = True) -> List[ScoredFact]:
        """Hybrid retrieval: vector similarity + keyword overlap + graph
        entity match, best first."""
        query = (query or "").strip()
        if not query or not self.count():
            return []

        qvec, qname = None, None
        for _ in range(3):
            try:
                vectors, qname = self.embedder.embed([query], "query")
            except EmbeddingError:
                qvec = qname = None
                break
            qvec = vectors[0]
            used = self._ensure_embeddings(qname)
            if used is None or used == qname:
                break

        baseline = 0.0
        if qname:
            baseline = self.embedder.baseline_for(qname)

        qtokens = keyword_tokens(query)
        expanded = set(qtokens)
        for token in qtokens:
            relation = ATTR_ALIASES.get(token)
            if relation:
                expanded.update(relation.split("_"))

        scored = []
        for fact, blob, model in self._load():
            if kinds and fact.kind not in kinds:
                continue

            semantic = 0.0
            vec = _decode(blob) if (qvec and model == qname) else None
            if vec and len(vec) == len(qvec):
                cosine = sum(a * b for a, b in zip(qvec, vec))
                semantic = max(0.0, (cosine - baseline) / (1.0 - baseline))

            fact_tokens = set(normalize_text(
                " ".join([fact.text, fact.subject or "",
                          (fact.relation or "").replace("_", " "),
                          fact.obj or ""])).split())
            lexical = 0.0
            if qtokens:
                matched = sum(1 for t in qtokens if _match(t, fact_tokens))
                relation_hit = sum(
                    1 for t in expanded - set(qtokens)
                    if _match(t, fact_tokens))
                lexical = min(1.0, (matched + 0.5 * relation_hit)
                              / len(qtokens))

            graph = 0.0
            entity_tokens = set()
            for part in (fact.subject, fact.obj):
                if part and normalize_text(part) != USER:
                    entity_tokens.update(normalize_text(part).split())
            if entity_tokens and any(_match(t, entity_tokens)
                                     for t in qtokens):
                graph = 1.0

            score = (W_SEMANTIC * semantic + W_LEXICAL * lexical
                     + W_GRAPH * graph)
            score *= 1.0 + 0.05 * (fact.importance - 3)
            if score >= min_score:
                scored.append(ScoredFact(fact, score, semantic, lexical,
                                         graph))

        scored.sort(key=lambda s: (s.score, s.fact.importance,
                                   s.fact.updated_at, s.fact.id),
                    reverse=True)
        top = scored[:max(1, k)]
        if touch:
            self._touch(s.fact.id for s in top)
        return top

    def context_for_prompt(self, query: str, k: int = 6,
                           max_always: int = 5) -> str:
        """Facts to put in the LLM prompt: the most important ones plus
        whatever is relevant to `query`. Empty string if none."""
        always = sorted(
            (f for f in self.all_facts() if f.importance >= 5),
            key=lambda f: (f.updated_at, f.id), reverse=True,
        )[:max_always]
        chosen = list(always)
        seen = {f.id for f in chosen}
        for item in self.retrieve(query, k=k):
            if item.fact.id not in seen:
                seen.add(item.fact.id)
                chosen.append(item.fact)
        return "\n".join(f"- {f.text}" for f in chosen)

    def preferred_language(self) -> Optional[str]:
        """Latest saved language preference. 'Mujhe Hinglish pasand nahi'
        (a dislike) is ignored."""
        ordered = sorted(self.all_facts(("preference",)),
                         key=lambda f: (f.updated_at, f.id), reverse=True)
        for fact in ordered:
            if fact.relation == "dislikes":
                continue
            tokens = set(normalize_text(fact.text).split())
            if fact.relation is None and tokens & NEGATIONS:
                continue
            for language in LANGUAGE_WORDS:
                if language in tokens:
                    return language
        return None

    # -- question answering ----------------------------------------------------

    def answer_question(self, question: str) -> Optional[str]:
        """Answer 'Mera project kya hai?', 'Meri umar kitni hai?',
        'Mera favourite colour kya hai?' from the knowledge graph.
        Returns None if the text is not such a question."""
        spec = question_spec(question)
        if spec is None:
            return None

        user_facts = [f for f in self.all_facts()
                      if f.subject and normalize_text(f.subject) == USER
                      and f.relation and f.obj]
        facts: List[Fact] = []

        if spec.mode == "relation":
            facts = [f for f in user_facts if f.relation == spec.relation]
        elif spec.mode == "favorite":
            wanted = slugify(spec.noun) if spec.noun else None
            facts = [f for f in user_facts
                     if f.relation.startswith("favorite_")
                     and (wanted is None or f.relation == "favorite_" + wanted)]
        else:
            facts = self._preference_facts(question, user_facts)

        facts.sort(key=lambda f: (f.updated_at, f.id), reverse=True)
        facts = facts[:5]
        if not facts:
            return NOT_SAVED_MESSAGE
        self._touch(f.id for f in facts)
        if len(facts) == 1:
            return f"Saved memory ke hisaab se: {facts[0].text}"
        return "Saved memory ke hisaab se:\n" + "\n".join(
            f"- {f.text}" for f in facts)

    def _preference_facts(self, question: str, user_facts: List[Fact]):
        generic = {"pasand", "prefer", "preference", "preferences", "like",
                   "likes"}
        specific = [t for t in keyword_tokens(question) if t not in generic]
        if not specific:
            return [f for f in user_facts
                    if f.relation in ("likes", "prefers")]
        hits = self.retrieve(question, k=3, kinds=("preference",),
                             min_score=0.1, touch=False)
        return [h.fact for h in hits if h.fact.relation in PREFERENCE_RELATIONS
                or h.fact.relation is None]

    # -- extraction -------------------------------------------------------------

    def extract(self, text: str, llm: Optional[Callable[[str, str], str]] = None
                ) -> List[Candidate]:
        """Find memorable information in `text` (rules, plus the local LLM
        when `llm(system_prompt, user_text)` is given). Nothing is saved;
        already-known facts are filtered out."""
        candidates = parse_statement(text)
        if llm is not None:
            raw = llm(EXTRACT_SYSTEM, text)
            candidates += [c for c in parse_llm_output(raw, text)]

        out, seen = [], set()
        for cand in candidates:
            key = normalize_text(cand.text)
            triple = (normalize_text(cand.subject or ""), cand.relation,
                      normalize_text(cand.obj or ""))
            if key in seen or (cand.relation and triple in seen):
                continue
            seen.add(key)
            if cand.relation:
                seen.add(triple)
            if self._duplicate_of(cand):
                continue
            out.append(cand._replace(
                source="extracted" if cand.source == "manual" else cand.source
            ))
        return out

    # -- knowledge graph ----------------------------------------------------------

    def triples(self, subject: Optional[str] = None,
                relation: Optional[str] = None,
                obj: Optional[str] = None) -> List[Fact]:
        """Graph edges (facts that have a triple), optionally filtered."""
        ns = normalize_text(subject) if subject else None
        no = normalize_text(obj) if obj else None
        rel = slugify(relation) if relation else None
        out = []
        for fact in self.all_facts():
            if not (fact.subject and fact.relation and fact.obj):
                continue
            if ns and normalize_text(fact.subject) != ns:
                continue
            if rel and fact.relation != rel:
                continue
            if no and normalize_text(fact.obj) != no:
                continue
            out.append(fact)
        return out

    def neighbors(self, entity: str) -> List[Tuple[Fact, str]]:
        """Edges touching `entity`: (fact, 'out') when it is the subject,
        (fact, 'in') when it is the object."""
        name = normalize_text(entity)
        out = []
        for fact in self.triples():
            if normalize_text(fact.subject) == name:
                out.append((fact, "out"))
            if normalize_text(fact.obj) == name:
                out.append((fact, "in"))
        return out

    def entities(self) -> List[Tuple[str, int]]:
        """All nodes of the graph with their degree, busiest first."""
        degree: Counter = Counter()
        display = {}
        for fact in self.triples():
            for part in (fact.subject, fact.obj):
                key = normalize_text(part)
                degree[key] += 1
                display.setdefault(key, part)
        return [(display[k], n) for k, n in degree.most_common()]


# ---------------------------------------------------------------------------
# Self-test:  python semantic_memory.py --selftest
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

    def triple(c):
        return (c.subject, c.relation, c.obj)

    folder = tempfile.mkdtemp()

    # ---- statement parsing ------------------------------------------------
    cases = [
        ("Mera project Ragnarok hai.", (USER, "project", "Ragnarok")),
        ("Mere project ka naam Ragnarok hai", (USER, "project", "Ragnarok")),
        ("Mera naam Rahul hai", (USER, "name", "Rahul")),
        ("Meri umar 20 saal hai", (USER, "age", "20 saal")),
        ("Mera favourite colour blue hai", (USER, "favorite_color", "blue")),
        ("Mera favourite food dark chocolate hai",
         (USER, "favorite_food", "dark chocolate")),
        ("Mujhe Python pasand hai", (USER, "likes", "Python")),
        ("Mujhe Python bahut pasand hai", (USER, "likes", "Python")),
        ("Mujhe simple Hinglish mein explanations pasand hain.",
         (USER, "likes", "simple Hinglish mein explanations")),
        ("Mujhe Java pasand nahi hai", (USER, "dislikes", "Java")),
        ("Mujhe Java nahi pasand", (USER, "dislikes", "Java")),
        ("My name is Rahul", (USER, "name", "Rahul")),
        ("My favorite color is blue", (USER, "favorite_color", "blue")),
        ("My project name is Ragnarok", (USER, "project", "Ragnarok")),
        ("I like Python", (USER, "likes", "Python")),
        ("I really love hiking", (USER, "likes", "hiking")),
        ("I don't like spicy food", (USER, "dislikes", "spicy food")),
        ("I hate Java", (USER, "dislikes", "Java")),
        ("I prefer tea over coffee", (USER, "prefers", "tea")),
        ("I live in Nagpur", (USER, "lives_in", "Nagpur")),
        ("I study at VNIT", (USER, "studies_at", "VNIT")),
        ("I work at Acme", (USER, "works_at", "Acme")),
        ("Main Nagpur mein rehta hoon", (USER, "lives_in", "Nagpur")),
        ("Main VNIT mein padhta hoon", (USER, "studies_at", "VNIT")),
        ("Mera project v2.0 hai", (USER, "project", "v2.0")),
        ("Mera project Ragnarok hai, jo ek local AI hai",
         (USER, "project", "Ragnarok")),
    ]
    for text, expected in cases:
        found = parse_statement(text)
        check(len(found) == 1 and triple(found[0]) == expected,
              f"parse {text!r} -> {found}")

    check(parse_statement("Mera project Ragnarok hai.")[0].text
          == "Mera project Ragnarok hai.", "sentence text kept")
    check(parse_statement("Mujhe simple Hinglish mein explanations pasand "
                          "hain.")[0].importance == 5,
          "language preference importance 5")
    check(parse_statement("Mera naam Rahul hai")[0].importance == 5,
          "name importance 5")
    check(parse_statement("Mujhe Python pasand hai")[0].kind == "preference",
          "likes is a preference")

    two = parse_statement("Mera naam Rahul hai aur mujhe Python pasand hai")
    check([triple(c) for c in two] ==
          [(USER, "name", "Rahul"), (USER, "likes", "Python")],
          "two clauses split")

    for text in ["Mera project kya hai?", "Mujhe kya pasand hai",
                 "What is my name?", "Recursion kya hota hai?",
                 "Hello", "Mera code error de raha hai", "I like"]:
        check(parse_statement(text) == [], f"not a statement: {text!r}")

    check(is_question("Mera naam kya hai") and is_question("is it ok")
          and not is_question("Mera naam Rahul hai"), "is_question")
    check(split_clauses("Version 2.0 is out. Done!") ==
          ["Version 2.0 is out", "Done"], "split keeps decimals")

    kn = parse_statement("Python ek programming language hai", "knowledge")
    check(len(kn) == 1 and triple(kn[0]) ==
          ("Python", "is_a", "programming language")
          and kn[0].kind == "knowledge", "knowledge ek")
    kn = parse_statement("Python is a programming language", "knowledge")
    check(triple(kn[0]) == ("Python", "is_a", "programming language"),
          "knowledge is a")
    kn = parse_statement("API ka matlab Application Programming Interface "
                         "hai", "knowledge")
    check(triple(kn[0]) == ("API", "means",
                            "Application Programming Interface"),
          "knowledge matlab")
    kn = parse_statement("RAM means random access memory", "knowledge")
    check(triple(kn[0]) == ("RAM", "means", "random access memory"),
          "knowledge means")
    check(parse_statement("I am a student", "knowledge") == [],
          "first-person is not knowledge")

    free = classify_free_text("python acchi hai")
    check(free.kind == "fact" and free.relation is None
          and free.text == "python acchi hai.", "free text")
    check(classify_free_text("Mujhe chai bahut pasand").kind == "preference",
          "free text preference")

    # ---- embedders ----------------------------------------------------------
    hasher = HashEmbedder()
    (v1, v2, v3), name = hasher.embed(
        ["Mujhe Python pasand hai", "Python mujhe pasand hai",
         "Aaj mausam accha hai"])
    cos = lambda a, b: sum(x * y for x, y in zip(a, b))
    check(name == "local-hash-v1" and len(v1) == 512, "hash embedder shape")
    check(abs(cos(v1, v1) - 1.0) < 1e-9, "unit length")
    check(cos(v1, v2) > 0.99 and cos(v1, v3) < 0.2,
          "hash similarity: same words close, unrelated far")
    check(hasher.embed(["Python"])[0][0] == hasher.embed(["Python"])[0][0],
          "hash embedder deterministic")
    check(cos(hasher.embed(["tasks"])[0][0], hasher.embed(["task"])[0][0])
          > 0.5, "plural similarity")
    check(set(hasher.embed(["hai kya"])[0][0]) == {0.0}, "stopword-only text")
    check(_decode(_encode([0.5, -1.0])) == [0.5, -1.0], "vector blob round trip")

    # Ollama embedder against a mocked server
    class FakeResponse:
        def __init__(self, status=200, payload=None):
            self.status_code, self.payload = status, payload or {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(str(self.status_code))

        def json(self):
            return self.payload

    sent = []
    original_post = requests.post

    def fake_post(url, json=None, timeout=None):
        sent.append((url, json))
        if url.endswith("/api/embed"):
            return FakeResponse(200, {"embeddings": [[3.0, 4.0]
                                                     for _ in json["input"]]})
        raise AssertionError("unexpected url " + url)

    requests.post = fake_post
    try:
        ollama = OllamaEmbedder("http://x:1", "nomic-embed-text")
        vecs, name = ollama.embed(["a", "b"], "query")
        check(name == "ollama:nomic-embed-text" and len(vecs) == 2
              and abs(vecs[0][0] - 0.6) < 1e-9 and abs(vecs[0][1] - 0.8) < 1e-9,
              "ollama embeddings normalized")
        check(sent[0][1]["input"] == ["search_query: a", "search_query: b"],
              "nomic query prefix")
        ollama.embed(["a"], "document")
        check(sent[-1][1]["input"] == ["search_document: a"],
              "nomic document prefix")
        check(OllamaEmbedder("http://x", "other").embed(["a"])[0] and
              sent[-1][1]["input"] == ["a"], "no prefix for other models")

        # legacy endpoint fallback
        def legacy_post(url, json=None, timeout=None):
            if url.endswith("/api/embed"):
                return FakeResponse(404)
            return FakeResponse(200, {"embedding": [1.0, 0.0]})
        requests.post = legacy_post
        vecs, _ = OllamaEmbedder("http://x").embed(["a", "b"])
        check(vecs == [[1.0, 0.0], [1.0, 0.0]], "legacy /api/embeddings")

        # failures
        def broken_post(url, json=None, timeout=None):
            raise requests.exceptions.ConnectionError("down")
        requests.post = broken_post
        try:
            OllamaEmbedder("http://x").embed(["a"])
            check(False, "should raise EmbeddingError")
        except EmbeddingError:
            check(True, "connection error -> EmbeddingError")

        def bad_shape_post(url, json=None, timeout=None):
            return FakeResponse(200, {"embeddings": [[1.0]]})
        requests.post = bad_shape_post
        try:
            OllamaEmbedder("http://x").embed(["a", "b"])
            check(False, "should raise on wrong length")
        except EmbeddingError:
            check(True, "wrong length -> EmbeddingError")

        # hybrid fallback
        requests.post = broken_post
        hybrid = HybridEmbedder(OllamaEmbedder("http://x"), HashEmbedder())
        check(hybrid.probe() == "local-hash-v1" and hybrid.using_fallback
              and "failed" in hybrid.last_error, "hybrid falls back")
        requests.post = fake_post
        check(hybrid.probe() == "ollama:nomic-embed-text"
              and not hybrid.using_fallback, "hybrid recovers on probe")
    finally:
        requests.post = original_post

    # ---- legacy database migration ---------------------------------------------
    legacy_db = os.path.join(folder, "legacy.db")
    connection = sqlite3.connect(legacy_db)
    connection.execute(
        "CREATE TABLE facts (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "fact TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO facts (fact, created_at) VALUES (?, ?)",
        [("Mera project Ragnarok hai.", "2026-10-01T10:00:00"),
         ("Mujhe simple Hinglish mein explanations pasand hain.",
          "2026-10-01T10:01:00"),
         ("Hackathon mein prize mila tha.", "2026-10-01T10:02:00")])
    connection.commit()
    connection.close()
    legacy = SemanticMemory(legacy_db)
    old = {f.id: f for f in legacy.all_facts()}
    check(old[1].relation == "project" and old[1].obj == "Ragnarok"
          and old[1].importance == 4 and old[1].category == "project",
          "legacy project upgraded")
    check(old[2].kind == "preference" and old[2].importance == 5
          and old[2].relation == "likes", "legacy preference upgraded")
    check(old[3].kind == "fact" and old[3].relation is None
          and old[3].category == "achievement", "legacy free text kept")
    check(legacy.preferred_language() == "hinglish", "legacy language pref")
    SemanticMemory(legacy_db)
    check(legacy.count() == 3, "upgrade is idempotent")

    # ---- add / duplicate / conflict ---------------------------------------------
    mem = SemanticMemory(os.path.join(folder, "t.db"),
                         clock=lambda: datetime(2026, 10, 25, 12, 0, 0))
    r = mem.remember("Mera naam Rahul hai")[0]
    check(r.status == "added" and r.fact.relation == "name"
          and r.fact.source == "manual" and r.fact.confidence == 1.0,
          "remember structured")
    check(mem.remember("Mera naam Rahul hai")[0].status == "duplicate",
          "duplicate by text")
    check(mem.remember("My name is Rahul")[0].status == "duplicate",
          "duplicate by triple")
    r = mem.remember("My name is Rohan")[0]
    check(r.status == "updated" and r.fact.id == 1
          and r.replaced[0].obj == "Rahul" and mem.count() == 1,
          "single-valued fact replaced in place")
    mem.remember("Meri umar 20 saal hai")
    check(mem.remember("Meri umar 21 saal hai")[0].status == "updated"
          and mem.count() == 2, "age updated")
    mem.remember("Mere project ka naam Ragnarok hai")
    mem.remember("Mera project JalNetra hai")
    check(len(mem.triples(USER, "project")) == 2,
          "multi-valued relation keeps both")
    mem.remember("Mujhe Java pasand hai")
    r = mem.remember("Mujhe Java pasand nahi hai")[0]
    check(r.status == "updated" and r.fact.relation == "dislikes"
          and not mem.triples(USER, "likes", "Java"),
          "likes -> dislikes replaces")
    mem.remember("Mera favourite colour blue hai")
    mem.remember("Mera favourite food pizza hai")
    check(mem.remember("Mera favourite colour green hai")[0].status
          == "updated" and len(mem.triples(USER, "favorite_color")) == 1
          and len(mem.triples(USER, "favorite_food")) == 1,
          "favourites are per-noun single valued")
    results = mem.remember("Mera naam Rahul hai. Aaj mausam accha hai. "
                           "Mujhe Python pasand hai")
    check([x.status for x in results] == ["updated", "added", "added"]
          and results[1].fact.relation is None
          and results[2].fact.relation == "likes", "mixed clauses")
    check(raises(mem.remember, "   "), "empty remember rejected")
    check(raises(mem.add, Candidate("bogus", "x", None, None, None)),
          "bad kind rejected")
    check(not mem.find_conflicts(Candidate("fact", "x", None, None, None)),
          "free text has no conflicts")
    r = mem.add(Candidate("fact", "Mera naam Sam hai.", USER, "name", "Sam"),
                replace_conflicts=False)
    check(r.status == "added" and len(mem.triples(USER, "name")) == 2,
          "replace_conflicts=False keeps both")
    mem.delete(r.fact.id)

    # ---- knowledge ----------------------------------------------------------------
    r = mem.learn("Python ek programming language hai")[0]
    check(r.fact.kind == "knowledge" and r.fact.source == "learned"
          and r.fact.relation == "is_a", "learn structured")
    r = mem.learn("Photosynthesis uses sunlight")[0]
    check(r.fact.kind == "knowledge" and r.fact.relation is None,
          "learn free text")

    # ---- graph ---------------------------------------------------------------------
    names = [n for n, _ in mem.entities()]
    check(USER in names and "Python" in names, "entities")
    edges = mem.neighbors("Python")
    check({d for _, d in edges} == {"in", "out"}, "neighbors in/out")
    check(len(mem.triples(relation="is_a")) == 1, "triples by relation")
    check(mem.triples("nobody") == [], "triples unknown subject")

    # ---- retrieval -------------------------------------------------------------------
    rmem = SemanticMemory(os.path.join(folder, "r.db"))
    for sentence in ["Mera project Ragnarok hai", "Meri umar 20 saal hai",
                     "Mera naam Rahul hai", "Mujhe Python pasand hai",
                     "Mujhe simple Hinglish mein explanations pasand hain",
                     "Mera favourite colour blue hai",
                     "Mera favourite food pizza hai",
                     "Main VNIT mein padhta hoon"]:
        rmem.remember(sentence)
    rmem.learn("Python ek programming language hai")

    def top(query, **kw):
        found = rmem.retrieve(query, touch=False, **kw)
        return found[0].fact.text if found else None

    check(top("project ke baare mein batao") == "Mera project Ragnarok hai.",
          "retrieve project")
    check(top("meri age kitni hai") == "Meri umar 20 saal hai.",
          "relation alias age->umar")
    check(top("favourite colour") == "Mera favourite colour blue hai.",
          "retrieve favourite colour")
    check(top("kaunsa college") == "Main VNIT mein padhta hoon.",
          "retrieve college")
    check(top("programming language") ==
          "Python ek programming language hai.", "retrieve knowledge")
    check(top("Python") in ("Mujhe Python pasand hai.",
                            "Python ek programming language hai."),
          "retrieve entity")
    check(rmem.retrieve("zebra quantum") == [], "unrelated -> nothing")
    check(rmem.retrieve("") == [], "empty query")
    only_pref = rmem.retrieve("Python", kinds=("preference",), touch=False)
    check(all(s.fact.kind == "preference" for s in only_pref) and only_pref,
          "kind filter")
    scored = rmem.retrieve("project", touch=False)[0]
    check(0 < scored.score <= 1.6 and scored.lexical > 0, "score parts")
    before = rmem.get(1).access_count
    rmem.retrieve("project")
    check(rmem.get(1).access_count == before + 1
          and rmem.get(1).last_accessed, "access tracking")

    context = rmem.context_for_prompt("Recursion kya hota hai?")
    check("Rahul" in context and "Hinglish" in context
          and "pizza" not in context and "VNIT" not in context,
          "context: important facts only for unrelated query")
    context = rmem.context_for_prompt("mera favourite food")
    check("pizza" in context and "Rahul" in context, "context: relevant added")
    check(SemanticMemory(os.path.join(folder, "e.db")).context_for_prompt("x")
          == "", "empty memory context")

    # ---- embedder switching -----------------------------------------------------------
    class FakeOllama:
        name = "ollama:fake"
        baseline = 0.0
        calls = 0

        def baseline_for(self, name):
            return 0.0

        def embed(self, texts, kind="document"):
            FakeOllama.calls += 1
            return HashEmbedder().embed(texts, kind)[0], self.name

    smem = SemanticMemory(os.path.join(folder, "s.db"))
    smem.remember("Mera project Ragnarok hai")
    check(smem.stats()["models"] == {"local-hash-v1": 1}, "stored with hash")
    smem.embedder = FakeOllama()
    found = smem.retrieve("project", touch=False)
    check(found and found[0].semantic > 0, "stale vectors re-embedded")
    check(smem.stats()["models"] == {"ollama:fake": 1}, "now fake space")
    smem.embedder = HashEmbedder()
    check(smem.reindex() == (1, "local-hash-v1")
          and smem.stats()["models"] == {"local-hash-v1": 1}, "reindex")

    class BrokenEmbedder:
        name = "broken"

        def baseline_for(self, name):
            return 0.0

        def embed(self, texts, kind="document"):
            raise EmbeddingError("nope")

    bmem = SemanticMemory(os.path.join(folder, "b.db"),
                          embedder=BrokenEmbedder())
    r = bmem.remember("Mera project Ragnarok hai")[0]
    check(r.status == "added" and bmem.stats()["embedded"] == 0,
          "save works without embeddings")
    check(bmem.retrieve("project", touch=False)[0].fact.id == r.fact.id,
          "retrieve falls back to keywords")

    # ---- question answering ---------------------------------------------------------------
    qa = SemanticMemory(os.path.join(folder, "q.db"))
    for sentence in ["Mera project Ragnarok hai", "Mera project JalNetra hai",
                     "Mera naam Rahul hai", "Meri umar 20 saal hai",
                     "Mera favourite food pizza hai",
                     "Mujhe simple Hinglish mein explanations pasand hain",
                     "Mujhe Python pasand hai",
                     "Main VNIT mein padhta hoon"]:
        qa.remember(sentence)

    ask = qa.answer_question
    check("Ragnarok" in ask("Mere project ka naam kya hai?")
          and "JalNetra" in ask("Mere project ka naam kya hai?"),
          "QA: both projects, project beats name")
    check(ask("Mera naam kya hai?") == "Saved memory ke hisaab se: "
          "Mera naam Rahul hai.", "QA name")
    check("20 saal" in ask("Meri umar kitni hai?"), "QA age")
    check("20 saal" in ask("How old am I?"), "QA age english")
    check("pizza" in ask("Mera favourite food kya hai?"), "QA favourite")
    check(ask("Mera favourite colour kya hai?") == NOT_SAVED_MESSAGE,
          "QA favourite colour not saved (food must not answer)")
    check("VNIT" in ask("Mera college kaunsa hai?"), "QA college")
    check("Hinglish" in ask("Mujhe kis language mein explanations pasand "
                            "hain?"), "QA language preference")
    check("Python" in ask("Mujhe kya pasand hai?")
          and "Hinglish" in ask("Mujhe kya pasand hai?"),
          "QA list likes")
    check(ask("Meri team kaunsi hai?") == NOT_SAVED_MESSAGE, "QA not saved")
    for text in ["Recursion kya hota hai?", "Hello", "Mera code error de "
                 "raha hai, kya karu, project fix karna hai please batao "
                 "jaldi", "My python script crashes, what should I do?",
                 "Mera naam Rahul hai"]:
        check(ask(text) is None, f"QA ignores {text!r}")

    # ---- preferred language --------------------------------------------------------------------
    lang = SemanticMemory(os.path.join(folder, "l.db"),
                          clock=iter_clock())
    check(lang.preferred_language() is None, "no language yet")
    lang.remember("Mujhe Hinglish pasand hai")
    check(lang.preferred_language() == "hinglish", "language hinglish")
    lang.remember("Mujhe English pasand hai")
    check(lang.preferred_language() == "english", "latest language wins")
    lang.remember("Mujhe English pasand nahi hai")
    check(lang.preferred_language() == "hinglish",
          "dislike ignored (English liked -> now disliked)")
    lang.remember("Please reply in Hindi, I like Hindi")
    check(lang.preferred_language() in ("hindi", "hinglish"), "free text")

    # ---- LLM output parsing ------------------------------------------------------------------------
    source = "Mera naam Rahul hai aur mujhe Python pasand hai"
    good = ('{"facts": [{"kind": "fact", "subject": "user", "relation": '
            '"name", "object": "Rahul", "text": "Mera naam Rahul hai.", '
            '"importance": 5}, {"kind": "preference", "subject": "I", '
            '"relation": "likes", "object": "Python", "text": "Mujhe '
            'Python pasand hai.", "importance": 9}]}')
    parsed = parse_llm_output(good, source)
    check(len(parsed) == 2 and parsed[0].source == "extracted"
          and parsed[0].confidence == 0.6 and parsed[1].subject == USER
          and parsed[1].importance == 5, "llm output parsed + sanitized")
    check(len(parse_llm_output("```json\n" + good + "\n```", source)) == 2,
          "fenced json")
    check(len(parse_llm_output("Sure! Here: " + good + " Hope it helps",
                               source)) == 2, "json inside prose")
    check(len(parse_llm_output('[{"text": "Mujhe Python pasand hai.", '
                               '"object": "Python"}]', source)) == 1,
          "bare list with minimal keys")
    halluc = ('{"facts": [{"kind": "fact", "subject": "user", "relation": '
              '"lives_in", "object": "Mumbai", "text": "Main Mumbai mein '
              'rehta hoon."}]}')
    check(parse_llm_output(halluc, source) == [], "ungrounded dropped")
    for junk in ["", "not json", "{", '{"facts": "oops"}', "[1, 2, 3]",
                 '{"facts": [{"kind": "fact"}]}', "null"]:
        check(parse_llm_output(junk, source) == [], f"junk {junk!r}")

    # ---- extraction pipeline -------------------------------------------------------------------------
    ex = SemanticMemory(os.path.join(folder, "x.db"))
    ex.remember("Mera naam Rahul hai")
    cands = ex.extract(source)
    check([triple(c) for c in cands] == [(USER, "likes", "Python")]
          and cands[0].source == "extracted",
          "extract skips known facts, marks source")
    check(ex.count() == 1, "extract saves nothing")

    def fake_llm(system, text):
        check(system == EXTRACT_SYSTEM and text == source, "llm gets prompts")
        return ('{"facts": [{"kind": "preference", "subject": "user", '
                '"relation": "likes", "object": "Python", "text": "Mujhe '
                'Python pasand hai."}, {"kind": "fact", "subject": "user", '
                '"relation": "age", "object": "99", "text": "Meri umar 99 '
                'hai."}]}')
    cands = ex.extract(source, llm=fake_llm)
    check(len(cands) == 1 and cands[0].relation == "likes",
          "llm + rules deduplicated, hallucination dropped")

    # ---- stats and timing --------------------------------------------------------------------------------
    st = rmem.stats()
    check(st["total"] == 9 and st["triples"] == 9
          and st["by_kind"]["knowledge"] == 1 and st["embedded"] == 9,
          "stats")
    big = SemanticMemory(os.path.join(folder, "big.db"))
    for i in range(200):
        big.add(Candidate("fact", f"Sample fact number {i} about topic{i % 17}"
                          f" and item{i}", None, None, None))
    started = time.perf_counter()
    big.retrieve("topic5 item42", touch=False)
    elapsed = (time.perf_counter() - started) * 1000
    check(elapsed < 800, f"retrieval over 200 facts took {elapsed:.0f} ms")

    check(mem.delete(9999) is None, "delete missing")
    check(mem.delete(1).id == 1 and mem.get(1) is None, "delete")

    print(f"All {checks} checks passed.")
    return 0


def iter_clock():
    """A clock that advances one minute per call (for ordering tests)."""
    state = {"t": datetime(2026, 10, 25, 9, 0, 0)}

    def clock():
        from datetime import timedelta
        state["t"] += timedelta(minutes=1)
        return state["t"]
    return clock


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("semantic_memory.py is a module used by ragnarok.py.")
    print("Run  python semantic_memory.py --selftest  to verify it.")