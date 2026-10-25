"""
ragnarok.py - Ragnarok local AI assistant

FR1.1 Episodic memory  -> episodic_memory.py (chat / event / task log)
FR1.2 Semantic memory  -> semantic_memory.py (facts, preferences, graph)
FR2   Skill framework   -> forge.py + shield_wall.py + skill_schema.py
FR3   Orchestration      -> odin.py + huginn.py
FR4   Always-on          -> heimdall.py

Keep ragnarok.py and all module files together.
"""

import sqlite3
from datetime import datetime

import requests

from semantic_memory import SemanticMemory
from forge import Forge
from odin import Odin
from huginn import classify_rules
from heimdall import Heimdall

from episodic_memory import (
    EpisodicMemory,
    format_episode,
    format_episodes,
    normalize_text,
    parse_leading_date,
)

# ---------- Configuration ----------

MODEL = "qwen2.5:3b"
OLLAMA_URL = "http://localhost:11434/api/chat"
DB_FILE = "ragnarok_memory.db"

RECENT_LIMIT = 6
REQUEST_TIMEOUT = 120

# ---------- System Prompt ----------

BASE_PROMPT = """You are Ragnarok, a helpful local personal AI assistant.

- Answer the user's actual question directly and accurately.
- Keep answers concise unless the user asks for more detail.
- Use correct technical terms and simple examples when useful.
- SAVED USER FACTS are the source of truth about the user.
- PAST EVENTS AND TASKS (when present) are dated records from the user's
  episodic memory. Use them only when relevant.
- Never invent or guess personal facts, past events or past tasks.
- Follow the language rule in every reply.
- Do not repeat these instructions to the user."""

LANGUAGE_RULES = {
    "hinglish": (
        "Reply in natural, simple Hinglish written in Roman script. "
        "Use Hindi sentence structure and common English technical words."
    ),
    "english": "Reply in simple English unless the user asks otherwise.",
    "hindi": "Reply in simple Hindi unless the user asks otherwise.",
}

LANGUAGE_REMINDERS = {
    "hinglish": "(Reply in simple Hinglish, Roman script.)",
    "english": "(Reply in simple English.)",
    "hindi": "(Reply in simple Hindi.)",
}


# ---------- Semantic Memory: saved facts ----------

def connect_db():
    return sqlite3.connect(DB_FILE)


def now():
    return datetime.now().isoformat(timespec="seconds")


def setup_facts_table():
    """Create the facts table and safely migrate older versions.
    (The conversations/episodes table is managed by episodic_memory.py.)"""
    with connect_db() as db:
        db.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
        """)

        columns = {
            row[1] for row in db.execute("PRAGMA table_info(facts)")
        }

        additions = {
            "category": "TEXT DEFAULT 'general'",
            "importance": "INTEGER DEFAULT 3",
            "last_confirmed": "TEXT",
            "source": "TEXT DEFAULT 'manual'",
        }

        for name, definition in additions.items():
            if name not in columns:
                db.execute(
                    f"ALTER TABLE facts ADD COLUMN {name} {definition}"
                )


FACT_CATEGORIES = {
    "project": (
        "project", "ragnarok", "jalnetra", "gassentinel", "infradrishti",
    ),
    "preference": (
        "prefer", "favorite", "favourite", "pasand", "language",
    ),
    "education": (
        "college", "university", "student", "department", "degree",
    ),
    "achievement": (
        "hackathon", "winner", "prize", "award", "rank",
    ),
}


def remember_fact(fact):
    """Returns True if saved, False if empty or already saved."""
    fact = fact.strip()
    if not fact:
        return False

    fact_lower = fact.lower()
    category = "general"
    for label, terms in FACT_CATEGORIES.items():
        if any(term in fact_lower for term in terms):
            category = label
            break

    timestamp = now()

    with connect_db() as db:
        cursor = db.execute(
            """
            INSERT OR IGNORE INTO facts
                (fact, created_at, category, importance,
                 last_confirmed, source)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (fact, timestamp, category, 3, timestamp, "manual"),
        )
        return cursor.rowcount > 0


def get_facts():
    with connect_db() as db:
        return db.execute(
            "SELECT id, fact, created_at FROM facts ORDER BY id"
        ).fetchall()


def delete_fact(fact_id):
    """Returns the deleted fact's text, or None if the id does not exist."""
    with connect_db() as db:
        row = db.execute(
            "SELECT fact FROM facts WHERE id = ?", (fact_id,)
        ).fetchone()
        if not row:
            return None
        db.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
        return row[0]


def show_facts():
    with connect_db() as db:
        facts = db.execute(
            """
            SELECT id, fact, created_at, category,
                   importance, last_confirmed, source
            FROM facts
            ORDER BY id
            """
        ).fetchall()

    if not facts:
        print("Ragnarok: Abhi koi saved fact nahi hai.\n")
        return

    print("\n--- Ragnarok's Saved Facts ---")

    for (fact_id, fact, created_at, category,
         importance, confirmed, source) in facts:
        print(
            f"{fact_id}. {fact}\n"
            f"   Category: {category or 'general'} | "
            f"Importance: {importance or 3} | "
            f"Source: {source or 'manual'}\n"
            f"   Saved: {created_at} | "
            f"Last confirmed: {confirmed or 'not set'}"
        )

    print()


# ---------- Personal fact shortcut ----------
# Short questions such as "Mera project ka naam kya hai?" are answered
# straight from the facts table. Everything else goes to the model.

FACT_STOP_WORDS = {
    "what", "which", "who", "how", "is", "the", "my", "tell", "me", "about",
    "please", "kya", "hai", "ka", "ki", "ke", "batao", "btao", "mujhe",
    "ya", "aur", "in", "on", "do", "you", "know", "does", "can", "could",
    "i", "am", "are", "was", "were", "it", "this", "that", "a", "an", "for",
    "to", "of", "hain", "ho", "hota", "hoti", "hote", "mera", "meri", "mere",
    "kis", "kitna", "kitni", "kaun", "kaunsa", "mein", "se", "hu", "hun",
}

PERSONAL_WORDS = {"my", "mera", "meri", "mere", "mujhe", "i"}

QUESTION_WORDS = {
    "kya", "kaun", "kaunsa", "kaunsi", "kis", "kitna", "kitni", "kitne",
    "what", "which", "who", "how", "whats",
}

CATEGORY_TERMS = {
    "project": {"project"},
    "favorite": {"favorite"},
    "name": {"name", "naam"},
    "age": {"age", "old", "umar", "umr"},
    "college": {"college", "university", "institute"},
    "team": {"team"},
    "hobby": {"hobby", "hobbies"},
    "birthday": {"birthday", "birth"},
    "preference": {"prefer", "preference", "pasand", "language"},
}

# "favorite color" must not be answered by "favorite food".
GENERIC_ONLY = {"favorite": {"favorite"}}


def fact_keywords(text):
    return {
        word
        for word in normalize_text(text).split()
        if word not in FACT_STOP_WORDS and len(word) > 1
    }


def personal_question_category(question):
    words = normalize_text(question).split()
    if not words or len(words) > 10:
        return None

    word_set = set(words)
    if not (word_set & PERSONAL_WORDS and word_set & QUESTION_WORDS):
        return None

    for category, terms in CATEGORY_TERMS.items():
        if word_set & terms:
            return category
    return None


def find_relevant_fact(question, category):
    query_words = fact_keywords(question)
    category_terms = CATEGORY_TERMS[category]
    generic = GENERIC_ONLY.get(category, set())

    best_fact, best_score = None, 0

    for _, fact, _ in get_facts():
        fact_words = set(normalize_text(fact).split())

        # A fact from another category must not answer this question.
        if not (category_terms & fact_words):
            continue

        score = len((query_words - generic) & fact_keywords(fact))
        if score > best_score:
            best_score, best_fact = score, fact

    return best_fact


def answer_personal_question(question, category):
    fact = find_relevant_fact(question, category)
    if fact:
        return f"Saved memory ke hisaab se: {fact}"
    return (
        "Ye information abhi meri saved memory mein nahi hai. "
        "Main guess nahi karunga."
    )


# ---------- Language preference ----------

def get_language_pref(facts):
    """The latest saved fact that mentions a language wins."""
    for _, fact, _ in reversed(facts):
        text = fact.lower()
        for language in ("hinglish", "hindi", "english"):
            if language in text:
                return language
    return None


# ---------- AI ----------

def build_system_message(facts, language, episodes_text=""):
    facts_text = "\n".join(
        f"- {fact}" for _, fact, _ in facts
    ) or "(No saved facts yet)"

    language_rule = LANGUAGE_RULES.get(
        language,
        "Reply in the language of the user's latest message, "
        "simple and natural.",
    )

    content = (
        BASE_PROMPT
        + "\n\nLANGUAGE:\n" + language_rule
        + "\n\nSAVED USER FACTS:\n" + facts_text
    )

    if episodes_text:
        content += (
            "\n\nPAST EVENTS AND TASKS (episodic memory, dated):\n"
            + episodes_text
        )

    return {"role": "system", "content": content}


def ask_ollama(memory, user_input, semantic=None):
    facts = get_facts()
    language = get_language_pref(facts)
    episodes_text = memory.context_for_prompt(user_input)
    semantic_text = semantic.context_for_prompt(user_input) if semantic else ""

    reminder = LANGUAGE_REMINDERS.get(language)
    prompt = f"{user_input}\n\n{reminder}" if reminder else user_input

    system_message = build_system_message(facts, language, episodes_text)
    if semantic_text:
        system_message["content"] += (
            "\n\nRELEVANT STRUCTURED SEMANTIC MEMORY:\n" + semantic_text
        )

    messages = (
        [system_message]
        + memory.chat_context(RECENT_LIMIT)
        + [{"role": "user", "content": prompt}]
    )

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.2, "repeat_penalty": 1.08},
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    answer = data.get("message", {}).get("content", "").strip()

    if not answer:
        raise ValueError("Ollama ne empty answer diya.")

    return answer


# ---------- Episodic memory commands ----------

def parse_count(arg, default, maximum=200):
    if not arg:
        return default
    try:
        number = int(arg.split()[0])
    except ValueError:
        raise ValueError("Number dena hoga, jaise: /history 20") from None
    if number < 1:
        raise ValueError("Number 1 ya usse bada hona chahiye.")
    return min(number, maximum)


def print_episodes(title, episodes, total=None):
    if not episodes:
        print("Ragnarok: Koi episode nahi mila.\n")
        return

    print(f"\n--- {title} ---")
    if total is not None and total > len(episodes):
        print(f"({total} mein se {len(episodes)} dikha raha hoon)")
    print(format_episodes(episodes))
    print()


def command_search(memory, arg):
    if not arg:
        print("Usage: /search <keywords>   "
              "(example: /search hackathon pichhle 30 din)\n")
        return

    result = memory.search(arg, limit=10)
    if not result.episodes and not result.keywords \
            and not result.time_range and not result.kinds:
        print("Ragnarok: Search ke liye koi meaningful keyword nahi mila.\n")
        return

    if not result.episodes:
        print("Ragnarok: Is query se related koi saved episode nahi mila.\n")
        return

    print_episodes(
        f"Episodic Search: {arg}", result.episodes, result.total
    )


def command_timeline(memory, arg):
    result = memory.timeline(arg or "today", limit=50)
    label = result.time_range.label if result.time_range else (arg or "today")
    print_episodes(f"Timeline: {label}", result.episodes, result.total)


def command_log(memory, kind, arg):
    usage = (
        f"Usage: /{kind} [YYYY-MM-DD ya DD-MM-YYYY] [HH:MM] <details>\n"
        f"Example: /{kind} 2026-10-20 14:30 Hackathon mein prize mila\n"
    )
    if not arg:
        print(usage)
        return

    when, details = parse_leading_date(arg)
    if not details:
        print(usage)
        return

    if kind == "task":
        episode_id = memory.log_task(details, when=when)
        name = "Task"
    else:
        episode_id = memory.log_event(details, when=when)
        name = "Event"

    stamp = (when or datetime.now()).isoformat(sep=" ", timespec="minutes")
    print(f"Ragnarok: {name} #{episode_id} episodic memory mein save ho "
          f"gaya ({stamp}).\n")


def command_episode(memory, arg):
    try:
        episode_id = int(arg)
    except ValueError:
        print("Usage: /episode <id>\n")
        return

    episode = memory.get(episode_id)
    if not episode:
        print("Ragnarok: Is ID ka koi episode nahi mila.\n")
        return

    print(
        f"\nEpisode #{episode.id}\n"
        f"  Time    : {episode.time.replace('T', ' ')}\n"
        f"  Kind    : {episode.kind}\n"
        f"  Role    : {episode.role}\n"
        f"  Session : {episode.session_id or 'n/a'}\n"
        f"  Content : {episode.content}\n"
    )


def command_delete_episode(memory, arg):
    try:
        episode_id = int(arg)
    except ValueError:
        print("Usage: /delepisode <id>\n")
        return

    episode = memory.get(episode_id)
    if not episode:
        print("Ragnarok: Is ID ka koi episode nahi mila.\n")
        return

    print(format_episode(episode))
    confirm = input("Ye episode permanently delete hoga. Continue? (y/n): ")
    if confirm.strip().lower() == "y":
        memory.delete_episode(episode_id)
        print("Ragnarok: Episode deleted.\n")
    else:
        print("Ragnarok: Cancelled.\n")


def command_clear(memory, arg):
    scopes = {
        "": ("chat",),
        "chat": ("chat",),
        "events": ("event",),
        "tasks": ("task",),
        "all": None,
    }
    scope = arg.strip().lower()
    if scope not in scopes:
        print("Usage: /clearhistory [chat|events|tasks|all]  "
              "(default: chat)\n")
        return

    label = scope or "chat"
    confirm = input(
        f"Episodic memory ({label}) delete hogi; saved facts safe rahenge. "
        "Continue? (y/n): "
    ).strip().lower()

    if confirm == "y":
        deleted = memory.clear(scopes[scope])
        print(f"Ragnarok: {deleted} episode(s) deleted. "
              "Saved facts safe hain.\n")
    else:
        print("Ragnarok: Cancelled.\n")


def command_stats(memory):
    stats = memory.stats()
    counts = stats["by_kind"]
    print(
        "\n--- Episodic Memory Stats ---\n"
        f"Total episodes : {stats['total']}\n"
        f"  chat   : {counts['chat']}\n"
        f"  events : {counts['event']}\n"
        f"  tasks  : {counts['task']}\n"
        f"First record   : {(stats['first'] or 'n/a').replace('T', ' ')}\n"
        f"Last record    : {(stats['last'] or 'n/a').replace('T', ' ')}\n"
        f"Session id     : {memory.session_id}\n"
    )


# ---------- Help ----------

def show_help():
    print("""
Facts (semantic memory):
  /remember <fact>        - Save an important fact
  /memory                 - View saved facts
  /forget <id>            - Delete a saved fact
  /semantic <query>       - Search semantic memory
  /semstats               - Semantic memory statistics

Episodic memory (chronological log):
  /history [n]            - Last n episodes (default 10)
  /search <words>         - Search the log, e.g. /search hackathon
                            (time words work: kal, pichhle 7 din, 2026-10-20)
  /timeline [when]        - Everything in a period: today, kal,
                            this week, last week, last month, 2026-10-20
  /event [date] <text>    - Save a contextual event
  /task [date] <text>     - Save a COMPLETED task
  /events [n]  /tasks [n] - List recent events / tasks
  /recall <question>      - e.g. /recall maine kal kya kiya
  /episode <id>           - Show one episode in full
  /delepisode <id>        - Delete one episode
  /stats                  - Episodic memory statistics
  /clearhistory [scope]   - chat (default), events, tasks or all

Skills (FR2 - Dwarven Forge):
  /skills                 - List all registered skills
  /skillinfo <name>       - Show skill details
  /run <name> [params]    - Run a skill, e.g. /run hello_world name=Odin
  /skilllog [n]           - Last n skill executions (default 5)

Background (FR4 - Heimdall):
  /watch <name> <path>    - Watch a directory for changes
  /unwatch <name>         - Stop watching
  /cron <name> <interval> <skill> - Schedule a skill, e.g. /cron check every 5m system_info
  /uncron <name>          - Remove scheduled job
  /heimdall               - Show Heimdall status
  /events [n]             - Recent Heimdall events

Orchestration (FR3 - Odin):
  /do <goal>              - Plan and execute a multi-step task
  /plan <goal>            - Show the plan without executing
  /graphs [n]             - Recent task graphs

General:
  /help                   - Show commands
  /exit                   - Quit Ragnarok

You can also just ask: "Maine kal kya kiya?" / "Maine kaunse tasks
complete kiye?" / "When did I win the hackathon?"
""")


# ---------- Command dispatcher ----------

def handle_command(memory, user_input, semantic=None, forge=None, odin=None, heimdall=None):
    """Run a slash command. Always returns True (the input was a command)."""
    command, _, arg = user_input.partition(" ")
    command = command.lower()
    arg = arg.strip()

    try:
        if command == "/help":
            show_help()

        elif command == "/memory":
            if semantic is None:
                show_facts()
            else:
                facts = semantic.all_facts()
                if not facts:
                    print("Ragnarok: Abhi koi saved fact nahi hai.\n")
                else:
                    print("\n--- Ragnarok Semantic Memory ---")
                    for fact in facts:
                        print(f"{fact.id}. [{fact.kind}] {fact.text}")
                        if fact.relation and fact.obj:
                            print(f"   {fact.subject} --{fact.relation}--> {fact.obj}")
                    print()

        elif command == "/remember":
            if not arg:
                print("Usage: /remember <fact>\n")
            elif semantic is not None:
                results = semantic.remember(arg)
                if any(r.status != "duplicate" for r in results):
                    memory.log_event(f"Fact saved: {arg}", role="system")
                    print("Ragnarok: Semantic fact save ho gaya.\n")
                else:
                    print("Ragnarok: Ye fact pehle se saved hai.\n")
            elif remember_fact(arg):
                memory.log_event(f"Fact saved: {arg}", role="system")
                print("Ragnarok: Fact local memory mein save ho gaya.\n")
            else:
                print("Ragnarok: Ye fact pehle se saved hai.\n")

        elif command == "/forget":
            try:
                fact_id = int(arg)
            except ValueError:
                print("Usage: /forget <id>\n")
            else:
                deleted = (semantic.delete(fact_id) if semantic is not None
                           else delete_fact(fact_id))
                if deleted:
                    deleted_text = deleted.text if semantic is not None else deleted
                    memory.log_event(f"Fact forgotten: {deleted_text}",
                                     role="system")
                    print("Ragnarok: Fact deleted.\n")
                else:
                    print("Ragnarok: Is ID ka koi fact nahi mila.\n")

        elif command == "/semantic":
            if not arg:
                print("Usage: /semantic <query>\n")
            else:
                for hit in semantic.retrieve(arg) if semantic else []:
                    print(f"{hit.fact.id}. ({hit.score:.2f}) {hit.fact.text}")
                print()

        elif command == "/semstats":
            print("Ragnarok Semantic Memory:", semantic.stats() if semantic else "not active", "\n")

        elif command == "/history":
            count = parse_count(arg, 10)
            print_episodes("Recent Episodic History", memory.recent(count))

        elif command == "/events":
            count = parse_count(arg, 10)
            print_episodes("Recent Events", memory.recent(count, ("event",)))

        elif command == "/tasks":
            count = parse_count(arg, 10)
            print_episodes("Completed Tasks", memory.recent(count, ("task",)))

        elif command == "/search":
            command_search(memory, arg)

        elif command == "/timeline":
            command_timeline(memory, arg)

        elif command == "/recall":
            print("\n" + memory.recall(arg).text + "\n")

        elif command in ("/event", "/task"):
            command_log(memory, command[1:], arg)

        elif command == "/episode":
            command_episode(memory, arg)

        elif command == "/delepisode":
            command_delete_episode(memory, arg)

        elif command == "/stats":
            command_stats(memory)

        elif command == "/clearhistory":
            command_clear(memory, arg)

        elif command == "/skills":
            if forge:
                print(forge.format_skill_list())
            else:
                print("Ragnarok: Skill framework not active.\n")

        elif command == "/skillinfo":
            if not arg:
                print("Usage: /skillinfo <skill_name>\n")
            elif forge:
                print(forge.format_skill_info(arg))
                print()
            else:
                print("Ragnarok: Skill framework not active.\n")

        elif command == "/run":
            if not forge:
                print("Ragnarok: Skill framework not active.\n")
            elif not arg:
                print("Usage: /run <skill_name> [key=value ...]\n")
            else:
                parts = arg.split()
                skill_name = parts[0]
                params = {}
                for part in parts[1:]:
                    if "=" in part:
                        k, _, v = part.partition("=")
                        params[k] = v
                skill_def = forge.get(skill_name)
                if not skill_def:
                    print(f"Ragnarok: Skill '{skill_name}' not found.\n")
                else:
                    def approval_cb(s):
                        resp = input(
                            f"Skill '{s.name}' has risk level "
                            f"'{s.risk_level}'. Allow? (y/n): "
                        )
                        return resp.strip().lower() == "y"
                    result = forge.run(
                        skill_name, params, approval_callback=approval_cb
                    )
                    memory.log_event(
                        f"Skill executed: {result.summary()}",
                        role="system"
                    )
                    print(f"\nRagnarok: {result.summary()}\n")

        elif command == "/watch":
            if not heimdall:
                print("Ragnarok: Heimdall not active.\n")
            else:
                parts = arg.split()
                if len(parts) < 2:
                    print("Usage: /watch <name> <path> [pattern]\n")
                else:
                    name, path = parts[0], parts[1]
                    patterns = [parts[2]] if len(parts) > 2 else ["*"]
                    heimdall.watch(name, path, patterns=patterns, recursive=True)
                    print(f"Ragnarok: Watching '{path}' as '{name}'.\n")

        elif command == "/unwatch":
            if heimdall and heimdall.unwatch(arg):
                print(f"Ragnarok: Watcher '{arg}' removed.\n")
            else:
                print(f"Ragnarok: Watcher '{arg}' not found.\n")

        elif command == "/cron":
            if not heimdall:
                print("Ragnarok: Heimdall not active.\n")
            else:
                parts = arg.split()
                if len(parts) < 3:
                    print("Usage: /cron <name> <interval> <skill>\nExample: /cron syscheck every_5m system_info\n")
                else:
                    name = parts[0]
                    interval = parts[1].replace("_", " ")
                    skill = parts[2]
                    params = {}
                    for p in parts[3:]:
                        if "=" in p:
                            k, _, v = p.partition("=")
                            params[k] = v
                    heimdall.schedule(name, interval, skill_name=skill, skill_params=params)
                    print(f"Ragnarok: Scheduled '{name}': {skill} {interval}.\n")

        elif command == "/uncron":
            if heimdall and heimdall.unschedule(arg):
                print(f"Ragnarok: Job '{arg}' removed.\n")
            else:
                print(f"Ragnarok: Job '{arg}' not found.\n")

        elif command == "/heimdall":
            if heimdall:
                print(heimdall.format_status())
                print()
            else:
                print("Ragnarok: Heimdall not active.\n")

        elif command in ("/events", "/bevents"):
            if heimdall:
                n = parse_count(arg, 10, 50)
                events = heimdall.recent(n)
                if not events:
                    print("Ragnarok: No Heimdall events yet.\n")
                else:
                    print("\n--- Heimdall Events ---")
                    for e in events:
                        print(f"  [{e['type']}] {e['timestamp']}: {e.get('data', {})}")
                    print()
            else:
                print("Ragnarok: Heimdall not active.\n")

        elif command == "/do":
            if not odin:
                print("Ragnarok: Orchestrator not active.\n")
            elif not arg:
                print("Usage: /do <goal>\nExample: /do greet Odin and show system info\n")
            else:
                def approval_cb(s):
                    resp = input(
                        f"Skill '{s.name}' has risk level "
                        f"'{s.risk_level}'. Allow? (y/n): "
                    )
                    return resp.strip().lower() == "y"
                print(f"\nRagnarok: Planning '{arg}'...\n")
                graph = odin.orchestrate(arg, approval_callback=approval_cb)
                memory.log_event(
                    f"Orchestrated: {graph.summary().replace(chr(10), ' ')}",
                    role="system"
                )
                print(f"\n{graph.format_plan()}\n")

        elif command == "/plan":
            if not odin:
                print("Ragnarok: Orchestrator not active.\n")
            elif not arg:
                print("Usage: /plan <goal>\n")
            else:
                from odin import plan as make_plan
                graph = make_plan(arg, forge)
                print(graph.format_plan())
                print()

        elif command == "/graphs":
            if odin:
                rows = odin.recent_tasks(parse_count(arg, 5, 20))
                if not rows:
                    print("Ragnarok: No task graphs yet.\n")
                else:
                    print("\n--- Recent Task Graphs ---")
                    for goal, status, created, gid in rows:
                        print(f"  [{status}] {created}: {goal} ({gid})")
                    print()
            else:
                print("Ragnarok: Orchestrator not active.\n")

        elif command == "/skilllog":
            if forge:
                count = parse_count(arg, 5, 50)
                print(forge.format_execution_log(count))
                print()
            else:
                print("Ragnarok: Skill framework not active.\n")

        else:
            print("Ragnarok: Unknown command. /help likho.\n")

    except (ValueError, sqlite3.Error) as error:
        print(f"Ragnarok: {error}\n")

    return True


# ---------- Main loop ----------

def main():
    setup_facts_table()
    memory = EpisodicMemory(DB_FILE)
    semantic = SemanticMemory(DB_FILE)
    forge = Forge()
    odin = Odin(forge, DB_FILE)
    heimdall = Heimdall(DB_FILE, forge)
    heimdall.start()

    print("=" * 45)
    print(" RAGNAROK AI - FR1 + FR2 + FR3 + FR4")
    print("=" * 45)
    print("Model:", MODEL)
    print("Database:", DB_FILE)

    stats = memory.stats()
    counts = stats["by_kind"]
    print(
        f"Episodic memory: {stats['total']} records "
        f"({counts['chat']} chat, {counts['event']} events, "
        f"{counts['task']} tasks)"
    )
    print("Semantic memory:", semantic.stats()["total"], "saved facts")
    print(f"Skill registry: {forge.skill_count()} skills loaded")
    print("Heimdall: Background agent started")
    show_help()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("/exit", "exit", "quit"):
                print("Ragnarok: Goodbye!")
                break

            if user_input.startswith("/"):
                handle_command(memory, user_input, semantic, forge, odin, heimdall)
                continue

            # 1) Past-occurrence questions -> answered from the episode log.
            # 2) Short personal-fact questions -> answered from facts.
            # 3) Everything else -> the model (with facts + relevant
            #    events/tasks as context).
            answer = memory.answer_recall_question(user_input)

            if answer is None:
                answer = semantic.answer_question(user_input)
            if answer is None:
                category = personal_question_category(user_input)
                if category:
                    answer = answer_personal_question(user_input, category)
                else:
                    answer = ask_ollama(memory, user_input, semantic)

            print(f"\nRagnarok: {answer}\n")

            # Saved only after a successful reply, both messages together.
            memory.log_exchange(user_input, answer)

            # Extract only explicit user statements; never store model guesses.
            # Do not automatically overwrite conflicting memories.
            for candidate in semantic.extract(user_input):
                if not semantic.find_conflicts(candidate):
                    semantic.add(candidate, replace_conflicts=False)

        except requests.exceptions.ConnectionError:
            print(
                "Ragnarok: Ollama se connect nahi ho paya. "
                "Check karo ki Ollama chal raha hai.\n"
            )

        except requests.exceptions.Timeout:
            print(
                "Ragnarok: Model ne time par jawab nahi diya. "
                "Dobara try karo.\n"
            )

        except requests.exceptions.RequestException as error:
            print("Ragnarok request error:", error, "\n")

        except (KeyError, ValueError) as error:
            print("Ragnarok response error:", error, "\n")

        except sqlite3.Error as error:
            print("Ragnarok database error:", error, "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nRagnarok shutting down.")
            break


if __name__ == "__main__":
    main()


