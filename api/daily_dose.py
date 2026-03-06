"""
Daily Dose Generator — Sanatan Sutra
Generates a ~500-word daily teaching from Krishna's sacred texts based on a
topic/question loaded from data/daily_topics.json.
"""

import json
import os
import sys
import datetime
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    _PYMONGO_AVAILABLE = True
except ImportError:
    _PYMONGO_AVAILABLE = False

load_dotenv()

# Path to the topics file (relative to the repo root)
TOPICS_FILE = Path(__file__).parent.parent / "data" / "daily_topics.json"

# ─── Journey start date ────────────────────────────────────────────────────────
# Day 1 of the 100-day journey.  Change this only once, when you want to
# reset the global cycle.  Everyone who visits on the same calendar date
# will see the same topic — no database or user accounts needed.
JOURNEY_START = datetime.date(2026, 3, 6)   # March 6 2026 = Day 1


def current_journey_day() -> int:
    """
    Return today's day number (1–100) calculated from JOURNEY_START.
    After 100 days the cycle resets back to Day 1 automatically.
    """
    delta = (datetime.date.today() - JOURNEY_START).days   # 0 on Day 1
    return (delta % 100) + 1   # 1-indexed, wraps at 100

# ─── Prompt ────────────────────────────────────────────────────────────────────

DAILY_DOSE_TEMPLATE = """
You are a devoted lifelong student of Shri Krishna, deeply immersed in the teachings
of the Bhagavad Gita, Mahabharata, Upanishads, and Srimad Bhagavatam.

Your task today is to write a **Daily Dose of Krishna's Wisdom** — a focused,
practical and inspiring message of approximately 500 words.

Topic for Today: {title}
Source Scripture(s): {source}
Core Question: {question}

INSTRUCTIONS:
1. Begin with a brief relatable opening that grounds the reader in this life challenge.
2. Introduce the relevant Krishna teaching or scriptural story that addresses the theme.
3. Extract the philosophical principle in plain, modern language without heavy Sanskrit jargon.
4. Translate that wisdom into 2–3 concrete, practical actions a person can take TODAY.
5. Close with a short inspiring reflection or quote paraphrased from the scriptures.

TONE:
- Warm, direct and compassionate
- Relatable to modern life
- Grounded in scripture but not preachy
- Each paragraph should be short (3–5 sentences), easy to read
- Avoid vague spirituality — be specific and practical

LENGTH: Write exactly around 500 words. Structure the message with natural paragraph breaks.

Begin the Daily Dose now:
"""

_llm_instance = None
_mongo_collection = None   # lazily initialised


# ─── MongoDB cache ─────────────────────────────────────────────────────────────

def _get_collection():
    """
    Return the MongoDB collection used for caching daily doses.
    Returns None when MongoDB is not configured or unavailable, so the
    app degrades gracefully by generating content on-the-fly.
    """
    global _mongo_collection
    if _mongo_collection is not None:
        return _mongo_collection
    if not _PYMONGO_AVAILABLE:
        return None
    uri = os.getenv("MONGODB_URI")
    if not uri:
        return None
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        db_name = os.getenv("MONGODB_DB", "sanatan_sutra")
        _mongo_collection = client[db_name]["daily_doses"]
        # Ensure a unique index on day so upsert is safe
        _mongo_collection.create_index("day", unique=True)
        print(f"[daily_dose] MongoDB cache connected (db={db_name})", file=sys.stderr)
    except PyMongoError as exc:
        print(f"[daily_dose] MongoDB unavailable, will generate live: {exc}", file=sys.stderr)
        _mongo_collection = None
    return _mongo_collection


def _cache_get(day: int) -> dict | None:
    """Return a cached dose dict for *day*, or None if not yet cached."""
    col = _get_collection()
    if col is None:
        return None
    try:
        doc = col.find_one({"day": day}, {"_id": 0})
        return doc
    except PyMongoError as exc:
        print(f"[daily_dose] Cache read failed for day {day}: {exc}", file=sys.stderr)
        return None


def _cache_set(dose: dict) -> None:
    """Upsert a dose dict into MongoDB, keyed by its day field."""
    col = _get_collection()
    if col is None:
        return
    try:
        col.update_one(
            {"day": dose["day"]},
            {"$set": {**dose, "cached_at": datetime.datetime.utcnow().isoformat()}},
            upsert=True,
        )
        _mark_topic_generated(dose["day"])
    except PyMongoError as exc:
        print(f"[daily_dose] Cache write failed for day {dose['day']}: {exc}", file=sys.stderr)


def _mark_topic_generated(day: int) -> None:
    """
    Set "generated": true on the matching topic in daily_topics.json so it's
    clear at a glance which topics already have a cached message in MongoDB.
    """
    try:
        with open(TOPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        changed = False
        for topic in data["topics"]:
            if topic["day"] == day and not topic.get("generated"):
                topic["generated"] = True
                changed = True
                break
        if changed:
            with open(TOPICS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[daily_dose] Could not mark day {day} as generated in JSON: {exc}", file=sys.stderr)


def _get_llm() -> ChatGroq:
    global _llm_instance
    if _llm_instance is None:
        groq_key = os.getenv("GROQ_API_KEY")
        if not groq_key:
            raise ValueError("GROQ_API_KEY environment variable is not set.")
        _llm_instance = ChatGroq(
            api_key=groq_key,   # type: ignore
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1024,
        )
    return _llm_instance


def load_topics() -> list[dict]:
    """Load and return all 100 daily topics."""
    with open(TOPICS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["topics"]


def get_topic_for_day(day_number: int | None = None) -> dict:
    """
    Return the topic for a given day number (1-100).
    If day_number is None, use current_journey_day() so the correct topic
    advances by one each real calendar day and everyone sees the same one.
    """
    topics = load_topics()
    resolved_day = day_number if day_number is not None else current_journey_day()
    resolved_day = max(1, min(100, resolved_day))
    return next((t for t in topics if t["day"] == resolved_day), topics[0])


def generate_daily_message(topic: dict) -> str:
    """
    Call the LLM to generate a ~500-word daily message for the given topic dict.
    Returns the generated text.
    """
    llm = _get_llm()
    prompt = PromptTemplate(
        template=DAILY_DOSE_TEMPLATE,
        input_variables=["title", "source", "question"],
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {
            "title": topic["title"],
            "source": topic["source"],
            "question": topic["question"],
        }
    )
    return (result or "").strip()


def get_daily_dose(day_number: int | None = None) -> dict:
    """
    High-level entry point used by the Flask route.

    Strategy:
      1. Resolve the requested day number (defaults to today).
      2. Check MongoDB for a previously generated and stored dose.
      3. If found, return the cached version — same content for every visitor.
      4. If not found, call the LLM, store the result, then return it.

    Returns a dict with: day, title, source, theme, question, message, date,
    journey_start, today_day, and cached (bool to indicate source).
    """
    today_day = current_journey_day()
    topic = get_topic_for_day(day_number)   # uses today_day when day_number is None
    resolved_day = topic["day"]

    # ── 1. Try cache ──────────────────────────────────────────────────────────
    cached = _cache_get(resolved_day)
    if cached:
        # Patch live fields so UI always shows correct "today" progress
        cached["today_day"] = today_day
        cached["journey_start"] = JOURNEY_START.strftime("%B %d, %Y")
        cached["cached"] = True
        return cached

    # ── 2. Generate ───────────────────────────────────────────────────────────
    message = generate_daily_message(topic)
    dose = {
        "day": resolved_day,
        "title": topic["title"],
        "source": topic["source"],
        "theme": topic["theme"],
        "question": topic["question"],
        "message": message,
        "date": datetime.date.today().strftime("%B %d, %Y"),
        "today_day": today_day,
        "journey_start": JOURNEY_START.strftime("%B %d, %Y"),
        "cached": False,
    }

    # ── 3. Persist ────────────────────────────────────────────────────────────
    _cache_set(dose)
    return dose


# ─── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate a Daily Dose of Sanatan Sutra wisdom.")
    parser.add_argument("--day", type=int, default=None, help="Topic day number (1-100). Defaults to today's auto-selected topic.")
    parser.add_argument("--list", action="store_true", help="List all 100 topics and exit.")
    args = parser.parse_args()

    if args.list:
        topics = load_topics()
        for t in topics:
            print(f"Day {t['day']:3d} | [{t['theme']}] {t['title']}")
    else:
        print("Generating today's Daily Dose...\n")
        dose = get_daily_dose(args.day)
        print(f"{'─' * 60}")
        print(f"Day {dose['day']} — {dose['date']}")
        print(f"Topic : {dose['title']}")
        print(f"Source: {dose['source']}")
        print(f"Theme : {dose['theme']}")
        print(f"{'─' * 60}\n")
        print(dose["message"])
        print(f"\n{'─' * 60}")
