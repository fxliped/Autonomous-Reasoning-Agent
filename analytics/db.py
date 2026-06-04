"""
SQLite backend for agent analytics and opponent memory.

DB: <project_root>/data/agent.db

Tables
------
  runs               — one row per game run (ingested from trace JSONs)
  react_steps        — one row per ReAct step (thought / action / argument)
  rounds             — one row per completed round (practice + tournament)
  deception_events   — cooperative-message + defect-action events
  opponents          — persistent opponent profiles (replaces flat JSON files)
  match_history      — per-match outcomes keyed to opponent
  opponent_messages  — messaging effectiveness log

Public API
----------
  init_db()
  ingest_trace(path)          — one JSON trace file → DB rows
  ingest_all(traces_dir)      — batch-ingest all traces
  ingest_match_rounds(...)    — tournament match data → analytics tables
  load_opponent(id)           — returns profile dict (migrates legacy JSON on first use)
  save_opponent(profile)      — upsert opponent row + child tables
  log_match(result)           — append a match result to match_history
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Generator

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "agent.db"

COOP_WORDS = ("cooperat", "together", "mutual", "both", "trust", "fair", "agree", "let's")

_PAYOFFS: dict[tuple[str, str], int] = {
    ("cooperate", "cooperate"): 2,
    ("cooperate", "defect"):   -1,
    ("defect",    "cooperate"): 5,
    ("defect",    "defect"):    0,
}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    game_name    TEXT,
    started_at   TEXT,
    finished_at  TEXT,
    outcome      TEXT,
    my_score     REAL DEFAULT 0,
    opp_score    REAL DEFAULT 0,
    source_file  TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS react_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT REFERENCES runs(run_id),
    round_num   INTEGER,
    step_num    INTEGER,
    thought     TEXT,
    action      TEXT,
    argument    TEXT,
    has_pause   INTEGER,
    decision    TEXT,
    parse_error TEXT,
    observation TEXT
);

CREATE TABLE IF NOT EXISTS rounds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT REFERENCES runs(run_id),
    round_num       INTEGER,
    my_action       TEXT,
    opp_action      TEXT,
    my_msg          TEXT,
    opp_msg         TEXT,
    my_pts          REAL,
    opp_pts         REAL,
    counterfactual_pts  REAL,   -- what I would have scored with the other action
    regret              REAL    -- counterfactual_pts - my_pts (positive = I left points on table)
);

CREATE TABLE IF NOT EXISTS deception_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT REFERENCES runs(run_id),
    round_num    INTEGER,
    actor        TEXT,
    msg_intent   TEXT,
    action_taken TEXT,
    is_deception INTEGER
);

CREATE TABLE IF NOT EXISTS opponents (
    opponent_id     TEXT PRIMARY KEY,
    classified_type TEXT,
    type_confidence REAL DEFAULT 0.0,
    total_my_score  REAL DEFAULT 0.0,
    total_opp_score REAL DEFAULT 0.0,
    matches_played  INTEGER DEFAULT 0,
    message_lies    INTEGER DEFAULT 0,
    msg_lie_rate    REAL DEFAULT 0.0,
    notes           TEXT DEFAULT '',
    last_updated    TEXT
);

CREATE TABLE IF NOT EXISTS match_history (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    opponent_id   TEXT NOT NULL REFERENCES opponents(opponent_id),
    match_date    TEXT,
    my_score      REAL,
    opp_score     REAL,
    strategy_used TEXT DEFAULT '',
    notes         TEXT DEFAULT '',
    rounds_json   TEXT DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS opponent_messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    opponent_id   TEXT NOT NULL REFERENCES opponents(opponent_id),
    message_text  TEXT,
    was_effective INTEGER
);
"""

_LEGACY_OPPONENTS_DIR = ROOT_DIR / "agent" / "memory" / "opponents"


# ─── CONNECTION ───────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create all tables and apply column migrations. Idempotent."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        # Column migrations: ALTER TABLE is safe to run repeatedly via existence check
        existing = {
            r[1]
            for r in conn.execute("PRAGMA table_info(rounds)").fetchall()
        }
        for col, definition in [
            ("counterfactual_pts", "REAL"),
            ("regret",             "REAL"),
        ]:
            if col not in existing:
                conn.execute(f"ALTER TABLE rounds ADD COLUMN {col} {definition}")
        # Backfill regret for any rows missing it (e.g. ingested before migration)
        stale = conn.execute(
            "SELECT id, my_action, opp_action, my_pts FROM rounds WHERE regret IS NULL"
        ).fetchall()
        for row in stale:
            my_a = (row[1] or "").lower()
            opp_a = (row[2] or "").lower()
            my_pts = row[3]
            if my_a and opp_a and my_pts is not None:
                alt = "defect" if my_a == "cooperate" else "cooperate"
                cf = float(_PAYOFFS.get((alt, opp_a), 0))
                reg = cf - float(my_pts)
                conn.execute(
                    "UPDATE rounds SET counterfactual_pts=?, regret=? WHERE id=?",
                    (cf, reg, row[0]),
                )


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def _msg_intent(text: str) -> str:
    return "cooperative" if any(w in (text or "").lower() for w in COOP_WORDS) else "neutral"


def _insert_round_and_deception(
    conn: sqlite3.Connection,
    run_id: str,
    round_num: int | None,
    my_action: str,
    opp_action: str,
    my_msg: str,
    opp_msg: str,
    my_pts: float | None,
    opp_pts: float | None,
) -> None:
    # Compute counterfactual regret: what would the other action have scored?
    alt_action = "defect" if my_action == "cooperate" else "cooperate"
    cf_pts: float | None = None
    regret: float | None = None
    if my_action and opp_action and my_pts is not None:
        cf_pts = float(_PAYOFFS.get((alt_action, opp_action), 0))
        regret = cf_pts - float(my_pts)

    conn.execute(
        "INSERT INTO rounds"
        " (run_id, round_num, my_action, opp_action, my_msg, opp_msg, my_pts, opp_pts,"
        "  counterfactual_pts, regret)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (run_id, round_num, my_action, opp_action, my_msg or "", opp_msg or "",
         my_pts, opp_pts, cf_pts, regret),
    )
    for actor, msg, action in [("agent", my_msg, my_action), ("opponent", opp_msg, opp_action)]:
        intent = _msg_intent(msg)
        is_dec = int(intent == "cooperative" and action == "defect" and bool(msg))
        if intent == "cooperative" or action == "defect":
            conn.execute(
                "INSERT INTO deception_events (run_id, round_num, actor, msg_intent, action_taken, is_deception)"
                " VALUES (?,?,?,?,?,?)",
                (run_id, round_num, actor, intent, action, is_dec),
            )


# ─── TRACE INGESTION ──────────────────────────────────────────────────────────

def ingest_trace(path: Path) -> bool:
    """
    Parse one trace JSON file into the DB. Returns True if newly ingested.
    Idempotent: skips files already recorded by source_file path.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    run_id = data.get("run_id", path.stem)
    game_name = data.get("game_name", "unknown")

    # Derive scores from final_result or round_results
    final = data.get("final_result") or {}
    my_score = float(final.get("my_score") or 0)
    opp_score = float(final.get("opp_score") or 0)
    if my_score == opp_score == 0:
        rrs = data.get("round_results") or []
        my_score = sum(float(r.get("my_pts") or 0) for r in rrs)
        opp_score = sum(float(r.get("opp_pts") or 0) for r in rrs)
    outcome = "WIN" if my_score > opp_score else "LOSS" if opp_score > my_score else "DRAW"

    with get_conn() as conn:
        if conn.execute("SELECT 1 FROM runs WHERE source_file=?", (str(path),)).fetchone():
            return False

        conn.execute(
            "INSERT OR IGNORE INTO runs"
            " (run_id, game_name, started_at, finished_at, outcome, my_score, opp_score, source_file)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (run_id, game_name, data.get("started_at"), data.get("finished_at"),
             outcome, my_score, opp_score, str(path)),
        )

        for step in data.get("steps", []):
            parsed = step.get("parsed") or {}
            conn.execute(
                "INSERT INTO react_steps"
                " (run_id, round_num, step_num, thought, action, argument, has_pause, decision, parse_error, observation)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, step.get("round"), step.get("step"),
                    parsed.get("thought"), parsed.get("action"), parsed.get("argument"),
                    int(bool(parsed.get("has_pause"))), parsed.get("decision"),
                    parsed.get("parse_error"), step.get("observation"),
                ),
            )

        for rr in data.get("round_results", []):
            _insert_round_and_deception(
                conn, run_id,
                rr.get("round"),
                rr.get("my_action", ""),
                rr.get("opp_action", ""),
                rr.get("my_msg", "") or "",
                rr.get("opp_msg", "") or "",
                rr.get("my_pts"), rr.get("opp_pts"),
            )

    return True


def ingest_all(traces_dir: Path | None = None) -> int:
    """Ingest all trace JSONs not yet in the DB. Returns count of newly ingested files."""
    td = traces_dir or (ROOT_DIR / "traces")
    count = 0
    for p in sorted(td.glob("*.json")):
        try:
            if ingest_trace(p):
                count += 1
        except Exception as exc:
            print(f"  [ingest skip] {p.name}: {exc}")
    return count


def ingest_match_rounds(
    run_id: str,
    game_name: str,
    match_rounds: list[dict],
    my_avg: float = 0.0,
    opp_avg: float = 0.0,
    opponent_id: str | None = None,
) -> None:
    """
    Ingest tournament match_rounds (from TournamentAgent.end_match) into analytics tables.
    match_rounds: [{round, my_msg, opp_msg, my_action, opp_action, my_pts, opp_pts}, ...]

    If opponent_id is provided, each opponent message is logged to opponent_messages with
    was_effective=1 when they followed through (coop msg → cooperate) and 0 when they lied
    (coop msg → defect). This builds a per-opponent raw message credibility record.
    """
    outcome = "WIN" if my_avg > opp_avg else "LOSS" if opp_avg > my_avg else "DRAW"
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO runs (run_id, game_name, outcome, my_score, opp_score) VALUES (?,?,?,?,?)",
            (run_id, game_name, outcome, my_avg, opp_avg),
        )
        for r in match_rounds:
            _insert_round_and_deception(
                conn, run_id,
                r.get("round"),
                r.get("my_action", ""),
                r.get("opp_action", "") or "",
                r.get("my_msg", "") or "",
                r.get("opp_msg", "") or "",
                r.get("my_pts"), r.get("opp_pts"),
            )
            # Per-round opponent message credibility: log raw text + whether they followed through
            if opponent_id:
                opp_msg = (r.get("opp_msg") or "").strip()
                opp_action = (r.get("opp_action") or "").lower()
                if opp_msg and _msg_intent(opp_msg) == "cooperative":
                    was_effective = int(opp_action == "cooperate")
                    conn.execute(
                        "INSERT INTO opponent_messages (opponent_id, message_text, was_effective)"
                        " VALUES (?,?,?)",
                        (opponent_id, opp_msg, was_effective),
                    )


# ─── OPPONENT MEMORY ──────────────────────────────────────────────────────────

def _safe_id(opponent_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(opponent_id))


def _load_legacy_json(opponent_id: str) -> dict | None:
    p = _LEGACY_OPPONENTS_DIR / f"{_safe_id(opponent_id)}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _row_to_profile(row: sqlite3.Row, conn: sqlite3.Connection) -> dict:
    profile = dict(row)
    rows_mh = conn.execute(
        "SELECT * FROM match_history WHERE opponent_id=? ORDER BY id", (profile["opponent_id"],)
    ).fetchall()
    profile["match_history"] = [
        {
            "my_score": r["my_score"],
            "opp_score": r["opp_score"],
            "strategy_used": r["strategy_used"] or "unknown",
            "notes": r["notes"] or "",
            "rounds": json.loads(r["rounds_json"] or "[]"),
        }
        for r in rows_mh
    ]
    eff = conn.execute(
        "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=1",
        (profile["opponent_id"],),
    ).fetchall()
    fail = conn.execute(
        "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=0",
        (profile["opponent_id"],),
    ).fetchall()
    profile["effective_messages"] = [r["message_text"] for r in eff]
    profile["failed_messages"] = [r["message_text"] for r in fail]
    return profile


def load_opponent(opponent_id: str) -> dict:
    """
    Load opponent profile from DB. Automatically migrates legacy JSON profiles on first access.
    Returns a dict matching the schema expected by agent/memory.py's format_opponent_context().
    """
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM opponents WHERE opponent_id=?", (opponent_id,)).fetchone()
        if row:
            return _row_to_profile(row, conn)

    # Not in DB — try legacy JSON migration
    existing = _load_legacy_json(opponent_id)
    if existing:
        save_opponent(existing)
        return existing

    return {
        "opponent_id": opponent_id,
        "classified_type": None,
        "type_confidence": 0.0,
        "total_my_score": 0.0,
        "total_opp_score": 0.0,
        "matches_played": 0,
        "message_lies": 0,
        "msg_lie_rate": 0.0,
        "notes": "",
        "last_updated": None,
        "match_history": [],
        "effective_messages": [],
        "failed_messages": [],
    }


def save_opponent(profile: dict) -> None:
    """
    Upsert opponent profile row and sync child tables.
    Appends only new match_history entries (compares against current DB count).
    """
    opp_id = profile["opponent_id"]
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO opponents
               (opponent_id, classified_type, type_confidence, total_my_score, total_opp_score,
                matches_played, message_lies, msg_lie_rate, notes, last_updated)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(opponent_id) DO UPDATE SET
                 classified_type=excluded.classified_type,
                 type_confidence=excluded.type_confidence,
                 total_my_score=excluded.total_my_score,
                 total_opp_score=excluded.total_opp_score,
                 matches_played=excluded.matches_played,
                 message_lies=excluded.message_lies,
                 msg_lie_rate=excluded.msg_lie_rate,
                 notes=excluded.notes,
                 last_updated=excluded.last_updated
            """,
            (
                opp_id,
                profile.get("classified_type"),
                float(profile.get("type_confidence") or 0),
                float(profile.get("total_my_score") or 0),
                float(profile.get("total_opp_score") or 0),
                int(profile.get("matches_played") or 0),
                int(profile.get("message_lies") or 0),
                float(profile.get("msg_lie_rate") or 0),
                profile.get("notes", ""),
                str(date.today()),
            ),
        )

        # Sync messages (no duplicates)
        existing_eff = {
            r[0] for r in conn.execute(
                "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=1",
                (opp_id,),
            ).fetchall()
        }
        existing_fail = {
            r[0] for r in conn.execute(
                "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=0",
                (opp_id,),
            ).fetchall()
        }
        for msg in profile.get("effective_messages", []):
            if msg and msg not in existing_eff:
                conn.execute(
                    "INSERT INTO opponent_messages (opponent_id, message_text, was_effective) VALUES (?,?,1)",
                    (opp_id, msg),
                )
        for msg in profile.get("failed_messages", []):
            if msg and msg not in existing_fail:
                conn.execute(
                    "INSERT INTO opponent_messages (opponent_id, message_text, was_effective) VALUES (?,?,0)",
                    (opp_id, msg),
                )

        # Append new match_history entries
        existing_count = conn.execute(
            "SELECT COUNT(*) FROM match_history WHERE opponent_id=?", (opp_id,)
        ).fetchone()[0]
        for entry in profile.get("match_history", [])[existing_count:]:
            conn.execute(
                "INSERT INTO match_history"
                " (opponent_id, match_date, my_score, opp_score, strategy_used, notes, rounds_json)"
                " VALUES (?,?,?,?,?,?,?)",
                (
                    opp_id, str(date.today()),
                    float(entry.get("my_score") or 0),
                    float(entry.get("opp_score") or 0),
                    entry.get("strategy_used", ""),
                    entry.get("notes", ""),
                    json.dumps(entry.get("rounds", [])),
                ),
            )


def log_match(result: dict) -> None:
    """Append a raw tournament result dict to match_history. Ensures opponent row exists."""
    opp_id = result.get("opponent_id", "unknown")
    with get_conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO opponents (opponent_id, last_updated) VALUES (?,?)",
            (opp_id, str(date.today())),
        )
        conn.execute(
            "INSERT INTO match_history"
            " (opponent_id, match_date, my_score, opp_score, rounds_json)"
            " VALUES (?,?,?,?,?)",
            (
                opp_id, str(date.today()),
                float(result.get("my_avg_score") or 0),
                float(result.get("opp_avg_score") or 0),
                json.dumps(result.get("rounds", [])),
            ),
        )
