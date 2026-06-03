"""
Analytics queries over the agent DB.

All functions return plain Python primitives — no SQLite Row objects escape.

Usage:
    from analytics.queries import deception_rate, decision_trends, opponent_summary
"""

from __future__ import annotations

from analytics.db import get_conn


def deception_rate(
    actor: str = "agent",
    game_name: str | None = None,
) -> dict[str, float]:
    """
    What fraction of rounds involved a cooperative message followed by a defect action?

    Returns:
        {
          "deception_rate": 0.25,   # cooperative-msg + defect / all rounds with cooperative msg
          "lie_rounds": 4,
          "coop_msg_rounds": 16,
          "total_rounds": 80,
        }
    """
    game_filter = "AND r.run_id IN (SELECT run_id FROM runs WHERE game_name=?)" if game_name else ""
    params: list = [actor]
    if game_name:
        params.append(game_name)

    with get_conn() as conn:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN msg_intent='cooperative' THEN 1 ELSE 0 END) AS coop_msgs,
                SUM(is_deception) AS lies
            FROM deception_events r
            WHERE actor=?
            {game_filter}
            """,
            params,
        ).fetchone()

    total = row["total"] or 0
    coop = row["coop_msgs"] or 0
    lies = row["lies"] or 0
    return {
        "deception_rate": round(lies / coop, 3) if coop else 0.0,
        "lie_rounds": lies,
        "coop_msg_rounds": coop,
        "total_rounds": total,
    }


def decision_trends(
    game_name: str = "prisoners_dilemma",
    last_n_runs: int = 20,
) -> list[dict]:
    """
    Per-run cooperation and deception rates over the last N runs.

    Returns list of:
        {"run_id": ..., "game_name": ..., "coop_rate": 0.7, "agent_deception_rate": 0.1, "outcome": "WIN"}
    """
    with get_conn() as conn:
        run_ids = [
            r["run_id"]
            for r in conn.execute(
                "SELECT run_id FROM runs WHERE game_name=? ORDER BY rowid DESC LIMIT ?",
                (game_name, last_n_runs),
            ).fetchall()
        ]
        if not run_ids:
            return []

        results = []
        for run_id in reversed(run_ids):
            row = conn.execute(
                "SELECT outcome FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
            outcome = row["outcome"] if row else "UNKNOWN"

            rounds_row = conn.execute(
                "SELECT COUNT(*) AS n, SUM(CASE WHEN my_action='cooperate' THEN 1 ELSE 0 END) AS c"
                " FROM rounds WHERE run_id=?",
                (run_id,),
            ).fetchone()
            n_rounds = rounds_row["n"] or 0
            n_coop = rounds_row["c"] or 0

            dec_row = conn.execute(
                "SELECT SUM(is_deception) AS d, SUM(CASE WHEN msg_intent='cooperative' THEN 1 ELSE 0 END) AS cm"
                " FROM deception_events WHERE run_id=? AND actor='agent'",
                (run_id,),
            ).fetchone()
            dec_total = dec_row["d"] or 0
            coop_msgs = dec_row["cm"] or 0

            results.append({
                "run_id": run_id,
                "game_name": game_name,
                "outcome": outcome,
                "coop_rate": round(n_coop / n_rounds, 3) if n_rounds else 0.0,
                "agent_deception_rate": round(dec_total / coop_msgs, 3) if coop_msgs else 0.0,
                "n_rounds": n_rounds,
            })
        return results


def opponent_summary(opponent_id: str) -> dict:
    """
    Full stats for one opponent: profile + deception they showed against us.

    Returns dict with profile fields plus:
        "opp_deception_rate": fraction of their coop msgs followed by defect
        "our_deception_rate": fraction of our coop msgs followed by defect (this match)
        "run_ids": list of run_ids where this opponent appeared (if tracked)
    """
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM opponents WHERE opponent_id=?", (opponent_id,)).fetchone()
        if not row:
            return {"opponent_id": opponent_id, "error": "not found"}

        profile = dict(row)

        mh = conn.execute(
            "SELECT my_score, opp_score, strategy_used, notes FROM match_history WHERE opponent_id=? ORDER BY id",
            (opponent_id,),
        ).fetchall()
        profile["match_history_summary"] = [dict(r) for r in mh]

        eff = conn.execute(
            "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=1",
            (opponent_id,),
        ).fetchall()
        fail = conn.execute(
            "SELECT message_text FROM opponent_messages WHERE opponent_id=? AND was_effective=0",
            (opponent_id,),
        ).fetchall()
        profile["effective_messages"] = [r["message_text"] for r in eff]
        profile["failed_messages"] = [r["message_text"] for r in fail]

        # Opponent deception from deception_events
        dec = conn.execute(
            "SELECT SUM(is_deception) AS d, SUM(CASE WHEN msg_intent='cooperative' THEN 1 ELSE 0 END) AS cm"
            " FROM deception_events WHERE actor='opponent'",
        ).fetchone()
        cm = dec["cm"] or 0
        profile["opp_deception_rate"] = round((dec["d"] or 0) / cm, 3) if cm else 0.0

    return profile


def phase_adherence(game_name: str = "prisoners_dilemma") -> dict[str, dict]:
    """
    Per-arc-phase action breakdown — what fraction of each phase did the agent cooperate?

    Infers phase from round_num / total_rounds stored in rounds table.
    Note: total_rounds is not stored in rounds rows, so this approximates using
    max round_num per run as total_rounds.

    Returns:
        {"PROBE": {"cooperate": 18, "defect": 2, "coop_rate": 0.9}, ...}
    """
    with get_conn() as conn:
        run_totals = {
            r["run_id"]: r["max_round"]
            for r in conn.execute(
                "SELECT run_id, MAX(round_num) AS max_round FROM rounds"
                " WHERE run_id IN (SELECT run_id FROM runs WHERE game_name=?)"
                " GROUP BY run_id",
                (game_name,),
            ).fetchall()
        }

        rows = conn.execute(
            "SELECT r.run_id, r.round_num, r.my_action"
            " FROM rounds r"
            " JOIN runs u ON u.run_id=r.run_id"
            " WHERE u.game_name=?",
            (game_name,),
        ).fetchall()

    from tournament.core.context import arc_phase

    buckets: dict[str, dict[str, int]] = {}
    for r in rows:
        total = run_totals.get(r["run_id"], 10)
        phase_name, _ = arc_phase(r["round_num"] or 1, total)
        b = buckets.setdefault(phase_name, {"cooperate": 0, "defect": 0})
        action = (r["my_action"] or "").lower()
        if action in ("cooperate", "defect"):
            b[action] += 1

    result = {}
    for phase, counts in buckets.items():
        total_acts = counts["cooperate"] + counts["defect"]
        result[phase] = {
            **counts,
            "coop_rate": round(counts["cooperate"] / total_acts, 3) if total_acts else 0.0,
        }
    return result


def top_opponents(n: int = 5) -> list[dict]:
    """Return the N opponents we've played the most, with win rate and score differential."""
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT
                o.opponent_id,
                o.matches_played,
                o.classified_type,
                o.total_my_score,
                o.total_opp_score,
                ROUND(o.total_my_score - o.total_opp_score, 2) AS score_diff
            FROM opponents o
            ORDER BY o.matches_played DESC, score_diff DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
    return [dict(r) for r in rows]
