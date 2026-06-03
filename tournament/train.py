"""
Self-play training loop: TournamentAgent vs all scripted NPC strategies.

Runs N matches per strategy, LLM-judges each match, and appends strategic
lessons to agent/reflections/prisoners_dilemma.md. These reflections are
automatically injected into the agent's system prompt in future matches.

Run:
    python tournament/train.py                         # 2 games per strategy (all 7)
    python tournament/train.py --n 5                   # 5 games per strategy
    python tournament/train.py --strategy tit_for_tat  # one strategy only
    python tournament/train.py --no-judge              # skip LLM judging (faster)
    python tournament/train.py --quiet                 # suppress round-by-round output
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from agent.agent import append_reflection, create_client  # noqa: E402
from games.pd_game import PrisonersDilemma  # noqa: E402
from tournament.agent import TournamentAgent  # noqa: E402
from tournament.rating import bradley_terry_ratings, print_ratings_table  # noqa: E402

GAME_NAME = "prisoners_dilemma"

_JUDGE_SYSTEM = (
    "You are a strategic analyst for iterated Prisoner's Dilemma. "
    "Be concise and actionable. Respond with valid JSON only — no markdown fences."
)

_JUDGE_PROMPT = """\
Prisoner's Dilemma match: our agent vs NPC strategy "{strategy}" ({rounds} rounds).

Round-by-round record:
{rounds_str}

Final per-round averages: agent {my_avg:.3f} | NPC {opp_avg:.3f}
Outcome: {outcome}

Identify up to 3 specific strategic mistakes OR correct plays. Then write a 2-3 sentence
lesson the agent should apply in future matches against this opponent type.

JSON schema (no fences):
{{
  "analysis": [
    {{"round": 1, "category": "strategy_error|correct_play|missed_opportunity|deception_success|deception_failure", "explanation": "..."}}
  ],
  "lesson": "Paragraph lesson for future matches against {strategy} opponents."
}}
"""

_NPC_MESSAGES = {
    "always_cooperate": "I always cooperate — let's both benefit.",
    "always_defect":    "Do what you want. I play my own game.",
    "tit_for_tat":      "I mirror what you do. Cooperate and so will I.",
    "grim_trigger":     "Betray me once and I defect forever.",
    "pavlov":           "I adjust based on what worked last round.",
    "generous_tft":     "I cooperate by default and forgive the occasional mistake.",
    "random":           "Who knows? Let's see what happens.",
}


# =============================================================================
# MATCH RUNNER
# =============================================================================

def run_self_play_match(
    strategy: str,
    rounds: int = 10,
    verbose: bool = True,
) -> tuple[list[dict], float, float]:
    """
    Run one TournamentAgent match vs a scripted NPC strategy.
    Returns (match_rounds, my_avg_score, opp_avg_score).
    match_rounds is captured before end_match() clears internal state.
    """
    agent = TournamentAgent(opponent_id=f"npc_{strategy}", total_rounds=rounds)
    game = PrisonersDilemma(rounds=rounds, opponent_strategy=strategy)
    match_history: list[dict] = []
    npc_msg = _NPC_MESSAGES.get(strategy, "Let's play.")

    while not game.is_over():
        round_num = game.current_round
        my_score = float(game.agent_score)
        opp_score = float(game.opponent_score)

        # MESSAGING PHASE (blind — we don't see opponent's message yet)
        message = agent.compose_message(
            round_num=round_num,
            total_rounds=rounds,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
        )

        # MOVING PHASE (opponent's message revealed)
        action_int = agent.choose_action(
            round_num=round_num,
            total_rounds=rounds,
            opponent_message=npc_msg,
            my_message=message,
            match_history=match_history,
            my_score=my_score,
            opp_score=opp_score,
        )
        action = "cooperate" if action_int == 0 else "defect"
        game.make_move(action)

        last_a, last_o = game.history[-1]
        my_pts, opp_pts = PrisonersDilemma.PAYOFFS[(last_a, last_o)]

        agent.record_round_result(round_num, last_o, float(my_pts), float(opp_pts))
        match_history.append({
            "round":      round_num,
            "opp_msg":    npc_msg,
            "my_msg":     message,
            "my_action":  action,
            "opp_action": last_o,
            "my_pts":     my_pts,
            "opp_pts":    opp_pts,
        })

        if verbose:
            sym_a = "C" if action == "cooperate" else "D"
            sym_o = "C" if last_o == "cooperate" else "D"
            print(f"    R{round_num:02d}: agent={sym_a} npc={sym_o} | "
                  f"pts me{my_pts:+d} npc{opp_pts:+d} | "
                  f"running: me {game.agent_score} npc {game.opponent_score}")

    my_avg  = game.agent_score  / rounds
    opp_avg = game.opponent_score / rounds

    # Capture rounds BEFORE end_match() clears _match_rounds
    completed_rounds = agent.match_rounds
    agent.end_match(my_avg, opp_avg)

    return completed_rounds, my_avg, opp_avg


# =============================================================================
# LLM JUDGE
# =============================================================================

def judge_match(
    client,
    strategy: str,
    match_rounds: list[dict],
    my_avg: float,
    opp_avg: float,
    rounds: int,
) -> dict:
    """LLM judges a completed match. Returns {analysis, lesson}."""
    rounds_str = "\n".join(
        f"  R{r['round']:02d}: agent={r.get('my_action','?'):<10} "
        f"npc={r.get('opp_action','?'):<10} | "
        f"pts me{(r.get('my_pts') or 0):+d} npc{(r.get('opp_pts') or 0):+d} | "
        f"agent said: \"{r.get('my_msg', '')}\" | npc said: \"{r.get('opp_msg', '')}\""
        for r in match_rounds
    )
    outcome = "WIN" if my_avg > opp_avg else "LOSS" if opp_avg > my_avg else "DRAW"
    prompt = _JUDGE_PROMPT.format(
        strategy=strategy,
        rounds=rounds,
        rounds_str=rounds_str,
        my_avg=my_avg,
        opp_avg=opp_avg,
        outcome=outcome,
    )
    try:
        raw = client.complete(system=_JUDGE_SYSTEM, messages=[{"role": "user", "content": prompt}])
        text = re.sub(r"^```[a-z]*\n?", "", raw.strip())
        text = re.sub(r"\n?```$", "", text.strip())
        return json.loads(text)
    except Exception as exc:
        print(f"  [Judge parse error: {exc}]")
        return {}


# =============================================================================
# SELF-PLAY LOOP
# =============================================================================

def run_self_play(
    strategies: list[str] | None = None,
    n_per_strategy: int = 2,
    rounds: int = 10,
    judge: bool = True,
    verbose: bool = True,
) -> None:
    """
    Run the full self-play training loop.
    Lessons from judged matches are appended to agent/reflections/prisoners_dilemma.md
    and will be injected into the agent's system prompt on the next run.
    """
    strategies = strategies or list(PrisonersDilemma.OPPONENT_STRATEGIES)
    client = create_client() if judge else None

    results: list[dict] = []

    for strategy in strategies:
        print(f"\n{'='*62}")
        print(f"  SELF-PLAY: vs {strategy}  ({n_per_strategy} game(s), {rounds} rounds each)")
        print(f"{'='*62}")

        for game_num in range(1, n_per_strategy + 1):
            print(f"\n  --- Game {game_num}/{n_per_strategy} ---")
            match_rounds, my_avg, opp_avg = run_self_play_match(
                strategy=strategy, rounds=rounds, verbose=verbose,
            )
            outcome = "WIN" if my_avg > opp_avg else "LOSS" if opp_avg > my_avg else "DRAW"
            print(f"\n  Result: {outcome}  |  agent avg {my_avg:.3f}  |  npc avg {opp_avg:.3f}")
            results.append({"strategy": strategy, "game": game_num,
                            "outcome": outcome, "my_avg": my_avg, "opp_avg": opp_avg})

            if judge and client is not None:
                print("  [Judging...]")
                judgment = judge_match(client, strategy, match_rounds, my_avg, opp_avg, rounds)
                lesson = judgment.get("lesson", "").strip()
                if lesson:
                    print(f"  Lesson: {lesson}")
                    source = f"self_play_vs_{strategy}_game{game_num}"
                    append_reflection(GAME_NAME, lesson, source_file=source)
                    print(f"  Appended to agent/reflections/{GAME_NAME}.md")
                for item in judgment.get("analysis", []):
                    cat = item.get("category", "?")
                    expl = item.get("explanation", "")
                    print(f"  R{item.get('round','?')} [{cat}]: {expl}")

    # W/L/D summary table
    print(f"\n{'='*62}")
    print("  SELF-PLAY SUMMARY")
    print(f"{'='*62}")
    print(f"  {'Strategy':<22}  {'W':>3} {'L':>3} {'D':>3}  {'Avg score':>10}")
    print(f"  {'-'*22}  {'---':>3} {'---':>3} {'---':>3}  {'----------':>10}")
    for strat in strategies:
        strat_res = [r for r in results if r["strategy"] == strat]
        if not strat_res:
            continue
        counts = Counter(r["outcome"] for r in strat_res)
        avg = sum(r["my_avg"] for r in strat_res) / len(strat_res)
        print(f"  {strat:<22}  {counts['WIN']:>3} {counts['LOSS']:>3} {counts['DRAW']:>3}  {avg:>10.3f}")

    # Bradley-Terry ratings (calibrated strength with bootstrap confidence intervals)
    if len(results) >= 2:
        n_boot = min(1000, max(200, len(results) * 50))
        print(f"\n{'='*62}")
        print(f"  BRADLEY-TERRY RATINGS  (bootstrap n={n_boot})")
        print(f"{'='*62}")
        ratings = bradley_terry_ratings(results, n_bootstrap=n_boot)
        print_ratings_table(ratings)

    if judge:
        refl_path = ROOT_DIR / "agent" / "reflections" / f"{GAME_NAME}.md"
        print(f"\n  Reflections saved to: {refl_path}")
        print("  These will be injected into the agent's system prompt on next run.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play training loop for TournamentAgent")
    parser.add_argument("--n", type=int, default=2, metavar="N",
                        help="Games per strategy (default: 2)")
    parser.add_argument("--rounds", type=int, default=10,
                        help="Rounds per game (default: 10)")
    parser.add_argument("--strategy", default=None,
                        choices=PrisonersDilemma.OPPONENT_STRATEGIES,
                        help="Run against one strategy only (default: all 7)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip LLM judging — faster, no reflections written")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress round-by-round output")
    args = parser.parse_args()

    run_self_play(
        strategies=[args.strategy] if args.strategy else None,
        n_per_strategy=args.n,
        rounds=args.rounds,
        judge=not args.no_judge,
        verbose=not args.quiet,
    )
