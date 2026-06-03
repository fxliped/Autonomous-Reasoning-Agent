"""
Bradley-Terry rating model for self-play evaluation.

Converts raw W/L/D counts into calibrated strength ratings with confidence intervals.

Usage (standalone):
    from tournament.rating import bradley_terry_ratings
    ratings = bradley_terry_ratings(results, n_bootstrap=500)
    # returns {agent_name: {"mean": float, "std": float, "ci_low": float, "ci_high": float}}

Where `results` is the list of dicts returned by run_self_play():
    [{"strategy": "tit_for_tat", "outcome": "WIN", "my_avg": 1.8, "opp_avg": 1.2}, ...]

Theory
------
Each agent i has a latent strength β_i > 0.
P(i beats j) = β_i / (β_i + β_j)

Given observed outcomes, MLE finds the β values that maximise the likelihood of those results.
We solve via the Zermelo iterative algorithm (no scipy needed):

    β_i ← W_i / Σ_j  n_ij / (β_i + β_j)

where W_i = total (fractional) wins and n_ij = total games between i and j.
Draws count as 0.5 wins for each player.

Bootstrap resampling gives a distribution over β, capturing uncertainty from small sample sizes.
We report log(β) centred at 0 (log-strength), so positive = stronger than average.
"""

import math
import random
from collections import defaultdict


# =============================================================================
# CORE MLE
# =============================================================================

def _bt_mle(
    outcomes: list[tuple[str, str, float]],  # (winner, loser, weight) — weight=0.5 for draws
    max_iter: int = 500,
    tol: float = 1e-9,
) -> dict[str, float]:
    """
    Fit Bradley-Terry model via Zermelo iterative MLE.

    Returns {agent: log_strength} centred so the mean log-strength is 0.
    Agents with no wins get a very small strength but are still included.
    """
    agents = sorted(set(a for triple in outcomes for a in triple[:2]))
    n = len(agents)
    if n < 2:
        return {a: 0.0 for a in agents}

    idx = {a: i for i, a in enumerate(agents)}

    # Fractional wins and games matrix
    wins = [0.0] * n
    games = [[0.0] * n for _ in range(n)]
    for w, l, weight in outcomes:
        i, j = idx[w], idx[l]
        wins[i] += weight
        wins[j] += (1.0 - weight)
        games[i][j] += 1.0
        games[j][i] += 1.0

    # Initialise strengths uniformly
    beta = [1.0] * n

    for _ in range(max_iter):
        beta_new = [0.0] * n
        for i in range(n):
            denom = sum(
                games[i][j] / (beta[i] + beta[j])
                for j in range(n) if j != i and games[i][j] > 0
            )
            beta_new[i] = wins[i] / denom if denom > 0 else beta[i]

        # Normalise by geometric mean (keeps values stable)
        log_sum = sum(math.log(max(b, 1e-12)) for b in beta_new) / n
        geomean = math.exp(log_sum)
        beta_new = [b / geomean for b in beta_new]

        if max(abs(beta_new[i] - beta[i]) for i in range(n)) < tol:
            break
        beta = beta_new

    # Return log-strengths centred at 0
    log_beta = [math.log(max(b, 1e-12)) for b in beta]
    mean_log = sum(log_beta) / n
    return {agents[i]: log_beta[i] - mean_log for i in range(n)}


def _results_to_outcomes(results: list[dict]) -> list[tuple[str, str, float]]:
    """
    Convert self-play result dicts to (winner, loser, weight) triples.

    result keys: strategy, outcome (WIN/LOSS/DRAW), my_avg, opp_avg
    The agent is always called "agent"; the NPC is called by strategy name.
    """
    triples = []
    for r in results:
        strat = r["strategy"]
        outcome = r.get("outcome", "DRAW")
        if outcome == "WIN":
            triples.append(("agent", strat, 1.0))
        elif outcome == "LOSS":
            triples.append((strat, "agent", 1.0))
        else:  # DRAW
            triples.append(("agent", strat, 0.5))
    return triples


# =============================================================================
# BOOTSTRAP
# =============================================================================

def _bootstrap_ratings(
    outcomes: list[tuple[str, str, float]],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, list[float]]:
    """
    Run B-T MLE on n_bootstrap resampled datasets.
    Returns {agent: [log_strength_sample_1, ..., log_strength_sample_n]}.
    """
    rng = random.Random(seed)
    n = len(outcomes)
    samples: dict[str, list[float]] = defaultdict(list)

    for _ in range(n_bootstrap):
        resample = [outcomes[rng.randint(0, n - 1)] for _ in range(n)]
        try:
            ratings = _bt_mle(resample)
            for agent, rating in ratings.items():
                samples[agent].append(rating)
        except Exception:
            continue  # skip degenerate resamples (e.g. agent never appears)

    return dict(samples)


# =============================================================================
# PUBLIC API
# =============================================================================

def bradley_terry_ratings(
    results: list[dict],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict[str, dict]:
    """
    Fit Bradley-Terry model with bootstrap confidence intervals.

    Args:
        results:     list of self-play result dicts (from run_self_play)
        n_bootstrap: number of bootstrap resamples (default 1000; use 200 for speed)
        ci:          confidence interval width, e.g. 0.95 for 95% CI

    Returns:
        {
          "agent":        {"mean": 0.8, "std": 0.3, "ci_low": 0.2, "ci_high": 1.4, "n_games": 14},
          "tit_for_tat":  {"mean": 0.1, ...},
          ...
        }
        Positive mean = stronger than the field average.
        Negative mean = weaker than the field average.
    """
    if not results:
        return {}

    outcomes = _results_to_outcomes(results)
    if not outcomes:
        return {}

    # Point estimate from all data
    point = _bt_mle(outcomes)

    # Bootstrap distribution
    boot = _bootstrap_ratings(outcomes, n_bootstrap=n_bootstrap)

    # Count games per agent
    game_counts: dict[str, int] = defaultdict(int)
    for r in results:
        game_counts["agent"] += 1
        game_counts[r["strategy"]] += 1

    tail = (1.0 - ci) / 2.0
    out = {}
    for agent in point:
        samples = sorted(boot.get(agent, [point[agent]]))
        n_s = len(samples)
        mean = sum(samples) / n_s if samples else point[agent]
        variance = sum((s - mean) ** 2 for s in samples) / n_s if n_s > 1 else 0.0
        std = math.sqrt(variance)
        ci_low = samples[max(0, int(tail * n_s))]
        ci_high = samples[min(n_s - 1, int((1 - tail) * n_s))]
        out[agent] = {
            "mean": round(mean, 3),
            "std": round(std, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "n_games": game_counts.get(agent, 0),
        }

    return out


def print_ratings_table(ratings: dict[str, dict], ci: float = 0.95) -> None:
    """Print a formatted Bradley-Terry ratings table."""
    if not ratings:
        print("  (no data)")
        return

    ci_pct = int(ci * 100)
    ranked = sorted(ratings.items(), key=lambda kv: -kv[1]["mean"])

    col_w = max(len(a) for a in ratings) + 2
    print(f"\n  {'Agent':<{col_w}}  {'Rating':>8}  {'±Std':>6}  {f'{ci_pct}% CI':>14}  {'Games':>6}")
    print(f"  {'-'*col_w}  {'--------':>8}  {'------':>6}  {'----------':>14}  {'------':>6}")
    for agent, r in ranked:
        ci_str = f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        marker = " ◀" if agent == "agent" else ""
        print(
            f"  {agent:<{col_w}}  {r['mean']:>+8.3f}  {r['std']:>6.3f}  {ci_str:>14}  {r['n_games']:>6}{marker}"
        )
    print(
        f"\n  Rating = log-strength (centred at 0). "
        f"Positive = stronger than field average. "
        f"Wide CI = need more games."
    )
