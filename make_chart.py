"""Per-opponent results chart for the final-presentation deck.

Reads analysis/match_summary.csv (Secret Agent matches only), aggregates the
real-team opponents, and renders a grouped bar chart: our avg payoff/round vs
the opponent's, per team. Saved to analysis/per_opponent_results.png.
"""

import csv
import collections
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
# match_summary.csv already excludes our own agents; every row is a real team.
real = list(csv.DictReader(open(ROOT / "analysis" / "match_summary.csv")))

agg = collections.defaultdict(lambda: {"n": 0, "my": 0.0, "opp": 0.0, "W": 0, "T": 0, "L": 0})
for r in real:
    a = agg[r["opponent"]]
    a["n"] += 1
    a["my"] += float(r["my_avg"])
    a["opp"] += float(r["opp_avg"])
    a[r["result"]] += 1

# Order by our margin, best matchup first.
opps = sorted(agg, key=lambda o: -(agg[o]["my"] - agg[o]["opp"]) / agg[o]["n"])
my = [agg[o]["my"] / agg[o]["n"] for o in opps]
opp = [agg[o]["opp"] / agg[o]["n"] for o in opps]
labels = [f"{o}\n({agg[o]['W']}-{agg[o]['T']}-{agg[o]['L']})" for o in opps]

x = range(len(opps))
w = 0.38
fig, ax = plt.subplots(figsize=(12, 5.5))
b1 = ax.bar([i - w / 2 for i in x], my, w, label="Secret Agent", color="#2563eb")
b2 = ax.bar([i + w / 2 for i in x], opp, w, label="Opponent", color="#cbd5e1")

ax.axhline(0, color="#334155", linewidth=0.8)
ax.axhline(2.0, color="#16a34a", linewidth=0.9, linestyle="--", alpha=0.7)
ax.text(-0.45, 2.04, "mutual-coop (2.0)", color="#16a34a", fontsize=8, ha="left")

ax.set_ylabel("Avg payoff per round")
ax.set_title("Secret Agent vs. real teams — avg payoff/round  (W-T-L under each)", fontsize=12)
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=9)
ax.legend(loc="upper right")
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", (bar.get_x() + bar.get_width() / 2, h),
                    textcoords="offset points", xytext=(0, 3 if h >= 0 else -11),
                    ha="center", fontsize=8)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = ROOT / "analysis" / "per_opponent_results.png"
fig.savefig(out, dpi=160)
print(f"wrote {out}")
