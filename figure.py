"""Figure: return vs evaluation horizon for the three released one-shot networks."""
import json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results")
with open(os.path.join(OUT, "report.json")) as fh:
    rep = json.load(fh)

# dataviz reference palette, categorical slots 1-3 (light mode); validated all-pairs set
COL = {"breakout": "#2a78d6", "seaquest": "#eb6834", "asterix": "#1baf7a"}
PPO = {"breakout": 25.344, "seaquest": 88.88, "asterix": 55.20}
INK, INK2, GRID, SURF = "#15181f", "#4a515f", "#dde1e8", "#ffffff"

fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.6, 4.5), facecolor=SURF)

for name, a in ((("return"), ax), (("alive"), bx)):
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=9, length=3, color=GRID)
    a.grid(True, which="major", color=GRID, lw=0.7, alpha=0.9)
    a.set_axisbelow(True)

# left: mean return vs horizon
for g in ("breakout", "seaquest", "asterix"):
    rows = rep[g]["ladder"]
    h = [r["horizon"] for r in rows]
    m = [r["mean"] for r in rows]
    se = [r["se"] for r in rows]
    ax.errorbar(h, m, yerr=[1.96 * s for s in se], color=COL[g], lw=2, marker="o",
                ms=5.5, capsize=2.5, mec=SURF, mew=1.2, label=g, zorder=3)
    dy = {"breakout": 11, "asterix": 11, "seaquest": -17}[g]
    ax.annotate(f"{g}  {m[-1]:,.0f}", (h[-1], m[-1]), textcoords="offset points",
                xytext=(-6, dy), ha="right", color=COL[g], fontsize=9.5, weight="semibold")
ax.axhline(PPO["breakout"], color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=2)
ax.annotate("PPO 500M on Breakout, 25.3 — its episodes\nend by ~282 steps, so its return does not grow",
            (1150, 25.344), textcoords="offset points", xytext=(0, -34), fontsize=8.3, color=INK2)
ax.axvline(10_000, color=INK2, lw=1, ls=":", zorder=2)
ax.annotate("the paper's cap", (10_000, 1.6), rotation=90, fontsize=8.3,
            color=INK2, ha="right", va="bottom", textcoords="offset points", xytext=(-4, 0))
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("evaluation horizon (steps, log)", color=INK2, fontsize=9.5)
ax.set_ylabel("mean return over 1,000 episodes (log)", color=INK2, fontsize=9.5)
ax.set_title("Reported score is a point on a curve", color=INK, fontsize=12.5,
             loc="left", pad=11)
ax.set_xlim(85, 42_000); ax.set_ylim(1.2, 2200)
leg = ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2)

# right: share of episodes still running when the clock cuts them
for g in ("breakout", "seaquest", "asterix"):
    rows = rep[g]["ladder"]
    h = [r["horizon"] for r in rows]
    al = [100 * r["alive_frac"] for r in rows]
    bx.plot(h, al, color=COL[g], lw=2, marker="o", ms=5.5, mec=SURF, mew=1.2, label=g, zorder=3)
b = rep["breakout"]["ladder"][4]
bx.annotate(f"Breakout at the paper's cap:\n{100*b['alive_frac']:.1f}% of episodes never ended",
            (10_000, 100 * b["alive_frac"]), textcoords="offset points", xytext=(-10, 16),
            ha="right", fontsize=8.8, color=COL["breakout"], weight="semibold")
bx.axvline(10_000, color=INK2, lw=1, ls=":", zorder=2)
bx.set_xscale("log")
bx.set_xlabel("evaluation horizon (steps, log)", color=INK2, fontsize=9.5)
bx.set_ylabel("episodes still alive at the cap (%)", color=INK2, fontsize=9.5)
bx.set_title("How much of the score the clock decides", color=INK, fontsize=12.5,
             loc="left", pad=11)
bx.set_xlim(85, 42_000); bx.set_ylim(-4, 104)
bx.legend(frameon=False, fontsize=9, loc="lower left", labelcolor=INK2)

fig.suptitle("Experiment 5's one-shot networks, re-evaluated across evaluation horizons",
             color=INK, fontsize=13.5, x=0.012, ha="left", y=0.995, weight="semibold")
fig.text(0.012, 0.012,
         "1,000 episodes per game, each run once to 30,000 steps so horizons are nested and paired. "
         "MinAtar 1.0.15, sticky actions 0.1. Networks as released by the authors.\n"
         "Bars are 95% CI. PPO checkpoints were not released, so PPO appears only as the authors' "
         "reported mean at their own 10,000-step cap.",
         fontsize=7.8, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 0.945))
path = os.path.join(OUT, "horizon-sweep.png")
fig.savefig(path, dpi=190, facecolor=SURF)
print("wrote", os.path.abspath(path))
