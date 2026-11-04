"""Contact sheet: what each live hidden unit of the released Breakout network reads."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]; b1 = z["actor/dense1/bias"]
G = W1.reshape(10, 10, 6, 32)

UNITS = [0, 1, 2, 3, 4, 5]
CHANS = [(0, "paddle"), (2, "bricks"), (3, "ball"), (4, "trail")]
ROLE = {0: "landing error A", 1: "landing error B", 2: "landing error C", 3: "landing error D",
        4: "horizontal direction", 5: "vertical direction"}
INK, INK2, GRID, SURF = "#15181f", "#4a515f", "#dde1e8", "#ffffff"

fig, axes = plt.subplots(len(UNITS), len(CHANS), figsize=(9.2, 12.6), facecolor=SURF)
for i, j in enumerate(UNITS):
    vmax = float(np.abs(G[:, :, :, j]).max()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    for k, (ch, name) in enumerate(CHANS):
        ax = axes[i, k]
        m = G[:, :, ch, j]
        ax.imshow(m, cmap="RdBu_r", norm=norm, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        if i == 0:
            ax.set_title(name, fontsize=10.5, color=INK, pad=7)
        if k == 0:
            ax.set_ylabel(f"unit {j}\n{ROLE[j]}\n±{vmax:g}", fontsize=8.6, color=INK2, rotation=0,
                          ha="right", va="center", labelpad=52)
        if np.count_nonzero(m) == 0:
            ax.text(4.5, 4.5, "never read", ha="center", va="center", fontsize=9, color=INK2, alpha=0.8)

fig.suptitle("What the written Breakout network reads: 6 live hidden units of 32, by input channel",
             color=INK, fontsize=12.5, x=0.012, ha="left", y=0.985, weight="semibold")
fig.text(0.012, 0.012,
         "Each panel is the 10×10 grid of first-layer weights from that channel into that unit (red +, blue −, "
         "white 0; scale per row). Units 6–31 are all-zero.\nThe bricks column is empty for every unit: the policy "
         "never reads where the bricks are. It intercepts the ball and nothing else.",
         fontsize=8.2, color=INK2, va="bottom")
fig.tight_layout(rect=(0.02, 0.045, 1, 0.965))
out = os.path.join(os.path.dirname(__file__), "breakout-units.png")
fig.savefig(out, dpi=170, facecolor=SURF)
print("wrote", out)
