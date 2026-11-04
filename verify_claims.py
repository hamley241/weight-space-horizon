"""Re-derive every number quoted in the README directly from the raw per-episode records."""
import json, os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results")
H = [100, 300, 1000, 3000, 10000, 30000]
PPO = {"breakout": 25.344, "seaquest": 88.88, "asterix": 55.20}
PPO_LEN = {"breakout": 281.852}


def load(g):
    with open(os.path.join(OUT, f"{g}.episodes.json")) as fh:
        recs = json.load(fh)["records"]
    rets = {h: np.array([r["returns"][str(h)] for r in recs], float) for h in H}
    lens = np.array([r["length"] for r in recs], float)
    return rets, lens


print("1. CENSORING PREDICATE  (cut_at(h) = censored OR length > h; naive rules shown for contrast)")
print(f"{'game':9s} {'h':>6s} {'length>=h':>10s} {'length>h':>9s} {'used':>7s}")
for g in ("breakout", "seaquest", "asterix"):
    with open(os.path.join(OUT, f"{g}.episodes.json")) as fh:
        recs = json.load(fh)["records"]
    lens = np.array([r["length"] for r in recs], float)
    cens = np.array([bool(r["censored"]) for r in recs])
    for h in H:
        a_ge, a_gt, used = np.mean(lens >= h), np.mean(lens > h), np.mean(cens | (lens > h))
        if abs(a_ge - used) > 1e-12 or abs(a_gt - used) > 1e-12:
            print(f"{g:9s} {h:6d} {a_ge:10.3%} {a_gt:9.3%} {used:7.3%}")

print("\n2. SATURATION BEYOND A 3,000-STEP HORIZON")
for g in ("seaquest", "asterix"):
    _, lens = load(g)
    print(f"  {g:9s} alive>3000 = {np.mean(lens > 3000):.3%}   max length = {lens.max():.0f}   "
          f"n beyond 3000 = {int((lens > 3000).sum())}")

print("\n3. HORIZON SENSITIVITY ACROSS THE LADDER")
for g in ("breakout", "seaquest", "asterix"):
    rets, lens = load(g)
    lo, hi = rets[100].mean(), rets[30000].mean()
    beyond = rets[30000].mean() / rets[3000].mean()
    print(f"  {g:9s} 100->30k x{hi/lo:6.2f}   3k->30k x{beyond:5.3f}  "
          f"(materially sensitive beyond 3,000? {'YES' if beyond > 1.01 else 'no'})")

print("\n4. THE DECOMPOSITION AT THE MATCHED RUNG (10,000)")
rets, lens = load("breakout")
eff = np.minimum(lens, 10000)
ratio = rets[10000].mean() / PPO["breakout"]
dur = eff.mean() / PPO_LEN["breakout"]
dens = (rets[10000].sum() / eff.sum()) / (PPO["breakout"] / PPO_LEN["breakout"])
print(f"  ratio {ratio:.4f} = duration {dur:.4f} x density {dens:.5f} = {dur*dens:.4f}")

print("\n5. RETURN EARNED BEYOND THE PUBLISHED CAP")
inc = rets[30000] - rets[10000]
n = len(inc)
rng = np.random.RandomState(7)
boot = np.array([inc[rng.randint(0, n, n)].mean() for _ in range(10000)])
surv = lens > 10000
print(f"  mean increment 10k->30k = {inc.mean():.3f}  paired SE {inc.std(ddof=1)/np.sqrt(n):.3f}  "
      f"bootstrap 95% CI [{np.percentile(boot,2.5):.3f}, {np.percentile(boot,97.5):.3f}]")
print(f"  that is {100*inc.mean()/rets[30000].mean():.2f}% of the 30,000-step mean")
print(f"  episodes surviving 10,000: {surv.sum()}   of those, gained >0: {(inc[surv] > 0).sum()}   "
      f"mean gain among survivors {inc[surv].mean():.3f}")
print(f"  still censored at 30,000: {np.mean(lens >= 30000):.1%}  -> 910.6 is not a saturated return")

print("\n6. SHAPE: EXPONENT WITH A PAIRED EPISODE BOOTSTRAP")
x = np.log10(np.array(H[:5], float))
def slope(mat):
    y = np.log10(mat)
    return np.polyfit(x, y, 1)[0]
means = np.array([rets[h].mean() for h in H[:5]])
s0 = slope(means)
bs = []
for _ in range(2000):
    idx = rng.randint(0, n, n)
    bs.append(slope(np.array([rets[h][idx].mean() for h in H[:5]])))
bs = np.array(bs)
print(f"  OLS exponent {s0:.6f}   bootstrap 95% CI [{np.percentile(bs,2.5):.5f}, {np.percentile(bs,97.5):.5f}]")
loc = np.log10(rets[30000].mean()/rets[10000].mean()) / np.log10(3.0)
print(f"  local exponent 10k->30k = {loc:.5f}")

print("\n7. PER-EPISODE vs POOLED REWARD DENSITY")
for h in (100, 30000):
    e = np.minimum(lens, h)
    per = (rets[h] / e).mean()
    pooled = rets[h].sum() / e.sum()
    print(f"  h={h:6d}  per-episode {per:.6f}   pooled {pooled:.6f}")
cens = lens >= 30000
for h in (100, 30000):
    e = np.minimum(lens, h)
    print(f"  h={h:6d}  pooled among the {cens.sum()} censored-at-30k episodes: "
          f"{rets[h][cens].sum()/e[cens].sum():.6f}")
