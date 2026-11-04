"""Analysis for the horizon sweep. Outputs exactly what RULE-FROZEN-2026-11-03.md pre-declared."""
import json, os

import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "results")
HORIZONS = [100, 300, 1_000, 3_000, 10_000, 30_000]

# authors' reported values at their 10,000-step cap (exp5 eval/minatar_eval.json)
AUTHORS = {
    "breakout": {"mean": 517.411, "se": 7.416952829044303, "mean_length": 7417.202},
    "seaquest": {"mean": 53.110, "se": 1.3573821814015339, "mean_length": 1411.613},
    "asterix": {"mean": 53.178, "se": 1.023305169277215, "mean_length": 858.045},
}
# authors' best PPO comparator at the same cap, for context only (checkpoints not released)
PPO_500M = {"breakout": 25.344, "seaquest": 88.88, "asterix": 55.20}


def load(game):
    with open(os.path.join(OUT, f"{game}.episodes.json")) as fh:
        d = json.load(fh)
    recs = d["records"]
    rets = {h: np.array([r["returns"][str(h)] for r in recs], dtype=float) for h in HORIZONS}
    lens = np.array([r["length"] for r in recs], dtype=float)
    cens = np.array([bool(r["censored"]) for r in recs])
    return rets, lens, cens, len(recs)


def main():
    report = {}
    print("=" * 104)
    print("GATE: mean return at horizon 10,000 vs the authors' reported mean")
    print("=" * 104)
    print(f"{'game':9s} {'mine':>10s} {'SE':>8s} {'authors':>10s} {'SE':>8s} {'diff':>9s} {'SE_comb':>8s} {'z':>7s}  verdict")
    gate_all = True
    for g in AUTHORS:
        rets, lens, cens, n = load(g)
        mine = rets[10_000]
        m, se = mine.mean(), mine.std(ddof=1) / np.sqrt(n)
        am, ase = AUTHORS[g]["mean"], AUTHORS[g]["se"]
        sec = float(np.hypot(se, ase))
        z = (m - am) / sec
        ok = abs(z) <= 3.0
        gate_all &= ok
        print(f"{g:9s} {m:10.3f} {se:8.3f} {am:10.3f} {ase:8.3f} {m-am:+9.3f} {sec:8.3f} {z:+7.2f}  {'PASS' if ok else 'FAIL'}")
        report[g] = {"gate": {"mine": m, "se": se, "authors": am, "authors_se": ase, "z": z, "pass": bool(ok)}}
    print(f"\nGATE OVERALL: {'PASS' if gate_all else 'FAIL'}  (criterion |z| <= 3, frozen before the run)\n")

    for g in AUTHORS:
        rets, lens, cens, n = load(g)
        print("=" * 104)
        print(f"{g.upper()}   n={n} episodes, each run once to 30,000 steps (horizons nested and paired)")
        print("=" * 104)
        print(f"{'horizon':>8s} {'mean return':>12s} {'SE':>7s} {'alive at cap':>13s} {'mean length':>12s} "
              f"{'reward/step':>12s} {'x PPO-500M':>11s}")
        rows = []
        for h in HORIZONS:
            r = rets[h]
            # an episode is cut by the cap at h iff it had not terminated by step h.
            # `censored` means it never terminated at all (length == MAX_STEPS), so it is cut at
            # every rung; otherwise it is cut only if it ran strictly past h. Using `length >= h`
            # miscounts episodes that ended exactly at h; using `length > h` miscounts the top rung.
            alive = float(np.mean(cens | (lens > h)))
            eff = np.minimum(lens, h)
            rate = r.sum() / eff.sum()
            row = {
                "horizon": h,
                "mean": float(r.mean()),
                "se": float(r.std(ddof=1) / np.sqrt(n)),
                "alive_frac": alive,
                "mean_length": float(eff.mean()),
                "pooled_reward_per_step": float(rate),
                "ratio_to_ppo500m": float(r.mean() / PPO_500M[g]),
            }
            rows.append(row)
            print(f"{h:8d} {row['mean']:12.3f} {row['se']:7.3f} {alive:12.1%} {row['mean_length']:12.1f} "
                  f"{rate:12.4f} {row['ratio_to_ppo500m']:11.2f}")
        report[g]["ladder"] = rows

        # shape of return vs horizon: a doubling of the cap multiplies the return by what?
        print("\n  growth between consecutive horizons (return ratio vs horizon ratio):")
        for a, b in zip(HORIZONS, HORIZONS[1:]):
            ra, rb = rets[a].mean(), rets[b].mean()
            hr, rr = b / a, (rb / ra if ra > 0 else float("nan"))
            print(f"    {a:6d} -> {b:6d}   horizon x{hr:5.2f}   return x{rr:5.2f}   "
                  f"{'(linear would be x%.2f)' % hr}")
        report[g]["survival"] = {
            "mean_length_30k": float(lens.mean()),
            "median_length": float(np.median(lens)),
            "censored_at_30k": float(np.mean(cens)),
            "quantiles": {str(q): float(np.quantile(lens, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        }
        s = report[g]["survival"]
        print(f"\n  episode length: median {s['median_length']:.0f}, mean {s['mean_length_30k']:.0f}, "
              f"{s['censored_at_30k']:.1%} still alive at 30,000 (administrative censoring)")
        print(f"  authors' mean length at their 10,000 cap: {AUTHORS[g]['mean_length']:.1f}; "
              f"mine at the same cap: {np.minimum(lens, 10_000).mean():.1f}\n")

    with open(os.path.join(OUT, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", os.path.join(OUT, "report.json"))


if __name__ == "__main__":
    main()
