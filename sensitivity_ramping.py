"""Sensitivity check: does MinAtar's difficulty ramping affect the acceptance gate?

Breakout ignores ramping; Seaquest and Asterix use it. Re-run those two with ramping OFF and
report the change. Cap 10,000 only - this bears on the gate, not on the horizon ladder.
"""
import json, os, sys
from multiprocessing import Pool

import numpy as np
from minatar import Environment

sys.path.insert(0, os.path.dirname(__file__))
from sweep import CHANNEL_MAP, OFFSET, load_net  # noqa: E402

CAP = 10_000
AUTHORS = {"seaquest": (53.110, 1.3573821814015339), "asterix": (53.178, 1.023305169277215)}


def episode(args):
    game, ep, net, ramping = args
    w1, b1, w2, b2, w3, b3 = net
    env = Environment(game, sticky_action_prob=0.1, difficulty_ramping=ramping)
    off = OFFSET[game]
    env.seed(10_000_000 + off + ep)
    arng = np.random.RandomState(20_000_000 + off + ep)
    own = env.env.channels
    srcs = [[own[n] for n in names] for names in CHANNEL_MAP[game]]
    env.reset()
    grid = np.zeros((10, 10, 6), dtype=bool)
    cum = 0.0
    for step in range(1, CAP + 1):
        s = env.state()
        grid[:] = False
        for k, idx in enumerate(srcs):
            if not idx:
                continue
            if len(idx) == 1:
                grid[:, :, k] = s[:, :, idx[0]]
            else:
                acc = s[:, :, idx[0]]
                for j in idx[1:]:
                    acc = acc | s[:, :, j]
                grid[:, :, k] = acc
        nz = np.flatnonzero(grid.reshape(600))
        h1 = np.tanh(w1[nz].sum(axis=0) + b1) if nz.size else np.tanh(b1)
        h2 = np.tanh(h1 @ w2 + b2)
        lg = h2 @ w3 + b3
        p = np.exp(lg - lg.max())
        p /= p.sum()
        a = int(np.searchsorted(np.cumsum(p), arng.random_sample()))
        a = min(a, 5)
        r, done = env.act(a)
        cum += r
        if done:
            return cum, step
    return cum, CAP


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = {}
    print(f"n={n} episodes per cell, cap {CAP}\n")
    print(f"{'game':9s} {'ramping':8s} {'mean':>8s} {'SE':>7s} {'authors':>8s} {'z':>7s} {'mean_len':>9s}")
    for game in ("seaquest", "asterix"):
        net = load_net(game)
        out[game] = {}
        for ramping in (True, False):
            with Pool(9) as pool:
                res = pool.map(episode, [(game, ep, net, ramping) for ep in range(n)], chunksize=2)
            r = np.array([x[0] for x in res], dtype=float)
            L = np.array([x[1] for x in res], dtype=float)
            m, se = r.mean(), r.std(ddof=1) / np.sqrt(n)
            am, ase = AUTHORS[game]
            z = (m - am) / np.hypot(se, ase)
            out[game]["on" if ramping else "off"] = {
                "mean": float(m), "se": float(se), "z": float(z), "mean_length": float(L.mean())
            }
            print(f"{game:9s} {'on' if ramping else 'off':8s} {m:8.3f} {se:7.3f} {am:8.3f} {z:+7.2f} {L.mean():9.1f}")
    # does turning ramping off move the mean toward the authors'?
    print()
    for game in out:
        on, off = out[game]["on"], out[game]["off"]
        d = off["mean"] - on["mean"]
        sed = float(np.hypot(on["se"], off["se"]))
        print(f"{game:9s} off - on = {d:+7.3f} (SE {sed:.3f}, z {d/sed:+.2f}) ; "
              f"|z to authors| on {abs(on['z']):.2f} -> off {abs(off['z']):.2f}")
    with open(os.path.join(os.path.dirname(__file__), "results", "ramping_sensitivity.json"), "w") as fh:
        json.dump({"episodes": n, "cap": CAP, "results": out}, fh, indent=1)


if __name__ == "__main__":
    main()
