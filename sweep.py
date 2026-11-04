"""Horizon sweep, optimized inner loop. Same protocol as sweep.py (RULE-FROZEN-2026-11-03.md).

The observation is binary, so the first layer is a gather-and-sum over the rows of W1 whose input
bit is set (typically ~20-40 of 600) instead of a dense 600x32 matmul.
"""
import argparse, json, os
from multiprocessing import Pool

import numpy as np
from minatar import Environment

MAX_STEPS = 30_000
HORIZONS = [100, 300, 1_000, 3_000, 10_000, 30_000]

CHANNEL_MAP = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
    "seaquest": [
        ["sub_front", "sub_back"],
        ["enemy_fish", "enemy_sub", "enemy_bullet"],
        ["enemy_fish", "enemy_sub", "diver"],
        ["friendly_bullet"],
        ["trail", "sub_back"],
        ["oxygen_guage", "diver_guage"],
    ],
}
OFFSET = {"breakout": 0, "seaquest": 100_000, "asterix": 200_000}
NET_DIR = os.path.join(os.path.dirname(__file__), "data", "networks")


def load_net(game_or_path):
    """Load the six actor tensors from a game name (data/networks/<game>.hand.npz) or an npz path."""
    game_or_path = os.fspath(game_or_path)
    if game_or_path.endswith(".npz"):
        path = game_or_path
    else:
        path = os.path.join(NET_DIR, f"{game_or_path}.hand.npz")
    with np.load(path, allow_pickle=False) as z:
        k = {n: z[n] for n in z.files}
    return (
        np.ascontiguousarray(k["actor/dense1/kernel"], dtype=np.float64),
        k["actor/dense1/bias"].astype(np.float64),
        np.ascontiguousarray(k["actor/dense2/kernel"], dtype=np.float64),
        k["actor/dense2/bias"].astype(np.float64),
        np.ascontiguousarray(k["actor/logits/kernel"], dtype=np.float64),
        k["actor/logits/bias"].astype(np.float64),
    )


def run_episode(args):
    game, ep, net = args
    w1, b1, w2, b2, w3, b3 = net
    env = Environment(game, sticky_action_prob=0.1, difficulty_ramping=True)
    off = OFFSET[game]
    env.seed(10_000_000 + off + ep)
    arng = np.random.RandomState(20_000_000 + off + ep)
    own = env.env.channels
    # target channel k -> list of source channel indices in the game's own state()
    srcs = [[own[n] for n in names] for names in CHANNEL_MAP[game]]
    env.reset()

    grid = np.zeros((10, 10, 6), dtype=bool)
    uniform = np.full(6, 1.0 / 6)
    cum = 0.0
    marks = {}
    hi = 0
    term_step = None
    rand = arng.random_sample
    cumsum = np.cumsum

    for step in range(1, MAX_STEPS + 1):
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
        logits = h2 @ w3 + b3
        m = logits.max()
        p = np.exp(logits - m)
        tot = p.sum()
        if not np.isfinite(tot) or tot <= 0.0:
            p = uniform
        else:
            p = p / tot
        # inverse-CDF draw; equivalent in distribution to RandomState.choice(6, p=p)
        a = int(np.searchsorted(cumsum(p), rand()))
        if a > 5:
            a = 5

        r, done = env.act(a)
        cum += r
        while hi < len(HORIZONS) and step == HORIZONS[hi]:
            marks[HORIZONS[hi]] = cum
            hi += 1
        if done:
            term_step = step
            break

    for h in HORIZONS:
        marks.setdefault(h, cum)
    return {
        "ep": ep,
        "returns": {str(h): marks[h] for h in HORIZONS},
        "length": term_step if term_step is not None else MAX_STEPS,
        "censored": term_step is None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="+", default=["breakout", "seaquest", "asterix"])
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    for game in a.games:
        net = load_net(game)
        jobs = [(game, ep, net) for ep in range(a.episodes)]
        with Pool(a.procs) as pool:
            res = pool.map(run_episode, jobs, chunksize=2)
        with open(os.path.join(a.out, f"{game}.episodes.json"), "w") as fh:
            json.dump(
                {
                    "game": game,
                    "episodes": a.episodes,
                    "max_steps": MAX_STEPS,
                    "horizons": HORIZONS,
                    "sticky_action_prob": 0.1,
                    "difficulty_ramping": True,
                    "minatar_version": "1.0.15",
                    "records": res,
                },
                fh,
            )
        n = len(res)
        at10k = np.array([r["returns"]["10000"] for r in res])
        lens = np.array([r["length"] for r in res], dtype=float)
        print(
            f"{game:9s} n={n} cap10k mean={at10k.mean():9.3f} se={at10k.std(ddof=1)/np.sqrt(n):7.4f} "
            f"mean_len={lens.mean():9.1f} cens@30k={np.mean([r['censored'] for r in res]):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
