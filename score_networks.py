"""Batch scorer: run sweep.py's episode protocol over arbitrary network npz files.

    python score_networks.py --game breakout --npz PATH [PATH ...] --out-dir OUT [--episodes 1000] [--procs 9]

Each npz is scored with exactly the sweep's seeds, so scoring a released `*_hand.npz` reproduces
the records in results/<game>.episodes.json. Output: OUT/<npz-stem>.episodes.json in the same
schema as the results files, plus "source_npz" and "source_sha256".
"""
import argparse, hashlib, json, os, sys
from multiprocessing import Pool

import numpy as np

from sweep import HORIZONS, MAX_STEPS, load_net, run_episode


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def out_path(out_dir, npz):
    stem = os.path.splitext(os.path.basename(npz))[0]
    return os.path.join(out_dir, f"{stem}.episodes.json")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", required=True, choices=["breakout", "seaquest", "asterix"])
    ap.add_argument("--npz", nargs="+", required=True, help="network npz files (sweep.load_net key names)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--procs", type=int, default=9)
    ap.add_argument("--force", action="store_true", help="overwrite existing output files")
    a = ap.parse_args(argv)

    # Every input must map to its own output file. Two inputs with the same stem would collide
    # within this invocation, so reject that up front regardless of --force; --force only covers
    # outputs that already existed before the run.
    dests = {}
    for p in a.npz:
        dests.setdefault(out_path(a.out_dir, p), []).append(p)
    dupes = {d: srcs for d, srcs in dests.items() if len(srcs) > 1}
    if dupes:
        print(
            "duplicate output paths (inputs share an npz stem): "
            + "; ".join(f"{d} <- {', '.join(srcs)}" for d, srcs in dupes.items()),
            file=sys.stderr,
        )
        return 1
    if not a.force:
        existing = [d for d in dests if os.path.exists(d)]
        if existing:
            print("refusing to overwrite (use --force): " + ", ".join(existing), file=sys.stderr)
            return 1
    os.makedirs(a.out_dir, exist_ok=True)

    for npz in a.npz:
        net = load_net(npz)
        jobs = [(a.game, ep, net) for ep in range(a.episodes)]
        with Pool(a.procs) as pool:
            res = pool.map(run_episode, jobs, chunksize=2)
        with open(out_path(a.out_dir, npz), "w") as fh:
            json.dump(
                {
                    "game": a.game,
                    "episodes": a.episodes,
                    "max_steps": MAX_STEPS,
                    "horizons": HORIZONS,
                    "sticky_action_prob": 0.1,
                    "difficulty_ramping": True,
                    "minatar_version": "1.0.15",
                    "records": res,
                    "source_npz": npz,
                    "source_sha256": sha256(npz),
                },
                fh,
            )
        n = len(res)
        at10k = np.array([r["returns"]["10000"] for r in res])
        lens = np.array([r["length"] for r in res], dtype=float)
        se = at10k.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
        print(
            f"{os.path.basename(npz)} n={n} cap10k mean={at10k.mean():9.3f} se={se:7.4f} "
            f"mean_len={lens.mean():9.1f} cens@30k={np.mean([r['censored'] for r in res]):.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
