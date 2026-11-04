"""Analysis for the generation-reliability study (generation/PREREGISTRATION.md sections 3, 5, 6, 7, 8).

    python analyze_generation.py --game G --attempts DIR --scores DIR --released results/G.episodes.json \
        --ppo100 MEAN:SE --ppo500 MEAN:SE --out report.json [--markdown report.md]

``DIR/attempt-K.json`` are the driver's attempt records; ``SCORES/attempt-K.episodes.json`` are the
per-network score files in the ``results/`` schema; ``--released`` is the released network's own
score file on the same seeds.  Every rate is over the submitted-attempt denominator; the headline is
never a mean across networks.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

Z95 = 1.959964
CLAIM_MIN_EVALUABLE = 29
CLAIM_PERCENTILE = 0.9
RECORD_RE = re.compile(r"^attempt-(\d+)\.json$")


# ----------------------------------------------------------------------------- statistics
def wilson(count: int, n: int, z: float = Z95) -> dict:
    """Wilson score interval for ``count`` successes of ``n`` (no scipy)."""
    out = {"count": int(count), "denominator": int(n), "rate": None, "low": None, "high": None}
    if n <= 0:
        return out
    p = count / n
    z2n = z * z / n
    centre = (p + z2n / 2) / (1 + z2n)
    half = z * np.sqrt(p * (1 - p) / n + z2n / (4 * n)) / (1 + z2n)
    out.update({"rate": float(p), "low": float(centre - half), "high": float(centre + half)})
    return out


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks with ties, 1-based."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or len(x) != len(y):
        return None
    rx, ry = _ranks(x), _ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


# ----------------------------------------------------------------------------- inputs
def load_attempts(directory: str, game: str) -> list:
    recs = []
    for path in sorted(glob.glob(os.path.join(directory, "attempt-*.json"))):
        if not RECORD_RE.match(os.path.basename(path)):
            continue
        with open(path) as f:
            r = json.load(f)
        if r.get("game") == game:
            recs.append(r)
    recs.sort(key=lambda r: r["index"])
    return recs


def load_scores(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    recs = d["records"]
    eps = np.array([int(r["ep"]) for r in recs], dtype=int)
    if len(set(eps.tolist())) != len(eps):
        raise SystemExit(f"{path}: duplicate episode indices")
    r10 = np.array([r["returns"]["10000"] for r in recs], dtype=float)
    r30 = np.array([r["returns"]["30000"] for r in recs], dtype=float)
    lens = np.array([r["length"] for r in recs], dtype=float)
    cens = np.array([bool(r["censored"]) for r in recs])
    return {"ep": eps, "r10": r10, "r30": r30, "lengths": lens, "censored": cens}


def paired_panel(named_scores: list) -> np.ndarray:
    """Stack 10,000-step returns into a (networks x episodes) panel aligned on each record's ``ep``.

    ``named_scores`` is a list of ``(name, scores)``.  Every file must carry exactly the same set of
    episode indices; a missing or extra seed anywhere is an error, never a truncation.
    """
    if not named_scores:
        return np.zeros((0, 0))
    ref_name, ref = named_scores[0]
    ref_eps = sorted(ref["ep"].tolist())
    rows = []
    for name, s in named_scores:
        eps = sorted(s["ep"].tolist())
        if eps != ref_eps:
            raise SystemExit(f"episode indices of {name} do not match {ref_name}: "
                             f"missing {sorted(set(ref_eps) - set(eps))}, extra {sorted(set(eps) - set(ref_eps))}")
        order = np.argsort(s["ep"], kind="mergesort")
        rows.append(s["r10"][order])
    return np.stack(rows)


def performance(s: dict) -> dict:
    n = len(s["r10"])
    se = float(s["r10"].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
    return {
        "n_episodes": n,
        "mean_10000": float(s["r10"].mean()),
        "se_10000": se,
        "mean_30000": float(s["r30"].mean()),
        "mean_length": float(s["lengths"].mean()),
        "censored_share_10000": float(np.mean((s["lengths"] > 10_000) | s["censored"])),
        "censored_share_30000": float(np.mean(s["censored"])),
    }


def parse_mean_se(text: str) -> dict:
    m, se = text.split(":")
    return {"mean": float(m), "se": float(se)}


# ----------------------------------------------------------------------------- analysis
def variance_components(panel: np.ndarray) -> dict:
    """One-way random-effects decomposition on a (networks x episodes) panel of 10,000-step returns."""
    k, m = panel.shape
    out = {"n_networks": int(k), "n_episodes": int(m), "within_network_variance": None,
           "between_network_variance": None, "ratio": None, "paired_mean_variance_across_networks": None}
    if k < 1 or m < 2:
        return out
    # within-network variance needs only one network with two or more episodes
    within = float(np.mean(panel.var(axis=1, ddof=1)))
    out["within_network_variance"] = within
    if k < 2:
        return out
    between = float(max(panel.mean(axis=1).var(ddof=1) - within / m, 0.0))
    out.update({
        "between_network_variance": between,
        "ratio": (between / within) if within > 0 else None,
        "paired_mean_variance_across_networks": float(np.mean(panel.var(axis=0, ddof=1))),
    })
    return out


def analyze(game: str, attempts: list, scores: dict, released: dict, ppo100: dict, ppo500: dict) -> dict:
    submitted = [a for a in attempts if a["outcome"] != "not-submitted"]
    n_sub = len(submitted)
    replied = [a for a in submitted if a["outcome"] == "replied"]
    evaluable = [a for a in submitted if a.get("evaluable")]
    report = {
        "game": game,
        "n_attempts": len(attempts),
        "n_submitted": n_sub,
        "not_submitted_count": sum(a["outcome"] == "not-submitted" for a in attempts),
        "transport_count": sum(a["outcome"] == "transport-failure" for a in attempts),
        "replied_rate": wilson(len(replied), n_sub),
        "block_rate": wilson(sum(bool(a.get("block_found")) for a in submitted), n_sub),
        "evaluable_rate": wilson(len(evaluable), n_sub),
        "clean_rate": wilson(sum(bool(a.get("clean")) for a in submitted), n_sub),
        "ppo100": ppo100,
        "ppo500": ppo500,
    }

    networks = []
    for a in evaluable:
        idx = a["index"]
        if idx not in scores:
            raise SystemExit(f"evaluable attempt {idx} has no score file")
        net = {"index": idx, "wall_seconds": a.get("wall_seconds")}
        net.update(performance(scores[idx]))
        for key, ppo in (("100m", ppo100), ("500m", ppo500)):
            net["exceeds_" + key] = net["mean_10000"] > ppo["mean"]
            net["near_boundary_" + key] = bool(
                abs(net["mean_10000"] - ppo["mean"]) < 2 * np.sqrt(net["se_10000"] ** 2 + ppo["se"] ** 2))
        networks.append(net)
    report["networks"] = networks
    n_eval = len(networks)
    means = np.array([n["mean_10000"] for n in networks], dtype=float)

    report["thresholds"] = {
        "exceeds_100m": wilson(sum(n["exceeds_100m"] for n in networks), n_eval),
        "exceeds_500m": wilson(sum(n["exceeds_500m"] for n in networks), n_eval),
        "near_boundary_100m": {"count": sum(n["near_boundary_100m"] for n in networks), "denominator": n_eval},
        "near_boundary_500m": {"count": sum(n["near_boundary_500m"] for n in networks), "denominator": n_eval},
    }

    if n_eval:
        q25, q50, q75 = np.percentile(means, [25, 50, 75])
        dist = {"median": float(q50), "iqr": [float(q25), float(q75)],
                "min": float(means.min()), "max": float(means.max())}
    else:
        dist = {"median": None, "iqr": None, "min": None, "max": None}
    dist["ecdf"] = sorted(float(v) for v in means)
    report["distribution"] = dist

    rel = performance(released)
    rel_mean = rel["mean_10000"]
    allowed = n_eval >= CLAIM_MIN_EVALUABLE and bool(np.all(means < rel_mean))
    n_below = int(np.sum(means < rel_mean))
    rel.update({
        "percentile": (n_below / n_eval) if n_eval else None,
        "n_evaluable": n_eval,
        "percentile_claim_allowed": allowed,
        "percentile_claim_alpha": (CLAIM_PERCENTILE ** n_eval) if allowed else None,
        # attempt-level estimand: every submitted attempt without an evaluable network counts as below
        "percentile_attempt_level": ((n_sub - n_eval + n_below) / n_sub) if n_sub else None,
    })
    report["released"] = rel

    # the released file must sit on the same seeds as every generated network, so it joins the
    # alignment check; the panel itself holds only the generated networks
    paired_panel([("released", released)] + [(f"attempt {n['index']}", scores[n["index"]]) for n in networks])
    panel = paired_panel([(f"attempt {n['index']}", scores[n["index"]]) for n in networks])
    report["variance_components"] = variance_components(panel)

    shas = {}
    for a in replied:
        if a.get("reply_sha256"):
            shas.setdefault(a["reply_sha256"], []).append(a["index"])
    collisions = {k: v for k, v in shas.items() if len(v) > 1}
    cached = np.array([(a.get("usage") or {}).get("cached_input_tokens") for a in submitted
                       if (a.get("usage") or {}).get("cached_input_tokens") is not None], dtype=float)
    walls = [(a["index"], a["wall_seconds"]) for a in submitted if a.get("wall_seconds") is not None]
    report["exchangeability"] = {
        "spearman_wall_seconds_vs_index": spearman([w[0] for w in walls], [w[1] for w in walls]),
        "spearman_mean_10000_vs_index": spearman([n["index"] for n in networks], means),
        "reply_sha256_collisions": {"count": sum(len(v) - 1 for v in collisions.values()), "groups": collisions},
        "cached_input_tokens": {
            "n": int(len(cached)),
            "min": float(cached.min()) if len(cached) else None,
            "median": float(np.median(cached)) if len(cached) else None,
            "mean": float(cached.mean()) if len(cached) else None,
            "max": float(cached.max()) if len(cached) else None,
        },
    }
    return report


# ----------------------------------------------------------------------------- markdown
def _w(w: dict) -> str:
    if w["rate"] is None:
        return f"{w['count']}/{w['denominator']} (no denominator)"
    return f"{w['count']}/{w['denominator']} = {100 * w['rate']:.1f}% (Wilson 95% {100 * w['low']:.1f}–{100 * w['high']:.1f})"


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def markdown(r: dict) -> str:
    d, rel, t, v, e = r["distribution"], r["released"], r["thresholds"], r["variance_components"], r["exchangeability"]
    lines = [f"# Generation reliability — {r['game']}", "", "## Headline", "",
             f"- Evaluable rate: {_w(r['evaluable_rate'])}",
             f"- Clean rate: {_w(r['clean_rate'])}",
             f"- Replied: {_w(r['replied_rate'])}; block present: {_w(r['block_rate'])}; "
             f"not submitted {r['not_submitted_count']}; transport failures {r['transport_count']}",
             f"- ECDF of 10,000-step means over {rel['n_evaluable']} evaluable networks: "
             f"median {_f(d['median'])}, IQR {_f(d['iqr'][0]) if d['iqr'] else 'n/a'}–{_f(d['iqr'][1]) if d['iqr'] else 'n/a'}, "
             f"min {_f(d['min'])}, max {_f(d['max'])}",
             f"- Released network: mean {_f(rel['mean_10000'])} (SE {_f(rel['se_10000'])}); "
             f"percentile among evaluable networks {_f(rel['percentile'])}; "
             f"attempt-level percentile (non-evaluable counted as below) {_f(rel['percentile_attempt_level'])}",
             f"- Percentile claim (≥ {CLAIM_MIN_EVALUABLE} evaluable and released above all): "
             f"{'ALLOWED' if rel['percentile_claim_allowed'] else 'NOT ALLOWED'}"
             + (f", one-sided alpha {rel['percentile_claim_alpha']:.4f}" if rel["percentile_claim_allowed"] else ""),
             "", "## Thresholds (never phrased as beating PPO)", "",
             f"- Exceeds the 100M PPO mean {r['ppo100']['mean']} (SE {r['ppo100']['se']}): {_w(t['exceeds_100m'])}; "
             f"near boundary {t['near_boundary_100m']['count']}/{t['near_boundary_100m']['denominator']}",
             f"- Exceeds the 500M PPO mean {r['ppo500']['mean']} (SE {r['ppo500']['se']}): {_w(t['exceeds_500m'])}; "
             f"near boundary {t['near_boundary_500m']['count']}/{t['near_boundary_500m']['denominator']}",
             "", "## Per-network performance", "",
             "| attempt | mean 10,000 | SE | mean 30,000 | mean length | censored @10k | censored @30k | >100M | >500M |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in r["networks"]:
        lines.append(f"| {n['index']} | {n['mean_10000']:.3f} | {n['se_10000']:.3f} | {n['mean_30000']:.3f} | "
                     f"{n['mean_length']:.1f} | {n['censored_share_10000']:.3f} | {n['censored_share_30000']:.3f} | "
                     f"{'yes' if n['exceeds_100m'] else 'no'}{'*' if n['near_boundary_100m'] else ''} | "
                     f"{'yes' if n['exceeds_500m'] else 'no'}{'*' if n['near_boundary_500m'] else ''} |")
    lines += ["", "`*` within 2·sqrt(SE_gen² + SE_PPO²) of the threshold.", "",
              "## Variance components (paired seed panel)", "",
              f"- Networks {v['n_networks']}, episodes {v['n_episodes']}",
              f"- Within-network variance {_f(v['within_network_variance'])}; "
              f"between-network variance {_f(v['between_network_variance'])}; ratio {_f(v['ratio'])}",
              f"- Paired: mean over episode indices of the variance across networks {_f(v['paired_mean_variance_across_networks'])}",
              "", "## Exchangeability diagnostics (reported, not corrected)", "",
              f"- Spearman rho, wall seconds vs attempt index: {_f(e['spearman_wall_seconds_vs_index'])}",
              f"- Spearman rho, 10,000-step mean vs attempt index: {_f(e['spearman_mean_10000_vs_index'])}",
              f"- Reply SHA-256 collisions: {e['reply_sha256_collisions']['count']}",
              f"- cached_input_tokens over {e['cached_input_tokens']['n']} attempts: min {_f(e['cached_input_tokens']['min'], 0)}, "
              f"median {_f(e['cached_input_tokens']['median'], 0)}, mean {_f(e['cached_input_tokens']['mean'], 0)}, "
              f"max {_f(e['cached_input_tokens']['max'], 0)}", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", required=True)
    ap.add_argument("--attempts", required=True, help="directory of attempt-K.json records")
    ap.add_argument("--scores", required=True, help="directory of attempt-K.episodes.json score files")
    ap.add_argument("--released", required=True, help="released network's score file on the same seeds")
    ap.add_argument("--ppo100", required=True, help="MEAN:SE of the best-seed PPO at 100M")
    ap.add_argument("--ppo500", required=True, help="MEAN:SE of the best-seed PPO at 500M")
    ap.add_argument("--out", required=True)
    ap.add_argument("--markdown")
    a = ap.parse_args(argv)

    attempts = load_attempts(a.attempts, a.game)
    scores = {}
    for att in attempts:
        if att.get("evaluable"):
            p = os.path.join(a.scores, f"attempt-{att['index']}.episodes.json")
            if os.path.exists(p):
                scores[att["index"]] = load_scores(p)
    report = analyze(a.game, attempts, scores, load_scores(a.released),
                     parse_mean_se(a.ppo100), parse_mean_se(a.ppo500))
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write(markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
