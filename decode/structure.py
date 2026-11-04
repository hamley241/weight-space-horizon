"""Decode the released one-shot Breakout network: what does each hidden unit look at, and how do
units map to actions? Pure structural analysis of the tensors, no rollouts."""
import json, os
import numpy as np

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
CH = ["player", "danger", "reward", "projectile", "trail", "gauge"]      # prompt's 6 channels
LIVE = {0: "paddle", 2: "brick", 3: "ball", 4: "trail"}                  # breakout's sources; 1,5 empty
ACT = ["noop", "left", "up", "right", "down", "fire"]

with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]; b1 = z["actor/dense1/bias"]
    W2 = z["actor/dense2/kernel"]; b2 = z["actor/dense2/bias"]
    W3 = z["actor/logits/kernel"]; b3 = z["actor/logits/bias"]

G = W1.reshape(10, 10, 6, 32)   # row, col, channel, unit   (index = (row*10+col)*6 + ch)

print("NONZERO COUNTS  W1 %d  b1 %d  W2 %d  b2 %d  W3 %d  b3 %d  total %d" % tuple(
    int(np.count_nonzero(a)) for a in (W1, b1, W2, b2, W3, b3, np.concatenate([a.ravel() for a in (W1,b1,W2,b2,W3,b3)]))))

print("\nLAYER 1 — what each hidden unit reads (nonzero inputs per channel; rows/cols touched)")
live_units = []
for j in range(32):
    col = G[:, :, :, j]
    nz = int(np.count_nonzero(col))
    if nz == 0 and b1[j] == 0:
        continue
    live_units.append(j)
    parts = []
    for ch in range(6):
        m = col[:, :, ch]
        if np.count_nonzero(m) == 0:
            continue
        rows = sorted(set(np.nonzero(m)[0].tolist())); cols = sorted(set(np.nonzero(m)[1].tolist()))
        vals = m[m != 0]
        parts.append(f"{CH[ch]}({LIVE.get(ch,'DEAD')}): {np.count_nonzero(m)} cells rows{rows} cols{cols} "
                     f"w∈[{vals.min():+.2f},{vals.max():+.2f}]")
    print(f"  u{j:02d} b={b1[j]:+.2f} nz={nz:3d} | " + " ; ".join(parts))
print(f"\n  live layer-1 units: {len(live_units)} of 32 -> {live_units}")

print("\nLAYER 2 — hidden-to-hidden structure")
for k in range(32):
    src = np.nonzero(W2[:, k])[0]
    if len(src) == 0 and b2[k] == 0:
        continue
    print(f"  h2[{k:02d}] b={b2[k]:+.2f} <- " + ", ".join(f"u{s}({W2[s,k]:+.2f})" for s in src))

print("\nLAYER 3 — which layer-2 units drive which actions")
for a in range(6):
    src = np.nonzero(W3[:, a])[0]
    print(f"  {ACT[a]:5s} b={b3[a]:+.2f} <- " + (", ".join(f"h2[{s}]({W3[s,a]:+.2f})" for s in src) or "(nothing)"))

# dead-channel check: any weight on channels that are always zero in breakout is inert
dead = int(np.count_nonzero(G[:, :, [1, 5], :]))
print(f"\nweights on always-empty channels (danger, gauge): {dead}  (inert if > 0)")

json.dump({"live_units": live_units}, open(os.path.join(os.path.dirname(__file__), "structure.json"), "w"))
