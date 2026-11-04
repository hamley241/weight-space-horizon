"""Gating with realistic error magnitudes, and the shape of the ascending landing maps."""
import os
import numpy as np

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]; b1 = z["actor/dense1/bias"]
    W2 = z["actor/dense2/kernel"]; b2 = z["actor/dense2/bias"]
G = W1.reshape(10, 10, 6, 32)

print("5'. GATING, with the error units in their real range")
# an error unit's pre-activation is 0.02*(landing_col - paddle_col); columns run 1..9 so |delta| <= 8
for delta in (0, 1, 2, 8, -1, -8):
    u = np.tanh(0.02 * delta)
    right_quadrant = np.tanh(1000 * u + 200 + 200 + b2[1])     # h2[1]: +1000*u0, +200 u4, +200 u5
    one_gate_wrong = np.tanh(1000 * u + 200 - 200 + b2[1])
    both_wrong = np.tanh(1000 * u - 200 - 200 + b2[1])
    print(f"   paddle off by {delta:+d} col: h2 pre = {1000*u + 400 + b2[1]:+7.1f} -> matching quadrant {right_quadrant:+.3f} | "
          f"one gate wrong {one_gate_wrong:+.3f} | both wrong {both_wrong:+.3f}")
print("   => fires only in its own quadrant AND only when the paddle is >= 1 column off the predicted landing. "
      "Delta = 0 gives no-op: that is the dead-band.")

print("\n7. DO THE DESCENDING LANDING MAPS ENCODE WALL REFLECTIONS LITERALLY?")
print("   Convention from check 4: u4 sign = dc (+ = moving right), u5 sign = dr (+ = moving down, toward the paddle).")
print("   u0 is gated on (u4+, u5+) = down-right; u1 on (u4-, u5+) = down-left.")

def land(r, c, dc):
    """Where a ball at (r,c) moving diagonally with horizontal step dc reaches row 8, reflecting off cols 0 and 9."""
    col, d = c, dc
    for _ in range(8 - r):
        nxt = col + d
        if nxt <= 0 or nxt >= 9:      # wall at col 0 / col 9: reflect
            d = -d
            nxt = col + d
        col = nxt
    return col

for j, dc, name in ((0, +1, "down-right"), (1, -1, "down-left")):
    m = G[:, :, 3, j] / 0.02          # weight = 0.02 * predicted column?
    ok = tot = 0; mism = []
    for r in range(0, 9):
        for c in range(1, 9):
            if m[r, c] == 0:
                continue
            tot += 1
            pred = int(round(m[r, c]))
            if pred == land(r, c, dc):
                ok += 1
            else:
                mism.append((r, c, pred, land(r, c, dc)))
    print(f"   u{j} ({name}): weight/0.02 is an integer landing column matching a reflecting bounce "
          f"for {ok}/{tot} ball cells; first mismatches: {mism[:6]}")

print("\n   The two ascending maps (u2 up-right, u3 up-left) — what do they encode?")
for j, name in ((2, "up-right"), (3, "up-left")):
    m = G[:, :, 3, j] / 0.02
    vals = m[m != 0]
    frac_int = np.mean(np.abs(vals - np.round(vals)) < 1e-6)
    print(f"   u{j} ({name}): {int((m!=0).sum())} cells, values/0.02 in [{vals.min():.1f},{vals.max():.1f}], "
          f"integer-valued: {frac_int:.0%}; row-9 (paddle row) cells used: {int((m[9]!=0).sum())}")
    # does it depend on row at all, or is it 'hold current column' heuristics?
    for r in (1, 4, 7):
        print(f"      row {r}: predicted cols {np.round(m[r, 1:9]).astype(int).tolist()}")
