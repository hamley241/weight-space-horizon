"""Do the descending landing maps equal MinAtar's exact ball trajectory to the catch frame?

Uses the bounce rules from minatar/environments/breakout.py act():
  - side wall: new_x outside 0..9 is clamped to the wall column and horizontal direction flips
    (the ball sits on the wall column for that frame);
  - catch test happens when the ball would enter row 9: caught if paddle == ball_x (current)
    or paddle == new_x (next); on a catch the ball stays at row 8.
So the catch interval when the ball is at row 8 with horizontal step dc is {x8, clamp(x8+dc)}.
The predictor cannot see bricks (it never reads that channel), so trajectories assume no bricks.
"""
import os
import numpy as np

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]
G = W1.reshape(10, 10, 6, 32)

def trace_to_row8(r, c, dc):
    """Exact MinAtar movement, descending, no bricks. Returns (x at row 8, dc at row 8)."""
    x, y, d = c, r, dc
    while y < 8:
        nx, ny = x + d, y + 1
        if nx < 0 or nx > 9:
            nx = 0 if nx < 0 else 9
            d = -d
        x, y = nx, ny
    return x, d

def catch_interval(x8, d):
    nx = min(9, max(0, x8 + d))
    return sorted({x8, nx})

for j, dc, name in ((0, +1, "descending-right"), (1, -1, "descending-left")):
    m = G[:, :, 3, j] / 0.02
    print(f"\nu{j} {name}: map value / 0.02 versus exact trajectory")
    print(f"{'row':>4s} {'cells':>6s} {'= first catch col (ball_x@8)':>30s} {'= second (next col)':>20s} "
          f"{'= interval midpoint':>20s} {'other':>6s}")
    for r in range(0, 9):
        first = second = mid = other = n = 0
        others = []
        for c in range(0, 10):
            v = m[r, c]
            if v == 0:
                continue
            n += 1
            x8, d8 = trace_to_row8(r, c, dc)
            iv = catch_interval(x8, d8)
            if abs(v - x8) < 1e-6:
                first += 1
            elif len(iv) == 2 and abs(v - iv[1 if iv[0] == x8 else 0]) < 1e-6:
                second += 1
            elif abs(v - np.mean(iv)) < 1e-6:
                mid += 1
            else:
                other += 1; others.append((c, round(v, 1), iv))
        print(f"{r:4d} {n:6d} {first:30d} {second:20d} {mid:20d} {other:6d}  {others[:3] if others else ''}")

print("\nAstra's own row comments: rows 0-5 'ball row N'; row 6: 'descending targets become catch-interval midpoints'.")
