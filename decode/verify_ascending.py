"""What do the ascending landing units (u2 up-right, u3 up-left) assume, given they cannot see bricks?

Hypothesis (from Astra's comment "anticipate the earliest possible brick bounce"): assume a brick
sits in the lowest brick row (row 3) directly on the path. MinAtar's brick-hit rule (breakout.py
act()): when the ball would enter a brick cell, it keeps its row (new_y = last_y), its x still
advances (with wall clamp), and vertical direction flips. So from row 4 going up, the next frame is
(4, x+dc) going down. Then descend under the exact wall rules to row 8; the target is the first
catch column (ball_x at row 8), or the catch-interval midpoint near the bottom, as in the
descending units.
Inside the brick zone (rows <= 3) the ball is already among bricks; test the same rule: bounce on
the very next upward step.
"""
import os
import numpy as np

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]
G = W1.reshape(10, 10, 6, 32)

def step_x(x, d):
    nx = x + d
    if nx < 0 or nx > 9:
        nx = 0 if nx < 0 else 9
        d = -d
    return nx, d

def predict(r, c, dc):
    """Ascending from (r,c) with horizontal step dc. Returns (x at row 8, dc at row 8)."""
    x, y, d = c, r, dc
    # ascend until the next row is a brick row (1..3) or above the top (-1). Row 0 holds no bricks,
    # so a ball at row 1 continues to row 0 and bounces off the top wall there.
    while True:
        nxt = y - 1
        if nxt in (1, 2, 3):      # brick hit: stay at row y, x advances, vertical flips
            x, d = step_x(x, d); break
        if nxt < 0:               # top wall: same effect (new_y clamped to 0, vertical flips)
            x, d = step_x(x, d); break
        x, d = step_x(x, d); y = nxt
    if r == 9:                    # unreachable in play; Astra labels it "terminal-row fallback"
        return c, dc
    # descend to row 8
    while y < 8:
        x, d = step_x(x, d); y += 1
    return x, d

def interval(x8, d):
    nx = min(9, max(0, x8 + d))
    return sorted({x8, nx})

for j, dc, name in ((2, +1, "ascending-right"), (3, -1, "ascending-left")):
    m = G[:, :, 3, j] / 0.02
    print(f"\nu{j} {name}: map/0.02 vs 'bounce at the lowest brick row, then exact descent'")
    print(f"{'row':>4s} {'cells':>6s} {'first catch col':>16s} {'midpoint':>9s} {'other (c, map, predicted interval)':>40s}")
    for r in range(0, 10):
        n = first = mid = 0; others = []
        for c in range(10):
            v = m[r, c]
            if v == 0:
                continue
            n += 1
            x8, d8 = predict(r, c, dc)
            iv = interval(x8, d8)
            if abs(v - x8) < 1e-6:
                first += 1
            elif abs(v - np.mean(iv)) < 1e-6:
                mid += 1
            else:
                others.append((c, round(float(v), 1), iv))
        print(f"{r:4d} {n:6d} {first:16d} {mid:9d}   {others[:4] if others else ''}")
