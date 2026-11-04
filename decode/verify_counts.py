"""Recount every landing-map match with zero-valued cells (target column 0) included, and show the two
bounce frames on which the direction units misreport."""
import os
import numpy as np

NET = os.path.join(os.path.dirname(__file__), "..", "data", "networks", "breakout.hand.npz")
with np.load(NET, allow_pickle=False) as z:
    W1 = z["actor/dense1/kernel"]; b1 = z["actor/dense1/bias"]
G = W1.reshape(10, 10, 6, 32)

def pre(j, ball, trail):
    x = np.zeros(600); x[(ball[0]*10+ball[1])*6+3] = 1; x[(trail[0]*10+trail[1])*6+4] = 1
    return float(x @ W1[:, j] + b1[j])

print("Bounce frame 1: second-column paddle catch, ball (8,7) trail (8,8), true post-catch dir = up-RIGHT")
p4 = pre(4, (8,7), (8,8)); print(f"   u4 pre = {p4:+.1f}  -> unit says {'right' if p4>0 else 'left'}")
print("Bounce frame 2: descending-left ball (0,3) hits brick at (1,2): stays row 0 -> ball (0,2) trail (0,3), true dir = UP")
p5 = pre(5, (0,2), (0,3)); print(f"   u5 pre = {p5:+.1f}  -> unit says {'down' if p5>0 else 'up'}")

print("\nmin |pre| over ALL legal same-row / same-col ball-trail pairs (bounce frames), not only diagonals")
m4 = m5 = 1e9
for r in range(10):
    for c in range(10):
        for r2 in range(10):
            for c2 in range(10):
                if (r, c) == (r2, c2): continue
                if abs(r-r2) <= 1 and abs(c-c2) <= 1:      # any one-step displacement incl. axis-aligned
                    m4 = min(m4, abs(pre(4,(r,c),(r2,c2)))); m5 = min(m5, abs(pre(5,(r,c),(r2,c2))))
print(f"   min |u4 pre| = {m4:.0f}   min |u5 pre| = {m5:.0f}    ")

print("\nZero-valued cells encode target column 0 and are included below.")
for j, name in ((0,"u0 desc-right"),(1,"u1 desc-left"),(2,"u2 asc-right"),(3,"u3 asc-left")):
    m = G[:, :, 3, j] / 0.02
    zero_cells = int((m == 0).sum()); zero_in_rows_0_8 = int((m[:9] == 0).sum())
    print(f"   {name}: cells with value 0 = {zero_cells} ({zero_in_rows_0_8} in rows 0-8)")

print("\nDescending units: rows 0-5 first-catch column, rows 6-8 interval midpoint")
# re-implement locally to avoid re-running its prints
def trace_to_row8(r, c, dc):
    x, y, d = c, r, dc
    while y < 8:
        nx, ny = x + d, y + 1
        if nx < 0 or nx > 9: nx = 0 if nx < 0 else 9; d = -d
        x, y = nx, ny
    return x, d
def interval(x8, d):
    nx = min(9, max(0, x8 + d)); return sorted({x8, nx})
for j, dc, name in ((0,+1,"u0"),(1,-1,"u1")):
    m = G[:, :, 3, j] / 0.02
    a = sum(1 for r in range(6) for c in range(10) if abs(m[r,c]-trace_to_row8(r,c,dc)[0])<1e-6)
    b = 0; both = 0
    for r in range(6,9):
        for c in range(10):
            x8,d8 = trace_to_row8(r,c,dc); iv = interval(x8,d8)
            if abs(m[r,c]-np.mean(iv))<1e-6:
                b += 1
                if len(iv)==1: both += 1
    print(f"   {name}: rows 0-5 first-catch {a}/60 ; rows 6-8 midpoint {b}/30 (of which {both} singleton intervals where first == midpoint)")

print("\nAscending units: rows 0-8 first-catch column under the full-band assumption")
def predict_asc(r, c, dc):
    x, y, d = c, r, dc
    while True:
        nxt = y - 1
        if nxt in (1,2,3) or nxt < 0:
            nx = x + d
            if nx < 0 or nx > 9: nx = 0 if nx < 0 else 9; d = -d
            x = nx; break
        nx = x + d
        if nx < 0 or nx > 9: nx = 0 if nx < 0 else 9; d = -d
        x, y = nx, nxt
    while y < 8:
        nx = x + d
        if nx < 0 or nx > 9: nx = 0 if nx < 0 else 9; d = -d
        x, y = nx, y + 1
    return x
for j, dc, name in ((2,+1,"u2"),(3,-1,"u3")):
    m = G[:, :, 3, j] / 0.02
    ok = sum(1 for r in range(9) for c in range(10) if abs(m[r,c]-predict_asc(r,c,dc))<1e-6)
    print(f"   {name}: rows 0-8 first-catch {ok}/90")
