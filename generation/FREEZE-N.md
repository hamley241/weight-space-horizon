# FREEZE-N — sample size locked
Locked at: **2026-11-04T16:20:17Z**, before any main-study prompt was submitted and before any pilot network was scored.
Inputs (attempt records only; no reply text, parse detail or score consulted):
- pilot attempts: 9; outcomes: {'replied': 9}
- r (not-submitted) = **0**
- t (median wall seconds over submitted attempts) = **1048.0 s**  ->  300 * t = 314,389 s (limit 432,000 s)
Rule: n = 100 iff r == 0 and 300*t <= 432,000; else 30.
## n = **100 per game** (300 main attempts)
Window: main attempts run from this lock until 2026-11-18 23:59 UTC; an incomplete run at that time is
reported as incomplete at the achieved count. n is never redefined.
Per-attempt wall seconds (submitted): [1132, 994, 1048, 1288, 1307, 749, 905, 1612, 848]
