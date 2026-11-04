# Generation reliability — seaquest

## Headline

- Evaluable rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Clean rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Replied: 3/3 = 100.0% (Wilson 95% 43.9–100.0); block present: 3/3 = 100.0% (Wilson 95% 43.9–100.0); not submitted 0; transport failures 0
- ECDF of 10,000-step means over 3 evaluable networks: median 22.639, IQR 13.470–38.623, min 4.301, max 54.607
- Released network: mean 51.028 (SE 1.340); percentile among evaluable networks 0.667; attempt-level percentile (non-evaluable counted as below) 0.667
- Percentile claim (≥ 29 evaluable and released above all): NOT ALLOWED

## Thresholds (never phrased as beating PPO)

- Exceeds the 100M PPO mean 5.617 (SE 0.0493): 2/3 = 66.7% (Wilson 95% 20.8–93.9); near boundary 0/3
- Exceeds the 500M PPO mean 88.884 (SE 1.5618): 0/3 = 0.0% (Wilson 95% -0.0–56.1); near boundary 0/3

## Per-network performance

| attempt | mean 10,000 | SE | mean 30,000 | mean length | censored @10k | censored @30k | >100M | >500M |
|---|---|---|---|---|---|---|---|---|
| 1 | 54.607 | 0.969 | 54.607 | 2082.0 | 0.000 | 0.000 | yes | no |
| 5 | 22.639 | 0.602 | 22.639 | 1542.5 | 0.000 | 0.000 | yes | no |
| 8 | 4.301 | 0.161 | 4.301 | 393.2 | 0.000 | 0.000 | no | no |

`*` within 2·sqrt(SE_gen² + SE_PPO²) of the threshold.

## Variance components (paired seed panel)

- Networks 3, episodes 1000
- Within-network variance 442.540; between-network variance 647.712; ratio 1.464
- Paired: mean over episode indices of the variance across networks 1084.306

## Exchangeability diagnostics (reported, not corrected)

- Spearman rho, wall seconds vs attempt index: 1.000
- Spearman rho, 10,000-step mean vs attempt index: -1.000
- Reply SHA-256 collisions: 0
- cached_input_tokens over 3 attempts: min 2432, median 2432, mean 4267, max 7936
