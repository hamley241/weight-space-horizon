# Generation reliability — asterix

## Headline

- Evaluable rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Clean rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Replied: 3/3 = 100.0% (Wilson 95% 43.9–100.0); block present: 3/3 = 100.0% (Wilson 95% 43.9–100.0); not submitted 0; transport failures 0
- ECDF of 10,000-step means over 3 evaluable networks: median 73.539, IQR 52.366–86.788, min 31.193, max 100.037
- Released network: mean 53.262 (SE 1.081); percentile among evaluable networks 0.333; attempt-level percentile (non-evaluable counted as below) 0.333
- Percentile claim (≥ 29 evaluable and released above all): NOT ALLOWED

## Thresholds (never phrased as beating PPO)

- Exceeds the 100M PPO mean 25.82 (SE 0.4472): 3/3 = 100.0% (Wilson 95% 43.9–100.0); near boundary 0/3
- Exceeds the 500M PPO mean 55.204 (SE 0.6359): 2/3 = 66.7% (Wilson 95% 20.8–93.9); near boundary 0/3

## Per-network performance

| attempt | mean 10,000 | SE | mean 30,000 | mean length | censored @10k | censored @30k | >100M | >500M |
|---|---|---|---|---|---|---|---|---|
| 3 | 100.037 | 2.297 | 100.037 | 1290.2 | 0.000 | 0.000 | yes | yes |
| 4 | 73.539 | 1.176 | 73.539 | 1031.6 | 0.000 | 0.000 | yes | yes |
| 7 | 31.193 | 0.710 | 31.193 | 664.1 | 0.000 | 0.000 | yes | no |

`*` within 2·sqrt(SE_gen² + SE_PPO²) of the threshold.

## Variance components (paired seed panel)

- Networks 3, episodes 1000
- Within-network variance 2386.759; between-network variance 1203.417; ratio 0.504
- Paired: mean over episode indices of the variance across networks 3552.862

## Exchangeability diagnostics (reported, not corrected)

- Spearman rho, wall seconds vs attempt index: -0.500
- Spearman rho, 10,000-step mean vs attempt index: -1.000
- Reply SHA-256 collisions: 0
- cached_input_tokens over 3 attempts: min 0, median 0, mean 2005, max 6016
