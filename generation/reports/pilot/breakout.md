# Generation reliability — breakout

## Headline

- Evaluable rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Clean rate: 3/3 = 100.0% (Wilson 95% 43.9–100.0)
- Replied: 3/3 = 100.0% (Wilson 95% 43.9–100.0); block present: 3/3 = 100.0% (Wilson 95% 43.9–100.0); not submitted 0; transport failures 0
- ECDF of 10,000-step means over 3 evaluable networks: median 496.986, IQR 490.704–545.221, min 484.423, max 593.455
- Released network: mean 505.332 (SE 7.763); percentile among evaluable networks 0.667; attempt-level percentile (non-evaluable counted as below) 0.667
- Percentile claim (≥ 29 evaluable and released above all): NOT ALLOWED

## Thresholds (never phrased as beating PPO)

- Exceeds the 100M PPO mean 15.779 (SE 0.0926): 3/3 = 100.0% (Wilson 95% 43.9–100.0); near boundary 0/3
- Exceeds the 500M PPO mean 25.344 (SE 0.1211): 3/3 = 100.0% (Wilson 95% 43.9–100.0); near boundary 0/3

## Per-network performance

| attempt | mean 10,000 | SE | mean 30,000 | mean length | censored @10k | censored @30k | >100M | >500M |
|---|---|---|---|---|---|---|---|---|
| 2 | 593.455 | 6.980 | 1283.016 | 17928.7 | 0.697 | 0.329 | yes | yes |
| 6 | 484.423 | 4.948 | 1144.983 | 20695.9 | 0.753 | 0.468 | yes | yes |
| 9 | 496.986 | 5.419 | 1048.094 | 17900.8 | 0.705 | 0.295 | yes | yes |

`*` within 2·sqrt(SE_gen² + SE_PPO²) of the threshold.

## Variance components (paired seed panel)

- Networks 3, episodes 1000
- Within-network variance 34189.907; between-network variance 3524.489; ratio 0.103
- Paired: mean over episode indices of the variance across networks 35846.422

## Exchangeability diagnostics (reported, not corrected)

- Spearman rho, wall seconds vs attempt index: -0.500
- Spearman rho, 10,000-step mean vs attempt index: -0.500
- Reply SHA-256 collisions: 0
- cached_input_tokens over 3 attempts: min 2432, median 2432, mean 2432, max 2432
