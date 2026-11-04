# Attribution

This repository analyses artifacts released with *Sparks of Weight-Space Programming* by Rikiya
Takehi, Zhi Rui Tam (Teoh), Zhang and Omar Khattab, MIT CSAIL, November 2026.

- Blog post: https://rikiyat.github.io/blog/weight-space-programming/
- Release: https://github.com/RikiyaT/lm_understand-logs

**Nothing from that release is redistributed here.** `fetch_data.py` downloads the three networks
and the evaluation file directly from the upstream repository and verifies their SHA-256 digests.

| Upstream content | Licence |
|---|---|
| Logs, data and networks | CC BY 4.0 |
| The authors' scripts | MIT |
| MinAtar source embedded in the Experiment 5 prompts | GPL-3.0 |

The prompts are not fetched or used by this repository, precisely because they embed GPL-3.0
licensed MinAtar source.

MinAtar (Kenny Young and Tian Tian) is a separate dependency installed from PyPI and is licensed
GPL-3.0; it is imported at runtime and no part of it is vendored here.

Code in this repository is © 2026 and released under the MIT Licence (see LICENSE).
