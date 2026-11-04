# Preregistration — generation reliability
Frozen 2026-11-04, before any generation attempt. The protocol, categories, thresholds, sample-size
rule and decision rules below are fixed in advance; nothing is changed after the first attempt.

## The question
The published Experiment 5 rests on one generated network per game. The horizon sweep established
each released artifact's performance to tight precision; it says nothing about how often the
generation procedure produces a working network. This study estimates that.

## 1. Generation — executable, isolated per attempt
- CLI `codex-cli 0.163.0`.
- **Home template** `codex_home_template/`, exactly three files, refused otherwise by the driver:
  - `config.toml` — the authors' released `harness/codex_config.toml` with one edit (their
    redaction placeholder `<codex_home>/instructions_neutral.md` → this template's absolute
    path). SHA-256 `83e2fcf97c512b456074865075790f287309d56fa435e42edee81d2e05a3f6d0`.
  - `instructions_neutral.md` — the authors' released `harness/instructions.md`, byte-identical
    ("Answer the user's request."). SHA-256 `b33fe0624137cd6f190e42a8659046baf16261f3c29cd6455734523ba473afeb`.
  - `auth.json` — this machine's credential, mode 0600, never hashed, never logged.
- **Per attempt:** the template is copied to a fresh `attempt-K.home/`; the CLI runs with
  `CODEX_HOME` set to it; afterwards `auth.json` is deleted from that home and the list of files the
  CLI left behind is recorded (names only). Nothing is shared between attempts. (A single shared home accumulates
  sqlite state, logs and session history across runs; verified by inspection.)
- **Command:** `CODEX_HOME=<attempt home> codex exec --json --skip-git-repo-check -c model=gpt-6-astra -c model_reasoning_effort=xhigh -c model_reasoning_summary=detailed -`
  with the game's prompt on stdin. Overrides are per call, as the authors' README describes
  (their config defaults to luna/medium).
- **Prompts**, the authors' byte-exact `message.txt` per game:
  breakout `766ee251d59a7893ed86d5eaea99ddb847529ab2092afc69f90f55fb0a22e876`
  seaquest `81e51857d7cffac29b12a389dee6ffcda2256a9e03f6cf91f4fd96e1eb855bf6`
  asterix  `9e8243d5545956e42b8bd130bceba6ed90ecb807768be3ae0f498f765197f5da`
- **Order:** attempts numbered before the run; within each block of three, game order is a random
  permutation from `numpy.random.RandomState(20261104)`, written to `plan.json` before attempt 1.

## 2. Reader — frozen
`edits_parser.py` in the public repo at commit `61c42915fc4ef01a3bdb713356d95263706862ed`.
Oracle: the three released replies reproduce the three released networks exactly, with the authors'
statement counts 92/122/55 and non-zero totals 815/803/637; plus negative and boundary fixtures.
No reader change after attempt 1 is submitted; a later-found reader bug is a reported limitation.

## 3. Attempt outcomes — categories, all over the submitted-attempt denominator
*not submitted* (refused before sending; index not consumed; pause and resume at the same index,
pause logged) · *transport failure* (sent, no reply; consumed; never re-run) · *replied* → block
present / cut off / statements read, applied, skipped with reasons / **clean** (all applied, no
errors) / **evaluable** (block present and ≥ 1 applied; a network exists to score). Clean rate is
reported beside evaluable rate. No reply is ever regenerated.

## 4. Evaluation — paired, frozen seeds, cap 30,000
`score_networks.py` at the same commit: 1,000 episodes, env seed `10,000,000 + offset + ep`, action
seed `20,000,000 + offset + ep`, offsets 0 / 100,000 / 200,000; sticky 0.1; softmax sampling; nested
ladder 100…30,000; every length and censor flag retained. Oracle: it reproduces the committed sweep
records exactly. The released network is re-scored on the same seeds as the reference point.

## 5. Thresholds and proximity
Endpoints "exceeds the reported best-seed PPO mean at 100M / at 500M under the 10,000-step
evaluation", using the authors' values and **their reported SEs**:
breakout 15.779 (SE 0.093) / 25.344 (SE 0.121); seaquest 5.617 (0.049) / 88.884 (1.562);
asterix 25.820 (0.447) / 55.204 (0.636); seed counts as the authors state (ten for the two 100M
Breakout/Asterix sets, three otherwise). A generated network is flagged *near a boundary* when
|its mean − threshold| < 2 × sqrt(SE_gen² + SE_PPO²). Never phrased as "beats PPO".

## 6. Variance components
Per-network SE from its 1,000 episodes; one-way random-effects decomposition of between- vs
within-network variance on the paired seed panel; the paired analysis reported alongside.

## 7. Sample size
1. **Pilot:** 3 attempts per game (9), mechanics and cost only, excluded from all inference.
2. **Permitted main sizes: n ∈ {30, 100} per game.** Nothing else.
3. **Mapping, deterministic, from pilot quantities only:** let `t` = median wall seconds per
   submitted pilot attempt and `r` = number of pilot attempts with outcome *not submitted*.
   **n = 100 iff r = 0 and 300·t ≤ 5 days (432,000 s); otherwise n = 30.** Reply text, parse
   results, validity and any score play no part.
4. **Lock:** n is written to `FREEZE-N.md` with a timestamp **before the first main prompt is
   submitted**.
5. **Window:** main attempts run from the lock until **2026-11-18 23:59 UTC**. Pauses resume at the
   same index. If the locked n is not complete by then, an administrative stop is declared with its
   timestamp and the study is reported as incomplete at the achieved count; n is never redefined.
6. **Percentile estimand and denominator:** per game, the claim "the released network is at or
   above the 90th percentile of generated networks" is made only if ≥ 29 *evaluable* networks exist
   and the released network's score exceeds **all** of them; scores are the frozen-panel means at
   10,000; a tie counts as *not below*. Confidence is **per game** at one-sided 95%
   (0.9³⁰ = 0.042); no simultaneous three-game claim is made (that would need ≥ 39 per game).
   An attempt-level estimand (non-evaluable counted as below) is reported separately, labelled.
7. **Decision rules by achieved n:** at 30 — the percentile claim (if its conditions hold), the
   evaluable and clean rates with Wilson 95% intervals (half-width ≈ 17 points at 50%), the ECDF,
   median, IQR, and the released network's percentile; at 100 — the same with half-width ≈ 10 points.
   Never the mean across networks as the headline.

## 8. Exchangeability detectors (reported, not corrected)
Model/build/tier fields from every event stream; `cached_input_tokens` per attempt; reply SHA-256
collisions; longest common substring > 200 chars between replies to the same prompt; validity,
length, score and latency by attempt index, by calendar block, and pre/post any pause.

## 9. Artifacts per evaluable attempt
Reply text; full event stream; attempt record (all fields in the driver spec); parsed tensors as
`.npz`; parser commit; application log; per-episode evaluation records. Falsifier: if a later
decode cannot run from these alone, the retention list was insufficient.

## 10. Costs
Editor turns on the subscription (pilot 9; main 90 or 300; the authors' Exp 5 replies ran ~500–580 s
each). Scoring on CPU: 30 Breakout networks ≈ 3 h, 100 ≈ 10 h. No OpenRouter spend under this
protocol.
