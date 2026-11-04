# Generation reliability

`PREREGISTRATION.md` is the frozen protocol for re-running Experiment 5's one-shot network
generation many times and scoring every result under the sweep's seeds. It was committed before
the first attempt.

`codex_home_template.*` are the two configuration files of the per-attempt CLI home: the authors'
released `harness/codex_config.toml` with their redaction placeholder for the instructions path
replaced (shown here as `<template dir>`; the hash in the preregistration is of the file with the
real absolute path substituted), and their one-line `harness/instructions.md`, byte-identical.
The third file in the template, the credential, is never committed.

The prompts are the authors' byte-exact `message.txt` per game; they embed MinAtar source under
GPL-3.0 and so are not redistributed here. Their SHA-256s are in the preregistration.

Attempt records, parsed networks and per-episode scores land under `generation/attempts/` as the
study runs. The driver is `generate.py`; the scorer is `score_networks.py`.
