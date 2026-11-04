"""Generation driver: one attempt at a time, fully logged, never regenerates.

    python generate.py plan --attempts-per-game N --seed SEED --out plan.json
    python generate.py run --plan plan.json --index K --prompts-dir DIR --home-template TEMPLATE \
        --out-dir OUT [--provider codex|stub --stub-reply FILE --stub-mode replied|transport|not-submitted]

``plan`` fixes the attempt order up front; ``run`` performs one attempt of it and writes
``OUT/attempt-K.json`` (the record), ``attempt-K.events.jsonl`` (stdout verbatim),
``attempt-K.stderr.txt`` and, when the reply is evaluable, ``attempt-K.npz``.  An index is consumed
once; only a ``not-submitted`` attempt (nothing reached the model) leaves it free.

Exit codes: 0 replied, 2 transport-failure, 3 not-submitted, 4 index already consumed,
5 template directory refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edits_parser import parse_reply, to_npz  # noqa: E402

REPO = os.path.dirname(os.path.abspath(__file__))
GAMES = ("breakout", "seaquest", "asterix")
TEMPLATE_FILES = ("config.toml", "instructions_neutral.md", "auth.json")
CODEX_COMMAND = [
    "codex", "exec", "--json", "--skip-git-repo-check",
    "-c", "model=gpt-6-astra",
    "-c", "model_reasoning_effort=xhigh",
    "-c", "model_reasoning_summary=detailed",
    "-",
]
EXIT = {"replied": 0, "transport-failure": 2, "not-submitted": 3}
EXIT_CONSUMED = 4
EXIT_TEMPLATE = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- plan

def plan(attempts_per_game: int, seed: int, out: str) -> None:
    rs = np.random.RandomState(seed)
    attempts = []
    for _ in range(attempts_per_game):
        for g in rs.permutation(len(GAMES)):
            attempts.append({"index": len(attempts) + 1, "game": GAMES[int(g)]})
    with open(out, "w") as f:
        json.dump({"seed": seed, "attempts": attempts}, f, indent=2)
        f.write("\n")


# ----------------------------------------------------------------------------- run

def run_child(command: list[str], env: dict[str, str], prompt: bytes,
              events_path: str, stderr_path: str) -> tuple[int, str | None]:
    """Spawn ``command`` with the raw ``prompt`` bytes on stdin; stdout verbatim to ``events_path``,
    stderr to ``stderr_path``.  Returns (exit code, timestamp of the first stdout byte or None)."""
    first: str | None = None
    with open(events_path, "wb") as out, open(stderr_path, "wb") as err:
        proc = subprocess.Popen(command, env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=err)

        def feed():
            try:
                proc.stdin.write(prompt)
            except BrokenPipeError:
                pass
            finally:
                proc.stdin.close()

        threading.Thread(target=feed, daemon=True).start()
        while True:
            chunk = proc.stdout.read1(65536)
            if not chunk:
                break
            if first is None:
                first = _now()
            out.write(chunk)
            out.flush()
        code = proc.wait()
    return code, first


def run_stub(mode: str, reply_file: str, events_path: str, stderr_path: str) -> tuple[int, str | None]:
    """Synthetic event stream standing in for ``codex exec --json``; no network."""
    events: list[dict] = []
    code = 0
    if mode in ("replied", "transport"):
        events.append({"type": "thread.started", "thread_id": "stub-thread"})
    if mode == "replied":
        with open(reply_file) as f:
            text = f.read()
        events.append({"type": "turn.started"})
        events.append({"type": "item.completed",
                       "item": {"id": "item_0", "type": "agent_message", "text": text}})
        events.append({"type": "turn.completed",
                       "usage": {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}})
    if mode == "not-submitted":
        code = 1
    with open(events_path, "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    open(stderr_path, "w").close()
    return code, (_now() if events else None)


def _check_template(template: str) -> str | None:
    """Reason the template directory is refused, or None when it holds exactly the expected files."""
    if not os.path.isdir(template):
        return f"template {template!r} is not a directory"
    present = set(os.listdir(template))
    expected = set(TEMPLATE_FILES)
    if present != expected:
        return (f"template {template!r} must contain exactly {sorted(expected)}; "
                f"missing {sorted(expected - present)}, extra {sorted(present - expected)}")
    if not all(os.path.isfile(os.path.join(template, n)) for n in TEMPLATE_FILES):
        return f"template {template!r}: every entry must be a regular file"
    return None


def _residue(home: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(home):
        for name in files:
            paths.append(os.path.relpath(os.path.join(root, name), home))
    return sorted(paths)


def _read_events(events_path: str) -> list[dict]:
    events = []
    with open(events_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(e, dict):
                events.append(e)
    return events


def run(a: argparse.Namespace) -> int:
    with open(a.plan) as f:
        p = json.load(f)
    attempt = next((x for x in p["attempts"] if x["index"] == a.index), None)
    if attempt is None:
        print(f"index {a.index} is not in the plan", file=sys.stderr)
        return 1
    game = attempt["game"]
    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.join(a.out_dir, f"attempt-{a.index}")
    record_path = stem + ".json"
    if os.path.exists(record_path):
        with open(record_path) as f:
            prior = json.load(f)
        if prior.get("consumed", True):
            print(f"{record_path} exists and is consumed; refusing to regenerate", file=sys.stderr)
            return EXIT_CONSUMED
    reason = _check_template(a.home_template)
    if reason:
        print(reason, file=sys.stderr)
        return EXIT_TEMPLATE
    if a.provider == "stub":
        if not a.stub_reply:
            print("--stub-reply is required with --provider stub", file=sys.stderr)
            return 1
        if a.stub_mode == "replied" and not os.path.isfile(a.stub_reply):
            print(f"--stub-reply {a.stub_reply!r} is not a file", file=sys.stderr)
            return 1

    # Every input is validated before the credential is copied anywhere.
    prompt_path = os.path.join(a.prompts_dir, f"{game}.message.txt")
    if not os.path.isfile(prompt_path):
        print(f"prompt {prompt_path!r} is not a file", file=sys.stderr)
        return 1
    with open(prompt_path, "rb") as f:
        prompt = f.read()
    prompt_sha256 = _sha256_file(prompt_path)
    template_hashes = {n: _sha256_file(os.path.join(a.home_template, n))
                       for n in TEMPLATE_FILES if n != "auth.json"}
    events_path, stderr_path = stem + ".events.jsonl", stem + ".stderr.txt"

    # A fresh home per attempt, copied from the template.  Whatever happens after this point,
    # the credential is removed from the home before control leaves this function.
    home = stem + ".home"
    auth = os.path.join(home, "auth.json")
    if os.path.exists(home):
        shutil.rmtree(home)
    shutil.copytree(a.home_template, home)
    try:
        os.chmod(auth, 0o600)
        if a.provider == "codex":
            command = list(CODEX_COMMAND)
            env = dict(os.environ, CODEX_HOME=home)
        else:
            command = ["stub", "--stub-mode", a.stub_mode, "--stub-reply", a.stub_reply]
        before = _now()
        t0 = time.monotonic()
        if a.provider == "codex":
            code, first = run_child(command, env, prompt, events_path, stderr_path)
        else:
            code, first = run_stub(a.stub_mode, a.stub_reply, events_path, stderr_path)
        wall = time.monotonic() - t0
        exited = _now()
    finally:
        if os.path.lexists(auth):
            os.remove(auth)

    events = _read_events(events_path)
    started = next((e for e in events if e.get("type") == "thread.started"), None)
    messages = [e["item"]["text"] for e in events
                if e.get("type") == "item.completed" and isinstance(e.get("item"), dict)
                and e["item"].get("type") == "agent_message" and isinstance(e["item"].get("text"), str)]
    usage = next((e.get("usage") for e in events if e.get("type") == "turn.completed"), None)
    if started is None:
        outcome = "not-submitted"
    elif not messages:
        outcome = "transport-failure"
    else:
        outcome = "replied"

    record = {
        "index": a.index,
        "game": game,
        "plan_seed": p["seed"],
        "provider": a.provider,
        "command": command,
        "codex_home": home,
        "template_hashes": template_hashes,
        "home_residue": _residue(home),
        "prompt_sha256": prompt_sha256,
        "thread_id": started.get("thread_id") if started else None,
        "thread_started": started,
        "usage": usage,
        "timestamps": {"before_spawn": before, "first_stdout": first, "exit": exited},
        "wall_seconds": wall,
        "child_exit_code": code,
        "agent_message_count": len(messages),
        "reply_sha256": None,
        "reply_length": None,
        "outcome": outcome,
        "consumed": outcome != "not-submitted",
    }
    if outcome == "replied":
        # The reply is the exact text of the final completed agent_message item; nothing is
        # joined or inserted.  Earlier messages are counted in ``agent_message_count`` and
        # remain verbatim in the events file.
        reply = messages[-1]
        r = parse_reply(reply)
        evaluable = r.block_found and r.applied >= 1
        record.update({
            "reply_sha256": _sha256_text(reply),
            "reply_length": len(reply),
            "block_found": r.block_found,
            "cut_off": r.cut_off,
            "statements": r.statements,
            "applied": r.applied,
            "skipped": r.statements - r.applied,
            "errors": [list(e) for e in r.errors],
            "clean": r.applied == r.statements and not r.errors and r.block_found and not r.cut_off,
            "evaluable": evaluable,
        })
        if evaluable:
            npz = stem + ".npz"
            to_npz(r, npz)
            record["npz_sha256"] = _sha256_file(npz)
            record["parser_commit"] = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True,
            ).stdout.strip()
    with open(record_path, "w") as f:
        json.dump(record, f, indent=2)
        f.write("\n")
    return EXIT[outcome]


# ----------------------------------------------------------------------------- cli

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("plan", help="write the attempt order")
    sp.add_argument("--attempts-per-game", type=int, required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--out", required=True)
    sr = sub.add_parser("run", help="perform one attempt of the plan")
    sr.add_argument("--plan", required=True)
    sr.add_argument("--index", type=int, required=True)
    sr.add_argument("--prompts-dir", required=True)
    sr.add_argument("--home-template", required=True)
    sr.add_argument("--out-dir", required=True)
    sr.add_argument("--provider", choices=["codex", "stub"], default="codex")
    sr.add_argument("--stub-reply")
    sr.add_argument("--stub-mode", choices=["replied", "transport", "not-submitted"], default="replied")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        plan(a.attempts_per_game, a.seed, a.out)
        return 0
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
