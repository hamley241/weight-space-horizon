import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import generate  # noqa: E402
from edits_parser import NPZ_KEYS  # noqa: E402

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
AUTH = '{"OPENAI_API_KEY": "sk-template-secret-0123456789"}'
ENV_SECRET = "sk-env-secret-9876543210"


class PlanTests(unittest.TestCase):
    def test_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
            self.assertEqual(generate.main(["plan", "--attempts-per-game", "2", "--seed", "20261104", "--out", a]), 0)
            self.assertEqual(generate.main(["plan", "--attempts-per-game", "2", "--seed", "20261104", "--out", b]), 0)
            with open(a, "rb") as fa, open(b, "rb") as fb:
                self.assertEqual(fa.read(), fb.read())
            with open(a) as f:
                p = json.load(f)
        self.assertEqual(p["seed"], 20261104)
        self.assertEqual([x["index"] for x in p["attempts"]], list(range(1, 7)))
        games = [x["game"] for x in p["attempts"]]
        self.assertEqual(Counter(games), {g: 2 for g in generate.GAMES})
        for i in range(0, 6, 3):
            self.assertEqual(set(games[i:i + 3]), set(generate.GAMES))


class RunBase(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d)
        self.plan = os.path.join(self.d, "plan.json")
        generate.main(["plan", "--attempts-per-game", "1", "--seed", "7", "--out", self.plan])
        with open(self.plan) as f:
            self.attempts = json.load(f)["attempts"]
        self.prompts = os.path.join(self.d, "prompts")
        os.mkdir(self.prompts)
        for g in generate.GAMES:
            with open(os.path.join(self.prompts, f"{g}.message.txt"), "w") as f:
                f.write(f"prompt for {g}\n")
        self.template = os.path.join(self.d, "template")
        os.mkdir(self.template)
        for name, body in (("config.toml", "model = 'x'\n"), ("instructions_neutral.md", "be neutral\n"),
                           ("auth.json", AUTH)):
            with open(os.path.join(self.template, name), "w") as f:
                f.write(body)
        self.out = os.path.join(self.d, "out")
        os.environ["OPENAI_API_KEY"] = ENV_SECRET
        self.addCleanup(os.environ.pop, "OPENAI_API_KEY", None)

    def index_of(self, game):
        return next(x["index"] for x in self.attempts if x["game"] == game)

    def run_stub(self, index, mode, reply):
        return generate.main(["run", "--plan", self.plan, "--index", str(index), "--prompts-dir", self.prompts,
                              "--home-template", self.template, "--out-dir", self.out,
                              "--provider", "stub", "--stub-reply", reply, "--stub-mode", mode])

    def record(self, index):
        with open(os.path.join(self.out, f"attempt-{index}.json")) as f:
            return json.load(f)

    def record_text(self, index):
        with open(os.path.join(self.out, f"attempt-{index}.json")) as f:
            return f.read()

    def patch_child(self, fake):
        original = generate.run_child
        generate.run_child = fake
        self.addCleanup(setattr, generate, "run_child", original)


class StubTests(RunBase):
    def test_replied_breakout(self):
        k = self.index_of("breakout")
        self.assertEqual(self.run_stub(k, "replied", os.path.join(FIXTURES, "breakout_reply.md")), 0)
        r = self.record(k)
        self.assertEqual(r["outcome"], "replied")
        self.assertTrue(r["consumed"])
        self.assertTrue(r["clean"])
        self.assertTrue(r["evaluable"])
        self.assertEqual((r["statements"], r["applied"], r["skipped"]), (92, 92, 0))
        self.assertEqual(r["errors"], [])
        self.assertRegex(r["parser_commit"], r"^[0-9a-f]{40}$")
        self.assertEqual(r["game"], "breakout")
        self.assertEqual(r["plan_seed"], 7)
        self.assertEqual(r["thread_id"], "stub-thread")
        self.assertEqual(r["thread_started"]["type"], "thread.started")
        self.assertEqual(r["usage"]["cached_input_tokens"], 0)
        self.assertEqual(r["child_exit_code"], 0)
        self.assertIsInstance(r["wall_seconds"], float)
        for key in ("before_spawn", "first_stdout", "exit"):
            self.assertTrue(r["timestamps"][key])
        self.assertRegex(r["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(r["reply_sha256"], r"^[0-9a-f]{64}$")
        with open(os.path.join(FIXTURES, "breakout_reply.md")) as f:
            self.assertEqual(r["reply_length"], len(f.read()))
        self.assertEqual(set(r["template_hashes"]), {"config.toml", "instructions_neutral.md"})
        self.assertIsInstance(r["home_residue"], list)
        self.assertNotIn("auth.json", r["home_residue"])
        self.assertFalse(os.path.exists(os.path.join(self.out, f"attempt-{k}.home", "auth.json")))
        self.assertRegex(r["npz_sha256"], r"^[0-9a-f]{64}$")
        saved = np.load(os.path.join(self.out, f"attempt-{k}.npz"))
        hand = np.load(os.path.join(FIXTURES, "breakout_hand.npz"))
        self.assertEqual(set(saved.files), set(NPZ_KEYS.values()))
        for key in NPZ_KEYS.values():
            self.assertEqual(saved[key].dtype, np.float32)
            self.assertTrue(np.array_equal(saved[key], hand[key]), key)
        self.assertTrue(os.path.exists(os.path.join(self.out, f"attempt-{k}.events.jsonl")))
        self.assertTrue(os.path.exists(os.path.join(self.out, f"attempt-{k}.stderr.txt")))

    def test_no_edits_block_not_evaluable(self):
        reply = os.path.join(self.d, "prose.md")
        with open(reply, "w") as f:
            f.write("Just prose, no block.\n")
        self.assertEqual(self.run_stub(1, "replied", reply), 0)
        r = self.record(1)
        self.assertEqual(r["outcome"], "replied")
        self.assertFalse(r["block_found"])
        self.assertFalse(r["evaluable"])
        self.assertFalse(r["clean"])
        self.assertNotIn("parser_commit", r)
        self.assertFalse(os.path.exists(os.path.join(self.out, "attempt-1.npz")))

    def test_transport_failure(self):
        self.assertEqual(self.run_stub(2, "transport", os.path.join(FIXTURES, "breakout_reply.md")), 2)
        r = self.record(2)
        self.assertEqual(r["outcome"], "transport-failure")
        self.assertTrue(r["consumed"])
        self.assertIsNone(r["reply_sha256"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "attempt-2.npz")))

    def test_not_submitted_then_rerun_then_consumed_refused(self):
        reply = os.path.join(FIXTURES, "breakout_reply.md")
        self.assertEqual(self.run_stub(3, "not-submitted", reply), 3)
        r = self.record(3)
        self.assertEqual(r["outcome"], "not-submitted")
        self.assertFalse(r["consumed"])
        self.assertEqual(r["child_exit_code"], 1)
        self.assertIsNone(r["thread_id"])
        # The index is still free.
        self.assertEqual(self.run_stub(3, "replied", reply), 0)
        self.assertTrue(self.record(3)["consumed"])
        # Now it is consumed: a third run refuses and changes nothing.
        before = {n: os.stat(os.path.join(self.out, n)).st_mtime_ns for n in os.listdir(self.out)}
        text = self.record_text(3)
        self.assertEqual(self.run_stub(3, "transport", reply), 4)
        after = {n: os.stat(os.path.join(self.out, n)).st_mtime_ns for n in os.listdir(self.out)}
        self.assertEqual(before, after)
        self.assertEqual(self.record_text(3), text)

    def test_record_never_leaks_secrets(self):
        self.assertEqual(self.run_stub(1, "replied", os.path.join(FIXTURES, "breakout_reply.md")), 0)
        text = self.record_text(1)
        self.assertNotIn(AUTH, text)
        self.assertNotIn("sk-template-secret", text)
        self.assertNotIn(ENV_SECRET, text)
        self.assertNotIn("OPENAI_API_KEY", text)

    def test_template_refused_before_spawn(self):
        reply = os.path.join(FIXTURES, "breakout_reply.md")
        calls = []
        original_child, original_stub = generate.run_child, generate.run_stub
        generate.run_child = generate.run_stub = lambda *a, **k: calls.append(a) or (0, None)
        self.addCleanup(setattr, generate, "run_child", original_child)
        self.addCleanup(setattr, generate, "run_stub", original_stub)
        with open(os.path.join(self.template, "extra.txt"), "w") as f:
            f.write("x")
        self.assertEqual(self.run_stub(1, "replied", reply), 5)
        os.remove(os.path.join(self.template, "extra.txt"))
        os.remove(os.path.join(self.template, "config.toml"))
        self.assertEqual(self.run_stub(1, "replied", reply), 5)
        self.assertEqual(calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.out, "attempt-1.json")))
        self.assertFalse(os.path.exists(os.path.join(self.out, "attempt-1.home")))


class CodexCommandTests(RunBase):
    def test_codex_command_env_and_home(self):
        k = self.index_of("seaquest")
        home = os.path.join(self.out, f"attempt-{k}.home")
        seen = {}

        def fake_child(command, env, prompt, events_path, stderr_path):
            seen["command"], seen["env"], seen["prompt"] = command, env, prompt
            seen["home_files"] = sorted(os.listdir(env["CODEX_HOME"]))
            seen["auth_mode"] = os.stat(os.path.join(env["CODEX_HOME"], "auth.json")).st_mode & 0o777
            with open(os.path.join(env["CODEX_HOME"], "auth.json")) as f:
                seen["auth"] = f.read()
            with open(os.path.join(env["CODEX_HOME"], "sessions.log"), "w") as f:
                f.write("left behind\n")
            with open(events_path, "w") as f:
                f.write(json.dumps({"type": "thread.started", "thread_id": "t-1", "extra": 5}) + "\n")
                f.write("not json\n")
                f.write(json.dumps({"type": "item.completed", "item": {
                    "id": "item_0", "type": "agent_message",
                    "text": "```edits\nb1[0] = 1\n```\n"}}) + "\n")
                f.write(json.dumps({"type": "turn.completed", "usage": {
                    "input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 2}}) + "\n")
            open(stderr_path, "w").close()
            return 0, "2026-11-04T00:00:00+00:00"

        original = generate.run_child
        generate.run_child = fake_child
        self.addCleanup(setattr, generate, "run_child", original)
        code = generate.main(["run", "--plan", self.plan, "--index", str(k), "--prompts-dir", self.prompts,
                              "--home-template", self.template, "--out-dir", self.out, "--provider", "codex"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["command"], [
            "codex", "exec", "--json", "--skip-git-repo-check",
            "-c", "model=gpt-6-astra", "-c", "model_reasoning_effort=xhigh",
            "-c", "model_reasoning_summary=detailed", "-"])
        self.assertEqual(seen["env"]["CODEX_HOME"], home)
        self.assertEqual(seen["prompt"], b"prompt for seaquest\n")
        self.assertEqual(seen["home_files"], ["auth.json", "config.toml", "instructions_neutral.md"])
        self.assertEqual(seen["auth"], AUTH)
        self.assertEqual(seen["auth_mode"], 0o600)
        r = self.record(k)
        self.assertEqual(r["command"], seen["command"])
        self.assertEqual(r["codex_home"], home)
        self.assertEqual(r["thread_id"], "t-1")
        self.assertEqual(r["thread_started"], {"type": "thread.started", "thread_id": "t-1", "extra": 5})
        self.assertEqual(r["usage"]["cached_input_tokens"], 4)
        self.assertEqual(r["timestamps"]["first_stdout"], "2026-11-04T00:00:00+00:00")
        self.assertEqual(r["outcome"], "replied")
        self.assertTrue(r["evaluable"])
        self.assertEqual(r["home_residue"], ["config.toml", "instructions_neutral.md", "sessions.log"])
        self.assertFalse(os.path.exists(os.path.join(home, "auth.json")))
        text = self.record_text(k)
        self.assertNotIn("sk-template-secret", text)
        self.assertNotIn(ENV_SECRET, text)

    def test_prompt_hash_is_raw_file_bytes_and_stdin_is_verbatim(self):
        k = self.index_of("asterix")
        raw = b"line one\r\nline two\r\n\xef\xbb\xbfno newline at end"
        with open(os.path.join(self.prompts, "asterix.message.txt"), "wb") as f:
            f.write(raw)
        seen = {}

        def fake_child(command, env, prompt, events_path, stderr_path):
            seen["prompt"] = prompt
            open(events_path, "w").close()
            open(stderr_path, "w").close()
            return 1, None

        self.patch_child(fake_child)
        code = generate.main(["run", "--plan", self.plan, "--index", str(k), "--prompts-dir", self.prompts,
                              "--home-template", self.template, "--out-dir", self.out, "--provider", "codex"])
        self.assertEqual(code, 3)
        self.assertEqual(seen["prompt"], raw)
        self.assertEqual(self.record(k)["prompt_sha256"], hashlib.sha256(raw).hexdigest())

    def test_reply_is_last_agent_message_verbatim(self):
        k = self.index_of("seaquest")
        first_text = "Thinking out loud...\n"
        last_text = "```edits\nb1[0] = 1\n```\n"

        def fake_child(command, env, prompt, events_path, stderr_path):
            with open(events_path, "w") as f:
                f.write(json.dumps({"type": "thread.started", "thread_id": "t-2"}) + "\n")
                for i, text in enumerate((first_text, last_text)):
                    f.write(json.dumps({"type": "item.completed", "item": {
                        "id": f"item_{i}", "type": "agent_message", "text": text}}) + "\n")
            open(stderr_path, "w").close()
            return 0, None

        self.patch_child(fake_child)
        code = generate.main(["run", "--plan", self.plan, "--index", str(k), "--prompts-dir", self.prompts,
                              "--home-template", self.template, "--out-dir", self.out, "--provider", "codex"])
        self.assertEqual(code, 0)
        r = self.record(k)
        self.assertEqual(r["agent_message_count"], 2)
        self.assertEqual(r["reply_length"], len(last_text))
        self.assertEqual(r["reply_sha256"], hashlib.sha256(last_text.encode("utf-8")).hexdigest())
        self.assertTrue(r["evaluable"])

    def test_auth_removed_when_child_raises(self):
        k = self.index_of("breakout")
        home = os.path.join(self.out, f"attempt-{k}.home")

        def exploding_child(command, env, prompt, events_path, stderr_path):
            self.assertTrue(os.path.exists(os.path.join(env["CODEX_HOME"], "auth.json")))
            raise OSError("spawn failed")

        self.patch_child(exploding_child)
        with self.assertRaises(OSError):
            generate.main(["run", "--plan", self.plan, "--index", str(k), "--prompts-dir", self.prompts,
                           "--home-template", self.template, "--out-dir", self.out, "--provider", "codex"])
        self.assertTrue(os.path.isdir(home))
        self.assertFalse(os.path.exists(os.path.join(home, "auth.json")))

    def test_missing_prompt_refused_before_home_is_created(self):
        k = self.index_of("breakout")
        os.remove(os.path.join(self.prompts, "breakout.message.txt"))
        calls = []
        self.patch_child(lambda *a, **kw: calls.append(a) or (0, None))
        code = generate.main(["run", "--plan", self.plan, "--index", str(k), "--prompts-dir", self.prompts,
                              "--home-template", self.template, "--out-dir", self.out, "--provider", "codex"])
        self.assertEqual(code, 1)
        self.assertEqual(calls, [])
        self.assertFalse(os.path.exists(os.path.join(self.out, f"attempt-{k}.home")))
        self.assertFalse(os.path.exists(os.path.join(self.out, f"attempt-{k}.json")))

    def test_help(self):
        for args in (["--help"], ["plan", "--help"], ["run", "--help"]):
            with contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit) as cm:
                generate.main(args)
            self.assertEqual(cm.exception.code, 0)
            self.assertIn("usage:", out.getvalue())


if __name__ == "__main__":
    unittest.main()
