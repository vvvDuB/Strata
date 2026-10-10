"""Pi checkpoints use their own thinking policy; ordinary agent turns keep theirs."""
import copy
import contextlib
import io
import json
import unittest
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate, openai_to_messages
from serve.pi_compaction import CHECKPOINT_GUIDANCE, CHECKPOINT_REQUEST_GUIDANCE, PI_SUMMARIZATION_SYSTEM_PROMPT, prepare_pi_compaction
from serve.server import ByteTokenizer, EngineDied, MockEngine, Service, engine_unavailable_message, serve
from serve.test_server import ThinkingEngine

ROOT = Path(__file__).resolve().parents[1]


def checkpoint():
    return [{"role": "system", "content": PI_SUMMARIZATION_SYSTEM_PROMPT},
            {"role": "user", "content": "<conversation>\n[User]: keep the current goal\n</conversation>\n\nSummarize."}]


class PiCompaction(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "short checkpoint", max_context=32768), self.tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_automatic_detection_before_tokenization_is_silent_and_source_is_unchanged(self):
        original = checkpoint()
        saved = copy.deepcopy(original)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ids, thinking, cap = self.svc.prepare(original, None, {"reasoning_effort": "xhigh"}, 8192)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        prompt = self.tok.decode(ids)
        self.assertFalse(thinking)
        self.assertEqual(cap, 8192)
        self.assertIn(CHECKPOINT_GUIDANCE, prompt)
        self.assertIn(original[1]["content"], prompt)
        self.assertIn(CHECKPOINT_REQUEST_GUIDANCE, prompt)
        self.assertRegex(prompt, r"<think>\s*</think>\s*$")
        self.assertEqual(original, saved)

    def test_guidance_follows_the_full_original_request_without_mutation(self):
        original = checkpoint()
        saved = copy.deepcopy(original)
        messages, _, recognized = prepare_pi_compaction(original, None, {})
        self.assertTrue(recognized)
        self.assertEqual(messages[1]["content"], original[1]["content"] + CHECKPOINT_REQUEST_GUIDANCE)
        self.assertEqual(original, saved)

    def test_split_turn_prefix_is_also_summarized_without_thinking(self):
        messages = checkpoint()
        messages[1]["content"] = "# Conversation\n[User]: pending work\n\n# Instructions\nThe messages above are earlier context from an ongoing conversation. Later messages are stored separately."
        ids, thinking, cap = self.svc.prepare(messages, None, {"reasoning_effort": "xhigh"}, 16384)
        self.assertFalse(thinking)
        self.assertEqual(cap, 16384)
        self.assertIn(CHECKPOINT_GUIDANCE, self.tok.decode(ids))

    def test_ordinary_summarization_and_quoted_signature_are_unchanged(self):
        for messages in [[{"role": "user", "content": "Summarize this file"}],
                         [{"role": "user", "content": PI_SUMMARIZATION_SYSTEM_PROMPT}],
                         [*checkpoint(), {"role": "assistant", "content": "history"}],
                         [{"role": "system", "content": "A different summarizer"}, checkpoint()[1]]]:
            with self.subTest(messages=messages):
                ids, thinking, _ = self.svc.prepare(messages, None, {"reasoning_effort": "xhigh"}, 20)
                self.assertTrue(thinking)
                self.assertNotIn(CHECKPOINT_GUIDANCE, self.tok.decode(ids))

    def test_tools_and_forced_calls_are_not_checkpoints(self):
        for tools, force in [([{"name": "read"}], None), (None, "forced")]:
            _, _, recognized = prepare_pi_compaction(checkpoint(), tools, {}, force)
            self.assertFalse(recognized)

    def test_no_shrinking_of_client_output_limit(self):
        for limit in [1, 100, 8192, 26214]:
            self.assertEqual(self.svc.prepare(checkpoint(), None, {}, limit)[2], limit)

    def test_actual_openai_shape_and_length_reporting(self):
        server = serve(self.svc, port=0)
        try:
            body = {"messages": checkpoint(), "reasoning_effort": "high", "max_tokens": 5}
            req = urllib.request.Request(f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions",
                                         data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.load(response)
            self.assertEqual(result["choices"][0]["finish_reason"], "length")
            self.assertEqual(result["choices"][0]["message"]["content"], "short")
            self.assertFalse(result["choices"][0]["message"].get("reasoning_content"))
        finally:
            server.shutdown()
            server.server_close()


class ThinkingBudgetAlias(unittest.TestCase):
    def setUp(self):
        tok = ByteTokenizer()
        self.engine = ThinkingEngine(tok)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_alias_and_native_precedence(self):
        self.assertEqual(self.svc.reasoning_budget({"thinking_budget_tokens": 20}), 20)
        self.assertEqual(self.svc.reasoning_budget({"thinking_budget_tokens": 40, "reasoning_budget_tokens": 20}), 20)
        self.svc.reasoning_budget_tokens = 30
        self.assertEqual(self.svc.reasoning_budget({}), 30)
        self.assertIsNone(self.svc.reasoning_budget({"thinking_budget_tokens": 0}))

    def test_alias_is_validated(self):
        for bad in [True, "20", 2.5, []]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                self.svc.reasoning_budget({"thinking_budget_tokens": bad})

    def test_alias_reaches_the_real_generation_path(self):
        import threading
        ids, thinking, cap = self.svc.prepare([{"role": "user", "content": "2+2?"}], None, {}, 400)
        events = list(self.svc.run(ids, thinking, None, cap, {"thinking_budget_tokens": 20}, threading.Event()))
        self.assertEqual(len(self.engine.prompts), 2)
        self.assertEqual(events[-1][1]["finish"], "stop")


class TransientEngineError(unittest.TestCase):
    def test_stream_message_keeps_cause_and_recovery_instruction(self):
        message = engine_unavailable_message(EngineDied("the engine stopped unexpectedly (exit code -6)"))
        self.assertTrue(message.startswith("Service unavailable:"))
        self.assertIn("exit code -6", message)
        self.assertIn("next request restarts it", message)


if __name__ == "__main__":
    unittest.main()
