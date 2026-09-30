"""Cache counters must report actual engine reuse, for streaming and collected APIs."""
import threading
import unittest
import io
import queue
from types import SimpleNamespace
from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import (EngineDied, StrataEngine, ByteTokenizer, MockEngine, Service, openai_chunks, openai_collect,
                          anthropic_events, anthropic_collect)

class CacheReportingEngine(MockEngine):
    def generate(self, *args, **kwargs):
        try:
            yield from super().generate(*args, **kwargs)
        finally:
            # Deliberately only available when Service drains/closes the generator.
            self.last = {"reused": 7}

class CacheUsage(unittest.TestCase):
    def service(self, cached=True):
        tok = ByteTokenizer()
        engine = (CacheReportingEngine if cached else MockEngine)(tok, "4")
        svc = Service(engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        return svc

    def test_service_captures_reuse_before_done(self):
        svc = self.service()
        done = list(svc.run([1]*30, False, None, 2, {}, threading.Event()))[-1][1]
        self.assertEqual(done.get("cached_tokens"), 7)
        svc.engine.last["reused"] = 99
        self.assertEqual(done["cached_tokens"], 7)

    def test_openai_stream_and_collect(self):
        for cached in (False, True):
            svc = self.service(cached)
            chunks = list(openai_chunks(svc, {}, [1]*30, False, None, 2, threading.Event()))
            for usage in (chunks[-1]["usage"], openai_collect(iter(chunks))["usage"]):
                self.assertEqual(usage.get("prompt_tokens_details", {}).get("cached_tokens"), 7 if cached else 0)
                self.assertEqual(usage["prompt_tokens"], 30)
                self.assertEqual(usage["total_tokens"], 30 + usage["completion_tokens"])

    def test_anthropic_stream_and_collect(self):
        for cached in (False, True):
            svc = self.service(cached)
            events = list(anthropic_events(svc, {}, [1]*30, False, None, 2, threading.Event()))
            delta = next(e for name,e in events if name == "message_delta")
            for usage in (delta["usage"], anthropic_collect(iter(events))["usage"]):
                self.assertEqual(usage.get("cache_read_input_tokens"), 7 if cached else 0)
                self.assertEqual(usage["input_tokens"], 23 if cached else 30)

class CacheProtocol(unittest.TestCase):
    def test_new_done_preserves_coverage_diagnostics(self):
        e = StrataEngine.__new__(StrataEngine)
        e._parse_done("DONE 8 42617 12.5 33.0 stop 4 7 14336 12 20 16046 1710 checkpoint")
        self.assertEqual(e.last["reused"], 14336)
        self.assertEqual(e.last["common_prefix_tokens"], 16046)
        self.assertEqual(e.last["replay_gap_tokens"], 1710)
        self.assertEqual(e.last["cache_source"], "checkpoint")

    def test_old_done_does_not_leak_previous_diagnostics(self):
        e = StrataEngine.__new__(StrataEngine)
        e._parse_done("DONE 8 42617 12.5 33.0 stop 4 7 14336 12 20 16046 1710 checkpoint")
        e._parse_done("DONE 0 32 1.0 0 cancel 0 0 0 0 0")
        self.assertEqual(e.last["reused"], 0)
        self.assertNotIn("replay_gap_tokens", e.last)

    def test_request_error_clears_old_usage(self):
        e = StrataEngine.__new__(StrataEngine)
        e.last = {"reused": 12345, "replay_gap_tokens": 2048}
        e.proc = SimpleNamespace(stdin=io.StringIO())
        e.lines = queue.Queue()
        e.lines.put("ERR invalid diagnostic request")
        with self.assertRaisesRegex(ValueError, "invalid diagnostic request"):
            list(e.generate([1, 2], 8, {}, threading.Event()))
        self.assertEqual(e.last, {})

    def test_dead_input_pipe_clears_old_usage(self):
        class DeadInput:
            def write(self, _):
                raise BrokenPipeError("closed")
        e = StrataEngine.__new__(StrataEngine)
        e.last = {"reused": 12345}
        e.proc = SimpleNamespace(stdin=DeadInput())
        e.exit_code = lambda: -9
        with self.assertRaises(EngineDied):
            list(e.generate([1, 2], 8, {}, threading.Event()))
        self.assertEqual(e.last, {})

if __name__ == '__main__': unittest.main()
