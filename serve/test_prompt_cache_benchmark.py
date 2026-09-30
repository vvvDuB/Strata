"""HTTP benchmark parser tests with synthetic SSE only; no model/server required."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.benchmark_prompt_cache import run_case, save_report


def stream(*events):
    raw = ": heartbeat\n\n" + "".join("data: " + json.dumps(e) + "\n\n" for e in events)
    return io.BytesIO((raw + "data: [DONE]\n\n").encode())


class PromptCacheBenchmark(unittest.TestCase):
    def test_stream_usage_and_answer(self):
        messages = [{"role": "user", "content": "diagnostic"}]
        with patch("urllib.request.urlopen", return_value=stream(
            {"choices": [{"delta": {"content": "7"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 96}}}
        )) as opener:
            result, answer = run_case("http://localhost:11434", "test", messages, {}, 5, 16)
        self.assertEqual(answer, "7")
        self.assertEqual(result["cached_tokens"], 96)
        self.assertEqual(result["read_tokens"], 4)
        self.assertEqual(result["cache_ratio"], .96)
        self.assertIsNotNone(result["ttft_seconds"])
        self.assertEqual(result["finish"], "stop")
        payload = json.loads(opener.call_args.args[0].data)
        self.assertEqual(payload["messages"], messages)
        self.assertTrue(payload["stream_options"]["include_usage"])

    def test_missing_usage_is_not_silently_reported_as_zero(self):
        with patch("urllib.request.urlopen", return_value=stream({"choices": []})):
            with self.assertRaisesRegex(RuntimeError, "Missing prompt_tokens"):
                run_case("http://localhost", "test", [], {}, 5, 16)

    def test_error_event_is_not_reported_as_a_success(self):
        with patch("urllib.request.urlopen", return_value=stream({"error": {"message": "failure"}})):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                run_case("http://localhost", "test", [], {}, 5, 16)

    def test_report_is_private_and_contains_counts_only(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "report.json"
            save_report(path, {"results": [{"cached_tokens": 96}]})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text())["results"][0]["cached_tokens"], 96)


if __name__ == "__main__":
    unittest.main()
