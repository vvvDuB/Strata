"""Direct shell-profile arguments, without reading settings or loading a model."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from serve import server


class ServerCLI(unittest.TestCase):
    def parse(self, argv):
        self.assertTrue(callable(getattr(server, "parse_server_options", None)),
                        "server must accept direct engine options without a JSON profile")
        return server.parse_server_options(argv)

    def direct(self):
        return ["--engine", "strata", "--exe", "/engine/strata", "--tokenizer", "/pack/tokenizer",
                "--cwd", "/engine", "--log", "/logs/engine.log", "--alias", "q3",
                "--lib-dir", "/cuda/lib64", "--host", "0.0.0.0", "--port", "11434",
                "--api-key", "", "--gpu", "0", "--", "--native", "/model path/q3.gguf",
                "--max-context", "131072", "--mmap-experts"]

    def test_direct_options_are_exact_and_do_not_read_json(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("no settings file may be read")):
            _, a, cfg = self.parse(self.direct())
        self.assertIsNone(a.config)
        self.assertEqual((a.host, a.port, a.tokenizer), ("0.0.0.0", 11434, "/pack/tokenizer"))
        self.assertEqual(cfg, {"exe": "/engine/strata", "cwd": "/engine", "log": "/logs/engine.log",
                              "model_name": "q3", "tokenizer": "/pack/tokenizer", "gpu": 0,
                              "lib_dirs": ["/cuda/lib64"],
                              "args": ["--native", "/model path/q3.gguf", "--max-context", "131072",
                                       "--mmap-experts"]})

    def test_native_option_names_are_not_parsed_as_server_options(self):
        argv = self.direct() + ["--host", "native-host", "--log", "native-log"]
        _, a, cfg = self.parse(argv)
        self.assertEqual(a.host, "0.0.0.0")
        self.assertEqual(cfg["log"], "/logs/engine.log")
        self.assertEqual(cfg["args"][-4:], ["--host", "native-host", "--log", "native-log"])

    def test_prefill_stream_threshold_is_forwarded_literally(self):
        _, _, cfg = self.parse(self.direct() + ["--prefill-stream-min", "4096"])
        self.assertEqual(cfg["args"][-2:], ["--prefill-stream-min", "4096"])
        self.assertEqual(server.engine_args(cfg)[-2:], ["--prefill-stream-min", "4096"])

    def test_direct_layer_split_keeps_the_gpu_list(self):
        argv = self.direct()
        argv[argv.index("--gpu") + 1] = "0,2"
        _, _, cfg = self.parse(argv)
        self.assertEqual(cfg["gpu"], "0,2")
        self.assertEqual(server.gpu_list(cfg), [0, 2])

    def test_repeated_library_directories_preserve_order(self):
        argv = self.direct()
        argv[argv.index("--"):argv.index("--")] = ["--lib-dir", "/another lib"]
        _, _, cfg = self.parse(argv)
        self.assertEqual(cfg["lib_dirs"], ["/cuda/lib64", "/another lib"])

    def test_explicit_empty_api_key_does_not_use_environment(self):
        with patch.dict(server.os.environ, {"STRATA_API_KEY": "environment-secret"}):
            _, a, _ = self.parse(self.direct())
        self.assertEqual(a.api_key, "")

    def test_legacy_config_and_bom_are_still_supported(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.json"
            original = {"exe": "old-engine", "args": ["--native", "old-model"],
                        "tokenizer": "/legacy/tokenizer", "host": "127.0.0.2"}
            p.write_text(json.dumps(original), encoding="utf-8-sig")
            _, a, cfg = self.parse(["--engine", "strata", "--config", str(p), "--gpu", "0"])
        self.assertEqual(cfg, dict(original, gpu=0))
        self.assertEqual(a.host, "127.0.0.2")
        self.assertEqual(a.tokenizer, "/legacy/tokenizer")

    def test_cli_host_still_overrides_legacy_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "old.json"
            p.write_text('{"exe":"x","args":[],"host":"127.0.0.2"}')
            _, a, _ = self.parse(["--engine", "strata", "--config", str(p), "--host", "0.0.0.0"])
        self.assertEqual(a.host, "0.0.0.0")

    def test_mock_default_is_unchanged(self):
        _, a, cfg = self.parse([])
        self.assertEqual((a.engine, a.host, a.port, cfg), ("mock", "127.0.0.1", 8095, {}))

    def test_empty_legacy_strata_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.json"
            p.write_text("{}")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                self.parse(["--engine", "strata", "--config", str(p)])
        self.assertEqual(error.exception.code, 2)

    def test_invalid_or_ambiguous_sources_fail_before_loading(self):
        cases = [
            ["--engine", "strata"],
            ["--engine", "strata", "--exe", "x", "--", "--native", "m"],
            ["--engine", "strata", "--exe", "x", "--tokenizer", "t"],
            ["--engine", "strata", "--exe", "x", "--tokenizer", "t", "--"],
            ["--engine", "strata", "--config", "missing.json", "--exe", "x"],
            ["--engine", "strata", "--config", "missing.json", "--", "--native", "m"],
            ["--engine", "strata", "--cwd", "x"],
            ["--engine", "mock", "--exe", "x", "--tokenizer", "t", "--", "--native", "m"],
            ["--engine", "strata", "--exe", "x", "--tokenizer", "t", "native", "m"],
            ["--engine", "strata", "--ex", "x"],
        ]
        for argv in cases:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                self.assertTrue(callable(getattr(server, "parse_server_options", None)))
                with self.assertRaises(SystemExit) as error:
                    server.parse_server_options(argv)
                self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
