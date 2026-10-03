"""Native prefill threshold parsing: no weights or inference required."""
import os
from pathlib import Path
import subprocess
import unittest


class PrefillStreamCLI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exe = Path(os.environ.get("STRATA_TEST_EXE", Path(__file__).resolve().parents[1] / "build/strata"))
        if not cls.exe.is_file():
            raise unittest.SkipTest("build the native engine or set STRATA_TEST_EXE")

    def run_cli(self, *args):
        return subprocess.run([str(self.exe), *args],
                              capture_output=True, text=True, timeout=15)

    def test_help_documents_threshold_and_precedence(self):
        result = self.run_cli("--help")
        self.assertEqual(result.returncode, 0)
        self.assertIn("--prefill-stream-min N", result.stderr)
        self.assertIn("overrides STRATA_PREFILL_STREAM_MIN", result.stderr)

    def test_explicit_positive_integers_are_accepted(self):
        for value in ("1", "2048", "4096", "8192", "9223372036854775807"):
            with self.subTest(value=value):
                result = self.run_cli("--prefill-stream-min", value, "--help")
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_values_are_rejected_before_loading(self):
        for value in ("0", "-1", "abc", "4096x", "4.5", "+4096", " 4096", "4096 ",
                      "9223372036854775808", ""):
            with self.subTest(value=value):
                result = self.run_cli("--prefill-stream-min", value, "--help")
                self.assertEqual(result.returncode, 2)
                self.assertIn("--prefill-stream-min requires a positive integer", result.stderr)

    def test_missing_value_is_rejected(self):
        result = self.run_cli("--prefill-stream-min")
        self.assertEqual(result.returncode, 2)
        self.assertIn("--prefill-stream-min needs a value", result.stderr)


if __name__ == "__main__":
    unittest.main()
