"""The owner streaming default is independent of prefill/checkpoint interval."""
import re
import unittest
from pathlib import Path
class OwnerPrefillDefault(unittest.TestCase):
    def test_owner_streaming_default_2048(self):
        source = (Path(__file__).resolve().parents[1] / 'src/prefill/prefill.cpp').read_text()
        fallback = re.search(r'return e \? \(int64_t\) std::atoll\(e\) : \(int64_t\) (\d+)', source)
        self.assertIsNotNone(fallback, 'streaming override must retain an explicit literal fallback')
        self.assertEqual(int(fallback[1]), 2048)
if __name__ == '__main__': unittest.main()
