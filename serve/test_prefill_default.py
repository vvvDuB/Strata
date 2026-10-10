"""The owner streaming default is independent of prefill/checkpoint interval."""
import re
import unittest
from pathlib import Path
class OwnerPrefillDefault(unittest.TestCase):
    def test_owner_streaming_default_2048(self):
        source = (Path(__file__).resolve().parents[1] / 'src/prefill/prefill.cpp').read_text()
        function = source.split('inline int64_t stream_all_min() {', 1)[1].split('\n}', 1)[0]
        fallback = re.search(r'return g_stream_min_share > 0 \? g_stream_min_share : \(int64_t\) (\d+)', function)
        self.assertIsNotNone(fallback, 'streaming override must retain an explicit literal fallback')
        self.assertEqual(int(fallback[1]), 2048)
        self.assertLess(function.index('if (g_stream_min_override > 0)'), function.index('if (env >= 0)'))
if __name__ == '__main__': unittest.main()
