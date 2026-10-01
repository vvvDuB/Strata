"""Native 0.1.31 tiers coexist with the owner cache coverage protocol."""
import unittest
from serve.server import StrataEngine

class DoneProtocol031(unittest.TestCase):
    def parse(self, tail):
        engine = StrataEngine.__new__(StrataEngine)
        engine._parse_done('DONE 4 10026 125.0 20.0 stop 4 4 10019 9 10 ' + tail)
        return engine.last

    def test_upstream_positional_tiers_are_not_cache_coverage(self):
        last = self.parse('11 12 13.5')
        self.assertTrue(all(k in last for k in ('ram_blobs','file_blobs','file_mb')), 'expert tier metrics missing')
        self.assertEqual((last['ram_blobs'], last['file_blobs'], last['file_mb']), (11, 12, 13.5))
        self.assertNotIn('cache_source', last)
        self.assertEqual(last['reused'], 10019)

    def test_owner_coverage_timing_and_keyed_tiers(self):
        last = self.parse('10026 6 checkpoint cache_select_ms=2.5 ram_blobs=11 file_blobs=12 file_mb=13.5')
        self.assertEqual(last['cache_source'], 'checkpoint')
        self.assertEqual(last['common_prefix_tokens'], 10026)
        self.assertEqual(last['cache_timings'], {'selection_ms': 2.5})
        self.assertTrue(all(k in last for k in ('ram_blobs','file_blobs','file_mb')), 'expert tier metrics missing')
        self.assertEqual((last['ram_blobs'], last['file_blobs'], last['file_mb']), (11, 12, 13.5))

    def test_invalid_tier_extensions_do_not_poison_counters(self):
        last = self.parse('10026 6 checkpoint ram_blobs=-1 file_blobs=nan file_mb=inf')
        for key in ('ram_blobs', 'file_blobs', 'file_mb'):
            self.assertNotIn(key, last)
        self.assertEqual(last['reused'], 10019)

if __name__ == '__main__':
    unittest.main()
