"""Optional phase timings never change usage or legacy DONE compatibility."""
import unittest
from serve.server import StrataEngine, request_timings

BASE = 'DONE 4 40022 125.0 20.0 stop 4 4 40015 9 10 40022 6 checkpoint'
class CacheTimings(unittest.TestCase):
    def engine(self, extras=''):
        engine = StrataEngine.__new__(StrataEngine)
        engine._parse_done(BASE + extras)
        return engine

    def test_phase_fields_are_separate_from_inclusive_prompt_time(self):
        engine = self.engine(' cache_select_ms=2.5 cache_park_ms=40 cache_read_ms=30 cache_restore_ms=20'
                             ' checkpoint_ms=10 prefill_replay_ms=5 prefill_new_ms=1 prefill_mixed_ms=6 first_generated_ms=128')
        self.assertEqual(engine.last.get('cache_timings'), {
            'selection_ms':2.5, 'parking_ms':40.0, 'snapshot_read_ms':30.0,
            'snapshot_restore_ms':20.0, 'checkpoint_ms':10.0,
            'prefill_replay_ms':5.0, 'prefill_new_ms':1.0, 'prefill_mixed_ms':6.0,
            'first_generated_ms':128.0})
        self.assertEqual(engine.last['prompt_ms'],125.0)
        self.assertEqual(engine.last['reused'],40015)

    def test_timing_payload_is_copied_to_api_without_changing_old_counters(self):
        engine = self.engine(' cache_select_ms=2.5')
        result = request_timings(40022,4,engine.last)
        self.assertEqual(result.get('cache_timings'),{'selection_ms':2.5})
        self.assertEqual((result['cache_n'],result['prompt_n'],result['prompt_ms']),(40015,7,125.0))
        engine.last['cache_timings']['selection_ms']=999
        self.assertEqual(result['cache_timings']['selection_ms'],2.5)

    def test_bad_or_unknown_extensions_are_ignored(self):
        engine = self.engine(' cache_select_ms=nan cache_park_ms=-1 cache_read_ms=inf cache_restore_ms=nope'
                             ' unknown_ms=88 checkpoint_ms=3 malformed prefill_new_ms=0')
        self.assertEqual(engine.last.get('cache_timings'),{'checkpoint_ms':3.0,'prefill_new_ms':0.0})

    def test_old_done_clears_previous_metrics(self):
        engine = self.engine(' cache_select_ms=2.5')
        self.assertIn('cache_timings',engine.last)
        engine._parse_done('DONE 0 32 1.0 0 cancel 0 0 0 0 0')
        self.assertNotIn('cache_timings',engine.last)
        self.assertNotIn('cache_timings',request_timings(32,0,engine.last))

    def test_padded_cancel_keeps_coverage_and_metrics_parseable(self):
        engine = self.engine()
        engine._parse_done('DONE 0 40022 42.0 0 cancel 0 0 0 0 0 0 0 cancel cache_select_ms=5.5')
        self.assertEqual(engine.last['cache_source'],'cancel')
        self.assertEqual(engine.last.get('cache_timings'),{'selection_ms':5.5})

if __name__ == '__main__': unittest.main()
