"""A budget continuation is one HTTP request, not two cache hits/timing records."""
import threading
import unittest
from pathlib import Path
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service
from serve.test_server import ThinkingEngine

class ClockedThinkingEngine(ThinkingEngine):
    def generate(self, *args, **kwargs):
        try:
            yield from super().generate(*args, **kwargs)
        finally:
            segment = len(self.prompts)
            self.last = dict(reused=7 if segment == 1 else len(self.prompts[-1])-1,
                             prompt_ms=10.0*segment, decode_ms=20.0*segment,
                             generated=3*segment, hits=2*segment, lookups=4*segment,
                             ram_blobs=segment, file_blobs=segment, file_mb=0.5*segment,
                             cache_source='checkpoint' if segment == 1 else 'live',
                             cache_timings={'prefill_new_ms':8.0*segment, 'first_generated_ms':15.0*segment})

class ReasoningCacheTimings(unittest.TestCase):
    def run_budget(self):
        tok=ByteTokenizer(); engine=ClockedThinkingEngine(tok)
        svc=Service(engine,tok,ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        done=list(svc.run([1]*30,True,None,400,{'reasoning_budget_tokens':20},threading.Event()))[-1][1]
        self.assertEqual(len(engine.prompts),2)
        return svc,done

    def test_continuation_does_not_count_its_internal_prefix_as_client_hit(self):
        svc,done=self.run_budget()
        self.assertEqual(done['cached_tokens'],7)
        self.assertEqual(svc.totals['reused'],7)
        self.assertEqual(done['timings']['prompt_n'],23)

    def test_both_segments_contribute_to_clock_and_expert_counters(self):
        svc,done=self.run_budget()
        self.assertEqual(done['timings']['prompt_ms'],30.0)
        self.assertEqual(done['timings']['predicted_ms'],60.0)
        self.assertEqual(svc.history[-1]['engine_generated'],9)
        self.assertEqual(svc.history[-1]['file_blobs'],3)
        self.assertEqual(svc.history[-1]['file_mb'],1.5)
        self.assertEqual(svc.history[-1]['cache_source'],'checkpoint')
        self.assertEqual(done['timings']['cache_timings']['prefill_new_ms'],24.0)
        self.assertEqual(done['timings']['cache_timings']['first_generated_ms'],15.0)

if __name__=='__main__': unittest.main()
