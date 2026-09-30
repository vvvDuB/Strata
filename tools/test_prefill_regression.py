"""Opt-in real-model regression: run against a ready local Strata Coder server.

Requires batched prefill enabled and --short-read 64 (or 0), not the 4096 bypass.
STRATA_TEST_URL selects the endpoint; STRATA_TEST_RESULTS stores raw SSE results.
"""
import json
import os
from pathlib import Path
import unittest
import urllib.request

CODE = ('Write only Python code, no explanations. Implement merge_intervals(intervals): '
        'given a list of [start, end] integer pairs with start <= end, return a sorted list '
        'of merged overlapping or touching intervals. Handle empty input. Do not modify '
        'the input or its nested lists. No imports. Function signature: def merge_intervals(intervals):')

@unittest.skipUnless(__name__ == "__main__" or os.environ.get("STRATA_TEST_URL"),
                     "Opt-in: requires a running model server with batched prefill")
class BatchedPrefillRegression(unittest.TestCase):
    def request(self, name, prompt, limit=16):
        body = dict(model='Qwen3.8-Flash-Next-GSQ-RCO-Coder',
                    messages=[dict(role='user', content=prompt)], temperature=0,
                    max_tokens=limit, reasoning_effort='none', stream=True,
                    stream_options={'include_usage': True})
        url = os.environ.get('STRATA_TEST_URL', 'http://127.0.0.1:11435')
        request = urllib.request.Request(url + '/v1/chat/completions',
            data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        events, content, finish, usage, errors = [], '', None, {}, []
        with urllib.request.urlopen(request, timeout=300) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data:') or line[5:].strip() == '[DONE]':
                    continue
                event = json.loads(line[5:]); events.append(event)
                if event.get('error'):
                    errors.append(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    content += choice.get('delta', {}).get('content') or ''
                    finish = choice.get('finish_reason') or finish
        if os.environ.get('STRATA_TEST_RESULTS'):
            out = Path(os.environ['STRATA_TEST_RESULTS']); out.mkdir(parents=True, exist_ok=True)
            (out / (name + '.json')).write_text(json.dumps(dict(request=body, events=events), indent=2))
        self.assertEqual(errors, [], 'Batched prefill must not terminate the engine')
        self.assertIn(finish, ['stop', 'length'])
        self.assertTrue(content.strip(), 'Missing generated content')
        self.assertGreater(usage.get('prompt_tokens', 0), 64)
        return content

    def test_01_original_coding_crash(self):
        self.request('code', CODE, 256)

    def test_02_multichunk_prompt(self):
        prompt = 'Read the configuration and return only final_answer.\n'
        prompt += '\n'.join(f'parameter_{i:03d} = {i * 7 + 11}' for i in range(128))
        prompt += '\nfinal_answer = 8642\nReturn only the integer final_answer.'
        self.assertEqual(self.request('long', prompt).strip(), '8642')

    def test_03_relayout_after_long_prompt(self):
        self.request('code-after-long', CODE.replace('merge_intervals', 'merge_ranges'), 256)

if __name__ == '__main__':
    unittest.main()
