"""Real Q3 A/B/A regression, on a caller-owned localhost diagnostic server.

Never executes model tool calls. Saves private HTTP evidence and asserts cache
reuse only when --expect-cache is passed. Synthetic Pi/Claude-style roots are
not captures of either application's actual system prompt.
"""
import argparse
import json
import os
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--expect-cache', action='store_true')
    args = p.parse_args()
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.umask(0o077)
    rows = []
    systems = {
        'a': 'Sei un assistente per il progetto ALFA. Rispondi solo al messaggio utente.\n'
             + 'Il codice del progetto ALFA è QUARZO. Non usare strumenti.\n' * 24,
        'b': 'Sei un assistente per il progetto BETA. Rispondi solo al messaggio utente.\n'
             + 'Il codice del progetto BETA è AMBRA. Non usare strumenti.\n' * 24,
    }
    histories = {}
    for i, name in enumerate(['a', 'b', 'a', 'b']):
        messages = histories.get(name, [{'role': 'user', 'content': 'Qual è il codice del progetto? Rispondi solo con il codice.'}])
        if name in histories:
            messages = messages + [{'role': 'user', 'content': 'Ripeti il codice del progetto, senza aggiungere altro.'}]
        body = {'model': 'test', 'system': systems[name], 'messages': messages,
                'thinking': {'type': 'disabled'}, 'max_tokens': 64, 'temperature': 0, 'stream': False}
        (args.output / f'input-{i}.json').write_text(json.dumps(body, ensure_ascii=False, indent=2))
        start = time.monotonic()
        req = urllib.request.Request('http://127.0.0.1:11435/v1/messages',
                                     data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=300) as r:
            data = json.load(r)
        (args.output / f'output-{i}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2))
        answer = ''.join(c.get('text', '') for c in data['content']).strip()
        row = {'branch': name, 'iteration': i, 'text': answer, 'seconds': round(time.monotonic()-start, 3),
               'stop_reason': data['stop_reason'], **data['usage']}
        rows.append(row)
        (args.output / 'results.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2))
        print(json.dumps(row, ensure_ascii=False), flush=True)
        assert ('QUARZO' if name == 'a' else 'AMBRA') in answer, 'wrong conversation state'
        histories[name] = messages + [{'role': 'assistant', 'content': data['content']}]
    if args.expect_cache:
        for row in rows[2:]:
            assert row.get('cache_read_input_tokens', 0) > 300, 'A/B/A loses conversation prefix'


if __name__ == '__main__':
    main()
