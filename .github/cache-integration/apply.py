#!/usr/bin/env python3
"""Reconstruct the reviewed tree from pinned public sources; never changes a ref."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

EXPECTED_PAYLOAD = 'ba8e2691812274498b16b0498f9f52331ce5b68db80314cfa980a96f934518e7'
PAYLOAD_TREE = 'bb00a92650e41a51f5093413abb9ba5a578b6784'
EXPECTED_TREE = 'a2cc50536916acadb4a6427b4cc8b7d25c812497'


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('unexpected merge-fix preimage')
    return text.replace(old, new, 1)


def main():
    source = Path(sys.argv[1]).resolve()
    root = Path.cwd().resolve()
    encoded = ''.join((source / f'payload.{i}').read_text().strip() for i in (1, 2))
    compressed = base64.b64decode(encoded, validate=True)
    if digest(compressed) != EXPECTED_PAYLOAD:
        raise RuntimeError('integration payload digest mismatch')
    spec = json.loads(gzip.decompress(compressed))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    merge = subprocess.check_output(['git', 'rev-parse', 'MERGE_HEAD'], text=True).strip()
    if head != spec['fork'] or merge != spec['upstream'] or spec['tree'] != PAYLOAD_TREE:
        raise RuntimeError('unexpected integration provenance')
    for script in spec['scripts']:
        run(sys.executable, '-c', script)
    for item in spec['supplements']:
        path = (root / item['path']).resolve()
        if root not in path.parents:
            raise RuntimeError('unsafe output path')
        before = path.read_bytes() if path.exists() else b''
        if digest(before) != item['before']:
            raise RuntimeError(f"preimage mismatch: {item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
        run('git', 'apply', '-', input=item['patch'].encode())
        if digest(path.read_bytes()) != item['after']:
            raise RuntimeError(f"postimage mismatch: {item['path']}")
    run('git', 'add', '-A')
    tree = subprocess.check_output(['git', 'write-tree'], text=True).strip()
    if tree != PAYLOAD_TREE:
        raise RuntimeError(f'unexpected payload tree: {tree}')
    # Full-engine compilation exposed two merge-resolution errors. The fork's
    # existing progress counters and option scope are authoritative.
    generate = root / 'src/program/generate.cpp'
    text = generate.read_text()
    text = replace_once(text, '        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;\n', '')
    text = replace_once(text, '            }\n        else if (a == "--conversation-cache-disk")',
                        '            }\n        }\n        else if (a == "--conversation-cache-disk")')
    generate.write_text(text)
    workflow = root / '.github/workflows/cache-regression.yml'
    workflow.write_text(replace_once(workflow.read_text(), 'pip install jinja2 regex\n',
                                     'pip install jinja2 regex numpy\n'))
    run('git', 'add', '-A')
    run('git', 'diff', '--cached', '--check')
    tree = subprocess.check_output(['git', 'write-tree'], text=True).strip()
    if tree != EXPECTED_TREE:
        raise RuntimeError(f'unexpected reconstructed tree: {tree}')
    print(f'Verified integration tree: {tree}', flush=True)


if __name__ == '__main__':
    main()
