#!/usr/bin/env python3
"""Compare two teacher-forced Strata --dump-logits files, row by row.

Use identical tokens, weights and run flags except --kv; --max-new 1,
--prefill 0 and --spec 0. The CLI's dump is little-endian [int32 vocab,
int32 rows, float32 logits...]. --stride must match --logits-stride.
Native packs: use STRATA_DUMP_FIRST_LOGITS and --raw-first to compare the
single prediction after the full prompt. This is NOT a perplexity test.
This reports differences; it does not certify losslessness or model quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys
import numpy as np


def load_dump(path: Path) -> np.memmap:
    with path.open('rb') as f:
        header = f.read(8)
    if len(header) != 8:
        raise ValueError(f'{path}: missing 8-byte logits header')
    vocab, rows = struct.unpack('<ii', header)
    if vocab <= 0 or rows <= 0:
        raise ValueError(f'{path}: invalid dimensions')
    expected = 8 + 4 * vocab * rows
    if path.stat().st_size != expected:
        raise ValueError(f'{path}: expected {expected} bytes, got {path.stat().st_size}; incomplete dump or wrong execution path')
    return np.memmap(path, dtype='<f4', mode='r', offset=8, shape=(rows, vocab))


def positions_for(total: int, stride: int) -> list[int]:
    if total <= 0 or stride <= 0:
        raise ValueError('token count and stride must be positive')
    positions = list(range(0, total, stride))
    if positions[-1] != total - 1:
        positions.append(total - 1)
    return positions


def compare(baseline: Path, candidate: Path, tokens: list[int], stride: int = 1) -> dict:
    a, b = load_dump(baseline), load_dump(candidate)
    if a.shape != b.shape:
        raise ValueError('dump dimensions differ')
    positions = positions_for(len(tokens), stride)
    if a.shape[0] != len(positions):
        raise ValueError('dump row count does not match tokens and stride')
    if any(t < 0 or t >= a.shape[1] for t in tokens):
        raise ValueError('token id outside vocabulary')
    agreement = 0
    kl_sum = max_kl = max_abs = nll_a = nll_b = 0.0
    scored = 0
    for row, pos in enumerate(positions):
        x, y = np.asarray(a[row], dtype=np.float64), np.asarray(b[row], dtype=np.float64)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'nonfinite logits at position {pos}')
        agreement += int(np.argmax(x) == np.argmax(y))
        max_abs = max(max_abs, float(np.max(np.abs(x-y))))
        x -= np.max(x); y -= np.max(y)
        lx = x - np.log(np.exp(x).sum())
        ly = y - np.log(np.exp(y).sum())
        kl = max(0.0, float(np.sum(np.exp(lx)*(lx-ly))))
        kl_sum += kl; max_kl = max(max_kl, kl)
        if pos + 1 < len(tokens):
            nll_a -= float(lx[tokens[pos+1]])
            nll_b -= float(ly[tokens[pos+1]])
            scored += 1
    result = {
        'rows': len(positions), 'vocab': a.shape[1], 'input_tokens': len(tokens),
        'stride': stride, 'scored_next_tokens': scored,
        'tokens_sha256': hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest(),
        'top1_agreement': agreement / len(positions),
        'mean_kl_baseline_to_candidate_nats': kl_sum/len(positions),
        'max_kl_baseline_to_candidate_nats': max_kl,
        'max_abs_logit_delta': max_abs,
        'baseline_mean_nll_nats': nll_a/scored if scored else None,
        'candidate_mean_nll_nats': nll_b/scored if scored else None,
        'delta_mean_nll_nats': (nll_b-nll_a)/scored if scored else None,
        'scope': 'all conditioned next tokens' if stride == 1 else 'sampled conditioned positions only',
        'assumptions': 'Both dumps must come from these same tokens and identical settings except KV format; file bytes alone cannot establish this.',
        'decision': 'Metrics only. No universal pass threshold or losslessness claim.'
    }
    return result


def compare_first(baseline: Path, candidate: Path, tokens: list[int]) -> dict:
    """Native-pack diagnostic: one raw float32 vector after the entire prompt."""
    size = baseline.stat().st_size
    if size <= 0 or size % 4 or candidate.stat().st_size != size:
        raise ValueError('raw first-logit files must have equal, positive float32 sizes')
    vocab = size // 4
    if not tokens or any(t < 0 or t >= vocab for t in tokens):
        raise ValueError('nonempty prompt with token ids inside the vocabulary required')
    x = np.fromfile(baseline, dtype='<f4').astype(np.float64)
    y = np.fromfile(candidate, dtype='<f4').astype(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('nonfinite first logits')
    delta = float(np.max(np.abs(x-y)))
    top_a, top_b = int(np.argmax(x)), int(np.argmax(y))
    x -= np.max(x); y -= np.max(y)
    lx = x - np.log(np.exp(x).sum()); ly = y - np.log(np.exp(y).sum())
    return {
        'rows': 1, 'vocab': vocab, 'input_tokens': len(tokens),
        'position': len(tokens)-1,
        'tokens_sha256': hashlib.sha256(np.asarray(tokens,dtype='<i8').tobytes()).hexdigest(),
        'baseline_top1': top_a, 'candidate_top1': top_b,
        'top1_agreement': int(top_a == top_b),
        'kl_baseline_to_candidate_nats': max(0.0, float(np.sum(np.exp(lx)*(lx-ly)))),
        'max_abs_logit_delta': delta,
        'scope': 'one native-pack prediction after the full conditioned prompt',
        'assumptions': 'Raw files must be STRATA_DUMP_FIRST_LOGITS outputs for these same tokens and settings except KV format.',
        'decision': 'Single-position diagnostic only, not perplexity or a model quality pass.'
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    parser.add_argument('--tokens-file', type=Path, required=True)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--raw-first', action='store_true',
                        help='compare raw STRATA_DUMP_FIRST_LOGITS vectors from native packs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        text = args.tokens_file.read_text().strip()
        if not text or re.fullmatch(r'[0-9,\s]+', text) is None:
            raise ValueError('tokens file must contain nonnegative integer ids separated by commas/whitespace')
        tokens = [int(x) for x in re.split(r'[,\s]+',text) if x]
        if args.raw_first and args.stride != 1:
            raise ValueError('--raw-first does not accept a sampled stride')
        report = (compare_first(args.baseline,args.candidate,tokens) if args.raw_first
                  else compare(args.baseline,args.candidate,tokens,args.stride))
        rendered = json.dumps(report,indent=2,allow_nan=False) + '\n'
        if args.output:
            args.output.write_text(rendered)
        print(rendered,end='')
        return 0
    except (OSError, ValueError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
