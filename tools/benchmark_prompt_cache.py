#!/usr/bin/env python3
"""Opt-in HTTP benchmark against an already running, caller-owned Strata server.

Reports real prompt/cache counts and observed TTFT. Does not start/stop servers,
execute tools, save conversation text, or claim GPU parity from synthetic tests.
Use a quiet diagnostic server: other traffic can change its active cache branch.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request


def request_json(url: str, headers: dict[str, str], timeout: float) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout) as response:
        return json.load(response)


def run_case(base: str, model: str, messages: list[dict], headers: dict[str, str],
             timeout: float, max_tokens: int) -> tuple[dict, str]:
    body = dict(model=model, messages=messages, temperature=0, max_tokens=max_tokens,
                reasoning_effort="none", stream=True, stream_options={"include_usage": True})
    request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={**headers, "Content-Type": "application/json"})
    started = time.perf_counter()
    ttft = None
    usage: dict = {}
    text: list[str] = []
    finish = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            usage = event.get("usage") or usage
            for choice in event.get("choices", []):
                delta = choice.get("delta") or {}
                if ttft is None and (delta.get("content") or delta.get("reasoning_content")):
                    ttft = time.perf_counter() - started
                text.append(delta.get("content") or "")
                finish = choice.get("finish_reason") or finish
    prompt = usage.get("prompt_tokens")
    reused = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if not isinstance(prompt, int) or prompt <= 0 or not isinstance(reused, int):
        raise RuntimeError("Missing prompt_tokens / prompt_tokens_details.cached_tokens in final SSE usage")
    if not 0 <= reused <= prompt:
        raise RuntimeError("Invalid cache usage counters")
    return dict(prompt_tokens=prompt, cached_tokens=reused, read_tokens=prompt-reused,
                cache_ratio=round(reused/prompt, 5), ttft_seconds=None if ttft is None else round(ttft, 4),
                total_seconds=round(time.perf_counter()-started, 4), finish=finish), "".join(text)


def save_report(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        os.fchmod(out.fileno(), 0o600)
        json.dump(data, out, ensure_ascii=False, indent=2)
        out.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:11434")
    parser.add_argument("--model", help="Defaults to the first model from /v1/models")
    parser.add_argument("--rows", type=int, default=1024, help="Synthetic history size, not a token count")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, default=Path("cache-benchmark.json"))
    parser.add_argument("--api-key-env", default="STRATA_API_KEY")
    parser.add_argument("--expect-cache", action="store_true", help="Require >=90%% hit on the exact repeat")
    args = parser.parse_args()
    if args.rows < 32 or args.max_tokens < 1 or args.timeout <= 0:
        parser.error("rows must be >=32; max-tokens and timeout must be positive")
    base = args.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    key = os.environ.get(args.api_key_env, "")
    headers = {"Authorization": "Bearer " + key} if key else {}
    model = args.model
    if not model:
        models = request_json(base + "/v1/models", headers, args.timeout).get("data", [])
        if not models:
            raise RuntimeError("/v1/models returned no model")
        model = models[0]["id"]
    def records(start: int, end: int) -> str:
        return "\n".join(f"record_{i:05d}: checksum={i*17+11}; status=stable; group={i%13}." for i in range(start,end))
    messages = [
        {"role": "system", "content": "Diagnostic session ALFA. Read the supplied records; reply only with 7."},
        {"role": "user", "content": records(0, args.rows//3)},
        {"role": "assistant", "content": "7"},
        {"role": "user", "content": "Temporary goal revision: AAA. Continue reading."},
        {"role": "assistant", "content": "7"},
        {"role": "user", "content": records(args.rows//3, args.rows) + "\nReply only with 7."},
    ]
    report = {"kind": "live HTTP benchmark", "model": model, "rows": args.rows, "results": []}
    def measure(name: str, prompt: list[dict]) -> str:
        row, text = run_case(base, model, prompt, headers, args.timeout, args.max_tokens)
        row["case"] = name
        report["results"].append(row)
        save_report(args.output, report)
        print(json.dumps(row), flush=True)
        return text
    measure("initial", messages)
    answer = measure("exact_repeat", messages)
    if args.expect_cache and report["results"][-1]["cache_ratio"] < .90:
        raise RuntimeError("Exact repeat reused less than 90%; check cache configuration/other server traffic")
    appended = messages + [{"role": "assistant", "content": answer}, {"role": "user", "content": "Again: reply only with 7."}]
    measure("append", appended)
    edited = copy.deepcopy(appended)
    edited[3]["content"] = "Temporary goal revision: BBB. Continue reading."
    measure("history_edit", edited)
    edited[3]["content"] = "Temporary goal revision: CCC. Continue reading."
    measure("same_boundary_edit", edited)
    other = copy.deepcopy(edited)
    other[0]["content"] = "Diagnostic session BETA. Read the supplied records; reply only with 7."
    measure("other_system", other)
    measure("return_to_previous_branch", edited)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, urllib.error.HTTPError) as exc:
        raise SystemExit(f"Cache benchmark failed: {exc}") from exc
