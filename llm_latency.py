#!/usr/bin/env python3
"""llm-latency: measure time to first token, total latency, output speed and success rate
of any OpenAI-compatible or Anthropic-native streaming endpoint.

Standard library only. Python 3.9+.

    export LLM_API_KEY=sk-...
    python3 llm_latency.py --base-url https://api.openai.com/v1 --model gpt-6.1-sol -n 20
    python3 llm_latency.py --api anthropic --base-url https://api.anthropic.com --model claude-opus-5-5 -n 20

What is measured, per request, from the client:
  ttft     seconds from sending the request to the first streamed content token
           (for reasoning models this includes thinking time unless --count-reasoning is set)
  total    seconds from sending the request to the end of the stream
  tps      output tokens / (total - ttft); tokens come from the usage block when the server
           sends one, otherwise they are estimated from characters (marked with ~)
  ok       HTTP 2xx, at least one content token, and a clean end of stream
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import http.client
import json
import os
import ssl
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional
from urllib.parse import urlparse

__version__ = "0.1.0"
DEFAULT_PROMPT = "Write a 200-word explanation of how a hash table handles collisions."


@dataclass
class Result:
    model: str
    ok: bool
    status: int = 0
    ttft: Optional[float] = None
    total: Optional[float] = None
    output_tokens: Optional[int] = None
    tokens_estimated: bool = False
    tps: Optional[float] = None
    error: str = ""


def build_request(api: str, base_url: str, model: str, prompt: str, max_tokens: int, key: str):
    """Return (path, headers, body) for one streaming request."""
    u = urlparse(base_url)
    prefix = u.path.rstrip("/")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": f"llm-latency/{__version__}"}
    if api == "anthropic":
        path = prefix + ("/messages" if prefix.endswith("/v1") else "/v1/messages")
        headers.update({"x-api-key": key, "anthropic-version": "2023-06-01"})
        body = {"model": model, "max_tokens": max_tokens, "stream": True,
                "messages": [{"role": "user", "content": prompt}]}
    else:
        path = prefix + "/chat/completions"
        headers["Authorization"] = f"Bearer {key}"
        body = {"model": model, "max_completion_tokens": max_tokens, "stream": True,
                "stream_options": {"include_usage": True},
                "messages": [{"role": "user", "content": prompt}]}
    return path, headers, json.dumps(body).encode()


def parse_event(api: str, data: dict, count_reasoning: bool):
    """Return (content_text, output_tokens_or_None, error_or_None) for one SSE data object."""
    if api == "anthropic":
        t = data.get("type")
        if t == "error":
            return "", None, json.dumps(data.get("error", data))[:200]
        if t == "content_block_delta":
            d = data.get("delta", {})
            if d.get("type") == "text_delta":
                return d.get("text", ""), None, None
            if count_reasoning and d.get("type") == "thinking_delta":
                return d.get("thinking", ""), None, None
        if t == "message_delta":
            return "", (data.get("usage") or {}).get("output_tokens"), None
        return "", None, None
    if data.get("error"):
        return "", None, json.dumps(data["error"])[:200]
    text = ""
    for ch in data.get("choices") or []:
        d = ch.get("delta") or {}
        text += d.get("content") or ""
        if count_reasoning:
            text += d.get("reasoning_content") or d.get("reasoning") or ""
    usage = data.get("usage") or {}
    return text, usage.get("completion_tokens"), None


def one_request(api, base_url, model, prompt, max_tokens, key, timeout, count_reasoning, insecure=False) -> Result:
    u = urlparse(base_url)
    path, headers, body = build_request(api, base_url, model, prompt, max_tokens, key)
    r = Result(model=model, ok=False)
    conn = None
    t0 = time.perf_counter()
    try:
        if u.scheme == "https":
            ctx = ssl._create_unverified_context() if insecure else ssl.create_default_context()
            conn = http.client.HTTPSConnection(u.netloc, timeout=timeout, context=ctx)
        else:
            conn = http.client.HTTPConnection(u.netloc, timeout=timeout)
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()
        r.status = resp.status
        if resp.status // 100 != 2:
            r.total = time.perf_counter() - t0
            r.error = resp.read(400).decode("utf-8", "replace").replace("\n", " ")
            return r
        chars, done = 0, False
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            text, tokens, err = parse_event(api, data, count_reasoning)
            if err:
                r.error = err
                break
            if text:
                if r.ttft is None:
                    r.ttft = time.perf_counter() - t0
                chars += len(text)
            if tokens is not None:
                r.output_tokens = tokens
            if api == "anthropic" and data.get("type") == "message_stop":
                done = True
                break
        r.total = time.perf_counter() - t0
        if r.error:
            return r
        if r.ttft is None:
            r.error = "stream ended without content"
            return r
        if not done:
            r.error = "stream ended without a terminator"
            return r
        if r.output_tokens is None:
            r.output_tokens, r.tokens_estimated = max(1, round(chars / 4)), True
        gen = r.total - r.ttft
        r.tps = r.output_tokens / gen if gen > 0.05 else None
        r.ok = True
        return r
    except Exception as e:  # timeouts, resets, TLS errors: all count as failures
        r.total = time.perf_counter() - t0
        r.error = f"{type(e).__name__}: {e}"[:200]
        return r
    finally:
        if conn is not None:
            conn.close()


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize(results):
    ok = [r for r in results if r.ok]

    def q(vals, p):
        v = quantile(vals, p)
        return None if v is None else round(v, 3)

    ttft = [r.ttft for r in ok]
    total = [r.total for r in ok]
    tps = [r.tps for r in ok if r.tps]
    return {
        "model": results[0].model if results else "",
        "requests": len(results),
        "success_rate": round(100.0 * len(ok) / len(results), 2) if results else None,
        "ttft_p50_s": q(ttft, 0.5), "ttft_p95_s": q(ttft, 0.95),
        "total_p50_s": q(total, 0.5), "total_p95_s": q(total, 0.95),
        "tps_p50": None if not tps else round(quantile(tps, 0.5), 1),
        "tokens_estimated": any(r.tokens_estimated for r in ok),
        "errors": sorted({(r.error or f"HTTP {r.status}")[:80] for r in results if not r.ok}),
    }


def run(api, base_url, model, prompt, max_tokens, key, n, concurrency, timeout, count_reasoning,
        insecure=False, warmup=0, progress=None):
    args = (api, base_url, model, prompt, max_tokens, key, timeout, count_reasoning, insecure)
    for _ in range(warmup):
        one_request(*args)
    results = []
    with cf.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for r in pool.map(lambda _: one_request(*args), range(n)):
            results.append(r)
            if progress:
                progress(r)
    return results


def fmt(v, suffix=""):
    return "-" if v is None else f"{v}{suffix}"


def print_table(rows, out=sys.stdout):
    head = ["model", "n", "success", "ttft p50", "ttft p95", "total p50", "total p95", "tok/s p50"]
    table = [head]
    for s in rows:
        table.append([s["model"], str(s["requests"]), fmt(s["success_rate"], "%"),
                      fmt(s["ttft_p50_s"], " s"), fmt(s["ttft_p95_s"], " s"),
                      fmt(s["total_p50_s"], " s"), fmt(s["total_p95_s"], " s"),
                      ("~" if s["tokens_estimated"] else "") + fmt(s["tps_p50"])])
    widths = [max(len(r[i]) for r in table) for i in range(len(head))]
    for i, r in enumerate(table):
        print("  ".join(c.ljust(widths[j]) if j == 0 else c.rjust(widths[j]) for j, c in enumerate(r)), file=out)
        if i == 0:
            print("  ".join("-" * w for w in widths), file=out)
    for s in rows:
        for e in s["errors"]:
            print(f"  {s['model']}: {e}", file=out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="llm-latency", description=__doc__.split("\n\n")[0])
    p.add_argument("--base-url", required=True, help="e.g. https://api.openai.com/v1 or https://api.anthropic.com")
    p.add_argument("--model", required=True, action="append", help="model id; repeat to compare several")
    p.add_argument("--api", choices=["openai", "anthropic"], default="openai", help="wire format (default: openai)")
    p.add_argument("--api-key", default=os.environ.get("LLM_API_KEY", ""), help="defaults to $LLM_API_KEY")
    p.add_argument("-n", "--requests", type=int, default=20, help="measured requests per model (default: 20)")
    p.add_argument("-c", "--concurrency", type=int, default=1, help="parallel requests (default: 1)")
    p.add_argument("--warmup", type=int, default=1, help="unmeasured requests before measuring (default: 1)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--prompt-file", help="read the prompt from a file")
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=120.0, help="socket timeout in seconds (default: 120)")
    p.add_argument("--count-reasoning", action="store_true",
                   help="count streamed reasoning/thinking text as the first token")
    p.add_argument("--insecure", action="store_true", help="skip TLS verification (local testing only)")
    p.add_argument("--json", metavar="FILE", help="write summaries and per-request results as JSON")
    p.add_argument("--csv", metavar="FILE", help="write per-request results as CSV")
    p.add_argument("--version", action="version", version=__version__)
    a = p.parse_args(argv)
    if not a.api_key:
        p.error("no API key: pass --api-key or set LLM_API_KEY")
    prompt = open(a.prompt_file, encoding="utf-8").read() if a.prompt_file else a.prompt

    summaries, everything = [], []
    for model in a.model:
        def tick(r):
            sys.stderr.write("." if r.ok else "x")
            sys.stderr.flush()
        sys.stderr.write(f"{model} ")
        res = run(a.api, a.base_url, model, prompt, a.max_tokens, a.api_key, a.requests, a.concurrency,
                  a.timeout, a.count_reasoning, a.insecure, a.warmup, tick)
        sys.stderr.write("\n")
        summaries.append(summarize(res))
        everything.extend(res)
    print_table(summaries)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"base_url": a.base_url, "api": a.api, "concurrency": a.concurrency, "max_tokens": a.max_tokens,
                       "summaries": summaries, "results": [asdict(r) for r in everything]}, f, indent=1)
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(everything[0]).keys()))
            w.writeheader()
            w.writerows(asdict(r) for r in everything)
    return 0 if all(s["success_rate"] == 100.0 for s in summaries) else 1


if __name__ == "__main__":
    sys.exit(main())
