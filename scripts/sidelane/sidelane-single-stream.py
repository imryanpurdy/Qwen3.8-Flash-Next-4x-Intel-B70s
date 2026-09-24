#!/usr/bin/env python3
"""sidelane-single-stream.py — single-stream latency, production method.

Contract (mirrors tests/verify.sh "## 5. Single-stream"):
  * N=20 SEQUENTIAL /v1/chat/completions (no concurrency)
  * FIRST request is the COLD measurement and is DISCARDED
  * per-request rate = completion_tokens / wall_seconds
  * reported = median over the remaining 19, plus min/max and the discarded
    value; every run-pertinent value printed (RUN_i_tokps/ctok/s)
  * prompt and max_tokens identical to production verify.sh row 5
    ("Summarize the plot of Romeo and Juliet in a few paragraphs.", 600)
  * gate = 20/20 successful requests (any error fails the gate)
"""
import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request

PROMPT = "Summarize the plot of Romeo and Juliet in a few paragraphs."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--model", default="qwen-256k")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

    base = "http://%s:%d/v1/chat/completions" % (args.host, args.port)
    print("HARNESS=sidelane-single-stream.py (mirror of tests/verify.sh row 5)", flush=True)
    print("ENDPOINT=%s" % base, flush=True)
    print("MODEL=%s" % args.model, flush=True)
    print("FORMULA=rate=completion_tokens/wall per request; MEDIAN over 19 (first of 20 DISCARDED as cold)", flush=True)
    print("PROMPT=%s" % PROMPT, flush=True)
    print("PARAMS=runs=20 max_tokens=%d temperature=0.0 top_p=1.0 top_k=-1 min_p=0.0 presence_penalty=0.0 repetition_penalty=1.0" % args.max_tokens, flush=True)

    token_bytes = json.dumps({
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
    }).encode("utf-8")

    lat = []
    errors = 0
    for i in range(1, args.runs + 1):
        try:
            req = urllib.request.Request(base, data=token_bytes, method="POST",
                                         headers={"Content-Type": "application/json"})
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=args.timeout) as r:
                o = json.loads(r.read().decode("utf-8", errors="replace"))
            wall = time.perf_counter() - t0
            ct = int((o.get("usage") or {}).get("completion_tokens", 0))
            pt = int((o.get("usage") or {}).get("prompt_tokens", 0))
            rate = ct / wall if wall > 0 else 0.0
            latch = rate if ct > 0 else 0.0
            lat.append(latch)
            print("RUN_%d_tokps=%.3f RUN_%d_ctok=%d RUN_%d_ptok=%d RUN_%d_s=%.3f" % (i, rate, i, ct, i, pt, i, wall), flush=True)
        except Exception as exc:
            errors += 1
            print("RUN_%d_err=%s" % (i, str(exc).replace("\n", " ")[:200]), flush=True)

    if lat:
        discard, rest = lat[0], lat[1:]
        med = statistics.median(rest) if rest else float("nan")
        print("discarded_first=%.1f" % discard, flush=True)
        print("median19=%.1f" % med, flush=True)
        print("min1=%.1f max19=%.1f" % (min(rest) if rest else 0.0, max(rest) if rest else 0.0), flush=True)
        print("N_OK=%d ERRORS=%d" % (len(rest), errors), flush=True)
    else:
        print("median19=nan N_OK=0 ERRORS=%d" % errors, flush=True)

    return 0 if (errors == 0 and len(lat) == args.runs) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
