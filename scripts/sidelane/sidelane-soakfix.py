#!/usr/bin/env python3
"""sidelane-soakfix.py — thin variant of flashnext-recipe/scripts/soakfix.py
for the port-8022 side-lane engine. The ORIGINAL is NOT edited (it hardcodes
URL http://localhost:8021 and the production model name and takes no --port);
this file keeps the measurement contract byte-for-byte identical so the
numbers are comparable to production's.

Contract (identical to flashnext-recipe/scripts/soakfix.py):
  * N concurrent streams, 4 rounds, max_tokens=600 per stream
  * prompt: open-ended technical-essay instruction that runs TO the 600-token
    cap (early-EOS-shaped short replies would measure a different regime)
  * per-round aggregate: agg = sum(completion_tokens) / round_wall
    (Python-side timing, no shell date math)
  * round 1 is WARMUP and is DISCARDED;
    SUSTAINED = mean(agg of rounds 2..4)
  * reported per round and as a summary line with spread (min/max of aggs)
  * gate = clean: every round errs==0 AND wall < 400s AND /v1/models answers
    afterwards.

Sampling: temperature 0, and we EXPLICITLY pass presence_penalty=0 /
repetition_penalty=1.0 / top_p=1.0 / top_k=-1 / min_p=0.0. The side-lane
engine's launcher pins --override-generation-config to temperature 0.7,
top_p 0.80, top_k 20, presence_penalty 1.5, repetition_penalty 1.0; without
explicit neutralization every side-lane measurement would silently carry a
presence penalty production does not have (incomparable numbers).
"""
import argparse
import concurrent.futures
import json
import os
import sys
import tempfile
import time
import urllib.request

PROMPT = (
    "The history of computing began when humans first learned to count. "
    "Write a detailed technical essay about the development of computing "
    "machinery."
)


def one(url, payload, timeout):
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode("utf-8", errors="replace"))
        return d.get("usage", {}).get("completion_tokens", 0), 0
    except Exception:
        return 0, 1


def main():
    ap = argparse.ArgumentParser(description="Side-lane sustained soak (soakfix contract)")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8022)
    ap.add_argument("--model", default="qwen-256k")
    ap.add_argument("--n", type=int, default=16, help="concurrent streams (default 16)")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--timeout", type=float, default=500.0)
    ap.add_argument("--json-out", default=None,
                    help="path for the machine-readable JSON results file")
    args = ap.parse_args()

    url = "http://%s:%d/v1/completions" % (args.host, args.port)
    payload = json.dumps({
        "model": args.model,
        "prompt": PROMPT,
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "top_p": 1.0, "top_k": -1, "min_p": 0.0,
        "presence_penalty": 0.0, "repetition_penalty": 1.0,
    }).encode("utf-8")

    # comparability law: every number below is tagged with harness, formula,
    # prompt shape and sampling params
    print("HARNESS=sidelane-soakfix.py (thin variant of flashnext-recipe/scripts/soakfix.py)", flush=True)
    print("ENDPOINT=%s" % url, flush=True)
    print("MODEL=%s" % args.model, flush=True)
    print("FORMULA=agg=sum(completion_tokens)/round_wall; SUSTAINED=mean(agg rounds 2..4); round1=warmup DISCARDED", flush=True)
    print("PROMPT=open-ended technical-essay instruction ('The history of computing ...'), runs TO the %d-token cap" % args.max_tokens, flush=True)
    print("PARAMS=n=%d rounds=%d max_tokens=%d temperature=0.0 top_p=1.0 top_k=-1 min_p=0.0 presence_penalty=0.0 repetition_penalty=1.0" % (args.n, args.rounds, args.max_tokens), flush=True)

    aggs = []
    sustains = []
    rounds_meta = []
    ok = True
    for r in range(args.rounds):
        tag = "r%d" % (r + 1)
        t0 = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.n) as ex:
            res = list(ex.map(lambda _i: one(url, payload, args.timeout), range(args.n)))
        wall = time.time() - t0
        toks = sum(t for t, e in res)
        errs = sum(e for t, e in res)
        agg = toks / wall if wall > 0 else 0.0
        print("%s round: tok=%d errs=%d wall=%.2fs agg=%.1f per_stream=%.1f" % (tag, toks, errs, wall, agg, agg / args.n), flush=True)
        aggs.append(agg)
        rounds_meta.append({"round": r + 1, "tok": toks, "errs": errs, "wall": wall, "agg": agg})
        if r > 0:
            sustains.append(agg)
        ok = ok and errs == 0 and wall < 400

    # liveness
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (args.host, args.port), timeout=8) as resp:
            post = resp.status
    except Exception:
        post = 0
        ok = False

    to_out = sum(aggs) / len(aggs) if aggs else 0.0
    sus_out = sum(sustains) / len(sustains) if sustains else 0.0
    print("summary: sustained_agg=%.1f (r2-r4) mean_agg=%.1f min_agg=%.1f max_agg=%.1f post_models=%s clean=%s" % (sus_out, to_out, min(aggs) if aggs else 0, max(aggs) if aggs else 0, post, "YES" if ok else "NO"), flush=True)

    if args.json_out:
        path = args.json_out
    else:
        path = os.path.join(tempfile.gettempdir(), "sidelane_soakfix_%dway.json" % args.n)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"n": args.n, "rounds": rounds_meta, "mean": to_out,
                       "sustained": sus_out, "post_models": post, "clean": ok}, fh)
        print("JSON_OUT=%s" % path, flush=True)
    except OSError as exc:
        print("JSON_OUT_ERROR=%s" % exc, flush=True)

    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
