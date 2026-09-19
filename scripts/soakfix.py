#!/usr/bin/env python3
# Timer-fixed sustained soak: N-way x 4 rounds x 600 tok, Python-side timing (no bash date math).
import json, sys, time, glob, os
import urllib.request, concurrent.futures

URL = "http://localhost:8021/v1/completions"
PAYLOAD = json.dumps({"model": "qwen3.8-flash-next",
    "prompt": "The history of computing began when humans first learned to count. Write a detailed technical essay about the development of computing machinery.",
    "max_tokens": 600, "temperature": 0}).encode()

def one(i, tag):
    req = urllib.request.Request(URL, data=PAYLOAD, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=500) as r:
            d = json.loads(r.read())
            return d.get("usage", {}).get("completion_tokens", 0), 0
    except Exception:
        return 0, 1

def round_run(n, tag):
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(lambda i: one(i, tag), range(n)))
    wall = time.time() - t0
    toks = sum(t for t, e in res)
    errs = sum(e for t, e in res)
    agg = toks / wall if wall > 0 else 0
    print(f"{tag} round: tok={toks} errs={errs} wall={wall:.2f}s agg={agg:.1f} per_stream={toks/wall/n:.1f}")
    return agg, errs, wall

if __name__ == "__main__":
    n = int(sys.argv[1])
    aggs = []
    sustains = []  # r2-r4 = honest steady-decode result; r1 = warmup (all-N prefill artifact)
    ok = True
    for r in range(4):
        agg, errs, wall = round_run(n, f"r{r+1}")
        aggs.append(agg)
        if r > 0:
            sustains.append(agg)
        ok = ok and errs == 0 and wall < 400
    # liveness
    try:
        with urllib.request.urlopen("http://localhost:8021/v1/models", timeout=8) as resp:
            post = resp.status
    except Exception:
        post = 0
    to_out = sum(aggs) / len(aggs)
    sus_out = sum(sustains) / len(sustains) if sustains else 0.0
    print(f"summary: sustained_agg={sus_out:.1f} (r2-r4) mean_agg={to_out:.1f} min_agg={min(aggs):.1f} max_agg={max(aggs):.1f} post_models={post}")
    open(f"/tmp/soakfix_{n}way.json", "w").write(json.dumps(
        {"n": n, "rounds": aggs, "mean": to_out, "sustained": sus_out, "post_models": post, "clean": ok}))
