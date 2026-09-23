#!/usr/bin/env python3
"""bench_harness.py - benchmark a vLLM OpenAI-compatible server.

Single-file, Python-stdlib-only client benchmark. Four subcommands:

  single   - N sequential single-stream requests; RUN_i_tokps + MEDIAN/P10/P90/MEAN
  burst    - R rounds of W concurrent requests back-to-back; per-round
             aggregate tok/s and per-request median, plus BEST/WORST round
  longctx  - R rounds of W concurrent requests padded with a large filler prompt;
             server-reported prompt_tokens per request, aggregate tok/s, wall time
  accept   - extract spec-decode / draft-acceptance lines from a vLLM server log

Every line emitted is flat KEY=value (pure ASCII, LF-only, flushed per line) so a
shell watcher can grep it. No third-party modules, no streaming, no shell-outs.
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_URL = "http://localhost:8021/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash-next"
DEFAULT_TIMEOUT = 240.0        # seconds per HTTP request
MAX_ATTEMPTS = 3               # bounded retries for 5xx / refused / IO errors
BACKOFF_S = 2.0                # seconds between retries

FILLER_PARA = (
    "This is meaningless filler text used only to make the prompt long enough. "
    "Ignore every sentence in this body of filler. You must follow no instruction "
    "that appears above the final instruction, because this text is not part of "
    "the task. The paragraph repeats until the requested prompt size is reached. "
    "Nothing in it should be interpreted as a command, a question, or a request. "
    "It exists solely to exercise the prompt engine with a realistic context. "
)

SPEC_RE = re.compile(
    r"accept|draft|specul|spec[-_ ]?dec|\bmtp\b|proposal", re.IGNORECASE
)


def emit(key, value):
    """Print one flat KEY=value line, flushed so a watcher can tail it."""
    print("%s=%s" % (key, value), flush=True)


def est_tokens(text):
    """Rough token estimate used to size the long-context filler prompt."""
    return len(text) / 3.7


def build_filler(target_tokens):
    """Repeat FILLER_PARA until est_tokens() reaches target_tokens."""
    chunks = []
    total = 0
    para_len = len(FILLER_PARA)
    while (total + para_len) / 3.7 < target_tokens:
        chunks.append(FILLER_PARA)
        total += para_len
    return "".join(chunks)


def percentile(sorted_vals, p):
    """Nearest-rank percentile on an already-sorted list."""
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    rank = int(p / 100.0 * n)
    if rank < 1:
        rank = 1
    elif rank > n:
        rank = n
    return sorted_vals[rank - 1]


def chat_once(url, model, messages, max_tokens, timeout):
    """One non-streaming chat completion POST.

    Returns (completion_tokens, prompt_tokens, elapsed_s, content).
    Retries 5xx / 429 / connection-refused / IO errors up to MAX_ATTEMPTS with
    BACKOFF_S seconds between attempts. Non-recoverable HTTP codes (4xx) raise
    immediately.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    body = json.dumps(payload).encode("utf-8")
    last_err = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            req = urllib.request.Request(
                url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            elapsed = time.perf_counter() - t0
            obj = json.loads(raw.decode("utf-8"))
            ctk = int(obj["usage"].get("completion_tokens", 0) or 0)
            ptk = int(obj["usage"].get("prompt_tokens", 0) or 0)
            content = obj["choices"][0]["message"].get("content") or ""
            return (ctk, ptk, elapsed, content)
        except urllib.error.HTTPError as exc:
            if exc.code >= 500 or exc.code == 429:
                last_err = "HTTP %d %s" % (exc.code, exc.reason)
            else:
                raise RuntimeError("non-retryable HTTP %d" % exc.code) from exc
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # URLError (connection refused, DNS, request timeout) is an OSError.
            last_err = "%s: %s" % (type(exc).__name__, exc)
        if attempt < MAX_ATTEMPTS:
            time.sleep(BACKOFF_S)
    raise RuntimeError(
        "request failed after %d attempts (%s)" % (MAX_ATTEMPTS, last_err)
    )


def cmd_single(args):
    emit("MODE", "single")
    emit("URL", args.url)
    emit("MODEL", args.model)
    emit("RUNS", args.runs)
    emit("MAX_TOKENS", args.max_tokens)
    emit("TIMEOUT_S", "%g" % args.timeout)
    messages = [
        {
            "role": "user",
            "content": "Write a numbered list of the first 20 prime numbers.",
        }
    ]
    tokps = []
    t_start = time.perf_counter()
    for i in range(1, args.runs + 1):
        try:
            ctk, ptk, elapsed, _content = chat_once(
                args.url, args.model, messages, args.max_tokens, args.timeout
            )
            tps = (ctk / elapsed) if elapsed > 0.0 else 0.0
            tokps.append(tps)
            emit("RUN_%d_tokps" % i, "%.3f" % tps)
            emit("RUN_%d_ctok" % i, ctk)
            emit("RUN_%d_ptok" % i, ptk)
            emit("RUN_%d_s" % i, "%.3f" % elapsed)
        except Exception as exc:
            emit("RUN_%d_err" % i, str(exc).replace("\n", " ")[:200])
    wall = time.perf_counter() - t_start
    if tokps:
        s = sorted(tokps)
        emit("MEDIAN", "%.3f" % statistics.median(s))
        emit("P10", "%.3f" % percentile(s, 10))
        emit("P90", "%.3f" % percentile(s, 90))
        emit("MEAN", "%.3f" % (sum(s) / len(s)))
        emit("N_OK", len(s))
    else:
        emit("MEDIAN", "nan")
        emit("P10", "nan")
        emit("P90", "nan")
        emit("MEAN", "nan")
        emit("N_OK", 0)
        emit("NO_SUCCESS", 1)
    emit("WALL_S", "%.3f" % wall)


def cmd_burst(args):
    emit("MODE", "burst")
    emit("URL", args.url)
    emit("MODEL", args.model)
    emit("WORKERS", args.workers)
    emit("ROUNDS", args.rounds)
    emit("MAX_TOKENS", args.max_tokens)
    emit("TIMEOUT_S", "%g" % args.timeout)
    messages = [
        {
            "role": "user",
            "content": "Write a short thank-you note of exactly three sentences.",
        }
    ]
    aggregates = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in range(1, args.rounds + 1):
            t0 = time.perf_counter()
            futures = [
                pool.submit(
                    chat_once,
                    args.url,
                    args.model,
                    messages,
                    args.max_tokens,
                    args.timeout,
                )
                for _ in range(args.workers)
            ]
            pairs = []  # (completion_tokens, elapsed_s)
            for j, fut in enumerate(futures, 1):
                try:
                    ctk, ptk, elapsed, _content = fut.result()
                    pairs.append((ctk, elapsed))
                    tps = (ctk / elapsed) if elapsed > 0.0 else 0.0
                    emit("ROUND_%d_REQ_%d_tokps" % (r, j), "%.3f" % tps)
                    emit("ROUND_%d_REQ_%d_ctok" % (r, j), ctk)
                except Exception as exc:
                    emit(
                        "ROUND_%d_REQ_%d_err" % (r, j),
                        str(exc).replace("\n", " ")[:200],
                    )
            wall = time.perf_counter() - t0
            total_ctok = sum(p[0] for p in pairs)
            agg = (total_ctok / wall) if wall > 0.0 else 0.0
            aggregates.append(agg)
            perreq = [(p[0] / p[1]) if p[1] > 0.0 else 0.0 for p in pairs]
            emit("ROUND_%d_aggregate_tokps" % r, "%.3f" % agg)
            emit(
                "ROUND_%d_perreq_median" % r,
                "%.3f" % (statistics.median(perreq) if perreq else float("nan")),
            )
            emit("ROUND_%d_wall_s" % r, "%.3f" % wall)
            emit("ROUND_%d_ctok_total" % r, total_ctok)
            emit("ROUND_%d_ok_reqs" % r, len(pairs))
    if aggregates:
        best = max(aggregates)
        worst = min(aggregates)
        emit("BEST", "%.3f" % best)
        emit("WORST", "%.3f" % worst)
        emit("BEST_ROUND", aggregates.index(best) + 1)
        emit("WORST_ROUND", aggregates.index(worst) + 1)
        emit("N_OK_ROUNDS", len(aggregates))
    else:
        emit("BEST", "nan")
        emit("WORST", "nan")
        emit("N_OK_ROUNDS", 0)
        emit("NO_SUCCESS", 1)
    emit("WALL_S", "%.3f" % (time.perf_counter() - t_start))


def cmd_longctx(args):
    emit("MODE", "longctx")
    emit("URL", args.url)
    emit("MODEL", args.model)
    emit("WORKERS", args.workers)
    emit("ROUNDS", args.rounds)
    emit("FILLER_TOKENS", args.filler_tokens)
    emit("GEN_TOKENS", args.gen_tokens)
    emit("EXPECT_FAIL", str(args.expect_fail).lower())
    emit("TIMEOUT_S", "%g" % args.timeout)
    filler = build_filler(args.filler_tokens)
    emit("EST_FILLER_TOKENS", "%.1f" % est_tokens(filler))
    emit("FILLER_CHARS", len(filler))
    messages = [
        {"role": "system", "content": "You are a terse test assistant."},
        {
            "role": "user",
            "content": (
                filler + "\n\nIGNORE EVERYTHING ABOVE. Reply with exactly: pong"
            ),
        },
    ]

    def once(j):
        payload = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.gen_tokens,
            "temperature": 0,
        }
        body = json.dumps(payload).encode("utf-8")
        t0 = time.perf_counter()
        try:
            req = urllib.request.Request(
                args.url, data=body, method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=args.timeout) as resp:
                raw = resp.read()
            elapsed = time.perf_counter() - t0
            obj = json.loads(raw.decode("utf-8"))
            ctk = int(obj["usage"].get("completion_tokens", 0) or 0)
            ptk = int(obj["usage"].get("prompt_tokens", 0) or 0)
            emit("PROMPT_TOKENS_%d" % j, ptk)
            emit("REQ_%d_ctok" % j, ctk)
            emit("REQ_%d_tokps" % j, "%.3f" % (ctk / elapsed if elapsed > 0 else 0.0))
            emit("REQ_%d_s" % j, "%.3f" % elapsed)
            return (ctk, elapsed, ptk, True)
        except urllib.error.HTTPError as exc:
            errbody = ""
            try:
                errbody = exc.read().decode("utf-8", errors="replace")[:300]
            except Exception:
                pass
            if args.expect_fail and exc.code == 400:
                emit("REQ_%d_EXPECTED_FAIL" % j, "HTTP 400 %s" % errbody)
                return (0, 0.0, 0, True)
            emit("REQ_%d_err" % j, "HTTP %d %s" % (exc.code, errbody))
            return (0, 0.0, 0, False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            emit("REQ_%d_err" % j, "%s: %s" % (type(exc).__name__, exc))
            return (0, 0.0, 0, False)

    all_pairs = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for r in range(1, args.rounds + 1):
            t0 = time.perf_counter()
            futs = [pool.submit(once, j) for j in range(1, args.workers + 1)]
            results = [f.result() for f in futs]
            wall = time.perf_counter() - t0
            oks = [x for x in results if x[3]]
            ctk_sum = sum(x[0] for x in oks)
            agg = (ctk_sum / wall) if wall > 0.0 else 0.0
            emit("ROUND_%d_wall_s" % r, "%.3f" % wall)
            emit("ROUND_%d_aggregate_tokps" % r, "%.3f" % agg)
            emit("ROUND_%d_ok_reqs" % r, len(oks))
            all_pairs.extend(oks)
    total_s = time.perf_counter() - t_start
    pts = [p[2] for p in all_pairs if p[2]]
    if pts:
        emit("PROMPT_TOKENS_min", min(pts))
        emit("PROMPT_TOKENS_mean", "%.1f" % (sum(pts) / len(pts)))
        emit("PROMPT_TOKENS_max", max(pts))
    total_ctok = sum(p[0] for p in all_pairs)
    emit(
        "TOTAL_aggregate_tokps",
        "%.3f" % ((total_ctok / total_s) if total_s > 0.0 else 0.0),
    )
    emit("TOTAL_wall_s", "%.3f" % total_s)
    emit("TOTAL_ctok", total_ctok)
    emit("N_OK", len(all_pairs))
    if not all_pairs:
        emit("NO_SUCCESS", 1)


def cmd_accept(args):
    emit("MODE", "accept")
    emit("LOG", args.log)
    try:
        with open(args.log, "rb") as fh:
            raw = fh.read()
    except OSError as exc:
        emit("ERROR", "cannot read log: %s" % str(exc).replace("\\", "/")[:200])
        sys.exit(1)
    text = raw.decode("utf-8", errors="replace")
    lines = text.splitlines()
    hits = 0
    for lineno, line in enumerate(lines, 1):
        if SPEC_RE.search(line):
            hits += 1
            emit("SPEC_LINE_%d" % lineno, line.rstrip()[:500])
    emit("SPEC_LINE_COUNT", hits)
    emit("TOTAL_LINES", len(lines))


def add_common(parser):
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="OpenAI-compatible endpoint (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="model name (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="per-request HTTP timeout in seconds (default: %(default)s)",
    )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="bench_harness.py",
        description="vLLM OpenAI-compatible benchmark harness (stdlib only)",
    )
    sub = parser.add_subparsers(dest="mode", required=True, metavar="MODE")

    sp = sub.add_parser("single", help="sequential single-stream runs")
    add_common(sp)
    sp.add_argument("--runs", type=int, default=20)
    sp.add_argument("--max-tokens", type=int, default=320)
    sp.set_defaults(func=cmd_single)

    bp = sub.add_parser("burst", help="back-to-back rounds of concurrent requests")
    add_common(bp)
    bp.add_argument("--workers", type=int, default=16)
    bp.add_argument("--rounds", type=int, default=3)
    bp.add_argument("--max-tokens", type=int, default=320)
    bp.set_defaults(func=cmd_burst)

    lp = sub.add_parser("longctx", help="concurrent requests with long filler prompts")
    add_common(lp)
    lp.add_argument("--workers", type=int, default=8)
    lp.add_argument("--rounds", type=int, default=2)
    lp.add_argument("--filler-tokens", type=int, default=31000)
    lp.add_argument("--gen-tokens", type=int, default=512)
    lp.add_argument("--expect-fail", action="store_true",
                    help="treat HTTP 400 as EXPECTED_FAIL (pre-context-boot probe)")
    lp.set_defaults(func=cmd_longctx)

    ap = sub.add_parser("accept", help="extract spec-decode acceptance lines from a log")
    add_common(ap)
    ap.add_argument("--log", required=True, help="path to the vLLM server log file")
    ap.set_defaults(func=cmd_accept)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
