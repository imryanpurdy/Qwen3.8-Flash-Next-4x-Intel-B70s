#!/usr/bin/env python3
"""needle_probe.py — engine-calibrated needle-in-haystack probe (v2.1, 2026-09-23)

History:
  v1    sized by hardcoded CHARS_PER_TOKEN=3.7 -> served only ~65K engine
        tokens when asked for "98K" (this filler is ~5.59 chars/token on the
        Qwen tokenizer). 2026-09-23 fresh-clone acceptance caught it.
  v2    calibrated against the engine (usage.prompt_tokens) but its first
        iteration could overshoot max_model_len by <1% -> HTTP 400 from
        vLLM's context-length validation.
  v2.1  self-corrects against the limit itself: on a 400 it parses the
        engine's own "maximum context length ... you requested N tokens"
        body, recomputes the exact chars/token ratio from N, steps the
        target safely below the ceiling, and retries.

The gate number is ALWAYS the engine-confirmed token count
(usage.prompt_tokens, or the 400 body's N), never an estimate.

PASS = prompt_tokens >= --min-prompt-tokens
       AND needle code found in the reply (content OR reasoning_content)
       AND zero NEW PLE staging timeouts in the server log.

Stdlib only.  Never writes to the log or the server.
"""
import argparse
import datetime
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request

MODEL = "qwen3.8-flash-next"
CODE = "QRX-88-SHELDON"
NEEDLE = "The procurement code for the Meridian account is QRX-88-SHELDON."
QUESTION = "What is the procurement code for the Meridian account? Reply with only the code."
DEFAULT_LOG = "/home/bonz/rollback-unit/.run/server.log"
INITIAL_CHARS_PER_TOKEN = 5.59   # 2026-09-23 engine calibration (v1 used 3.7)
MAX_TOKEN_MARGIN = 0.005         # converged when |measured-target|/target <= 0.5%
ITERATIONS = 6
LOG_TAIL_BYTES = 262144

# vLLM 400 bodies (two formats seen 2026-09-23):
#  classic: "This model's maximum context length is 98304 tokens. However,
#           you requested 98814 tokens (42 in the messages, 98772 in the
#           completion). ..."
#  w/ max_tokens: "... maximum context length is 98304 tokens. However, you
#           requested 256 output tokens and your prompt contains at least
#           98049 input tokens, for a total of at least 98305 tokens. ..."
LEN_CLASSIC_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?you requested (\d+) tokens"
    r"(?:\s*\((\d+) in the messages)?", re.S)
LEN_SPLIT_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?you requested (\d+) output tokens"
    r".*?prompt contains at least (\d+) input tokens", re.S)

FILLER = (
    "The Meridian account covers quarterly procurement of instrumentation: "
    "calibration rigs, sensor pods, and spare flight units. Every line item "
    "crosses the compliance desk before the purchase order clears, and each "
    "batch is reconciled against the previous cycle's consumption spread. "
    "Maintenance windows are reserved for Thursdays, and the audit trail is "
    "kept in the shared vault with a thirty-day retention policy. This is "
    "filler context that does not change between requests."
)


def build_prompt(target_tokens, chars_per_token, salt=None):
    overhead = len(NEEDLE) + len(QUESTION) + 8
    if salt:
        overhead += len(salt) + 2
    chars_needed = max(1, int(target_tokens * chars_per_token) - overhead)
    n = max(2, int(math.ceil(chars_needed / float(len(FILLER)))))
    depth = int(n * 0.5)
    blocks = [FILLER] * n
    if salt:
        # PREPEND: a differing FIRST block breaks vLLM's block-hash chain at
        # the root and forces a full prefill. (Appending left 99.9% of the
        # prompt identical — the prefix cache absorbed it: TTFT 4 s vs 289 s.)
        blocks.insert(0, "Session marker (unique per run; ignore its content "
                         "entirely): %s" % salt)
    blocks.insert(depth, NEEDLE)
    return "\n\n".join(blocks) + "\n\n" + QUESTION, n


def chat_completion(url, payload, timeout=3600):
    """Return (resp_dict, None) or (None, error_body_text)."""
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8")), None
    except urllib.error.HTTPError as e:
        try:
            return None, e.read().decode("utf-8", errors="replace")
        except Exception:
            return None, "HTTP %d (body unreadable)" % e.code


def tail_text(path, max_bytes=LOG_TAIL_BYTES):
    try:
        size = os.path.getsize(path)
        if size == 0:
            return ""
        start = max(0, size - max_bytes)
        with open(path, "rb") as f:
            f.seek(start)
            return f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


TS_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"|\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
)


def parse_timestamp(line):
    m = TS_RE.search(line)
    if not m:
        return None
    s = m.group(0).replace(",", ".").replace("T", " ")
    if re.match(r"^\d{2}-\d{2} ", s):
        s = "%d-%s" % (time.localtime().tm_year, s)
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.datetime.strptime(s, fmt).timestamp()
        except ValueError:
            continue
    return None


def count_staging_timeouts(path, since_epoch):
    count = 0
    matched = 0
    saw_ts = False
    for line in tail_text(path).splitlines():
        if "staging flag timeout" not in line:
            continue
        matched += 1
        ts = parse_timestamp(line)
        if ts is None:
            continue
        saw_ts = True
        if ts >= since_epoch:
            count += 1
    if count == 0 and not saw_ts and matched:
        return matched
    return count


def escape_one_line(text):
    return (
        text.replace("\\", "\\\\")
        .replace("\r", "\\r")
        .replace("\n", "\\n")
        .replace("\t", "\\t")
    )


def main():
    parser = argparse.ArgumentParser(description="Engine-calibrated needle probe (v2.1)")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--target-tokens", type=int, default=97800,
                        help="engine-confirmed prompt_tokens to serve (default 97800)")
    parser.add_argument("--min-prompt-tokens", type=int, default=97000,
                        help="gate floor on the ENGINE-CONFIRMED prompt_tokens")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="output budget per request (thinking template needs "
                             "room; 32 = FINISH_REASON=length with empty reply)")
    parser.add_argument("--salt", default=None,
                        help="unique marker appended to the prompt; forces a "
                             "FULL prefill (defeats the prefix cache) so "
                             "back-to-back needles genuinely re-prefill")
    args = parser.parse_args()

    url = args.endpoint or ("http://localhost:%d/v1/chat/completions" % args.port)
    wall0 = time.time()
    cpt = INITIAL_CHARS_PER_TOKEN
    prompt_tokens = -1
    prompt = ""
    n = 0
    resp = None
    wall_s = 0.0
    for it in range(1, ITERATIONS + 1):
        prompt, n = build_prompt(args.target_tokens, cpt, salt=args.salt)
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "stream": False,
        }
        t0 = time.monotonic()
        resp, err_body = chat_completion(url, payload, timeout=3600)
        wall_s = time.monotonic() - t0
        if resp is None:
            m = LEN_CLASSIC_RE.search(err_body or "")
            m2 = LEN_SPLIT_RE.search(err_body or "")
            if m2:  # format: "O output tokens ... at least P input tokens"
                limit, out_tok, in_tok = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
                print("ITER=%d HTTP400_LENGTH_LIMIT limit=%d engine_counted=%d out_budget=%d wall_s=%.1f"
                      % (it, limit, in_tok, out_tok, wall_s), flush=True)
                cpt = len(prompt) / float(in_tok)          # exact ratio, engine's own count
                args.target_tokens = min(args.target_tokens, in_tok - out_tok - 350)
                prompt_tokens = in_tok                      # engine-confirmed count
                continue
            if m:
                limit = int(m.group(1))
                requested = int(m.group(3) or (int(m.group(2)) - 32))
                print("ITER=%d HTTP400_LENGTH_LIMIT limit=%d engine_counted=%d wall_s=%.1f"
                      % (it, limit, requested, wall_s), flush=True)
                cpt = len(prompt) / float(requested)      # exact ratio, engine's own count
                args.target_tokens = min(args.target_tokens, requested - 350)
                prompt_tokens = requested                  # engine-confirmed count
                continue
            print("ERROR=HTTP: %s" % (err_body or "unknown")[:400], flush=True)
            return 1
        prompt_tokens = int((resp.get("usage") or {}).get("prompt_tokens", -1))
        print("ITER=%d est_chars=%d engine_prompt_tokens=%d wall_s=%.1f"
              % (it, len(prompt), prompt_tokens, wall_s), flush=True)
        if prompt_tokens < 0:
            print("ERROR=usage_missing", flush=True)
            return 1
        drift = abs(prompt_tokens - args.target_tokens) / float(args.target_tokens)
        if drift <= MAX_TOKEN_MARGIN:
            break
        cpt = len(prompt) / float(prompt_tokens)  # exact ratio from THIS tokenizer

    if resp is None:
        print("ERROR=no_converged_response prompt_tokens_last=%d" % prompt_tokens, flush=True)
        return 1

    try:
        msg = resp["choices"][0]["message"]
        content = msg.get("content") or ""
        reasoning = msg.get("reasoning_content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        content, reasoning = "", ""
    finish = ((resp.get("choices") or [{}])[0].get("finish_reason", "?"))
    staging = count_staging_timeouts(args.log, since_epoch=wall0)

    reply_all = content + "\n" + reasoning
    print("NEEDLE_N_blocks=%d" % n, flush=True)
    print("NEEDLE_reply_text=%s" % escape_one_line(content), flush=True)
    if reasoning:
        print("NEEDLE_reasoning_text=%s" % escape_one_line(reasoning[:300]), flush=True)
    print("FINISH_REASON=%s" % finish, flush=True)
    print("ENGINE_PROMPT_TOKENS=%d" % prompt_tokens, flush=True)
    print("CORRECT=%s" % ("YES" if CODE in reply_all else "NO"), flush=True)
    print("TTFT=%.2f" % wall_s, flush=True)
    print("STAGING_NEW=%d" % staging, flush=True)

    size_ok = prompt_tokens >= args.min_prompt_tokens
    print("SIZE_OK=%s" % ("YES" if size_ok else "NO"), flush=True)
    if not size_ok:
        print("GATE_NOTE=engine served only %d of >=%d requested tokens"
              % (prompt_tokens, args.min_prompt_tokens), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
