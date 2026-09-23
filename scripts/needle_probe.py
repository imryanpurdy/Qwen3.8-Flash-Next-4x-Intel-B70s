#!/usr/bin/env python3
"""needle_probe.py

Needle-in-haystack probe for a vLLM OpenAI-compatible server
(model qwen3.8-flash-next, default port 8021).

Builds a ~--target-tokens prompt from an in-file filler paragraph with the
needle sentence "The procurement code for the Meridian account is
QRX-88-SHELDON." buried at 50% depth, then asks for the code in one
non-streaming request (temperature 0, max_tokens 32, timeout 3600 s) and
checks whether the model answers with the code.  After the request, the
server log tail (read-only, --log) is grepped for NEW "staging flag
timeout" lines (timestamped after the send started) and the count is
printed as STAGING_NEW.

Usage:  python3 needle_probe.py [--port 8021] [--log /path/to/server.log]
                    [--target-tokens 170000]

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
import urllib.request

MODEL = "qwen3.8-flash-next"
CODE = "QRX-88-SHELDON"
NEEDLE = "The procurement code for the Meridian account is QRX-88-SHELDON."
QUESTION = "What is the procurement code for the Meridian account? Reply with only the code."
DEFAULT_LOG = "/home/bonz/rollback-unit/.run/server.log"
CHARS_PER_TOKEN = 3.7  # rough estimate
LOG_TAIL_BYTES = 262144  # 256 KiB tail is plenty for the grep

FILLER = (
    "The Meridian account covers quarterly procurement of instrumentation: "
    "calibration rigs, sensor pods, and spare flight units. Every line item "
    "crosses the compliance desk before the purchase order clears, and each "
    "batch is reconciled against the previous cycle's consumption spread. "
    "Maintenance windows are reserved for Thursdays, and the audit trail is "
    "kept in the shared vault with a thirty-day retention policy. This is "
    "filler context that does not change between requests."
)


def build_prompt(target_tokens):
    """Return (prompt, n_filler_blocks) for ~target_tokens total."""
    overhead = len(NEEDLE) + len(QUESTION) + 8  # separators
    chars_needed = max(1, int(target_tokens * CHARS_PER_TOKEN) - overhead)
    n = max(1, int(math.ceil(chars_needed / float(len(FILLER)))))
    depth = int(n * 0.5)  # needle at 50% depth (block index)
    blocks = [FILLER] * n
    blocks.insert(depth, NEEDLE)
    prompt = "\n\n".join(blocks) + "\n\n" + QUESTION
    return prompt, n


def chat_completion(url, payload, timeout=3600):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def tail_text(path, max_bytes=LOG_TAIL_BYTES):
    """Read-only tail of the log file (missing/unreadable -> empty string)."""
    try:
        size = os.path.getsize(path)
        if size == 0:
            return ""
        start = max(0, size - max_bytes)
        with open(path, "rb") as f:
            f.seek(start)
            raw = f.read()
        return raw.decode("utf-8", errors="replace")
    except OSError:
        return ""


TS_RE = re.compile(
    r"(?:\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
    r"|\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
)


def parse_timestamp(line):
    """Epoch (local clock) of a log line timestamp; None if unparseable.

    Handles vLLM's default "MM-DD HH:MM:SS,mmm" prefix (current year assumed)
    as well as ISO "YYYY-MM-DD HH:MM:SS(.mmm)".  Assumes probe and server
    share a clock (run the probe on the same host as the server).
    """
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
    """Count "staging flag timeout" lines timestamped >= since_epoch.

    If no matched line carries a parseable timestamp, falls back to the raw
    count of matches in the log tail.
    """
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
    parser = argparse.ArgumentParser(description="Needle-in-haystack probe")
    parser.add_argument("--port", type=int, default=8021, help="vLLM server port")
    parser.add_argument("--endpoint", default=None,
                        help="full chat completions URL; overrides --port")
    parser.add_argument("--log", default=DEFAULT_LOG, help="server log path (read-only)")
    parser.add_argument("--target-tokens", type=int, default=170000,
                        help="approx prompt length in tokens (default 170000)")
    args = parser.parse_args()

    prompt, n = build_prompt(args.target_tokens)
    url = args.endpoint or ("http://localhost:%d/v1/chat/completions" % args.port)
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }

    print("NEEDLE_N_blocks=%d" % n, flush=True)
    print("NEEDLE_est_chars=%d" % len(prompt), flush=True)
    print("NEEDLE_est_tokens=%d" % int(len(prompt) / CHARS_PER_TOKEN), flush=True)

    wall0 = time.time()  # send-start timestamp for the log greps
    t0 = time.monotonic()
    resp = chat_completion(url, payload, timeout=3600)
    wall_s = time.monotonic() - t0
    try:
        content = resp["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    prompt_tokens = int((resp.get("usage") or {}).get("prompt_tokens", -1))
    staging = count_staging_timeouts(args.log, since_epoch=wall0)

    print("NEEDLE_reply_text=%s" % escape_one_line(content), flush=True)
    print("CORRECT=%s" % ("YES" if CODE in content else "NO"), flush=True)
    print("TTFT=%.2f" % wall_s, flush=True)
    print("prompt_tokens=%d" % prompt_tokens, flush=True)
    print("STAGING_NEW=%d" % staging, flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
