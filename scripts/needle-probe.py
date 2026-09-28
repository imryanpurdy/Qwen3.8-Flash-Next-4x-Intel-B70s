#!/usr/bin/env python3
"""needle-probe.py — MODEL-selection wrapper around needle_probe.py (v2.1).

needle_probe.py v2.1 (the engine-calibrated, self-correcting needle harness)
hardcodes MODEL to "qwen3.8-flash-next" (the other served alias). This wrapper
overrides it from the NEEDLE_MODEL_ID env var (default "qwen-256k", the name
this kit serves) and executes the unchanged main() — every measurement
contract of v2.1 is preserved verbatim (engine-confirmed usage.prompt_tokens
sizing, 400-length self-correction, PREPEND salt, CORRECT/SIZE_OK/
STAGING_NEW/TTFT output).

Finds needle_probe.py in (in order):
  1. $NEEDLE_PROBE_DIR                (explicit)
  2. this repo's scripts/             (same layout as this file)
  3. ./vendor/qwen38-flash-next/scripts (vendored copy)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_needle_probe_dir():
    cands = [
        os.environ.get("NEEDLE_PROBE_DIR", ""),
        HERE,                                                      # repo layout: this file lives next to needle_probe.py
        os.path.join(HERE, "vendor", "qwen38-flash-next", "scripts"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "needle_probe.py")):
            return os.path.abspath(c)
    return None


def main():
    d = find_needle_probe_dir()
    if not d:
        print("ERROR=needle_probe.py v2.1 not found; set NEEDLE_PROBE_DIR to the "
              "directory containing needle_probe.py", flush=True)
        return 2
    sys.path.insert(0, d)
    import needle_probe  # noqa: E402  (must import after sys.path fix)

    model = os.environ.get("NEEDLE_MODEL_ID", "qwen-256k")
    needle_probe.MODEL = model
    print("WRAPPER=needle_probe_v2.1 MODEL_PATCHED=%s SRC=%s" % (model, d), flush=True)
    return needle_probe.main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
