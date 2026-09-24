#!/usr/bin/env python3
"""sidelane-needle-probe.py — thin MODEL-patching wrapper around the UNCHANGED
flashnext-recipe/scripts/needle_probe.py (v2.1).

Why a wrapper: needle_probe.py v2.1 (the engine-calibrated, self-correcting
needle harness) has MODEL hardcoded to "qwen3.8-flash-next" (production
alias). The side-lane engine on :8022 serves the electric-sheep Flash-Next
alias (default "qwen-256k"), so we patch needle_probe.MODEL from the
SIDELANE_MODEL_ID env var and execute the original main() — every measurement
contract of v2.1 is preserved verbatim (engine-confirmed usage.prompt_tokens
sizing, 400-length self-correction, PREPEND salt, CORRECT/SIZE_OK/
STAGING_NEW/TTFT output). The original file is never modified.

Finds the original in (in order):
  1. $NEEDLE_PROBE_DIR                (explicit)
  2. <sidelane dir>/../flashnext-recipe/scripts   (repo siblings)
  3. <sidelane dir>/vendor/flashnext-recipe/scripts
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def find_needle_probe_dir():
    cands = [
        os.environ.get("NEEDLE_PROBE_DIR", ""),
        os.path.join(HERE, ".."),                                  # repo layout: scripts/sidelane/ -> scripts/
        os.path.join(HERE, "..", "flashnext-recipe", "scripts"),
        os.path.join(HERE, "vendor", "flashnext-recipe", "scripts"),
    ]
    for c in cands:
        if c and os.path.isfile(os.path.join(c, "needle_probe.py")):
            return os.path.abspath(c)
    return None


def main():
    d = find_needle_probe_dir()
    if not d:
        print("ERROR=needle_probe.py v2.1 not found; set NEEDLE_PROBE_DIR to the "
              "flashnext-recipe/scripts directory", flush=True)
        return 2
    sys.path.insert(0, d)
    import needle_probe  # noqa: E402  (must import after sys.path fix)

    model = os.environ.get("SIDELANE_MODEL_ID", "qwen-256k")
    needle_probe.MODEL = model
    print("WRAPPER=needle_probe_v2.1 MODEL_PATCHED=%s SRC=%s" % (model, d), flush=True)
    return needle_probe.main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print("ERROR=%s: %s" % (type(exc).__name__, exc), flush=True)
        sys.exit(1)
