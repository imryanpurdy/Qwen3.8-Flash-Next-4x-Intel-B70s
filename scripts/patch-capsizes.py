#!/usr/bin/env python3
# patch-capsizes.py — v24h: capture EVERY decode batch size 1..MAX_NUM_SEQS.
# Why: an aligned decode batch at an uncaptured size falls back to eager on the
# XPU FULL_DECODE_ONLY path (12-way cliff: MNS=12 with list [1,2,4,8,16,24]
# pinned 12-way at 30.4 agg vs 163+ at 8-way; fast first rounds = staggered
# batches landed on captured sizes). Generating the list through MNS makes the
# trap structurally impossible; the guard refuses to boot if generation drifts.
import hashlib, sys, io

P = "/home/bonz/fn-recipe-int4/start.sh"
src = open(P, "r", encoding="utf-8", newline="").read()
h0 = hashlib.md5(src.encode()).hexdigest()

OLD = 'VLLM_ARGS+=(--compilation-config \'{\"cudagraph_mode\":\"FULL_DECODE_ONLY\"}\')'
NEW = '''# v24h (2026-09-19): capture EVERY decode batch size 1..MAX_NUM_SEQS.
# An aligned decode batch at an uncaptured size falls back to EAGER on the XPU
# full-graph path (12-way 5x cliff at MNS=12, list [1,2,4,8,16,24]; fast first
# rounds = staggered batches still landed on captured sizes). Generating
# through MNS makes the trap structurally impossible; the guard below refuses
# to boot if generation ever drifts.
CAP_SIZES_JSON="[$(seq -s, 1 "$MAX_NUM_SEQS")]"
case "$CAP_SIZES_JSON" in
  *",${MAX_NUM_SEQS}]"|"[${MAX_NUM_SEQS}]") : ;;
  *) err "v24h guard: capture list '${CAP_SIZES_JSON}' does not cover MAX_NUM_SEQS=${MAX_NUM_SEQS}"; exit 1 ;;
esac
VLLM_ARGS+=(--compilation-config "{\\"cudagraph_mode\\":\\"FULL_DECODE_ONLY\\",\\"cudagraph_capture_sizes\\":${CAP_SIZES_JSON}}")'''

n = src.count(OLD)
if n != 1:
    print(f"ABORT: marker found {n} times (need exactly 1); md5={h0}")
    sys.exit(1)
if "V24H_CAPSIZES" in src:
    print("ABORT: already patched")
    sys.exit(1)

out = src.replace(OLD, NEW + "\n# V24H_CAPSIZES\n")
open(P, "w", encoding="utf-8", newline="").write(out)
h1 = hashlib.md5(out.encode()).hexdigest()
print(f"PATCH_OK md5 {h0} -> {h1}")
print("V24H_CAPSIZES_APPLIED")
