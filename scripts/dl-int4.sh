#!/bin/bash
# Download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound snapshot using HOST python3
# (network in the vLLM image is fine; its entrypoint wrapper just spams stdout).
set -uo pipefail
export HF_HOME=/home/bonz/hf-int4
mkdir -p "$HF_HOME"
python3 - <<'PY'
import sys
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("NO_HF_HUB_ON_HOST"); sys.exit(3)
p = snapshot_download(
    "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound",
    revision="4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897",  # PINNED 2026-09-20 pre-wipe; guarantees same weights (blobs are content-hash-named)
    max_workers=8,
)
print("DONE", p)
PY
