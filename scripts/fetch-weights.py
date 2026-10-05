#!/usr/bin/env python3
"""fetch-weights.py — one-time weights/PLE bootstrap for the b70-lumnus lane.

Three steps, each idempotent (skips when its output already exists and verifies):
  1. download  wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16 @ 0939125 (--local-dir, no
     symlink-forest snapshot) and devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18d
     (the BF16 PLE table source AND the documented rollback checkpoint).
  2. snapshot  tools/awq_snapshot.py snapshot <awq-dir> <bf16-table> <out>
     (from the Lumnus repo clone — the serve snapshot the entrypoint consumes).
  3. int8-ple  tools/build_int8_ple.py build + verify (deterministic build; the
     verify step re-quantizes stratified sample rows and cross-checks against BF16).

Environment (or .env): HF_TOKEN (optional; public repo), AWQ_DIR, BF16_DIR
(rollback tree; carries ple_table_qwen4exp.pt), SNAPSHOT_DIR, INT8_PLE_DIR,
LUMNUS_CLONE_DIR (default <repo>/.build/lumnus-b70-flash-next — created by
scripts/build-image.sh).

Pinned revisions (identity of record):
  wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16  @ 0939125
  devan-carlin/Qwen3.8-Flash-Next-W4A16 @ 40b8f18d
Verified artifacts (sha256 of record):
  BF16 PLE table  ple_table_qwen4exp.pt : 1d12b3952c2ed42e50d2b22325556352ac10b41e558c446b9a8e51879c92b3b3
  INT8 PLE table  ple_ngram_int8_rowscale.safetensors : 038b8458a5c581be1cf5e761e90f69e0c825962e3e3fa41bafcf65ba0f334af0 (48.9 GiB)
  AWQ serve config (written by snapshot serve-config): c7a2b345927976d911cfd57d1083b71d1a75fee245f61a17b8f126b6717342c8

License: the wtdcode checkpoint carries no license tag on Hugging Face —
license to confirm before any redistribution. devan-carlin's tree and the
Lumnus tooling are Apache-2.0/MIT as documented in the README credits.
"""
import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AWQ_REPO = "wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16"
AWQ_REV = "0939125"
BF16_REPO = "devan-carlin/Qwen3.8-Flash-Next-W4A16"
BF16_REV = "40b8f18d"

BF16_TABLE_SHA = "1d12b3952c2ed42e50d2b22325556352ac10b41e558c446b9a8e51879c92b3b3"
INT8_TABLE_SHA = "038b8458a5c581be1cf5e761e90f69e0c825962e3e3fa41bafcf65ba0f334af0"
SERVE_CONFIG_SHA = "c7a2b345927976d911cfd57d1083b71d1a75fee245f61a17b8f126b6717342c8"

AWQ_DIR = os.environ.get("AWQ_DIR", "/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16")
BF16_DIR = os.environ.get("BF16_DIR", "/srv/hf-devan/Qwen3.8-Flash-Next-W4A16")
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "/data/awq-snapshot")
INT8_DIR = os.environ.get("INT8_PLE_DIR", "/data/int8-ple")
CLONE = os.environ.get("LUMNUS_CLONE_DIR", os.path.join(REPO, ".build", "lumnus-b70-flash-next"))

BF16_TABLE = os.path.join(BF16_DIR, "ple_table_qwen4exp.pt")
INT8_TABLE = os.path.join(INT8_DIR, "ple_ngram_int8_rowscale.safetensors")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def info(msg): print(f"[INFO]  {msg}", flush=True)
def ok(msg):   print(f"[ OK ]  {msg}", flush=True)
def err(msg):
    print(f"[ERR ]  {msg}", flush=True)
    sys.exit(1)


def hf_download(repo: str, rev: str, dest: str) -> None:
    """Download a pinned revision with snapshot_download (real files, no symlink forest).

    Uses the Python API, not the CLI: the `huggingface_hub.commands.huggingface_cli`
    module was removed in huggingface_hub 1.0 (the CLI now lives at `hf`), and a
    fresh `pip install huggingface_hub` on a new host would crash the bootstrap.
    """
    info(f"Downloading {repo} @ {rev} -> {dest}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        err("huggingface_hub not installed — pip install huggingface_hub (needs >=0.20)")
    try:
        snapshot_download(
            repo_id=repo,
            revision=rev,
            local_dir=dest,
            token=os.environ.get("HF_TOKEN") or None,
        )
    except Exception as e:  # network/auth/disk failures surface here
        err(f"download of {repo} failed: {e}. Check HF_TOKEN / disk space.")


def step_download():
    if os.path.isfile(os.path.join(AWQ_DIR, "config.json")) and \
       any(f.endswith(".safetensors") for f in os.listdir(AWQ_DIR)):
        ok(f"AWQ tree present: {AWQ_DIR}")
    else:
        hf_download(AWQ_REPO, AWQ_REV, AWQ_DIR)
    if os.path.isfile(BF16_TABLE):
        ok(f"BF16 PLE table present: {BF16_TABLE}")
    else:
        # The rollback tree is large; fetch only what the INT8 build + rollback need.
        hf_download(BF16_REPO, BF16_REV, BF16_DIR)


def step_snapshot():
    if os.path.isdir(SNAPSHOT_DIR) and os.path.isfile(os.path.join(SNAPSHOT_DIR, "model.safetensors.index.json")):
        ok(f"Snapshot present: {SNAPSHOT_DIR}")
    else:
        tool = os.path.join(CLONE, "tools", "awq_snapshot.py")
        os.path.isfile(tool) or err(f"awq_snapshot.py not found at {tool} — run scripts/build-image.sh first (it clones the Lumnus repo at the pin).")
        info(f"Snapshot: {tool} snapshot {AWQ_DIR} {BF16_TABLE} {SNAPSHOT_DIR}")
        r = subprocess.run([sys.executable, tool, "snapshot", AWQ_DIR, BF16_TABLE, SNAPSHOT_DIR])
        r.returncode == 0 or err("snapshot failed.")
    # Identity: the PLE symlink must resolve on the host, and the table must be the pinned one.
    ple = os.path.join(SNAPSHOT_DIR, "ple_table_qwen4exp.pt")
    os.path.isfile(ple) or err(f"snapshot PLE table missing: {ple}")
    if os.path.islink(ple) and not os.path.exists(ple):
        err(f"snapshot PLE symlink dangles: {ple} -> {os.readlink(ple)}")
    actual = sha256(ple)
    actual == BF16_TABLE_SHA or err(f"PLE table sha mismatch: {actual} != {BF16_TABLE_SHA}")
    ok(f"PLE table sha256 verified ({actual[:12]}…)")


def step_int8():
    os.makedirs(INT8_DIR, exist_ok=True)
    if os.path.isfile(INT8_TABLE):
        ok(f"INT8 table present: {INT8_TABLE}")
    else:
        tool = os.path.join(CLONE, "tools", "build_int8_ple.py")
        os.path.isfile(tool) or err(f"build_int8_ple.py not found at {tool} — run scripts/build-image.sh first.")
        env = dict(os.environ, BF16_PT=BF16_TABLE, OUT_DIR=INT8_DIR)
        info(f"Building INT8 PLE table (streams the 95 GiB BF16 table; nice/ionice it if the host is busy)")
        r = subprocess.run([sys.executable, tool, "build"], env=env)
        r.returncode == 0 or err("INT8 build failed.")
    # Verify: the tool's own verify (stratified re-quantisation cross-check)...
    tool = os.path.join(CLONE, "tools", "build_int8_ple.py")
    info("Running build_int8_ple.py verify (BF16 cross-check)...")
    env = dict(os.environ, BF16_PT=BF16_TABLE, OUT_DIR=INT8_DIR)
    r = subprocess.run([sys.executable, tool, "verify"], env=env)
    r.returncode == 0 or err("INT8 verify failed.")
    # ...and the sha256 of record.
    actual = sha256(INT8_TABLE)
    actual == INT8_TABLE_SHA or err(f"INT8 table sha mismatch: {actual} != {INT8_TABLE_SHA}")
    ok(f"INT8 table sha256 verified ({actual[:12]}…)")


def step_serve_config():
    """The AWQ serve config the entrypoint expects — byte-exact with production."""
    out = os.path.join(REPO, "serve-config.json")
    if os.path.isfile(out):
        actual = sha256(out)
        if actual == SERVE_CONFIG_SHA:
            ok(f"serve-config.json verified ({actual[:12]}…)")
            return
        err(f"serve-config.json exists with sha {actual} != of-record {SERVE_CONFIG_SHA} — investigate before overwriting.")
    tool = os.path.join(CLONE, "tools", "awq_snapshot.py")
    image_cfg = os.path.join(CLONE, "image", "files", "opt", "b70-flashnext", "serve-config.json")
    os.path.isfile(image_cfg) or err(f"image serve-config not found at {image_cfg}")
    info("Writing AWQ serve-config (awq_snapshot.py serve-config)...")
    r = subprocess.run([sys.executable, tool, "serve-config", image_cfg, out])
    r.returncode == 0 or err("serve-config generation failed.")
    actual = sha256(out)
    actual == SERVE_CONFIG_SHA or err(f"generated serve-config sha {actual} != of-record {SERVE_CONFIG_SHA}")
    ok(f"serve-config.json written and verified ({actual[:12]}…)")


if __name__ == "__main__":
    info("=== Weights / PLE bootstrap (idempotent) ===")
    step_download()
    step_snapshot()
    step_int8()
    step_serve_config()
    ok("Bootstrap complete. Point .env at these paths and run ./scripts/start.sh")
