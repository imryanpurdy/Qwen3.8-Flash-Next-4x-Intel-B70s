# First-Light Runbook — 2026-09-16 (post RAM upgrade)

Deploy of the Lane-5 A367 exact-GDN line on the rebuilt jobe box.
Everything below was executed and verified live; deviations from the kit
defaults are recorded with reasons.

## Box state at deploy
- Boot: SSD (`ssd-vg/root`, WDC 1TB SATA). NVMe = `/data` ext4 (weights).
- RAM: 123 GiB usable (was 30) — PLE 51.2 GiB pin + Lane-1 graphs retrial unblocked.
- Swap: 96G `/swapfile` on SSD, ON.
- GPUs: 4× B70 via `/dev/dri` renderD128–131.
- llama.cpp lane: container removed (retired). 27B image intact, not running.

## Kit
- Cloned to `~/fn-recipe` (repo `imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s`).
- **Image reused, not rebuilt**: `qwen38-flash-next-xpu:0.21.0-b1-rt2f829747`
  (25.4 GB, Steve's certified runtime stage `2f829747`) survived on the box —
  boots the exact certified kernel identity without a build.
- Stock `_xpu_C.abi3.so` in image sha256 `9d405591…`; A367 drop-in
  `593a7107…` bind-mounted over it (rescued copy at `~/xpu_artifacts/`,
  verified `import vllm_xpu_kernels` OK inside the image with GPUs attached).

## .env (Lane-5 first light)
Defaults from `.env.sample` changed:
- `MTP_NUM_SPECULATIVE_TOKENS=1` (Lane-5 certified line)
- `IMAGE=qwen38-flash-next-xpu:0.21.0-b1-rt2f829747`
- `PORT=8021` (keeps the established `http://100.122.128.100:8021` URL)
- `PREFLIGHT_DISK_GB=30` — floor checks FREE space on the weights mount;
  after 185.56 GB lands on the 222 GB `/data`, free ≈ 36 GB. 200 would false-fail.
- `HF_HOME=/data/hf`
- `EXTRA_DOCKER_ARGS` (see below)

## Kit bugs found + fixed (start.sh)
1. `--group-add render` → **docker hard-fails**: the image has no `render`
   group (only gid 991 on host = render). Patched to `--group-add 991`.
2. **No port publish** — `docker run` array has no `-p`; container would be
   bridge-isolated. Added `-p 8021:8021` via `EXTRA_DOCKER_ARGS`.
3. Selector envs + kernel drop-in ride `EXTRA_DOCKER_ARGS` (spliced before
   the image name, so both `-e`/`-v` work).

## EXTRA_DOCKER_ARGS (verbatim)
```
-p 8021:8021
-e VLLM_XPU_GDN_SERIAL_SPEC_DECODE=0
-e VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1
-e VLLM_SPU_GDN_SPEC_PERSISTENT_SCRATCH=1
-e VLLM_XPU_GDN_NATIVE_SPEC_COMPLETION_BARRIER=1
-v /home/bonz/xpu_artifacts/_xpu_C.abi3.so:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so:ro
-v /home/bonz/xpu_artifacts/libgrouped_gemm_xe_2.so:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgrouped_gemm_xe_2.so:ro
-v /home/bonz/xpu_artifacts/libgrouped_gemm_xe_default.so:/opt/venv/lib/python3.12/site-packages/vllm_xpu_kernels/libgrouped_gemm_xe_default.so:ro
```

## Weights
- Host had no HF CLI → installed `huggingface_hub[hf_transfer]` via pip
  (`--break-system-packages`), download driven by `~/dl-weights.py`
  (snapshot_download, HF_HOME=/data/hf, 8 workers, nohup + log).
- The image's python is unusable for downloads: its entrypoint imports vLLM
  which hard-fails without XPU context (`Failed to infer device type`).
- `start.sh` sees the completed snapshot (plain `source .env` propagates
  HF_HOME) and auto-skips its own download; `check-weights.sh` gates on
  identity `bcd9f01d…` + du -sL floor 170G.
- `/data` was root-owned post-Phase-D → `chown -R bonz:bonz /data`.

## Auto-launch
`~/firstlight.sh`: polls `dl-weights.log` for DOWNLOAD_COMPLETE (bails if
downloader dies), then runs `./start.sh` logging to `~/firstlight.log`.

## Measurement plan (Lane-5 protocol)
- Headline = median of prompt-class medians over 99 inter-token intervals
  after TTFT, cold 12-prompt realistic suite (harness `~/bench_flashnext.py`).
- Gates before quoting throughput: selector envs visible in container,
  MTP1 active in engine log, acceptance rate sane, then lane5 quality gates
  (exact-2K/4K byte pins are lab-side; reproduce measurement protocol and
  compare against 46.85 floor).
- Known receipt context: first-use rows ~29.3 vs warm ~47.2 (page-cache);
  warm the table before quoting.
