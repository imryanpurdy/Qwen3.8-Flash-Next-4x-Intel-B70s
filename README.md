# Qwen3.8-Flash-Next on 4x Intel Arc Pro B70

Serving kit for **Qwen3.8-Flash-Next AWQ W4A16** (`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ `0939125`) on **4x Intel Arc Pro B70 32GB, TP4+EP**, served by the **Lumnus `b70-flash-next` engine** (vLLM v0.30.0 `ced6857a` + patch series 0001–0019 — sub-lettered, 21 files; see `docs/engine/PROVENANCE.md`), 262,144-token context, INT8 PLE table served from NVMe, 64 GiB CPU KV tier, OpenAI-compatible API on port 8022 as `qwen-256k`.

**What changed in this revision:** the engine line moved from the `devan-carlin/vllm@xpu-qwen4exp` fork to the Lumnus `b70-flash-next` series (measured **+10.2 %** at 32 concurrent), the checkpoint is the AWQ build (long-context recall is why), and the PLE n-gram table is INT8 served from NVMe instead of 95.4 GiB of pinned BF16. The devan fork + W4A16 stack stays documented as the rollback (see Rollback).

## Quick start

Prerequisites (one-time host provisioning — `scripts/host-setup.sh` installs and verifies all of it; REBOOT required):

| Requirement | Value |
|---|---|
| GPU | 4x Intel Arc Pro B70 32GB (Xe2 / Battlemage) |
| Kernel | **6.17.0-1010-intel ONLY** — 7.x kernels are **banned** on this rig (they wedge permanently on B70; see Troubleshooting) |
| GuC firmware | 70.65 (linux-firmware `fb0889c0`, sha256-pinned by the setup script) |
| Kernel cmdline | `iommu=off` (grub) |
| Swap | ≥64 GiB on |
| Host RAM | ≥100 GiB free at boot |
| Container | Docker Engine + `buildx` (the image builds with `docker buildx build`) |
| Disk | ~390 GB total: AWQ tree ~168 GB + devan BF16 tree ~168 GB (**required, not optional** — BF16 PLE table source + rollback checkpoint) + INT8 PLE table 48.9 GiB + caches. On the rig it splits across two mounts: `/data-awq` (raw AWQ tree) and `/data` (snapshot, INT8 table, lane caches) — the 234 G NVMe keeps ~162 GiB free alongside |

Then four commands:

```bash
git clone https://github.com/imryanpurdy/Qwen3.8-Flash-Next-4x-Intel-B70s.git && cd Qwen3.8-Flash-Next-4x-Intel-B70s
cp .env.example .env          # fill MODEL_PATH / PLE_TABLE_PATH / PLE_BF16_DIR /
                              # PLE_INT8_DIR / AWQ_ORIG_DIR / LANE_CACHE / IMAGE —
                              # every knob is annotated in the file
cp lumnus.env.example lumnus.env   # engine env (docker --env-file); shipped defaults boot production
./scripts/start.sh            # preflight → weights gate → XPU gate → build-if-missing → launch → ready-poll → watchdog
```

`SERVE_ARGS="serve-args"` in `.env` already points at the shipped `serve-args`
file — there is no third file to create. Weights, snapshot, and the INT8 PLE
build are one-time steps: run `python3 scripts/fetch-weights.py` first, or
answer `y` when `scripts/start.sh`'s weights gate offers to run it (Item 5).

## Architecture

### Engine

The serving image is built from the **Lumnus [`b70-flash-next`](https://github.com/Lumnus/b70-flash-next)** repository's `image/Dockerfile`:

- Base: stock **vLLM v0.30.0** (`ced6857a`) XPU image (`vllm/vllm-openai-xpu`, pinned by digest), torch 2.13, vllm-xpu-kernels 0.1.14.1.
- **Patch series 0001–0019** (sub-lettered; **21 files** — 0011 is an unpublished draft, 0015–0017 absent; per-file manifest in `docs/engine/PROVENANCE.md`) (fork branch `b70/v0.30.0`, exported as source patches, sha-verified at build time): PLE quantization + NVMe serving (0006–0013b), thinking budgets and repetition stop (0009/0010), the KV-offload "same document, new question" fix (0014a–f), chunked CPU KV pool allocation (0018), dense-QSA indexer-tensor skip (0019).
- **Patches 0001–0005 and two closed binaries** (`libgdn_index64.so`, the Level Zero peer-residency shim) are **wu1ff's B70-LLM-Controller pack**, taken from `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0` by digest; 0001–0005 are wu1ff's Python changes re-derived as diffs, byte-identical to the pack files.

The image is built by `scripts/build-image.sh` (`docker buildx build -f image/Dockerfile -t b70-lumnus-trial:v1 ...` from a clone of the Lumnus repo at the pinned commit, digest-compared) and pinned in `.env` as `IMAGE` (default `b70-lumnus-trial:v1`). `scripts/start.sh` runs `scripts/build-image.sh` if the tag is missing.

### Checkpoint

`wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16` @ **`0939125`** — AWQ-calibrated int4 on the routed experts only (attention, GDN, shared experts, PLE, embeddings stay BF16). Chosen over devan's W4A16 because the int4 attention path in that build misreads long-context values (6/24 cold, 4/24 warm at 118K) while AWQ reads them at 0–1/24. **License note: the wtdcode checkpoint carries no license tag on Hugging Face — license to confirm before redistribution.**

### PLE table path

The model's PLE n-gram table (320M rows × 160 values) is a per-architecture asset that ships **BF16 (95.4 GiB) only in the devan tree** — the AWQ tree does not carry it. The serving path:

1. **Source:** `ple_table_qwen4exp.pt` from `devan-carlin/Qwen3.8-Flash-Next-W4A16` @ `40b8f18d` (keep this tree on disk — it is also the rollback checkpoint).
2. **Build:** `tools/build_int8_ple.py build` (from the Lumnus repo) streams the BF16 table into a per-row-scaled INT8 `.safetensors` — **48.9 GiB**, 0.66 % relative L2 error, no measurable quality loss. The build is deterministic; run `tools/build_int8_ple.py verify` afterwards and keep the sha256 — the loader checks the `lumnus-ple-int8-rowscale/v1` format tag and cross-checks rows against the BF16 table at boot.
3. **Serve:** patch 0013/0013b with `B70_PLE_INT8=1 B70_PLE_INT8_NVME=1 B70_PLE_INT8_NVME_READER=native` — the table stays on NVMe, read row-per-4-KiB with `O_DIRECT` by a native C reader, behind an **8 GiB pinned row cache** (total over 4 ranks). Frees ~39 GiB host RAM for the CPU KV tier; costs 2–4 % decode (measured; prefill unchanged).

### Storage layout

| Path | Contents |
|---|---|
| `/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16` | raw AWQ download @ `0939125` (~169 GB) |
| `/data/awq-snapshot-trial` | the **snapshot** the entrypoint serves: `tools/awq_snapshot.py snapshot <awq-dir> <devan-ple-table> <out>` symlinks the shards, writes a filtered index (PLE shard tensors + indexer tensors removed), links the PLE table |
| `/srv/hf-devan/Qwen3.8-Flash-Next-W4A16` | devan tree @ `40b8f18d` (~168 GB) — **required, not optional**: BF16 PLE table source **and** the documented rollback checkpoint |
| `/data/int8-ple/` | `ple_ngram_int8_rowscale.safetensors` (48.9 GiB, fast local NVMe — the native reader reads it with `O_DIRECT`) |

Total footprint ~390 GB (AWQ ~169 + devan ~168 + INT8 48.9 GiB + caches); plan the split across `/data-awq` and `/data` accordingly.

`scripts/start.sh` identity-gates all of it (shard presence, no symlinked-snapshot breakage, INT8 table format tag).

### Serving line

Port **8022**, served name **`qwen-256k`**, MML 262144, **MNS 32**, power-of-two decode graphs, **bf16 KV cache** (no `--kv-cache-dtype` flag — the vLLM default follows `--dtype bfloat16`; fp8 KV is the rollback es-lane's setting, not this line's), CPU KV tier **64 GiB** (patch 0018 chunking — one pinned allocation ≥ ~31 GiB/rank is refused by the driver), offload fix on (`B70_OFFLOAD_JUNCTION=1 B70_OFFLOAD_GDN_BACKSTEP=1`), watchdog mandatory (load-aware progress gate, py-spy capture, xe engine-reset monitor), night sentinel **alert-only** (writes `ALERT_NEEDS_ROLLBACK.flag`; checkpoint switches are operator calls, never automatic).

`UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` is **required** on this kernel — see Troubleshooting.

## Results

Production baseline = the devan fork engine running the **wtdcode AWQ checkpoint** (the previous production line; KV pool fingerprint 845,862 tokens = the AWQ build — see `STATUS-20261004.md`). Lumnus trial = this README's stack. Same host, same harnesses.

| Metric | Production baseline (devan fork engine + AWQ checkpoint — the pre-Lumnus production line) | Lumnus trial | Notes |
|---|---|---|---|
| 32-stream sustained (n32) | 1,015.0 tok/s | **1,118.4 tok/s (+10.2 %)** | same harness/formula |
| 16-stream (n16) | 622.0 tok/s | **652.9 tok/s (+5.0 %)** | |
| Single-stream decode | 49.3 tok/s | **52.4–52.7 tok/s** | |
| Recall ≤100K (100 lookups, 60K/100K docs) | 95/100 (historical) | **99/100** (d60k 50/50, d100k 49/50) | the one miss: record 2663, truth `DLJPY`, answered `EXPDE` |
| Prefix-scan r3 | — | **PASS** — drift cold-cold 0.0, cold-warm 0.0 (limit 0.06); warm hit 118,144 tokens; 48 responses, 0 errors, 0 degenerate; 1 misquote cold / 1 warm | the 0014 offload fix working as designed |
| 60-min soak | — | **389 waves, 3,112/3,112 OK, 0 errors, p50 8.9 s, max 10.0 s, 0 restarts, 0 engine resets** | 8-way concurrent, mixed depths |
| Heavy-file agent batch (8 tasks) | 7/8 | **8/8, 39 tool calls clean, 0 loops, 695 s** | delegated-children shape |
| PLE NVMe row cache | — | **hit rate 95.4 %** (238,641 / 250,112), **p50 0.76 ms** | native reader, 8 GiB cache |

## Client configuration

OpenAI-compatible chat completions at `http://<host>:8022/v1`, model **`qwen-256k`**, 262,144-token window. No authentication — bind to localhost or front it with an authenticating proxy.

```json
{
  "model": "qwen-256k",
  "temperature": 0.7,
  "top_p": 0.80,
  "top_k": 20,
  "min_p": 0.0,
  "presence_penalty": 1.5
}
```

- **These sampler values are the server pin and the measured ceiling.** Long-context recall holds at temp ≤ 0.7; every logged miss was a confident wrong value, and higher temperatures widen exactly that failure mode. If your client ignores the server default, send these explicitly.
- **Reasoning parser:** `qwen3` (the engine serves `reasoning`/`reasoning_content` separately). Thinking is switched off with `enable_thinking: false` or a `reasoning_effort` of `none`/`off`; per-effort thinking budgets are capped server-side (`B70_THINKING_BUDGET`: minimal/low 512 → max/ultra 12288 tokens).
- **Tool calls:** parser `qwen3_xml`; verified structurally correct including multi-tool and nested-args calls.
- **Keep client context ≤ ~100K tokens.** Inside that envelope recall is 99/100; mid-document recall degrades near/above ~200K (see Known limits). Client config for agent frameworks (Hermes delegation, context caps, stale timeouts): [`docs/hermes-clients.md`](docs/hermes-clients.md).

## Troubleshooting

- **Engine freezes, `xe ... Engine reset: engine_class=bcs` bursts in dmesg.** The Level-Zero copy-engine (bcs) reset signature: all 4 TP workers stuck in `async_tensor_h2d`, engine core blocked in `shm_broadcast.wait`, `/health` stays 200 while the engine is dead (probe with a 1-token completion). Fix is the required flag `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` — with it: 3 freezes → 0 across a 60-min prefix-cold soak. `sudo dmesg | grep -aE "Engine reset: engine_class=(ccs|bcs)"` (unprivileged dmesg may silently return nothing).
- **The empty-env-var trap.** `VAR=` (declared but empty) is **not** unset — oneCCL parses the empty string and crashes worker init (`unexpected value: , expected values: 0, 1`). In `scripts/start.sh`, pass such flags with `${VAR:+-e VAR=$VAR}`, never `-e VAR=${VAR:-}`; unset in `.env` → omitted from the docker line entirely.
- **Reboot after `DEVICE_LOST` bursts.** A burst of Level-Zero resets can leave devices in `DEVICE_LOST` that a container restart alone does not clear (re-launch fails `OUT_OF_RESOURCES` on all workers). Stop, retry one boot; if workers fail again on an idle host, reboot, then re-run the XPU gate and READY gate before declaring recovery.
- **7.x kernels are banned.** 7.0.0-31 carries the job-timeout fix but lacks the flat-CCS fix (landed 6.18.51), and 7.x/newer-GuC combinations wedge permanently on B70. Stay on 6.17.0-1010-intel; `scripts/start.sh` preflight rejects anything else.
- **Preflight / XPU gate / READY gate failures** → `scripts/start.sh` prints which floor failed; the gate catches JIT gaps in seconds; cold boots take ~4–5 min (graph capture inside torch.compile) before `Application startup complete`.

## Known limits

- **Mid-document recall degrades near/above ~200K tokens.** In a 200K-token document, recall at the ~118K position collapsed (12/34) while 60K and 200K-position records read fine — the failure is concentrated mid-document, not monotonic with depth. It does not appear at ≤100K. Keep client contexts ≤100K; re-test before raising the cap.
- **~1 % recall miss rate at ≤100K** (1 miss in 100 on the Lumnus line; misses are confident wrong values, never refusals) — the client-side verify habit in `docs/hermes-clients.md` is the mitigation.
- **NVMe PLE decode cost: −2 … −4 %** vs the table pinned in RAM (the per-step host sync, not the reads). Prefill unchanged. If you have the host RAM, pinning INT8 in RAM (`B70_PLE_INT8_NVME` unset) buys the 2–4 % back.
- **No authentication** on the API; bind to localhost or proxy it.
- **`MAX_NUM_SEQS` > 32 hard-fails** (KV-cache knee at MML 262144).

## Rollback

The previous production line stays documented and bootable in one `rollback/devan-fork/start.sh` cycle (the rollback launcher lives under `rollback/devan-fork/`, mirroring the rig's deployed `~/mns32-lane/` kit, separate from the production entrypoint `scripts/start.sh`):

1. **Rollback of record — last-known-good production:** the devan-carlin `vllm@xpu-qwen4exp` fork image (a69fba21) + the **wtdcode AWQ checkpoint** (`/data-awq/Qwen3.8-Flash-Next-AWQ-W4A16`) at **MNS 32** with fp8 KV — the lane that served before the Lumnus promotion and measured the baseline numbers above (n32 1,015.0 / n16 622.0 / 49.3 tok/s / 95/100 recall; KV pool fingerprint 845,862 = the AWQ build). Config of record: the deployed `~/mns32-lane/.env` on the rig (in-repo sanitized equivalent: `rollback/devan-fork/.env.example` with `MAX_NUM_SEQS=32`).
2. **Deeper fallback (footnote):** the older es-lane launcher (`~/es-lane-launch/start-qwen-256k-vllm.sh`, MNS 4, fp8 KV) serving the **devan-carlin W4A16** build @ `40b8f18d` (`/srv/hf-devan/Qwen3.8-Flash-Next-W4A16`, ~168 GB, kept on disk for this **and** as the BF16 PLE table source). This MNS-4 line predates the MNS-32 soak and is *not* the rollback of record.
3. **Either way: PLE never executes on the devan fork.** The engine's PLE table path is absent from its expected location and the guard short-circuits (forensics: zero `FileNotFoundError` in the boot log, forward never reached `_ensure_table` — only the capture-legal early-return branch ran). Treat the rollback line as a **dense fallback**, not a PLE line.

The night sentinel is alert-only by design: a trigger writes `ALERT_NEEDS_ROLLBACK.flag`; an operator makes the switch.

## Credits

- **Lumnus** — the [`b70-flash-next`](https://github.com/Lumnus/b70-flash-next) engine and patch series 0001–0019 (sub-lettered; 21 files), [`Lumnus/vllm`](https://github.com/Lumnus/vllm) `b70/v0.30.0` (Apache-2.0, with NOTICE).
- **wu1ff** — [B70-LLM-Controller](https://github.com/wu1ff/B70-LLM-Controller) (MIT): patches 0001–0005 and the two binaries (`libgdn_index64.so`, the Level Zero peer-residency shim) via `ghcr.io/wu1ff/qwen38-flashnext-b70:1.0.0`.
- **devan-carlin** — the early community XPU port (`devan-carlin/vllm@xpu-qwen4exp`), the [`Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/devan-carlin/Qwen3.8-Flash-Next-W4A16) weights (rollback checkpoint + the BF16 PLE table every INT8 build derives from).
- **wtdcode** — the [`Qwen3.8-Flash-Next-AWQ-W4A16`](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) checkpoint we serve (**license to confirm** — no license tag on the HF repo).
- **TSUMUGI-XE** — the Level Zero peer-residency analysis and first shim ([intel/compute-runtime#968](https://github.com/intel/compute-runtime/issues/968)).
- **Intel** — the XPU software stack, [llm-scaler](https://github.com/intel/llm-scaler), the XPU work in vLLM.
- **vLLM / vllm-xpu-kernels** (Apache-2.0) — everything here is built on them; patch 0014e is a backport of vllm-project/vllm#51787.
- **Qwen** — the [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) model, under the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).

## License

This repo's kit (scripts, docs, configs): MIT — see [LICENSE](LICENSE). The engine image is Apache-2.0 (Lumnus) carrying wu1ff's MIT files; see the Lumnus repo's NOTICE. Model weights are governed by their own terms — Qwen Community License 1.0 for the base model; the wtdcode AWQ checkpoint's license is **to be confirmed** before redistribution.
