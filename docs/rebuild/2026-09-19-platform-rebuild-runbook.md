# Platform Rebuild Runbook — jobe (4× Intel Arc Pro B70, Flash-Next)

**Date:** 2026-09-19 · **Scope:** clean OS reinstall + OMIX stack install + stage-v24g engine
container reconstruction + verification ladder + rollback
**Posture:** writing-only runbook, no rig access at authoring time; every gate below is executed
on the rig by an operator (or the next agent run with access). **Do not skip gates.**
**Author note:** OS-target choice (24.04-HWE/6.17 vs 26.04/7.0) was pending resolution by a
sibling subagent — this doc is parameterized on `TARGET_OS` and ships with `A` (24.04-HWE) as the
conservative default. Fill the decision box (Section 1) before executing Section 2.

---

## 0. Prerequisites (do these BEFORE touching the OS)

0.1 **Preserve the current working engine image — the rollback anchor.** On the rig:

```bash
docker image ls | grep -i qwen38            # record full tags + image IDs
docker save qwen38-flash-next-xpu:0.21.0-b1-rt2f829747-stage-applied-v24g \
    -o /srv/rollback/stage-v24g-pre-rebuild.tar    # ~15-40 GB; verify sha256sum after save
sha256sum /srv/rollback/stage-v24g-pre-rebuild.tar > /srv/rollback/stage-v24g-pre-rebuild.tar.sha256
```

(If the exact v24g tag is absent — e.g. the rig only has v24f — save the newest tag that
served jobs, and record `docker image ls` output; rollback Section 7 is written against
whichever tag you saved.)

0.2 **Preserve the serving config and the reconstruction source tree** (copy to the OS disk,
+ second copy to a USB stick or the weights NVMe — the OS disk gets wiped):

```bash
cp -a ~/flashnext-scout        /srv/preserve/flashnext-scout-20260919      # patch scripts, v3-src/, audit-src/, wedge-census/
cp -a ~/flashnext-recipe/docs  /srv/preserve/flashnext-recipe-docs-20260919
cp -a <engine-dir>/.env*       /srv/preserve/ 2>/dev/null || true          # .env, .env.bak — DO NOT LOSE
cp -a <engine-dir>/.run        /srv/preserve/run-20260919 2>/dev/null || true   # boot_clock.jsonl, wd-decisions.jsonl, manifest.json
```

0.3 **Record the pre-rebuild stack fingerprint** (input to the Appendix, and the "before"
side of the distinguishing evidence in Section 1): `uname -r`, `dpkg -l | grep -iE 'libze|compute-runtime|level-zero|intel'`, `python3 -c 'import torch;print(torch.__version__)'`,
`docker image inspect` of the engine image, watchdog script sha256.

0.4 **Installer media + network**: Ubuntu <TARGET_OS> ISO on USB; wired network (OMIX + torch
downloads are multi-GB); console/SSH access planned before boot.

0.5 **Weights disk**: the weights NVMe is NOT wiped by the OS reinstall — but verify the mount
does not depend on OS-disk fstab entries, and plan to re-add the mount post-reinstall.

---

## 1. DECISION BOX — OS target (fill before Section 3)

Background: the failing config was an unsupported **three-source mix** — Ubuntu-26.04-ish /
kernel 7.0.0-31 **+** Intel PPA `libze1` 1.32.0 + compute-runtime 26.x **+** PyPI-ish
torch 2.11.0+xpu. Thread evidence: kernel 7.0 + newer runtime = permanent multi-card wedges;
kernel 6.17 + older GuC = stable and faster. Intel's position: 26.04/kernel 7.0 is the
natively-supported OMIX target.

| Option | OS / kernel | Stack source | Verdict |
|---|---|---|---|
| **A (DEFAULT)** | Ubuntu 24.04 LTS + HWE kernel **6.17** | OMIX per Intel guide, clean | Choose unless B's evidence checks out |
| **B** | Ubuntu 26.04 + kernel **7.0.0-31** | OMIX per Intel guide, clean | Evidence required before choosing |

**Reconciliation rule (both statements can be true):** Intel's "native support" for 26.04/7.0
and the thread's "kernel 7.0 = permanent wedges" are both satisfiable if the wedge cause was the
**PPA mix**, not the kernel. The distinguishing experiment is exactly this rebuild:

- **Clean OMIX 26.04/7.0 install that still wedges (multi-card, under load) → real
  kernel/runtime bug** → fall back to Option A; file the reproduction with boot-ID + stack
  fingerprint.
- **Clean OMIX 26.04/7.0 install that is stable → the PPA mix was the cause** → B is correct,
  keep it (and record which torch build version pairs with it).

If no evidence was resolved before execution: **execute Option A (24.04-HWE/6.17)**. It is the
conservative choice: the thread's stable+fast configuration is on this line, and OMIX on
24.04-HWE is a long-published, widely deployed path.

**RECORD HERE:** `TARGET_OS=____________` `KERNEL=____________` `OMIX_VERSION=____________`
`selected_by=____________` `date=____________`

---

## 2. Host rebuild — clean OS

### 2.1 Clean install

1. Boot the <TARGET_OS> installer. Install with **minimal** packages (base system + ssh + sudo
   only). Do **not** let any installer option add graphics/compute runtime packages.
2. Verify after first boot: `uname -r` matches the box (6.17.x for A, 7.0.0-31 for B);
   `cat /etc/os-release | head -3` matches the box.
3. Configure: hostname, static IP, key-based SSH, timezone (`timedatectl`), apt update/upgrade
   baseline.
4. **WEIGHTS DOWNLOAD FIRST — the critical path starts the moment the fresh OS has network and a
   filesystem, BEFORE OMIX, before anything else (2026-09-20 directive).** 169 GB from HF is
   hours; OMIX install and container reconstruction are minutes-to-an-hour. Start it unattended
   and let the rest of the stack install while it runs:
   ```bash
   mkdir -p /home/bonz/hf-int4 && nohup env HF_HOME=/home/bonz/hf-int4 bash dl-int4.sh > /tmp/dl-int4.log 2>&1 &
   ```
   `dl-int4.sh` (scout/) is **revision-pinned** to `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`
   (the pre-wipe serving snapshot) — guarantees byte-identical weights regardless of what Intel
   pushes to `main` (HF blobs are content-hash-named). Verify after download against the
   preserved manifest `evidence/2026-09-20/weights-identity/blobs-manifest.txt` (29 blobs,
   largest 102400512256 B). Nothing blocks on it: OMIX (2.3+), container build (3), and the vLLM
   pip installs proceed during the download; every ladder gate needs the weights present (L1
   boots the engine), so the ladder starts only after it completes.

### 2.2 Firmware / GuC discipline (record, don't reflexively touch)

- **Record** (do not upgrade anything yet):
  ```bash
  sudo dmesg | grep -iE 'guc|xe |firmware|fw ' | head -50        # GuC/firmware version + xe probe lines
  sudo lspci -nn | grep -i arc                                      # 4× B70 present?
  ```
- Do **not** run firmware/MEU/BIOS upgrades as part of this rebuild unless a measured blocker
  emerges — the thread's stable config was **not** on a newer firmware. If a firmware change is
  ever made: record old + new version, then re-run the full ladder.
- **Reboot discipline:** reboot **before every measurement session**. The xe driver state
  accumulates across crashed starts (crashed device starts are not clean); a measurement taken
  without a fresh boot is not comparable to any other measurement. Rule: `uptime` must be a
  fresh boot (or the session must be logged with its boot provenance) before L1–L5 gates.
- Verify the **kernel module** path: B70 = Battlemage (Xe2, BMG G21) → driver is `xe`
  (`modinfo xe | head`, `sudo lsmod | grep -w xe`). `i915`/other should not bind the cards.

### 2.3 Purge any PPA leftovers (mandatory — OMIX docs demand a clean system)

The old mix's packages must be gone before OMIX lands; a mixed libze/compute-runtime
installation is the exact re-creation of the wedge-generating config:

```bash
sudo apt-get purge -y libze1 libze-intel-gpu1 libze-intel-gpu-dev \
    intel-graphics-compute-runtime intel-opencl-icd intel-gsc intel-metrics-discovery \
    intel-level-zero-gpu intel-ocloc 2>/dev/null
sudo rm -f /etc/apt/sources.list.d/*intel* /etc/apt/sources.list.d/*opencl*   # PPA/repo leftovers
sudo apt-get update
dpkg -l | grep -iE 'libze|compute-runtime|level-zero|opencl' || echo "CLEAN: no runtime packages"   # MUST print CLEAN
```

**Gate 2.3:** the `dpkg -l` output must be empty (or contain nothing but the echo line above).
Anything else = **HARD STOP**; a dirty host bakes the old wedge cause into the new stack.

### 2.4 OMIX install — per Intel's official procedure

- OMIX = **Intel Open Middleware Xe**. Install exactly per the official guide
  (dgpu-docs.intel.com — "Installing Intel Open Middleware Xe" + the current OMIX release-notes
  page for the pinned OMIX/compute-runtime versions). The OMIX docs require a clean system —
  2.3 is the precondition, do not skip it.
- Use the official repository/package set only (the OMIX release provides the matching
  `libze1` + `libze-intel-gpu1` + `intel-opencl-icd` + `intel-gsc` + `intel-metrics-discovery`
  set). **One source, one runtime version** — never mix Ubuntu universe, Intel PPA, and OMIX
  packages for the GPU runtime.
- Torch: install `torch`/`torch-xpu` 2.11.0+xpu **for the matching oneAPI/Level-Zero release**
  (single-source rule extends to the wheel's driver contract). Record exact wheel index used.

**Gate 2.4 (single-source proof):**

```bash
dpkg -l | grep -iE 'libze|compute-runtime|level-zero|opencl'     # each package ONE version row, all from the OMIX release
sycl-ls                                                           # level_zero:gpu expected — 4 devices
xpu-smi 2>/dev/null || xpu-smi dump; clinfo -l
ls -la /usr/lib/x86_64-linux-gnu/libze_intel_gpu.so*             # exactly one real version (no .1.32.0 leftover)
python3 -c 'import torch; print(torch.__version__, torch.xpu.device_count())'   # expects 4
```

Then **reboot** and re-verify device count = 4 (xe state does not survive across boots by
design — a crashy start is exactly what the pre-rebuild wedges produced).

### 2.5 Host prep (preflight floors, unchanged from the kit contract)

- RAM ≥ 100 GiB total visible (`free -g`), swap ≥ 64 GiB ON (`swapon --show`, add via
  `/etc/fstab` + `swapon -a` if missing — the kit hard-fails below floors).
- Weights disk: re-add the mount (≥ 200 GiB free; model tree ~185 GB), add to `/etc/fstab`,
  `mount -a` + verify.
- HF cache path bind-mounted into the container must exist and be owned read/write.
- Docker: `apt install docker.io` (or per target-OS procedure), `sudo usermod -aG docker $USER`,
  verify `docker run --rm hello-world`.
- `mkdir -p /tmp/pyspywheel/` and copy the saved py-spy wheel there (see 5.2 golden rule #6);
  if it was lost, `pip download py-spy` and keep the wheel in that dir permanently.
- Install `jq`, `curl`, `xpu-smi`/utilities used by the harness.

**Gate 2.5:** `free -g` (RAM ≥ 100), `swapon --show` (≥ 64), `df -h <weights mount>` (≥ 200 GiB
free), `docker info` OK, py-spy wheel present.

---

## 3. Container reconstruction — stage-v24g, patch by patch

### 3.0 Golden rules (violations are the known failure modes)

1. **Script-file-only patching. NEVER heredocs into the container.** Two silent heredoc failures
   occurred 09-19 (empty diff, exit 0) — always write the patch as a file (`docker cp` it in)
   and execute the file inside the container. (Heredocs *for creating the patch-script file on
   the host* are fine; they must not be the mechanism that delivers code into the container.)
2. **md5/verify gate on every target.** Before applying: `md5sum <target>` and compare to the
   preserved record (`a4280a54329bdadaca2cd8754a35c9a6` was the connector.py target gate on
   stage-v24f — recompute from the preserved source tree, do not trust memory). After applying:
   re-compare, marker-count grep, `python -m py_compile` (or `ast.parse`) the touched file.
   Mismatch anywhere = HARD NO-GO for that step.
3. **CRLF is real:** any script authored on Windows must be `tr -d '\r'`-ed before it enters the
   container (`sed -i 's/\r$//'` or `tr -d '\r' < src > dst`), else bash/python silently mis-step.
4. **The container does NOT share host `/tmp`.** All transfers: `docker cp`, bind mounts, or
   `docker exec` reading from a bind. Never assume a file written on the host exists inside.
5. **py-spy must be reinstalled in every new image** (it is wiped on rebuild):
   `docker exec <ctr> pip install /tmp/pyspywheel/py_spy*.whl` with the wheel bind/docker-cp'd
   in; verify `docker exec <ctr> /opt/venv/bin/py-spy --version`.
6. **Every `docker commit` must re-set ENTRYPOINT + CMD.** A commit from an `--entrypoint
   bash` build container carries bash into the image and breaks boot/serve (it happened on
   stage-v24e). Every commit:
   ```
   docker commit --change 'ENTRYPOINT ["/opt/venv/bin/python3"]' \
                 --change 'CMD ["/opt/venv/bin/vllm","serve","Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"]' \
                 <ctr> <new-tag>
   ```
   Verify after every commit: `docker inspect <tag> --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'`
   → must show `/opt/venv/bin/python3` and the serve args, **not** `bash`.
7. **Idempotency + marker discipline:** every patch script asserts its marker absent before
   applying and prints a `PATCH_OK`-style line; re-running must be a no-op. Markers are the
   reconstruction proof (grep counts in the verify-gate table below).

### 3.1 Base

- Base image: `intel/vllm:0.21.0` (+ torch 2.11.0+xpu install per the preserved build record),
  recreated as `qwen38-flash-next-xpu:0.21.0-b1-rt2f829747-stage-applied-v24g` (keep the
  naming; if v24g exists on the rig already, use `-v24g2` and record it in the manifest).
- Pin the toolchain as before: python 3.12 in `/opt/venv`, triton-xpu for torch 2.11.
- Break the container into `--entrypoint /bin/bash` at build time only; the FINAL image gets the
  python3 entrypoint back per rule 3.0.6.

### 3.2 Patch table (order matters — apply in listed order, verify as you go)

All paths are **inside the container** under `/opt/venv/lib/python3.12/site-packages/vllm/`
unless noted. Canonical patch scripts/sources live in `~/flashnext-scout/` (preserved in 0.2).

| # | Feature | Files | Patch script / source | Markers & verify gate |
|---|---|---|---|---|
| P1 | PLE mmap connector (custom PLE host-offload) | `v1/ple_offload/connector.py`, `v1/ple_offload/worker.py`, `v1/ple_offload/ple_mmap_v18.py`, `model_executor/layers/ple_offload_layer.py` | preserved scout lineage (v18 mmap generation; `ple-mlock-v17/v16.py`, `ple-conn-v8/v7.py`) | ensure the mmap/v18 connector is present — grep `ple_mmap_v18` in worker.py; absence = wrong generation, which predates the 82%-freeze extinction |
| P2 | is_int4 detection via `weight_quant_dtype` | `model_executor/layers/fused_moe/experts/xpu_moe.py` | `patch-v21.py` — `_is_int4 = (getattr(self.quant_config, 'weight_quant_dtype', None) == 'int4')`, passed as `is_int4=_is_int4` into `XpuFusedMoe` | grep `weight_quant_dtype` + `is_int4=_is_int4` in xpu_moe.py; `V21DIAG` ctor line optional, keep (one-line diag, cheap) — fallback derivation if quant_config absent: `w13.dtype == torch.int32` (patch-v19.py) |
| P3 | Connector **v3**: runner-thread stream-ordered D2H staging | `v1/ple_offload/connector.py`, `v1/worker/gpu/model_runner.py` | `patch-v3.py` + canonical `v3-src/connector.py`, `v3-src/model_runner.py` | `V3RUNNER` marker: **7 occurrences in connector.py, 1 in gpu_model_runner.py**; constructor logs `PLE v3 staging wired: staged=… tp_rank=…` per rank |
| P4 | GDN capture D2H fix | `v1/attention/backends/gdn_attn.py` (~line 546) | `patch-gdn-v3b.py` | `V3BGDN` marker; replacement: `num_decode_draft_tokens_cpu = torch.diff(m.query_start_loc_cpu) - 1` (host companion from `backend.py:378` `query_start_loc_cpu`) |
| P5 | ShortConv D2H fix (twin) | `v1/attention/backends/short_conv_attn.py` (~line 552) | `patch-shortconv-v3b.py` | `V3BSC` marker; same `torch.diff(m.query_start_loc_cpu) - 1` replacement |
| P6 | Watchdog **v2.5** (host-side, not in image) | host: `wedge-watchdog-v2.sh` (v2.5) | preserved scout script | header comment `v2.5 HARDENED`; self-guard flock+pidfile; gen-probe; boot_clock-aware phases (details below) |
| P7 | Campaign harness (host-side) | `campaign.sh`, `soakfix.py`, `stallspy.sh` | preserved scout scripts | `stallspy.sh` runs INSIDE the container (py-spy, rule 3.0.5); soakfix timer-fixed |

**P3 detail (the load-bearing patch — read before applying):**

- Producer moves to the **runner thread** (V2 model runner — `VLLM_USE_V2_MODEL_RUNNER=1`;
  `vllm/v1/worker/gpu/model_runner.py`). Runner enqueues **stream-ordered D2H** into
  2-slot pinned mirrors: `staging_input_ids [2, max_num_batched_tokens]` int32,
  `staging_query_start_loc [2, max_num_reqs+1]`, `staging_ngram_context [2, *source.shape]`.
- **Flag write LAST on the stream**, carrying a per-boot monotonic `seq` (int32) — D2H publish
  is complete-before-flag. Connector consumes via **plain numpy load only**, gated on
  `flag_np[slot]==seq`; **bounded 2 s poll**, **rate-limited legacy `copy_` fallback** on
  timeout (error-logged, not silent).
- **(slot, seq) staging token rides inside the queue payload** — atomic handoff, no shared
  mutable state between threads. `put_nowait` → bounded `queue.put` (a full queue previously
  crashed the engine with `queue.Full`).
- Canary line at boot, one per rank: `PLE v3 staging wired: staged=… tp_rank=…` — a
  missing/aberrant line is a miswire spelled out, not a mystery.
- This patch REQUIRES P1 already in place (the connector it reshapes is the mmap connector) and
  P2's is_int4 semantics to keep the model load path identical.

**P4/P5 rationale (write this into the evidence log):** `(num_accepted_tokens - 1).cpu()` at
gdn_attn.py:546 and short_conv_attn.py:552 is a **blocking D2H USM memcpy through the Level
Zero command-list manager mid-capture** — it wedged all 4 TP ranks on the MTP1 boot (boot
4010612; `docs/incidents/2026-09-19-gdn-capture-l0-wedge.md`). The host mirror `query_start_loc_cpu`
(defined at backend.py:378) carries identical values, so `torch.diff(...) - 1` is pure host
math: no device op, no L0 append.

**P6 watchdog v2.5 semantics (all evidence-backed):**

- **Engine-true health:** `/v1/models` alone is INVALID as health (stays 200 while EngineCore is
  dead — stall #4 hunts ran blind through it). Serving-phase health = **1-token generation
  probe** (`/v1/completions`, `max_tokens:1`), run only in serving phase (F2).
- Phase-aware kill: parsed from `boot_clock.jsonl` + server.log markers; **boot_clock ready
  timestamp compared against the container start epoch** (F7a fix — was compared against probe
  time, so it never fired), and `container_start_epoch()` uses `date -d "$iso"` (F7b fix — arg
  order was wrong, always returned 0); fallback phase detection via the per-boot server.log
  `Application startup complete` line (F6).
- Mute thresholds (v2 semantics, preserved): pre-KV no-mtime-kill 300 s; post-KV 900 s AND all
  4 worker wchans blocked; serving: 2 consecutive gen-bad probes → capture-and-restart even
  when mtime is fresh (F3), kill at 300 s mute.
- **Stat-excuse (F4):** a 1-token probe can be queued behind an 8-way burst and time out while
  the engine is healthy; a demonstrably NONZERO `Avg generation throughput` line in server.log
  with no wedge/deadline marker after it counts as alive (a wedged engine goes 0.0 then
  silence/60 s warnings).
- **Self-guard (F5):** flock (+ pidfile fallback) against duplicate watchdogs (a real
  duplicate was seen — stall #4, pid 3233940); cold-start grace so a reload that takes < 2
  probes isn't burned as an exit.
- **Never kill on warning count**; before every kill, wchan/stat capture of all workers into
  the wedge log; **every kill and near-miss is appended to `.run/wd-decisions.jsonl`** with
  `{probe_ts, phase, mtime_age_s, size_delta, health, wchans[4], cpu%, trigger_fired,
  threshold_used, verdict}` — that file is the primary evidence for the next census.
- Restart is via **docker** (stop/start the named container), owned by the watchdog alone when
  the campaign is running.

### 3.3 Build sequence (with gates)

```bash
# On the rig, FROM the engine directory (start.sh + .env live here), engine dir = <ED>
# 1. base container up (bash entrypoint for patching ONLY)
docker run -d --name v24g-build --entrypoint /bin/bash \
  --device /dev/dri --group-add video --group-add 991 \
  qwen38-flash-next-xpu:0.21.0-b1-rt2f829747-stage-applied-v24g -c sleep 900   # or fresh base build
docker cp ~/flashnext-scout/patch-v3.py    v24g-build:/root/
docker cp ~/flashnext-scout/patch-gdn-v3b.py v24g-build:/root/
docker cp ~/flashnext-scout/patch-shortconv-v3b.py v24g-build:/root/
# CRLF guard for every script copied in:
docker exec v24g-build bash -c "for f in /root/*.py; do sed -i 's/\r$//' \$f; done"

# 2. P1 (mmap connector) — from preserved PLE lineage scripts/v3-src; then verify (3.2 table)
# 3. P2 is_int4 (patch-v21.py inside container, then verify)
docker exec v24g-build python3 /root/patch-v21.py    # prints V21DIAG wiring + PATCH-OK
# 4. P3 connector v3 — docker cp v3-src/{connector.py,model_runner.py} over the targets, or
#    apply patch-v3.py in-container; then verify markers:
docker exec v24g-build bash -c "grep -c V3RUNNER /opt/venv/lib/python3.12/site-packages/vllm/v1/ple_offload/connector.py /opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/model_runner.py"
#    EXPECT: connector.py = 7; model_runner.py = 1
# 5. P4/P5 (gdn + shortconv):
docker exec v24g-build python3 /root/patch-gdn-v3b.py
docker exec v24g-build python3 /root/patch-shortconv-v3b.py
# 6. py-compile everything touched:
docker exec v24g-build bash -c "cd /opt/venv/lib/python3.12/site-packages/vllm && python -m py_compile v1/ple_offload/connector.py v1/worker/gpu/model_runner.py v1/attention/backends/gdn_attn.py v1/attention/backends/short_conv_attn.py && echo PYCOMPILE-OK"
# 7. reinstall py-spy (rule 5): wheel at /tmp/pyspywheel/ bind-mounted or cp'd
docker exec v24g-build pip install /tmp/pyspywheel/py_spy-*.whl
# 8. commit WITH metadata restore (rule 6) → tag
docker commit --change 'ENTRYPOINT ["/opt/venv/bin/python3"]' \
              --change 'CMD ["/opt/venv/bin/vllm","serve","Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"]' \
              v24g-build qwen38-flash-next-xpu:0.21.0-b1-rt2f829747-stage-applied-v24g
docker inspect qwen38-flash-next-xpu:0.21.0-b1-rt2f829747-stage-applied-v24g \
  --format '{{json .Config.Entrypoint}} {{json .Config.Cmd}}'   # MUST show python3 + serve args
docker rm -f v24g-build
```

**Gate 3.3:** all `PATCH-OK`/`PYCOMPILE-OK` prints, marker counts exactly 7/1/1/1, entrypoint
+ CMD verified, py-spy version prints. Then a **boot smoke**: start the container with `.env`
(see 3.4) and watch the boot log for the v3 staging canary line `PLE v3 staging wired:
staged=…` per rank (**4/4 staged=True** on the v24f reference; anything else = miswire, see 3.2
P3) before doing ANY measurement.

### 3.4 Env config (earned list — set in .env / compose environment)

```
# model identity
MODEL=Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
SERVED_MODEL_NAME=qwen3.8-flash-next
PORT=8021
TENSOR_PARALLEL_SIZE=4   ENABLE_EXPERT_PARALLEL=true
MAX_MODEL_LEN=4352

# earned engine knobs
VLLM_USE_V2_MODEL_RUNNER=1
VLLM_KV_CACHE_LAYOUT=BLHNC
VLLM_PLE_CPU_OFFLOAD=1
VLLM_XPU_GDN_SERIAL_SPEC_DECODE=0
VLLM_XPU_GDN_NATIVE_SPEC_RECURRENT_SERIAL_EXACT=1
VLLM_SPU_GDN_SPEC_PERSISTENT_SCRATCH=1
VLLM_XPU_GDN_NATIVE_SPEC_COMPLETION_BARRIER=1
VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=900

# under active test (legacy adapter = 0 currently favored)
SYCL_UR_USE_LEVEL_ZERO_V2=0
VLLM_XPU_ENABLE_XPU_GRAPH=0        # graphs faulted on V2 adapter with MTP1; legacy verdict pending — 0 until the MTP ladder says otherwise

# CCL posture (from the thread's stable configs — add before any LARGE-CONTEXT work)
CCL_ENABLE_SYCL_KERNELS=0
CCL_ALLREDUCE=ring
CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
TORCH_LLM_ALLREDUCE=1
CCL_ZE_CACHE_OPEN_IPC_HANDLES=0

# watchdog/campaign
CONTAINER_NAME=qwen38-flash-next
WEDGE_WATCHDOG_RETRIES=3
```

Notes: the `VLLM_SPU_GDN_*`/`VLLM_XPU_GDN_*` and friends print
`Unknown vLLM environment variable detected` warnings at boot — **expected**: they are custom
knobs read by the patched code via `os.environ`, not registered in stock `envs.py`. A missing
warning is not a requirement; the envs must be present in the environment for the patched code.

---

## 4. Verification ladder (in order; each gate must PASS before the next)

Every gate record uses the Appendix template (boot ID + stack fingerprint). A failure at any
gate = STOP, capture (py-spy dumps, wd-decisions.jsonl, dmesg), then go back to the relevant
patch — never forward.

**L1 — Boot to READY.** Fresh reboot (Section 2.2 rule). Start engine; wait for
`Application startup complete` + serving readiness via the **engine-true** probe (models + 1-token
gen) — 15-min readiness poll like start.sh. Record boot ID + phase-clock log (`boot_clock.jsonl`),
V3 staging canary 4/4, KV pool geometry, PLE pinned allocation. Gate: READY with all 4 ranks
`staged=True`.

**L2 — Known-answer smoke.** temp-0: capital-of-France prompt → "Paris"; alphabet prompt →
correct ordered output. Gate: both correct, engine alive post-probe (gen probe 200).

**L3 — py-spy L0-absence proof (the op-level falsification).** Arm `stallspy.sh` **inside the
container** (300 dumps target: 5 s cadence × ~4 process dumps × ~75 cycles, or DUR sized so
`ls /tmp/stallspy/*.dump | wc -l` ≥ 300) while running one burst. Gate: **0/300 dumps contain
`appendUSMMemcpy`** (pre-fix it was 27/27); the connector thread may appear (idle at
queue-get) — that's fine; additionally grep 0 `ur_command_list_manager::appendUSMMemcpy`,
0 `libur_adapter_level_zero_v2`. (Reference: 170 dumps, 0/170 on stage-v24f.)

**L4 — Burst campaign (stability gate, not throughput):** `BURSTS=15 CONCURRENCY=8
TOKENS=600` → 15 bursts × 4 rounds = **60 rounds**, per-round aggregates written to
`campaign-<TS>/…/burst-*.json`, post-burst engine-true probe gate (`POST_GATE=strict`),
watchdog v2.5 live (campaign refuses to run without it). Stall = any of: rounds collapse to 0,
gen-probe != 200 within 90 s of burst end, `EngineDead`/`TimeoutError` in server.log, watchdog
kill decision. **Pre-registered decision rule:** 0–1 stalls / 15 → v3 confirmed, ships as
default; 2–3 → weak evidence, extend 10 bursts before claiming; ≥4 → fix ineffective → lock
diagnostic (serialize PLE transfer vs replay) and/or `VLLM_PLE_CPU_OFFLOAD=1` discriminator.
Aggregate throughput = measurement only, not a gate. (If a different OS target was selected, the
campaign is the FIRST place A-vs-B wedge evidence lands: a clean-OMIX 26.04 wedging here = the
distinguishing evidence of Section 1.)

**L5 — MTP ladder.** MTP0 baseline (compare to pre-rebuild anchor), then MTP1 with graphs
(`VLLM_XPU_ENABLE_XPU_GRAPH=1` + `SYCL_UR_USE_LEVEL_ZERO_V2=0` legacy adapter while the
verdict is pending) — **capture-fault check**: MTP1 capture must not wedge (the P4/P5 fixes
remove the mid-capture D2H; boot 4010612 must not recur). Record MTP1/acceptance numbers vs the
pre-rebuild anchors; every number carries boot ID + stack version. Do not enable MTP1+graphed
as default until the ladder shows no capture fault.

---

## 5. Rollback

- **Anchors:** preserved image at `/data/preserve/` **(GAP A fix 2026-09-20: the original
  `/srv/rollback` is on the wiped SSD — real backups went to `/data/stage-v24h2-rollback.tar.gz`,
  sha256 `3fc1d174…c3a706`, gzip -t verified; second copy pulling off-box)**, the
  preserved `.env` (0.2, off-box at `evidence/2026-09-20/rollback-unit/`), the preserved scout
  tree. Rollback anchor image = `stage-v24h2:rollback` (ae528b55ee4e, NOT v24g — v24h2 carries
  the capture-list fixes). Rollback is only viable while these exist —
  never delete them until L5 passes on the new build (then archive, don't delete, for six weeks).
- **Procedure:** stop container + watchdog; `docker load -i
  /data/stage-v24h2-rollback.tar.gz` (or the off-box copy); retag as the serving name; restore `.env` from
  `/srv/preserve/`; restart via start.sh with watchdog; gate L1+L2 only — **do not assert L3–L5
  results from a different boot provenance** (the campaign proviso: numbers are per-boot; a
  rollback boot is its own record).
- **When to roll back:** clean-OMIX B (26.04) wedging at L1–L4 with no repro path in ~2 attempts
  → roll back the OS line (to Option A) rather than patching — the Section 1 reconciliation says
  a kernel/runtime bug is not our code; and any L1–L4 failure on the container = roll back just
  the image (not the OS) after capturing evidence.

---

## 6. Measurement protocol appendix

**Every number needs a boot ID + stack version.** A number without both is not a number.

```text
boot_id=________ (boot_clock.jsonl)      boot_epoch=________ (docker StartedAt)
kernel=$(uname -r)                        os_release=$(head -1 /etc/os-release)
libze1=$(dpkg -l libze1 | tail -1 | awk '{print $3}')
compute_runtime=$(dpkg -l libze-intel-gpu1 | tail -1 | awk '{print $3}')
torch=$(python3 -c 'import torch;print(torch.__version__)')
engine_image=$(docker images --no-trunc | grep stage-applied-v24g | awk '{print $3}')
watchdog_md5=$(sha256sum wedge-watchdog-v2.sh | cut -d' ' -f1)
vllm_commit=$(docker exec <ctr> sh -c 'pip show vllm 2>/dev/null | grep -i location')   # or image label
date_utc=$(date -u +%FT%TZ)
```

Rules:
- **Health probes:** health = 1-token generation probe (gen-probe). `/v1/models` is NOT health
  (it returned 200 through a dead EngineCore — stall #4 evidence). Never report `/v1/models`
  status as "alive".
- **Counter discipline:** a stall count is only valid when the watchdog that observed it ran the
  v2.5 semantics listed in P6 and every kill/near-miss landed in `wd-decisions.jsonl`.
- **Confounds to declare on every campaign row:** boot provenance (engine restart mid-campaign
  = new provenance record), watchdog version, OS/kernel line, graph mode, MTP level, context
  length, temperature, burst size.
- **Crash-state awareness:** never measure after a crashed engine start without a reboot
  (Section 2.2). `uptime` before any run; a `journalctl -b` boot count mismatch = invalid.
- **Reference anchors (pre-rebuild):** stage-v24f smoke on boot 3915112 — 8/8 bursts clean;
  pre-fix baseline 5 stalls / 7 bursts on stage-v24d; L0-absence 0/170 (`appendUSMMemcpy`).

## 7. Evidence anchors (docs already on disk)

- `docs/campaigns/2026-09-19-connector-v3-fix-and-smoke.md` — v3 design, smoke gates, commit footgun, md5 gate.
- `docs/campaigns/2026-09-19-v3-campaign-prereg.md` — campaign protocol + pre-registered decision rule.
- `docs/incidents/2026-09-19-gdn-capture-l0-wedge.md` — the boot-4010612 wedge this runbook's P4/P5 fix.
- `docs/incidents/2026-09-19-stall-mechanism-ple-l0-contention.md` — the op-level wedge mechanism.
- `docs/incidents/2026-09-19-wedge-census-resplit.md` + `docs/campaigns/2026-09-19-ple-extinction-and-watchdog-v2.md` — watchdog v2/v2.5 semantics.
- `docs/campaigns/2026-09-19-ple-cpoffload-gate-and-campaign.md` — VLLM_PLE_CPU_OFFLOAD gate + discriminator ladder.
- `docs/rebuild/phase3-hardware-upgrade.md` — post-upgrade rig context (host RAM / disk floors).

*End of runbook. Edit sections marked with ▢* before execution; never skip a gate.*
