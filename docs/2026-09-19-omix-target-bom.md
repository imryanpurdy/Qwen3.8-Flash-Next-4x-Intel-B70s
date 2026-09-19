# Target BOM: OMIX 0.3 on Ubuntu 24.04-HWE (6.17) vs Ubuntu 26.04 (7.0) — Jobe 4× B70

**Date:** 2026-09-19 · **Scope:** web research only, no rig access. Every version cited below is source-backed; items I could not source are tagged **UNVERIFIED** and never guessed.

---

## Recommendation (one paragraph)

**Move Jobe onto the Intel OMIX 0.3.0 (or newer 0.4.0) repository on Ubuntu 24.04 LTS + HWE kernel 6.17 — not Ubuntu 26.04/kernel 7.0.** The decisive evidence is not the support matrix (OMIX 0.3.0 is nominally supported on both 24.04 and 26.04 — [release notes](https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.3.0.html)) but the field record: the only fully clean, high-throughput dual-B70 datapoint on record is kernel 6.17 + GuC 70.44.1 (Zumbasam: 448 tok/s @ C=64, 0 resets, 0 failed requests — [vLLM #41663](https://github.com/vllm-project/vllm/issues/41663)), while *every* report of the permanent wedge (fault-error `-ENOENT/-EINVAL` + ccs/bcs reset that never recovers) is on kernel 7.0: marcorigodanzo 7.0.0-27/GuC 70.58 ([compute-runtime #948](https://github.com/intel/compute-runtime/issues/948)), fbiricotti 7.0.0-31/GuC 70.58 **and** 70.72.1, all four card/image/firmware combos FAIL ([#948 comments](https://github.com/intel/compute-runtime/issues/948)), localyouser 7.0.0-31/GuC 70.72.1/UMD 26.31, NateHag 7.0.0-30/GuC 70.58/UMD 26.35, marfrit 7.0.14/GuC 70.65/UMD 26.27. Second, OMIX actually *owns* the kernel on 24.04 — the OMIX 0.4 repo for `noble` ships `linux-image-6.17.0-1010-intel` ([repo Packages index](https://repositories.intel.com/gpu/ubuntu/dists/noble/intel-omix/0.4/unified/binary-amd64/Packages)), which is exactly the "Intel manages the kernel" promise in the task; on 26.04 the OMIX `resolute` repo ships **no** kernel package, so you are on stock 7.0 — the version correlated with all permanent wedges. Third, Intel validated the `intel/vllm:0.21.0` container with KMD **6.14.0-1011-intel with IOMMU OFF** on Ubuntu 25.04 — the 6.x kernel family, not 7.0 ([container release notes](https://raw.githubusercontent.com/intel/containers/main/dockerfiles/vllm/release_notes/0.21.0-xpu.md)). "Intel supports 26.04" and "kernel 7.0 wedges permanently" can both be true: every 7.0 wedge so far was on *non-OMIX* package mixes (distro 26.05 UMD, PPA 26.27/26.31/26.35 UMD, PPA L0 1.32.0, GuC 70.58/70.72.1), not on the OMIX-pinned set — so 26.04 may eventually work under OMIX 0.3 exactly, but 24.04-HWE is the only path that is simultaneously (a) Intel-tested, (b) kernel-pinned by Intel, and (c) backed by a clean, reproducible 448 tok/s community datapoint. **Plan:** purge the `intel-graphics` PPA, install the OMIX metapackage from the `noble/intel-omix/0.4` (or prefer `0.3` to stay on the task's target) repo, land on `linux-intel-6.17` (6.17.0-1010), keep the `intel/vllm:0.21.0-xpu` container as-is, verify GuC = 70.44.1/70.65 (NOT 70.72.1), set `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`, and run the 2–6 h soak before trusting it. Then re-test 26.04/OMIX on a spare disk as a controlled A/B (see §6).

---

## 1. What the field data says (quick evidence map)

| Reporter | OS / kernel | GuC | UMD (compute-runtime) | oneCCL path | Result |
|---|---|---|---|---|---|
| Zumbasam ([vLLM #41663](https://github.com/vllm-project/vllm/issues/41663)) | Ubuntu 24.04.4, **6.17.0-23** | **70.44.1** | container UMD (26.14/26.18 era) | 2021.17.2 (intel image) / source build; stable with `CCL_ENABLE_SYCL_KERNELS=0` | **STABLE**: 448 tok/s @ C=64 (444 baseline), 0/256 failures, 0 BCS resets, peak 66–67 °C. Set `CCL_ENABLE_SYCL_KERNELS=1` + `CCL_ALLREDUCE=ring` + `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` + `UR_L0_V2_FORCE_DISABLE_COPY_OFFLOAD=1` also got 448 — no perf cliff, no resets |
| marcorigodanzo ([compute-runtime #948](https://github.com/intel/compute-runtime/issues/948)) | Ubuntu 26.04, **7.0.0-27** | **70.58.0** | **26.05.37020.3** (distro, no PPA) | 2021.17-era + full "stable fallback" env from #41663 applied | **PERMANENT WEDGE** every 2–6 h under TP=2 load: `Fault response: Unsuccessful -ENOENT/-EINVAL`, ccs/bcs reset, Level-Zero context never recovers (EngineDeadError); 8/8 incidents correlate with dmesg resets; the env workarounds removed an earlier crash mode but not this one |
| davetha ([#948 comment](https://github.com/intel/compute-runtime/issues/948)) | Ubuntu 26.04, 7.0.0-27 | **70.58.0 → 70.65.0** | (single B580, no oneCCL/TP) | n/a | Firmware A/B: 70.58.0 → instant ccs/bcs reset + `-ENOENT`; **70.65.0 → faults gone** (~1 h heavy load, single-GPU; author explicitly says not yet proof for the multi-hour wedge) |
| marfrit ([#948 comment](https://github.com/intel/compute-runtime/issues/948)) | Ubuntu 26.04, 7.0.14-12 | **70.65.0** | 26.27.39122.11 | OpenVINO/OpenCL | **Faults on 70.65.0 too**. Allocation-log shows faulting VA = the runtime's own `SEMAPHORE_BUFFER` (evicted under VRAM pressure → live semaphore wait at non-present page). Driver/firmware can't explain this one alone |
| fbiricotti ([#948 comment](https://github.com/intel/compute-runtime/issues/948)) | Ubuntu 24.04.4, **7.0.0-31** | 70.58.0 **and** 70.72.1 | 26.05 and 26.27 | vLLM 0.28/0.29 XPU | 4× B70: **all four** (card pair × image × GuC) combos FAIL with `Timedout job`/ccs reset + cascade to both cards. Deterministic repro in 3–5 min |
| localyouser ([#948 comment](https://github.com/intel/compute-runtime/issues/948)) | Ubuntu 24.04, **7.0.0-31** | **70.72.1** | 26.31.39395.13-1~24.04~ppa1 (≈ OMIX 0.4 UMD, from PPA), L0 1.32.0-1~24.04~ppa1 | llama-server | Faults under long prefill (bcs `-EINVAL` then ccs timeout, same BCS VA across two ASIDs) |
| NateHag ([#948 comments](https://github.com/intel/compute-runtime/issues/948), 2026-09-19) | Ubuntu 26.04, 7.0.0-30 | 70.58.0 | 26.35.39758.10 | single B70, vLLM 0.25.1 | Dies under steady load (2,793 req in ~2 min). Four eliminations: driver version, userspace build, compiler (ocloc matched), and `SYCL_UR_USE_LEVEL_ZERO_V2=0` all still fault → "sits below all four" |

**Reading:** the `ccs/bcs reset + Fault response: Unsuccessful` family is broader than oneCCL — it shows up single-GPU, in llama.cpp, OpenCL and OpenVINO, on GuC 70.58/70.65/70.72.1 and UMDs 26.05–26.35. But the **permanent (unrecoverable) mode** correlates hard with kernel 7.0 + GuC ≥ 70.58; the only known-good dual-B70 production reference is 6.17 + GuC 70.44.1. Also new: a *separate* deterministic dual-GPU TP2 startup hang on **GuC 70.72.1** — GuC consumes `SCHED_CONTEXT` but never dispatches the LRC (GSD-13481, [compute-runtime #999](https://github.com/intel/compute-runtime/issues/999)) — so newer firmware is demonstrably not a safe upgrade.

---

## 2. Version table — OMIX pin vs Jobe current vs intel/vllm:0.21.0 BOM

OMIX pins from Intel's support matrix ([omix-support-matrix.html](https://dgpu-docs.intel.com/overview/support-matrix/omix-support-matrix.html)) and per-version release notes ([OMIX 0.3.0](https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.3.0.html), [0.4.0](https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.4.0.html)); container BOM from [intel/vllm:0.21.0-xpu release notes](https://raw.githubusercontent.com/intel/containers/main/dockerfiles/vllm/release_notes/0.21.0-xpu.md), the [0.21.0 Dockerfile](https://raw.githubusercontent.com/intel/containers/main/dockerfiles/vllm/0.21.0-ubuntu24.04.dockerfile), and the image config on Docker Hub (`image.omix.version=0.1.0`, `VLLM_VERSION=0.21.0`).

| Component | OMIX **0.3.0** pin | OMIX **0.4.0** pin (newer; same BOM shape) | **Jobe current** | **intel/vllm:0.21.0-xpu** (in-container) | Verdict (vs Jobe) |
|---|---|---|---|---|---|
| Ubuntu | 26.04 / 24.04.4 / 24.04+HWE | 24.04, 26.04, RHEL 9.8/10.2 | 26.04-ish | base image: `intel/omix:0.1.0-devel-ubuntu24.04` (itself validated on host 24.04.4; release notes BOM lists validation host = Ubuntu 25.04) | **Move to 24.04+HWE** (§3) |
| Kernel | 24.04: OMIX repo ships **`linux-intel-6.17` → 6.17.0-1010-intel**; 26.04: **stock 7.0 (repo ships no kernel)** | same (0.4 repo: 6.17.0-1010-intel; resolute 0.4: no kernel pkg) | **7.0.0-31** | validated with KMD **6.14.0-1011-intel, IOMMU OFF** (host) | **Clearly off-pin** — 7.0 is the wedge correlate; 6.14/6.17 is the validated family |
| Level Zero (loader/tool) | **1.28.6** | **1.32.0** | **1.32.0-1~26.04~ppa1** (PPA) | 1.28.2 (OMIX 0.1.0) | Loader version ≈ OMIX 0.4.0 pin, but from **PPA** — component version is fine, packaging is not (PPA is unvalidated + can drift) |
| Compute Runtime (UMD `libze_intel_gpu`) | **26.22.38646.7** | **26.31.39395.13** | **`libze_intel_gpu.so.1.15.39122`** → build 39122 ≈ **26.27.39122.x** (see marfrit env line "26.27.39122.11", [same thread](https://github.com/intel/compute-runtime/issues/948))* | 26.14.37833.4 (OMIX 0.1.0) | **Off-pin both directions**: newer than OMIX 0.3's UMD, older than 0.4's; PPA build (26.27.39122 line) — *not any OMIX pin* |
| GuC firmware (host linux-firmware) | **70.65** (required, all OMIX releases) + G31 **IFWI 775** | same | **UNVERIFIED** — task context: "70.58 era" | n/a (host-provided) | **Must check `sudo dmesg \| grep -i guc`**; 70.58 = wedge-era; 70.65 = validated; **do NOT install 70.72.1+** (§5) |
| oneCCL | **2022.1.1** | 2022.1.2 | host: n/a (container carries its own); container's = **2021.15.9.14** (Arc `-arc` branch) | **2021.15.9.14** (standalone installer `intel-oneccl-2021.15.9.14_offline.sh` replaces OMIX's 2021.17.2; PyTorch-bundled oneccl uninstalled to avoid conflicts) | **Load-bearing row — see §4.** OMIX 0.3's 2022.1.1 is past 2021.17, but the #212 stale-handle fix is **NOT confirmed** in any 2022.x release note |
| oneDNN | **NOT included** (removed in 0.3.0) | NOT included | n/a | from PyTorch 2.11 wheel (oneDNN 3.11.x per vLLM#41663) | Non-issue: oneDNN now rides with PyTorch; no OMIX decision needed |
| oneMKL | **2026.1.0** | 2026.1.0 | host n/a (container bundles it — OMIX 0.1.0 = 2025.3.1) | 2025.3.1 | Fine in-container; OMIX 0.3 host oneMKL is 2026.1.0 if jobe ever runs host-side SYCL |
| SYCL compiler (DPC++) / base oneAPI | **2026.1.0** (base oneAPI 2026.1.2) | 2026.1.1 (base 2026.1.3) | σ8 **UNVERIFIED** (host) | 2025.3.3 (OMIX 0.1.0) | In-container already Intel-validated; host compiler only matters if building PyTorch/IPEX extensions on host |
| IGC (OpenCL compiler) | **2.36.5** | 2.40.13 | **UNVERIFIED** | 2.32.7 | Container self-consistent |
| XPU Manager (xpu-smi) | **2.0.0** | 2.0.1 | **UNVERIFIED** | 1.3.6 | Monitor-only; low risk |
| PyTorch (in container) | n/a (OMIX is userspace/libs) | n/a | 2.11.0+xpu | **2.11** (2.11.0+xpu, per [requirements/xpu.txt @ v0.21.0](https://raw.githubusercontent.com/vllm-project/vllm/v0.21.0/requirements/xpu.txt)) | ✅ matches |
| vllm-xpu-kernels | n/a | n/a | n/a | release notes **0.1.8.2**; `xpu.txt` pins **0.1.7** — small internal discrepancy, note only | ✅ matches (0.21.0 line) |
| vLLM | n/a | n/a | 0.21.0 | **0.21.0** + Intel patches (`v21.patch`, `pr-43426.patch` from llm-scaler branch `omix-vllm-0.21.0`) | ✅ as shipped |

\* The parent-provided `libze_intel_gpu.so.1.15.39122` maps to compute-runtime build **39122**, which appears in the wild as **26.27.39122.x** (reported by marfrit and referenced by NateHag). Exact minor is **UNVERIFIED** — confirm with `apt-cache policy intel-opencl-icd libze-intel-gpu1` on the rig. Either way it is a non-OMIX PPA build (OMIX 0.3 = 26.22.38646.7, OMIX 0.4 = 26.31.39395.13).

**Verdict per row (summary):** in-container everything is internally consistent and Intel-validated (PyTorch 2.11, kernels, oneCCL 2021.15.9.14-arc, UMD 26.14.37833.4, L0 1.28.2). The *host* is the problem: kernel 7.0.0-31 + PPA-built libze1 1.32.0 + PPA UMD 26.27.39122 + GuC 70.58-era = precisely the "newer stack" cocktail every permanent-wedge report ran. Note that container UMD is self-contained (OMIX 0.1.0 userspace inside), so the host components that matter are **kernel (incl. xe/firmware) + GuC + IOMMU config**; do not bind-mount host `libze*`/`libze_intel_gpu*` into the container.

---

## 3. OS/kernel: 24.04-HWE/6.17 vs 26.04/7.0 — reasoning

**Recommend: Ubuntu 24.04 LTS + HWE kernel 6.17 (OMIX-repo `linux-intel-6.17` = 6.17.0-1010-intel).**

1. **Intel pins the kernel on 24.04; it doesn't on 26.04.** The OMIX repo for `noble` ships `linux-image-6.17.0-1009/1010-intel` + `linux-intel-6.17` ([Packages index](https://repositories.intel.com/gpu/ubuntu/dists/noble/intel-omix/0.4/unified/binary-amd64/Packages)); the `resolute` (26.04) OMIX repo contains **no** linux-* packages (verified for both the `0.3` and `0.4` suites) — on 26.04 you are on stock Ubuntu 7.0, and the OMIX install guide's "reboot to activate the required kernel" step is only meaningful on 24.04. OMIX owns the validation surface on 24.04.
2. **Every permanent-wedge report is kernel 7.0; the only gold-standard stable one is 6.17.** See §1. Zumbasam's 448 tok/s @ C=64 / 0 failures / 0 resets (2026-05-06 stable since) is kernel 6.17.0-23 + GuC 70.44.1. There is **no** published clean 7.0 datapoint anywhere.
3. **Intel's own container validation chose the 6.x family.** `intel/vllm:0.21.0-xpu` BOM: Host OS Ubuntu 25.04, KMD 6.14.0-1011-intel, IOMMU OFF. The same is true for the 0.17.0-xpu BOM (25.04, KMD 6.14.0) quoted in [vLLM #41663](https://github.com/vllm-project/vllm/issues/41663). Intel validates vLLM XPU on 25.04/6.14, not on 26.04/7.0.
4. **Kernel 6.8 GA won't do:** Battlemage (B70, G31) needs xe support that landed ≥6.12, hence 24.04 must be on HWE/OEM/OMIX kernel, not GA. Ubuntu 24.04.4 HWE officially ships 6.17 ([OMG!Ubuntu](https://www.omgubuntu.co.uk/2026/01/ubuntu-24-04-4-lts-hwe-update-kernel-mesa)); Ubuntu 26.04 GA ships 7.0 ([release notes summary](https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/)).

**Reconciling "Intel says 26.04" with "kernel 7.0 wedges":** both can hold. Intel's claim is that the *OMIX 0.3.0 pinned set* (UMD 26.22.38646.7, L0 1.28.6, oneCCL 2022.1.1, oneAPI 2026.1, GuC 70.65) is validated on 26.04. No wedge report to date used that OMIX-pinned set on any OS — they used distro UMD 26.05.37020.3 (marcorigodanzo — explicitly "distro-provided, no PPA", *still* a different UMD than the OMIX pin), PPA UMD 26.27.39122/26.31.39395/26.35.39758, and GuC 70.58/70.72.1. So the wedge reports say "newer *unvalidated* combos wedge", not "kernel 7.0 wedges per se".

**What evidence would distinguish the two hypotheses ((a) 7.0/xe/GuC is inherently wedge-y vs (b) the wedge came from the PPA/unpinned mix):**
- **A/B with OMIX exact on 26.04:** fresh 26.04, OMIX 0.3 repo (or 0.4), no PPA, no other Intel repos, `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`, GuC 70.65, ~6 h sustained TP=2 soak. Clean → hypothesis (b) (and 26.04 becomes acceptable long-term). Wedges → hypothesis (a), and Intel's 26.04 support is nominal/untested-in-practice.
- **Control A/B with OMIX on 24.04:** same rig/disk layout, 24.04 + OMIX + 6.17-intel + same soak. Known-good baseline (Zumbasam) sits here.
- **Signature triage at first fault** (cheap, decisive): `Fault response: Unsuccessful -ENOENT/-EINVAL` + engine reset = the fault-driven wedge family (#948); `Timedout job` with GuC consuming SCHED_CONTEXT but LRC never dispatched = GSD-13481 (GuC 70.72.1, [#999](https://github.com/intel/compute-runtime/issues/999)); fault VA inside `SEMAPHORE_BUFFER`/`COMMAND_BUFFER` = residency/eviction class (marfrit).
- **Keep oneCCL constant across the A/B** — if the 2021.15.9/2022.1.1 difference correlates with failure, it's the oneCCL/UMD layer, not the kernel.

---

## 4. oneCCL #212 — the load-bearing row

**Status of the bug:** oneCCL#212 ([uxlfoundation/oneCCL #212](https://github.com/uxlfoundation/oneCCL/issues/212), marcorigodanzo) — *"Stale cached opened Level-Zero IPC handle after peer buffer realloc → GPU page fault + infinite hang in ze_base_entry::is_event_completed (Battlemage, world_size=2)"*. Reproduced on **oneCCL 2021.17 / 2021.17.2** + PyTorch 2.12+xpu + vLLM with buffer reallocation between collectives; faults in 15–30 s; hang is permanent because `ze_base_entry::is_event_completed()` never returns and oneCCL's wait has no timeout. Measurements (C=10 soak): default = fault+hang ~3 min; `CCL_ZE_CACHE=0` = 0 faults but −39% throughput; **`CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` = 0 faults, −3%**.

**Reporter's root-cause analysis (source-level, tag 2021.17.2):** the GET-side `ipc_handle_cache` has an evict-on-free tracer on `zeMemFree` — but it is **only registered when `ZE_ENABLE_TRACING_LAYER` is set** (dead by default → silently dangerous). Recommended fix = handle→allocation lifetime tracking; robust fix = bounded wait in `ccl_executor::wait()`.

**Is it fixed in OMIX 0.3's oneCCL 2022.1.1?** **UNVERIFIED — do not assume yes.**
- OMIX 0.3.0 = oneCCL **2022.1.1**; OMIX 0.4.0 = 2022.1.2 — both *past* 2021.17, and 2022.0.0 is a genuinely new line with Arc-Pro-B-Series optimized scale-up (SPMD for allreduce/allgather/alltoall/reduce-scatter/broadcast/pt2pt, low-latency protocol, SYCL graph support — [oneCCL 2022.0.0 release notes](https://github.com/uxlfoundation/oneCCL/releases) / tag page).
- **But no 2022.x release note mentions the stale-IPC-handle or evict-on-free fix.** 2022.0.0 notes: Arc Pro B-Series support, SPMD, reductions, window/alloc APIs. 2022.1.0 notes: "Fixed a potential invalid memory access in alltoall", "Improving concurrent insertion to shared resources", "get_last_event no longer returns the runtime's stored last event", unaligned buffer handling, GPU_RDMA pt2pt. "Concurrent insertion to shared resources" *could* touch the IPC cache but no note claims the #212 class; issue #212 is **still open, no assignee, no maintainer response** (checked 2026-09-19).
- Therefore: **treat oneCCL 2022.1.1/2022.1.2 as "probably improved, unproven" and keep the workaround until Intel/UXL confirm or you A/B the reproducer (15–30 s to fail on 2021.17; run it against 2022.1.1).** Belt-and-suspenders on all stacks regardless of OMIX version:
  - `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` (the cheap, −3% fix) — plus the already-known stable set: `CCL_ENABLE_SYCL_KERNELS=0`, `CCL_ATL_TRANSPORT=ofi`, `CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0`, `CCL_ZE_IPC_EXCHANGE=pidfd|sockets`, `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE`, `ZE_AFFINITY_MASK=0,1,2,3`, `SYCL_UR_USE_LEVEL_ZERO_V2=0`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `VLLM_XPU_ENABLE_XPU_GRAPH=1` ([#41663 stable fallback](https://github.com/vllm-project/vllm/issues/41663)).
- **Note for jobe specifically:** jobe's container ships oneCCL **2021.15.9.14** (the `-arc` branch — different branch from 2021.17.x, but the same ZE-IPC-handle cache architecture). Whether 2021.15.9 shares the #212 stale-handle defect is **UNVERIFIED**; its release notes only say "bug fixes on Arc A and B Series" ([tag page](https://github.com/uxlfoundation/oneCCL/releases)). 2021.15.9 is what Intel validated 0.21.0 against, but run with `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` anyway.
- **Bonus:** OMIX 0.2.0+ moves to oneCCL 2022.x, which is the line with real B70 scale-up optimization (2022.0.0 "Intel Arc Pro B-Series support delivers optimized scale up performance leveraging low latency protocol"); OMIX 0.1.0's own release notes admit its oneCCL 2021.17.2 "does not include full optimization for PCIe collective communications. Multi-GPU use cases will not run at full performance." ([OMIX 0.1.0 notes](https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.1.0.html)) — so Host OMIX 0.3/0.4 + container 0.21.0 is the right pairing if you want both validated *and* fast scale-up.

---

## 5. GuC firmware — don't reflexively upgrade

- **Intel's validated firmware:** OMIX support matrix requires **GuC 70.65** (and G21/G31 IFWI 775) for *all* OMIX releases ([omix-support-matrix.html](https://dgpu-docs.intel.com/overview/support-matrix/omix-support-matrix.html), "Supported card firmware"). If jobe's cards are not at IFWI 775, flash before blaming software (check `xpu-smi` / `dmesg`).
- **Do not go to 70.72.1 or newer.** GuC 70.72.1 introduced a *new deterministic* dual-GPU TP2 startup hang (GSD-13481, [#999](https://github.com/intel/compute-runtime/issues/999)) — strictly worse than the sporadic wedge; fbiricotti's 4-card matrix failed on 70.72.1 too.
- **Field results by GuC version:** 70.44.1 = the only fully clean dual-B70 production datapoint ([#41663](https://github.com/vllm-project/vllm/issues/41663)); 70.58.0 = immediate/repeated `-ENOENT` faults, permanent wedge on 7.0 ([#948](https://github.com/intel/compute-runtime/issues/948), davetha A/B); 70.65.0 = davetha's A/B eliminated the faults (single B580, ~1 h — not conclusive) **but** marfrit faults on 70.65.0 too (B60, 7.0.14, UMD 26.27); 70.72.1 = new deterministic hang (#999) + still faults in fbiricotti's matrix.
- **Recommended course:** match the OMIX-validated **70.65** (it addresses the obvious 70.58 regression and is the only version Intel will stand behind), *if* your kernel/linux-firmware package carries it (Ubuntu `linux-firmware` has it since 2026-07-14 per davetha; on 24.04-HWE/6.17 check availability — a manual drop of `xe/bmg_guc_70.bin` + `update-initramfs -u` is the documented community method ([IDFS write-up](https://idfs.ai/blog/six-days-with-the-intel-arc-pro-b70))). If wedges persist after everything else is pinned, **A/B 70.44.1** (Zumbasam's exact firmware) before blaming the stack. Never take `latest` blindly: verify with `sudo dmesg | grep -i 'guc'` → `GuC firmware: ... 70.xx`.

---

## 6. Distinguishing experiment (what would settle 24.04 vs 26.04)

Two identical-toolchain soaks on the same hardware, same container, same model (Qwen3-30B-A3B, TP=4 or TP=2, ~8–12 concurrent, mixed 4k–32k contexts — the marcorigodanzo repro pattern):

1. Disk A: Ubuntu 24.04.4 + OMIX 0.3 (or 0.4) repo, `linux-intel-6.17`, GuC 70.44.1→70.65 A/B, no PPA, `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`, IOMMU off (match Intel's BOM), 6 h soak. Target: Zumbasam-ish 400+ tok/s with 0 resets.
2. Disk B: Ubuntu 26.04 + OMIX 0.3 repo, stock 7.0.0-x, same env, 6 h soak.

Outcomes: B clean → keep 26.04 (or upgrade later; still prefer 24.04 for the Intel-managed kernel); B wedges → stay on 24.04-HWE and treat Intel's 26.04 line as "supported on paper" until Intel ships a kernel for resolute or fixes the 7.0/KMD pair. Also capture at first fault: full `dmesg`, `journalctl -k -b`, `lspci -vvv -k`, `xpu-smi` config, `sysctl vm.overcommit`, and whether p2p is `ze_p2p` or IPC (jobe = no XeLink, so IPC path — that's the #212 family's terrain).

---

## 7. Immediate action checklist for Jobe

1. **Purge the PPA stack** (host): remove `intel-graphics` PPA and `libze-intel-gpu*`, `intel-opencl-icd`, `libze1` etc.; OMIX install guide explicitly demands a clean system — "Use a clean system without preinstalled Intel GPU user-mode packages from the PPA… newer Intel packages from the PPA or other repositories can cause dependency conflicts or require package downgrades" ([installing-omix.html](https://dgpu-docs.intel.com/installation-guides/installing-omix.html)); the [installation-path selector](https://dgpu-docs.intel.com/installation-guides/index.html) says the same: "Do not use this path on systems that rely on Intel® OMIX, because PPA packages may conflict with Intel OMIX pinned versions."
2. **OS/kernel:** install on/ restore 24.04 + OMIX `noble/intel-omix/0.3` (or 0.4) → `intel-omix` metapackage → boots `6.17.0-1010-intel`.
3. **Container:** keep `intel/vllm:0.21.0-xpu` as-is (its userspace is the validated set); do **not** bind-mount host `libze*`; run with `--ipc=host --shm-size=16g --group-add=render --group-add=video`.
4. **Env (container):** `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0` + the #41663 stable set (above). Consider `UR_L0_USE_IMMEDIATE_COMMANDLISTS=0` (Zumbasam's libze fix) and `CCL_ALLREDUCE=ring` if the SYCL-kernel path is wanted — but shipping the profile-A set (`CCL_ENABLE_SYCL_KERNELS=0`) is the only production-proven one.
5. **Firmware:** verify GuC (`dmesg`) and IFWI (775) — target 70.65, never 70.72.1.
6. **IOMMU:** Intel's BOM says IOMMU OFF — check/follow on the rig (or at least confirm no quirks in dmesg).
7. **Soak:** 6 h sustained TP=2/TP=4 load before calling it done.

---

## 8. UNVERIFIED / open items (explicit)

- **oneCCL 2022.1.1/2022.1.2 fix for oneCCL#212** — no release-note or maintainer confirmation; issue open, unassigned. Test with the #212 reproducer before dropping `CCL_ZE_CACHE_OPEN_IPC_HANDLES=0`.
- **Jobe's exact compute-runtime minor** (26.27.39122.x) — inferred from `.so.1.15.39122` + marfrit's env line; **UNVERIFIED**, check `apt-cache policy`.
- **Jobe's GuC version** — task context says "70.58 era"; must confirm on rig.
- **Whether 2021.15.9-arc shares the #212 defect** — same cache architecture, different branch; no data.
- **GuC 70.65 = clean long-term?** — only ~1 h single-GPU datapoint (davetha) + one contradictory fault (marfrit, B60). The only clean long-term firmware is 70.44.1 (6.17).
- **OneDNN row in the vLLM container** — bundled via PyTorch wheel (3.11.x per #41663); not independently verifiable from public release notes.
- **vllm-xpu-kernels 0.1.7 (xpu.txt) vs 0.1.8.2 (release notes BOM)** — minor doc/pin discrepancy; the shipped image used one of them; not load-bearing for the host decision.
- **OMIX kernel on 26.04** — confirmed absent from `resolute/intel-omix/{0.3,0.4}` Packages indexes; if Intel adds one later, revisit.

## Sources (primary)

- Intel OMIX release notes 0.1.0/0.2.0/0.3.0/0.4.0: https://dgpu-docs.intel.com/overview/release-notes/OMIX/0.1.0.html, `/0.2.0.html`, `/0.3.0.html`, `/0.4.0.html`
- Intel OMIX support matrix (versions + GuC 70.65 / IFWI 775): https://dgpu-docs.intel.com/overview/support-matrix/omix-support-matrix.html
- Installing OMIX (clean-system/PPA prohibition): https://dgpu-docs.intel.com/installation-guides/installing-omix.html ; path selector: https://dgpu-docs.intel.com/installation-guides/index.html
- OMIX container release notes (0.1.0, 0.3.0): https://dgpu-docs.intel.com/overview/release-notes/containers/OMIX/0.1.0.html, `/0.3.0.html`
- intel/vllm:0.21.0-xpu BOM: https://raw.githubusercontent.com/intel/containers/main/dockerfiles/vllm/release_notes/0.21.0-xpu.md ; Dockerfile: https://raw.githubusercontent.com/intel/containers/main/dockerfiles/vllm/0.21.0-ubuntu24.04.dockerfile ; image labels (omix 0.1.0) via Docker Hub registry config, digest sha256:f5eecec8…
- vLLM 0.21.0 XPU requirements: https://raw.githubusercontent.com/vllm-project/vllm/v0.21.0/requirements/xpu.txt
- oneCCL#212 (stale IPC handle): https://github.com/uxlfoundation/oneCCL/issues/212 ; releases: https://github.com/uxlfoundation/oneCCL/releases
- vLLM#41663 (Zumbasam 6.17/70.44.1 stable + 448 tok/s; 0.17.0 BOM quote): https://github.com/vllm-project/vllm/issues/41663
- compute-runtime#948 (marcorigodanzo wedge, kernel 7.0.0-27/GuC 70.58) + comments (davetha 70.58→70.65 A/B; marfrit; fbiricotti 4×B70 matrix; localyouser; NateHag): https://github.com/intel/compute-runtime/issues/948
- compute-runtime#999 (GuC 70.72.1 deterministic TP2 startup hang, GSD-13481): https://github.com/intel/compute-runtime/issues/999
- Ubuntu 26.04 = kernel 7.0: https://documentation.ubuntu.com/release-notes/26.04/summary-for-lts-users/ ; Ubuntu 24.04.4 HWE = 6.17: https://www.omgubuntu.co.uk/2026/01/ubuntu-24-04-4-lts-hwe-update-kernel-mesa
- OMIX repo Packages indexes: https://repositories.intel.com/gpu/ubuntu/dists/noble/intel-omix/0.4/unified/binary-amd64/Packages (kernel 6.17.0-1010-intel present) ; `…/resolute/intel-omix/0.3/…` and `/0.4/…` (no kernel packages)
