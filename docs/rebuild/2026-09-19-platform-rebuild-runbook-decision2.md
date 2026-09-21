# Runbook addendum 2 — vxk upgrade is part of the rebuild (2026-09-20, from vxk-gdn-capture-intel)

**The rebuild now includes vllm-xpu-kernels → v0.1.14+ (full conf configs), not
just OMIX/OS.** Source: docs/incidents/2026-09-19-vxk-gdn-capture-intel.md (committed
with this file).

Why this changes the MTP1 lane:
1. Our pinned vxk `0.1.8.3.dev0+g3cab97a` (base 2026-05-21) predates native
   GDN spec-decode (#368, 2026-05-26) by 5 days. Our MTP1 therefore runs the
   Triton/FLA fallback, which upstream #336 documents as ~10× slower and #487
   calls "breaks XPUGraph capture/replay."
2. **Working theory for the MTP1-capture DEVICE_LOST (all three reproductions):
   the Triton/FLA spec fallback breaking graph capture — a documented upstream
   failure mode, not an unknown platform bug.** Falsifiable prediction: on the
   rebuilt stack (vxk ≥ 0.1.14 native GDN spec path), MTP1+graphs either
   captures cleanly or fails with a NON-DEVICE_LOST error.
3. vxk fixes we inherit at 0.1.14: #537 (mixed spec/non-spec batches),
   #544/#545, #599/#600 ragged GDN (the FP8-lab +73% single-stream lane per
   HANDBACK), #535 closed (head_dim 512/576 register exhaustion, split-V),
   #548 closed (T%64==5 NaN, root cause vllm#53059).
4. Shape coverage: our 0.1.8.2 build AOT-compiles the FULL matrix (not our
   gap). 0.1.14 default wheels ship small presets — the rebuild MUST use the
   full `chunk_prefill_{default,full}.conf` / `paged_decode_{default,full}.conf`
   configs (format `headsize,paged,causal,local,sink,lse` /
   `qgroup,headsize,pagesize,causal,local,sink`), else local sliding-window
   layers and pagesize-128 shapes silently fall back.
5. Torch 2.13.0+xpu + oneAPI 2026.0 is the DOCUMENTED matrix of vxk 0.1.14
   (README + pyproject) — confirms the rebuild pair Ryan specified.
   (vxk main has moved to torch 2.14/ess 2026.1.2 — stay on the 0.1.14 line.)

Related open upstream issues to watch after rebuild: #559 (moe_gather
CAT/DEVICE_LOST on B70 — EXACTLY our workload: Flash-Next top-10 + MTP, TP=4),
#457 (Xe2 MoE GEMM not capturable at batch>1; workaround capture_sizes=[1] —
resonates with our ≤12-clean/16-fault capture pattern), #567. If MTP1 capture
still faults post-rebuild, #559/#457 are the first suspects, and the fix is
kernel work — Intel's new-sycl-kernel skill (repo .claude/skills/) covers the
CUDA→SYCL port workflow for that contingency.

Runbook patch table gains: **P8 — vxk 0.1.14+ build with full conf configs +
verify `pip show vllm-xpu-kernels` version + conf-file presence in the image.**
