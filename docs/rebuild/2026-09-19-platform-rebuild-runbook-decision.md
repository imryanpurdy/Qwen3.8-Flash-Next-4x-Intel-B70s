# Runbook addendum — DECISION FILLED (2026-09-20, from omix-target-bom doc)

**OS target: Ubuntu 24.04-HWE + OMIX 0.4-repo packages with Intel's pinned
6.17.0-1010-intel kernel. GuC 70.65 + G31 IFWI 775 per the OMIX support
matrix. NEVER GuC 70.72.1 (GSD-13481 deterministic TP2 startup hang).**

Why (source-backed, docs/rebuild/2026-09-19-omix-target-bom.md):
- 26.04 path ships NO kernel in the OMIX repo — it would keep jobe's
  7.0.0-31, which is the permanent-wedge kernel line in every tracker report.
- All permanent-wedge reports are kernel 7.0 on NON-OMIX mixes; the only clean
  high-throughput dual-B70 datapoint is 6.17 + GuC 70.44.1.
- OMIX 0.4.0: LZ 1.32.0, UMD 26.31.39395.13, oneCCL 2022.1.2; 0.3.0: LZ
  1.28.6, CR 26.22.38646.7, oneCCL 2022.1.1. oneCCL 2022.x does NOT claim the
  #212 fix (issue open/unassigned) → CCL_ZE_CACHE_OPEN_IPC_HANDLES=0 stays.
- intel/vllm:0.21.0-xpu BOM: OMIX 0.1.0 base, torch 2.11, oneCCL
  2021.15.9.14-Arc, validated Ubuntu 25.04/KMD 6.14/IOMMU OFF. Container OK;
  jobe's HOST was the unsupported mix.
- Current host baseline (recorded pre-rebuild): kernel 7.0.0-31, GuC
  70.58.0/HuC 8.2.10, libze1 1.32.0-1~26.04~ppa1, UMD 1.15.39122,
  torch 2.11.0+xpu (container).

Post-rebuild verification ladder unchanged (L1-L5). Rollback anchor:
stage-v24g image (docker save + sha256) + .env + flashnext-scout/.
NOTE: start.sh now also carries v24h/v24h2 (capture-size generator + guard +
CAP_SIZES_LIST override) — commit 3589249 — and .env carries
CAP_SIZES_LIST=1,2,3,4,5,6,7,8,12,16 and MAX_NUM_SEQS=16. Preserve both.
