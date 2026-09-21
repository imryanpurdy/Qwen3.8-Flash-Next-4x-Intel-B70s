# 2026-09-18 Forensic Addendum — Dead-Cgroup Counters Convicted

## Context
A hung-container cgroup (the "corpse") had shown kernel=121.4 GiB (≈128GB device VRAM —
suspicious coincidence), slab=31.7 GiB, memory.peak=58 TB. Claude flagged: dead cgroup
accounting is exactly where stale garbage lives; system-wide counters never moved when MA fell
26GB, which cuts against any real 121GB kernel charge. Ryan: cards are in plain PCIe slots,
nothing weird topology-wise.

## Measurement (launch59, stage-v24c, live container)
Idle x3 baseline then 10 samples across a sustained 8-way 256-tok decode:
- memory.current: 16.8 → 16.9 GiB (delta +64 MiB total)
- anon 13.4 GiB flat; file 2.8→2.9 GiB; slab 164→169 MiB; kernel 424→440 MiB; pagetables 248→258 MiB
- Host meminfo same window: AnonPages ~14.2 GiB, Slab ~1.87 GiB, flat to the MiB

## Verdict
1. Corpse fields were corrupt across the board: slab 31.7 GiB vs live 164 MiB; kernel 121.4 GiB
   vs live 410 MiB; peak 58 TB vs live 55.3 GiB. Stale dead-cgroup accounting, not measurement.
2. 121.4 ≈ 128GB was coincidence. During real decode nothing moves at GB scale. No mechanism
   exists for device VRAM to appear as cgroup kernel memory (cards are plain PCIe endpoints).
3. Driver memory lives in xe's own accounting (/sys/kernel/debug/dri/N/gtt_mm), readable ONLY
   host-side (privileged container). Reading gtt from inside the serving container yields 0 —
   debugfs mount artifact, not a real collapse. Host-side GTT ~311GB = PLE mmap host-backed by
   design; DtoD 252 GB/s (VRAM-verified) proves the decode hot set is device-resident.
4. Rule for future forensics: never quote a dead cgroup's memory.stat. Sample the live cgroup
   (docker exec ... /sys/fs/cgroup/memory.stat) during controlled load, or the host scope at
   /sys/fs/cgroup/system.slice/docker-<CID>.scope.
