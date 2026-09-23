# 2026-09-22 — NEO host-GTT mirror OOM on kernel 7.0.0-31

**Artifacts:** v1-lane OOM ×3 (2026-09-22 night), server logs, boot ledger (B6-*), pair A/B outputs.

## What happened

On kernel 7.0.0-31 with intel-compute-runtime (NEO) of the PPA-era stack, the v1 production config (98K KV pool) OOMs at boot ×3, consistently at the same point: NEO mirrors device allocations into host GTT, and at 98K the GTT mirror demand exceeds host RAM floors → boot dies.

## Why 6.17 is the platform of record

- On 6.17 + GuC 70.65 the same config boots and serves 98K (pair-verbatim 210, needle PASS, soak 195.9 with the t120 image).
- The GTT-mirror demand is driver-behavior, not a config bug: same .env on 6.17 boots, on 7.0 it OOMs ×3 at v1 knob set.
**v1 OOM ×3 was platform-specific (7.0), not config.**

## Operating rule

**6.17.0-1010-intel is the platform of record.** On 7.0.0-31 the v1 line does not fit: 98K KV unreachable, host RAM floor breached by the NEO GTT mirror. The testing platform is fine for short-context (pair A/Bs landed, soak 195.9, reconciliation engine) but not the v1 acceptance platform.

The 6.17 acceptance reboot (next step) is where the v1 numbers-of-record are measured.
