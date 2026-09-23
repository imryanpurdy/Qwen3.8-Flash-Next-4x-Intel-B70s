# 2026-09-22/23 — Platform rebuild as executed

**Artifacts:** boot ledger `.run/boot-ledger.txt` (box: `/home/bonz/rollback-unit/.run/`), receipts in `evidence/`, FW-RECEIPT (sha256 below).

## Timeline (as executed)

1. **Kernel/firmware stage (OMIX 0.4):** 6.17.0-1010-intel + intel-omix 0.4 installed; GuC 70.65 installed from linux-firmware commit `fb0889c0` — sha256 `70d74627e395…67bb`, 377,664 B (no ASCII version header in GuC blobs; hash + size + commit provenance IS the verification).
2. **6.17 boots B3w→B4t** — the full validation battery ran on 6.17 (needle verdicts, parsers, kvb, MML, DEVICE_LOST probes).
3. **Rollback lane** (`stage-v24h2:rollback`, then t120): image ID `0ca8598598ba…`, connector patched to 120 s staging. One variable per boot; ledger line at every boot with boot ID + md5s.
4. **Platform restore to 7.0.0-31 + GuC 70.58** (2026-09-22 evening) for the v1-lane soak after 6.17 v1 OOM ×3 (NEO host-GTT mirror — separate doc).
5. **t120 boots + validation** on 7.0 testing platform: SOAK-V1 195.9 (bench_harness-era), needle 98K PASS, pair A/Bs.
6. **Reconciliation (2026-09-23):** Saturday's soakfix.py on tonight's engine = 299.9 sustained — platform exonerated (see measurement-reconciliation doc).
7. **2026-09-23 02:xx:** 6.17 re-staged for the v1 acceptance run: GuC 70.65 restored live from verified backup `/lib/firmware/xe/bmg_guc_70.bin.v7065.bak` (live md5 201f80b9), grub `iommu=off` (panic removed), saved_entry 6.17.0-1010-intel. Staging only; reboot is the operator's call.

## Outcome

Validated production line: **MTP0, MML 98304, MNS 16, capture list `1,2,3,4,5,6,7,8,12,16,24,32`, kv-bytes 9494279680, parsers qwen3_xml/qwen3, LPT 1024, t120 image, watchdog v1** on kernel 6.17.0-1010-intel + GuC 70.65. The 6.17 acceptance reboot follows this doc (fresh clone → verify.sh → numbers-of-record).
