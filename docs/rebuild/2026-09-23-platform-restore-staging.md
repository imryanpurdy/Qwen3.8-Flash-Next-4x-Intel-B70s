# 2026-09-23 — Platform-restore attempt to 6.17 + GuC 70.65 (staging; acceptance pending)

**Artifacts:** restore-617.sh (read-back green), ledger staging lines, FW-RECEIPT sha256, live GuC md5 201f80b9.

## What was done

After the v1-lane soak and A/B batteries completed on kernel 7.0.0-31 + GuC 70.58 (testing platform), the box was re-staged for the 6.17 acceptance run:

1. **GuC 70.65 restored live** from the verified backup `/lib/firmware/xe/bmg_guc_70.bin.v7065.bak` (sha256 `70d74627…67bb`, 377,664 B — matches linux-firmware fb0889c0 blob; live md5 201f80b9). Prior live blob (70.58, md5 cc5dbf7d) recorded for rollback.
2. **Grub `iommu=off`** set (the panic entry dropped from cmdline); read-back `GRUB_CMDLINE_LINUX_DEFAULT="iommu=off"`.
3. **saved_entry = 6.17.0-1010-intel**; one-shot cleared (benign `unset next_entry` error); 6.17 kernel + initrd present.
4. **Running engine preserved** — staging was done hot; changes apply only at next boot.

## Conclusion

Staging is green; the acceptance reboot (into 6.17.0-1010-intel + GuC 70.65 + iommu=off) is the operator's call and is the first step of the fresh-clone acceptance: read-back (uname -r, GuC version via dmesg, iommu cmdline, LimitNOFILE), then fresh clone → `.env.example` → start.sh → verify.sh → numbers of record.

**Status: staged, not yet booted** — this doc is updated with read-back results at acceptance.
