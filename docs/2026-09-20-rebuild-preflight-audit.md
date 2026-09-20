# Pre-Rebuild Pre-Flight Audit — jobe (bonz@100.122.128.100, 4× Intel Arc Pro B70)

**Date:** 2026-09-20 · **Author:** subagent audit (no rig access — writing only)
**Scope:** one question per item — *if the SSD-only reinstall (Ubuntu 24.04-HWE + OMIX 0.4, kernel 6.17.0-1010-intel, GuC 70.65) proceeds, what fails to come back, and what needs physical access?*
**Reads:** `docs/2026-09-19-platform-rebuild-runbook.md` (+ decision/decision2 addenda), `2026-09-19-omix-target-bom.md`, `2026-09-14` migration scripts (`flashnext-scout/migrate-{B5,D,D2}.sh`, `fix-b.py`), `phase3-hardware-upgrade.md`, `firstlight-2026-09-16.md`, `2026-09-20-capture-size-cliff-and-16way.md`, session record (via task context), local repo state (`flashnext-recipe`, `flashnext-scout` — verified by git/sha).
**Convention:** `UNVERIFIED` = no evidence in any source; must be resolved by Section 3 enumerate or by hand after first boot.

---

## 0. Direct answer — what fails to come back

| Item | Returns? | Depends on |
|---|---|---|
| Rollback anchor `stage-v24g` image (25.4 GB class) | **NO** — `/var/lib/docker` is on the wiped SSD; only a `docker save` + sha256 executed **before wipe** brings it back | Section 1 #1 |
| Tailscale identity/state (`/var/lib/tailscale`) | **NO** — box unreachable at `100.122.128.100` until re-authed; **no pre-auth key ⇒ physical console** | Section 1 #11, R2 |
| `fn-recipe-int4/` (`start.sh` **v24h2**, `stop.sh`, `wedge-watchdog-v2.sh` **v2.5 HARDENED**, `check-weights.sh`, `scripts/`, `.env` (secrets), `.run/` evidence) | **NO** (on SSD) — local repo has the *unpatched* `start.sh` + patch scripts (`patch-capsizes.py` = commit 3589249, `patch-capoverride.py`), NOT the rig's live `v24h2` file and **not the `.env`** | Section 1 #4–6 |
| INT4 weights `/home/bonz/hf-int4` (≈169 GB, serving identity `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`) | **NO** (on SSD) — re-downloadable via preserved `dl-int4.sh` (HF repo is public; ~169 GB re-fetch, needs HF token + bandwidth) — relocation to NVMe is blocked by space (see #7) | Section 1 #7 |
| Vendor GPU kernels `~/xpu_artifacts/{_xpu_C.abi3.so, libgrouped_gemm_xe_2.so, libgrouped_gemm_xe_default.so}` (sha `593a7107…abc43b`, `2da4a494…90d13`, `4b1ca1e6…b5161`) | **YES — already preserved locally** at `files/a367-kernel/` (sha verified 2026-09-20, 99 MB); verify rig copy matches, else rsync | Section 1 #9 |
| FP8 weights `/data/hf` (185.56 GB) | **YES** — NVMe untouched by SSD-only install (verify `df`/mount after) | Section 1 #8 |
| Boot chain (SSD-local: Boot0002 `ubuntu-ssd` → sda1 PARTUUID `b160562b` → `EFI/UBUNTU-SSD/grub.cfg` → sda2 fs_uuid `25094d26…`) | Survives, but **goes STALE**: sda2 reformat changes the UUID; if sda1 is *repartitioned*, the PARTUUID anchor dies ⇒ **no boot** | R1 (CORRECTED 2026-09-20 — no NVMe chainloader exists post-migration; NVMe = single WEIGHTS partition) |
| Whole `/etc` customization (fstab, sshd_config, docker daemon.json, udev, modules-load.d, sysctl, sudoers, firewall, cron, systemd) | **NO** — recreated from scratch; Section 3 archives it | Section 3 |
| GuC firmware files `/lib/firmware/xe/*guc*` | **NO** — must come from `linux-firmware`/OMIX repo package post-install; if package is <70.65 → manual drop (BOM §5) | R4 |
| BIOS/NVRAM, card IFWI firmware | **YES** (card/firmware-resident; OS reinstall does not touch) — but **must be checked**: IFWI ≥775 and GuC target 70.65 | R4 |
| `/tmp` harness | **NO** (wiped by reboot) — **source of truth is local** `C:\Users\imrya\flashnext-scout` (verified present: `patch-v3.py`, `patch-gdn-v3b.py`, `patch-shortconv-v3b.py`, `patch-v21.py`, `v3-src/`, `wedge-watchdog-v2.sh` v2.5, `campaign.sh`, `stallspy.sh`, `soakfix.py`, `migrate-*.sh`, `dl-int4.sh`) | — |

**Runbook gaps found (fix before executing):**
- **GAP A — backup destination.** Runbook §0.1/§0.2 write to `/srv/rollback` and `/srv/preserve`. Nothing in the migration/firstlight evidence gives `/srv` its own mount; `/` is `ssd-vg/root` ⇒ `/srv` **is on the wiped SSD**. Unless `mount | grep /srv` (Section 3) says otherwise, redirect all backups to **`/data/preserve/`** (NVMe, survives) **+** USB stick **+** off-box (local `C:\Users\imrya`).
- **GAP B — no boot-chainloader step in the runbook.** §2 ends at "configure host" — there is no step that re-points the NVMe ESP chainloader at the new `/boot` UUID. This is the **#1 physical risk**; a mandatory post-install step must be added (Section 4 G8).
- **GAP C — runbook §0.1 saves only the image; §0.2 never mentions `.env` values that differ from `.env.sample`** (`PREFLIGHT_DISK_GB=30`, `CAP_SIZES_LIST=1,2,3,4,5,6,7,8,12,16`, `MAX_NUM_SEQS=16`, `PORT=8021`, `HF_HOME`, image tag; decision doc: "Preserve both").

---

## 1. BACKUP MANIFEST

Legend: **SSD** = wiped by install · **NVMe** = survives (do NOT let installer touch) · reuse the destination `DST=/data/preserve` unless §3 says `/srv` is separate.

| # | Item | Location on rig | Survives? | Backup action BEFORE wipe | Verify (after backup / after rebuild) |
|---|---|---|---|---|---|
| 1 | Rollback anchor image (stage-v24g; if absent, newest served tag, e.g. v24f — record tag) | `/var/lib/docker` (SSD) | NO | `docker image ls`; `docker save <tag> -o $DST/stage-v24g-pre-rebuild.tar` (~25–30 GB; `gzip` optional), `sha256sum > .sha256` | `sha256sum -c $DST/*.tar.sha256`; POST: `docker load -i` then `docker image ls` |
| 2 | Every other image (esp. `vllm-xpu-b70:26.31-test` = only path to resurrect `qwen28tp4m`; base images `intel/vllm:0.21.0…`, `qwen38-flash-next-xpu:*`) | `/var/lib/docker` (SSD) | NO | `docker save` each non-standard tag to `$DST/images/`; record tag→sha map | `docker image ls` POST; list matches archive |
| 3 | Docker launch facts of the running container (the real deploy recipe): env vars, binds, devices, group-add, published ports | running `qwen38-flash-next` (SSD state) | NO | `docker inspect qwen38-flash-next --format '{{json .Config.Env}}'` etc. → `$DST/container-inspect.json` + `docker ps -a --no-trunc` | POST: compare to start.sh |
| 4 | Engine dir `fn-recipe-int4/` — `start.sh` (v24h2), `stop.sh`, `wedge-watchdog-v2.sh`, `check-weights.sh`, `scripts/`, `launch*`, plus **`.env` + `.env.bak`** | `/home/bonz/fn-recipe-int4` (SSD) | NO | `tar czf $DST/fn-recipe-int4.tgz` (exclude `model` trees; ~small); **masked copy** of `.env` (sed HF_TOKEN) plus real copy on USB/off-box | `tar tzf … | wc -l`; grep `CAP_SIZES_LIST` + `MAX_NUM_SEQS=16` + `PREFLIGHT_DISK_GB` inside archived `.env`; POST diff `start.sh` sha256 |
| 5 | `.run/` evidence (the measurement basis): `boot_clock.jsonl`, `wd-decisions.jsonl`, `manifest.json`, `campaign-*/`, `wedge-*.log`, stall/dump bundles | `/home/bonz/fn-recipe-int4/.run` (SSD) | NO | rsync to `$DST/run-20260920/` **and** USB; heavy py-spy `.dump` dirs: keep if `$DST` space allows, else archive filelist + copy JSONLs (must-have) | `find $DST/run-* -type f | wc -l` vs rig `find .run -type f wc -l`; POST: `wd-decisions.jsonl` present |
| 6 | `~/flashnext-recipe` + `~/flashnext-scout` clones (docs, patches, sources) | `/home/bonz/…` (SSD) | NO | `rsync -a` to `$DST/` (they exist locally too — belt & suspenders); **no cost** | `diff -rq` vs local copy (best effort) |
| 7 | INT4 weights `hf-int4/` (~169 GB) | `/home/bonz/hf-int4` (SSD) | NO | **Choose one:** (a) rsync to external ≥256 GB drive (`rsync -a --info=progress2`; verify `du` ≈169 GB) — **OR** (b) documented decision: **re-download post-rebuild** via local `dl-int4.sh` (public HF repo; place `HF_HOME=/home/bonz/hf-int4` back, 8 workers; ~hours on wired net + HF token off-box) — **OR** (c) *only if* FP8 tree is expendable: delete `/data/hf` (~185 GB) → rsync INT4 onto NVMe ≈169 GB (fits; verify first: `df -h /data` and `du -sm /data/hf`) | (a) `du -sh` matches (b) `find …/snapshots -name '*.safetensors' | wc -l` + `check-weights.sh`-style gate (c) `df -h /data` free ≥180 GB, then `du` |
| 8 | FP8 weights `/data/hf` (185.56 GB, frozen identity `bcd9f01d…`) | `/data/hf` (NVMe) | YES | none required — but **must not be touched by installer**; record current `du -sh` + mount UUID | POST: `mountpoint /data`; `du -sh /data/hf` matches; `check-weights.sh` (if FP8 line still used) |
| 9 | `~/xpu_artifacts/` (3 vendored `.so`) | `/home/bonz` (SSD) | NO (local copy exists) | `sha256sum` rig copy vs local known-good: `593a7107d3d20304f3d37b7ea20ca00b2eea361fff2aad5c1107fd91c5abc43b` (`_xpu_C.abi3.so`), `2da4a494d4014e58e8a4a91e84cec39a22143982c6f847188a2e5e7e94f90d13`, `4b1ca1e660b80bc422dcc3bc7bd71f9e471615a2baf1f012b340593c62ab5161`; mismatch → rsync from `files/a367-kernel/` | `sha256sum -c` POST after re-copy to `~/xpu_artifacts` |
| 10 | SSH state: `~/.ssh/authorized_keys` (existence/count — no contents), host keys, known_hosts | `/home/bonz/.ssh` + `/etc/ssh` (SSD) | NO | snapshot metadata + **back up `authorized_keys`** (count + copy); plan: generate fresh host keys, install pubkey from off-box, `PasswordAuthentication` decision, `~/.ssh` perms | POST: `ssh -o BatchMode` from off-box works; `tailscale ssh` if enabled |
| 11 | Tailscale state | `/var/lib/tailscale` (SSD) | NO | record: `tailscale status`, `tailscale debug prefs`, `ip -br a`, `ip route`; **create pre-auth key now** (tailnet admin, one-shot/reusable=false, scope short, ephemeral=false; store OFF-BOX; never in repo) | POST: `tailscale up --authkey <KEY> --ssh`; `tailscale status` shows hostname + 100.122.128.100 |
| 12 | Boot chainloader + NVMe ESP originals | `/dev/nvme0n1p1` `EFI/ubuntu/{grub.cfg, shimx64.efi, grubx64.efi, mmx64.efi{,.nvme-bak}}` (NVMe) | YES (stale) | copy whole `EFI/ubuntu/` dir to `$DST/nvme-esp-backup/` **before anything**; record current `search.fs_uuid` + `efibootmgr -v` | POST (after rewrite): `grep fs_uuid` matches new `/dev/sda2` UUID; originals still present as `.nvme-bak` |
| 13 | `/etc` inventory (fstab, daemon.json, sshd config, udev rules, modules-load.d, sysctl.d, sudoers.d, firewall, apt sources incl. Intel PPA/OMIX, hosts, network plan) | `/etc` (SSD) | NO | Section 3 archive → `$DST/etc-20260920/`; **do not restore GPU packages from the Intel PPA** (re-install per OMIX clean path) | POST: diff vs archive for non-GPU entries (fstab `/data` + swap, docker group, etc.) |
| 14 | crontab(s) + enabled systemd units + user env | `/var/spool/cron`, `/etc/systemd` (SSD) | NO | Section 3 captures; restore as wanted | POST: `crontab -l` matches |
| 15 | Python/venv inventories (`b70top`, host py, `/opt` venvs, binned wheels) | host py envs (SSD) | NO | `pip freeze` per env → `$DST/pip/` + `pip download` any non-PyPI package (e.g. b70top source) | POST: `pip list` diff; import test |
| 16 | `/tmp` harness | `/tmp` (SSD) | NO (wiped by any reboot) | nothing to do — local `flashnext-scout` is source of truth; **only** save rig-only artifacts: `ls /tmp` + `du -sh` each; copy new items (e.g. census/campaign tarballs) to `$DST/tmp-extra/` | local clone readable; `diff -rq` where possible |
| 17 | Credentials: HF token (in `.env`), out-of-box SSH credential file (location **UNVERIFIED** — not in repo; keep it safe and off-box), any registry creds | `.env`, off-box | NO | `.env` backup covers token; credential file = human-handled (do not paste into any git) | POST: `curl -sI https://huggingface.co` auth check |

**Space reality check (mark and measure in §3):** `/data` = ~222 GB usable, FP8 tree ≈185.56 GB ⇒ **free ≈36 GB** (firstlight 09-16). A 25–30 GB image tar fits; INT4 (169 GB) does **not**. If both image + evidence + configs exceed free space → external drive/USB ≥256 GB or off-box transfer (scp from rig over Tailscale BEFORE wipe). **`du -sh /data/*` and `df -h /data` at enumerate time are the authority.**

---

## 2. PHYSICAL-ACCESS RISKS (ranked; machinery required on-site)

> **Who: ______  How: ______** — to be filled by Ryan/site operator. "How" = the recovery procedure; "Who" = who executes (name + role). Never invented here.

### R1 — Boot chain update after SSD reinstall — **HIGH** (CORRECTED 2026-09-20, live preflight overrides migration-era assumption)

**Verified live topology (efibootmgr + blkid + mounts, 2026-09-20):** BootCurrent **0002** =
`ubuntu-ssd`, anchored to **sda1 PARTUUID `b160562b-4c03-4455-b86c-66438834ffb8`** with path
`\EFI\UBUNTU-SSD\SHIMX64.EFI`; its grub.cfg does `search.fs_uuid 25094d26-b9fc-48df-b94b-391ec2d71260 root hd0,gpt2` (= **sda2** `/boot`). The NVMe is a **single ext4 `WEIGHTS` partition** (`/data`, a87f17fd…) — **no ESP, no old ubuntu-vg, no NVMe chainloader exists post-migration**; `EFI/ubuntu/` on sda1 (search.fs_uuid 46850fa9…) is a stale pre-migration leftover. BootOrder `0002,0004,0001,0003,0000`; Boot0003 (VenHw "Ubuntu") is dead weight from the NVMe era.

**Residual risks after reinstall (reduced from the migration-era shape):**
- **sda2 reformat ⇒ stale fs_uuid** in `EFI/UBUNTU-SSD/grub.cfg` → grub rescue / no boot. Fix is one line, SSD-local (no NVMe involvement).
- **sda1 repartition ⇒ PARTUUID `b160562b` changes ⇒ Boot0002's HD(1,GPT,…) anchor dies** (worse: no boot). Mitigation: installer must **reuse, not recreate, the sda1 ESP** (manual partitioning: mount sda1 as /boot/efi, do NOT reformat it, or reformat-only keeps PARTUUID). If it changes anyway: re-register via `efibootmgr -c` from the live session.
- Installer may add a fresh NVRAM entry — harmless while `ubuntu-ssd`/sda1 anchor survives.
- **Prevention (live session, before first reboot):** `lsblk -f` → new sda2 UUID → mount sda1 → `sed -i` the `search.fs_uuid` line in `EFI/UBUNTU-SSD/grub.cfg` to the new UUID → `efibootmgr -v` confirms `ubuntu-ssd` still anchors to sda1's PARTUUID → reboot. **Full pre-change ESP backup preserved off-box:** `evidence/2026-09-20/sda1-efi/sda1-efi-20260920.tgz` (5.2 MB, whole EFI tree).
- **Recovery if missed:** boot USB live media → `blkid` new sda2 → mount sda1 → rewrite the fs_uuid line (same edit) → reboot. The stale `EFI/ubuntu/` leftover and dead Boot0003 are irrelevant to this recovery.
- **Who:** ______ · **How:** ______

### R2 — Tailscale identity lost ⇒ box unreachable, full stop — **HIGH — THE ONLY REMOTE PATH (corrected 2026-09-20, Ryan)**

`100.122.128.100` is the only route to jobe. **The LAN IP (192.168.100.197) is NOT a recovery path — it is only reachable from inside Alex's LAN; neither Ryan nor GLM is on that network.** Post-wipe, fresh 24.04 has no Tailscale and no remote access until `tailscale up --authkey` runs. The pre-auth key is therefore **mandatory infrastructure, not belt-and-suspenders**.
- **Requirement (Ryan):** key is **reusable and non-expiring for the rebuild window**, and delivered **non-interactively** — baked into the autoinstall `user-data` on the install USB, or fetched by a first-boot script from a URL Ryan hosts (GLM cannot pull from the box before Tailscale is up; the box-side fetch must be a push-from-outside or a curl from a location Ryan controls). Never typed interactively on the console, never stored in git.
- **First-boot sequence (bake into user-data or /root/first-boot.sh):** install tailscale → `tailscale up --authkey <KEY> --ssh` → verify `tailscale status` shows `jobe` + `100.122.128.100` → GLM regains remote control for everything after.
- **Recovery if key missing/wrong:** physical console (Alex, on-site) → interactive `tailscale up` login. Until then the box is dark to both of us.
- **Who:** Alex creates the key (in flight, 2026-09-20) · **How:** user-data embed or first-boot curl — fill exact mechanism when the key exists

### R3 — Install media + first-boot console — **MEDIUM**
Ubuntu 24.04.4 LTS desktop/server ISO (HWE 6.17 must be chosen; GA 6.8 will NOT work — BOM §3.4) on USB; the firmware-skip-BootOrder quirk likely needs a **boot-menu hotkey** (F-key) or one-time boot from USB — physical or KVM HID. Wired network required for OMIX (multi-GB). Also needed: a **second USB ≥256 GB** for the backup tar / INT4 if external-drive path chosen, or confirm off-box scp path.
- **Recovery:** recreate ISO (verified sha) on any USB; that's it. If the installer's UEFI entry never appears → console + set boot; NVRAM writes may be ignored ⇒ always boot via hotkey.
- **Who:** ______ · **How:** ______

### R4 — GPU firmware (GuC 70.65 host-files; IFWI 775 card-resident) — **MEDIUM**
- GuC 70.65 is **not** a card setting — it's `/lib/firmware/xe/*guc*` on the **host** ⇒ must be re-provided after install. OMIX support matrix: **GuC 70.65 is required**; **NEVER 70.72.1** (GSD-13481 deterministic TP2 startup hang). Check after OMIX install: package `linux-firmware`/`intel-firmware` version; if it carries <70.65, manual drop of `xe/bmg_guc_70.bin` + `update-initramfs -u` (documented method, BOM §5) **or** A/B 70.44.1 (the only long-term-clean dual-B70 field firmware).
- IFWI (card-resident, survives OS reinstall): must be **≥775** (G31) per OMIX matrix. Current value **UNVERIFIED** — `sudo dmesg | grep -i ifwi` / `xpu-smi`. **If <775, flashing the card firmware is a physical-task + risk (brickable); decide BEFORE the rebuild** — video/no-video, tool = Intel GSS/meu `.fw` update, needs idle GPUs.
- **Recovery:** wrong GuC (70.58-era or absent) = familiar wedge family at L1; fix = firmware package/manual drop, no physical. IFWI flash failure = **physical** (boot-loop/card-no-display; un-dump recovery often requires a second card/motherboard or RMA — avoid).
- **Who:** ______ · **How:** ______

### R5 — BIOS settings — **MEDIUM (verify; change = physical)**
- **IOMMU:** Intel validated the vLLM XPU container with **IOMMU OFF** (BOM §2); current rig value **UNVERIFIED**. Check `sudo dmesg | grep -i dmar` / `efibootmgr`/BIOS. If ON → test first; if wedge reproduces → set OFF in BIOS (physical) or `iommu=off` on cmdline (not persistent across kernel reinstall — record both).
- **SecureBoot:** currently OFF (migrate-B5 note) — keep off (OMIX/compute-runtime + shim chain).
- **ReBAR / Resizable BAR:** **UNVERIFIED** — check `lspci -vvv` for `Memory behind bridge`; Arc benefits; only touch with evidence.
- **boot-order quirk** (R1) — cannot be fixed via NVRAM; workaround is the EFI chainloader. No BIOS change to attempt.
- **Recovery:** any BIOS screen change = console/KVM only.
- **Who:** ______ · **How:** ______

### R6 — Installer targeting the wrong disk — **HIGH (catastrophic, but preventable)**
The install must touch **/dev/sda only** (WDC SSD). The NVMe holds: `/data` (185.56 GB FP8 + future INT4 relocation if chosen), the **Boot0003 chainloader ESP + `.nvme-bak` originals**, and (per session record) an old deactivated `ubuntu-vg`. Selecting "erase all disks" / "wipe entire device" with the NVMe selected destroys all three. Use **manual partitioning**; after install (before reboot) `lsblk -f` must still show NVMe partitions.
- **Recovery:** NVMe is gone; weights re-downloadable (185 GB + INT4), chainloader originals irrecoverable (`.nvme-bak` = the old-OS fallback for R1) — hardware data-loss event.
- **Who:** ______ · **How:** ______

### R7 — Sudo/root for `bonz` — **MEDIUM (no physical; install-time decision)**
`bonz` has **no sudo today** (root ops only via the privileged docker container — unique to this box). After a fresh minimal install, sudo is granted by default to the *install-time admin user*. Requirement recorded: **install creates user `bonz` and puts it in `sudo` + `docker` + `video` + (gid 991 `render`)** groups; or set a root password during install; else every host op needs console. Verify after: `id bonz`, `sudo -n true`.
- **Recovery (no physical):** none — you can only add bonz to sudo from root, and root comes from the install (or single-user/`init=/bin/bash` edit — needs console).
- **Who:** ______ · **How:** ______

### R8 — Swap / fstab / mount plan — **LOW** (host config, post-install)
96 GiB `/swapfile` currently lives **on the SSD root** (firstlight: `96G /swapfile on SSD`); config in `migrate-D.sh`. Post-install: recreate (≥64 GiB floor; root has room on the new SSD, or on `/data` swapfile per phase3 plan — note `/data` only has ~36 GB free ⇒ **swapfile stays on SSD** unless weights are relocated). Re-add `/data` fstab entry via UUID (record in §3). Docker install + `/etc/docker/daemon.json` restore. `--group-add 991` (not `render` — firstlight kit-bug fix; image has no `render` group).
- **Recovery:** trivial via SSH once R2/R1 are solved.
- **Who:** ______ · **How:** ______

---

## 3. ENUMERATE-NOW COMMAND BLOCK (read-only; paste into a root-or-sudo shell on jobe)

```bash
# ============================================================
# enumerate-pre-wipe.sh — pre-rebuild state capture, READ-ONLY
# Run: bash enumerate-pre-wipe.sh   (root or bonz+sudo)
# Output: /tmp/preflight-<ts>/ + tarball; NOTHING is modified.
# ============================================================
set -uo pipefail
TS=$(date +%Y%m%d-%H%M%S); OUT=/tmp/preflight-$TS; mkdir -p "$OUT"
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo"
log(){ echo "== $*" | tee -a "$OUT/run.log" >&2; }
c(){ echo "--- $*" >> "$OUT/capture.log"; "$@" >> "$OUT/capture.log" 2>&1; }

log "HOST"; c uname -a; c hostname; c date -u
log "END OF STATE == disks"
lsblk -o NAME,SIZE,FSTYPE,FSVER,LABEL,UUID,PARTUUID,MOUNTPOINT
c blkid
c pvs; c vgs; c lvs;                # ssd-vg + old ubuntu-vg (may be deactivated)
c df -hT; c swapon --show
log "mounts incl. /srv question"
mount | tee "$OUT/mounts.txt"
log "== boot / EFI"
c efibootmgr -v
ls -la /dev/disk/by-id/ 2>/dev/null | head -20
ESP=$(lsblk -o NAME,MOUNTPOINT | awk '/nvme.*1.*boot|EFI/ {print $1}' | head -1)
ESP_DEV=${ESP:-nvme0n1p1}   # CONFIRM with lsblk above! Chainloader is on the NVMe ESP.
mkdir -p "$OUT/nvme-esp"; mountpoint -q /mnt2 || { mkdir -p /mnt2; $SUDO mount /dev/$ESP_DEV /mnt2 2>/dev/null || true; }
if mountpoint -q /mnt2; then cp -a /mnt2/EFI/ubuntu "$OUT/nvme-esp/" 2>/dev/null || true; cat /mnt2/EFI/ubuntu/grub.cfg >> "$OUT/chainloader-grub.cfg.txt" 2>/dev/null || true; fi
umount /mnt2 2>/dev/null || true
log "== kernel / GPU / firmware"
c uname -r; c "$SUDO" dmesg | grep -iE 'guc|ifwi|xe |firmware' | head -60
c "$SUDO" dmesg | grep -iE 'iommu|dmar|aim' | head -20
c lspci -nn | grep -iE 'arc|vga|display'; c lspci -vvv | grep -iE 'bar|iommu' | head -30
c "$SUDO" lsmod | grep -w xe || c lsmod | grep -w xe
c modinfo xe | head -8
ls -la /lib/firmware/xe/ | tee "$OUT/fw-xe-listing.txt"
c apt-cache policy linux-firmware intel-firmware linux-intel-6.17 2>/dev/null
c "$SUDO" xpu-smi dump 2>/dev/null | head -60 || xpu-smi 2>/dev/null | head -30 || echo "xpu-smi absent"
log "== docker / containers"
c docker image ls --no-trunc; c docker ps -a --no-trunc; c docker volume ls
c docker info 2>/dev/null | grep -iE 'root dir|storage|version' | head
c docker inspect qwen38-flash-next --format '{{json .Config.Env}}' 2>/dev/null | tee "$OUT/ctr-env.json"
c docker inspect qwen38-flash-next --format '{{json .HostConfig.Binds}} {{json .HostConfig.Devices}} {{json .HostConfig.GroupAdd}}' 2>/dev/null | tee "$OUT/ctr-binds.json"
c cat /etc/docker/daemon.json 2>/dev/null || echo "no daemon.json"
du -sh /var/lib/docker 2>/dev/null | tee "$OUT/docker-size.txt"
log "== weights & data"
c du -sh /data/hf /home/bonz/hf-int4 /home/bonz/xpu_artifacts /home/bonz/fn-recipe-int4 2>/dev/null | tee "$OUT/sizes.txt"
c du -sh /data/* 2>/dev/null | sort -h | tail -12; c df -h /data /
find /data/hf -name '*.safetensors' 2>/dev/null | wc -l | tee "$OUT/fp8-shards.count"
find /home/bonz/hf-int4 -name '*.safetensors' 2>/dev/null | wc -l | tee "$OUT/int4-shards.count"
log "== engine dir (hash) =="
cd /home/bonz/fn-recipe-int4 2>/dev/null && { sha256sum start.sh stop.sh wedge-watchdog-v2.sh check-weights.sh 2>/dev/null | tee "$OUT/engine-sha.txt"; ls -la . scripts/ .run/ 2>/dev/null | tee "$OUT/engine-ls.txt"; du -sh .run/ scripts/ 2>/dev/null; grep -E 'CAP_SIZES_LIST|MAX_NUM_SEQS|PREFLIGHT_DISK_GB|PORT=|IMAGE=|HF_HOME|MTP_NUM' .env 2>/dev/null | sed -E 's/(TOKEN|KEY|PASS|SECRET)[^ ]*=.*/\1=<redacted>/' | tee "$OUT/env-masked.txt"; }
log "== /tmp harness state =="; ls -la /tmp | tee "$OUT/tmp-ls.txt"; du -sh /tmp 2>/dev/null
log "== network =="; c ip -br a; c ip route; c tailscale status; c tailscale debug prefs 2>/dev/null | head -30
log "== users / ssh / sudo =="
c id; c getent group sudo docker video render input
c "$SUDO" grep -E '^bonz' /etc/passwd; c "$SUDO" grep -rE 'sudo|ALL' /etc/sudoers.d/ 2>/dev/null | head
c "$SUDO" -n true 2>&1 || echo "sudo needs password (expected: bonz no-sudo)"
ls -la /home/bonz/.ssh 2>/dev/null | tee "$OUT/ssh-ls.txt"; wc -l /home/bonz/.ssh/authorized_keys 2>/dev/null | tee "$OUT/ssh-aurh.count"
c "$SUDO" sshd -T 2>/dev/null | grep -Ei 'passwordauth|permitroot|port |pubkeyauth' | tee "$OUT/sshd-settings.txt"
log "== cron / systemd / modules / sysctl / udev / netfilter =="
c crontab -l 2>/dev/null; c "$SUDO" crontab -l 2>/dev/null
c "$SUDO" systemctl list-unit-files --state=enabled --no-pager; c "$SUDO" systemctl list-units --type=service --state=running --no-pager | head -60
c ls /etc/modules-load.d/; c cat /etc/modules-load.d/* 2>/dev/null
c ls /etc/udev/rules.d/; c cat /etc/udev/rules.d/*dri* /etc/udev/rules.d/*render* 2>/dev/null
c ls /etc/sysctl.d/ /etc/security/limits.d/; c cat /etc/sysctl.d/* /etc/security/limits.d/* 2>/dev/null | head -80
c "$SUDO" ufw status 2>/dev/null; c "$SUDO" iptables-save 2>/dev/null | head -40; c "$SUDO" nft list ruleset 2>/dev/null | head -40
c cat /etc/fstab | tee "$OUT/fstab.txt"; c cat /etc/apt/sources.list.d/*.list 2>/dev/null | tee "$OUT/apt-sources.txt"
log "== python / venvs / pip =="
c python3 -V; which -a python3 pip pip3
for v in $(ls -d /opt/*/bin 2>/dev/null; ls -d /home/bonz/.venv*/bin 2>/dev/null; ls -d /home/bonz/venv*/bin 2>/dev/null); do echo "--- $v" | tee -a "$OUT/pip.txt"; "$v/pip" list 2>/dev/null >> "$OUT/pip.txt"; done
c "$SUDO" dpkg -l | grep -iE 'libze|compute-runtime|level-zero|opencl|intel|guc' | tee "$OUT/dpkg-intel.txt"
log "== bash history (provenance) =="
cp -a /home/bonz/.bash_history "$OUT/" 2>/dev/null; wc -l "$OUT/.bash_history" 2>/dev/null
log "== DONE — archive =="
cd /tmp && tar czf "preflight-$TS.tar.gz" "preflight-$TS" && sha256sum "preflight-$TS.tar.gz" | tee "$OUT/archive.sha256"
du -h "preflight-$TS.tar.gz"
echo ">>> rsync/scp to off-box NOW: scp /tmp/preflight-$TS.tar.gz user@offbox:/path;  ALSO copy to $DST and the USB stick"
```

Notes: the script prints **no secrets** (`.env` masked, `authorized_keys` counted not dumped). If `sudo` is password-gated (bonz has none), run it from the root-capable context (the privileged `qwen38-flash-next` docker container or install-time single-user) or let the sudo-less lines skip. `ESP_DEV` must be confirmed against `lsblk` output — **identify the NVMe ESP partition before anything else**.

---

## 4. POINT-OF-NO-RETURN CHECKLIST (ordered gates — **ALL must PASS** before the USB lives in the box)

| Gate | Check | Pass criterion / action |
|---|---|---|
| **G0** | Install + backup media ready | Ubuntu **24.04.4 LTS** ISO (+ HWE kernel available on it), sha-verified, on USB-A; **second storage** ≥256 GB (or off-box scp target ready); ethernet cable + switch port confirmed |
| **G1** | Enumerate **now** | §3 script ran as root; tarball saved to `$DST/` + USB + off-box; `archive.sha256` matches after copy |
| **G2** | NVMe ESP + disk map confirmed | `lsblk -f` shows NVMe partitions incl. **p1 ESP**; recorded which is `/data`; recorded old `ubuntu-vg` state; **installer partner confirms: touch `sda` only** |
| **G3** | Rollback anchor saved + verified | `docker save` done; `sha256sum -c` PASS; tar on **survivor** (`/data/preserve`) AND off-box/USB; `docker image ls` list archived; `vllm-xpu-b70:26.31-test` also saved or its pull provenance confirmed |
| **G4** | Engine config off-box | `fn-recipe-int4.tgz` (+ `.env` real copy on USB/off-box; masked copy here) verified: `CAP_SIZES_LIST=1,…,16`, `MAX_NUM_SEQS=16`, `PREFLIGHT_DISK_GB`, `PORT=8021`, `HF_HOME` present; `start.sh` sha matches rig (v24h2); `patch-capsizes.py`/`patch-capoverride.py` in local repo = commit 3589249 |
| **G5** | Weights decided | INT4: external copy verified (du ≈169 GB) **or** re-download plan recorded (dl-int4.sh local, token off-box, ~hours, wired); FP8: `df -h /data` recorded, no plan touches NVMe; xpu_artifacts sha256 = the three hashes (already known-good locally) |
| **G6** | Tailscale plan | Pre-auth key created (reusable, non-expiring for the window) + delivered **non-interactively**: baked into autoinstall user-data on the USB or fetched by a first-boot script from a Ryan-hosted URL — GLM has no path to the box before Tailscale is up, and the LAN IP is inside Alex's LAN only (not a recovery path). Physical console at first boot is the fallback only. |
| **G7** | sudo/root plan | Install will create user **bonz** in `sudo`/`docker`/`video`/991-or-render; set root password at install; **or** single-user recovery documented. `id bonz` + `sudo -n true` expected on first shell |
| **G8** | Boot-chain plan | Explicit post-install step written into runbook (GAP B), per **corrected R1**: from live session before first reboot — `lsblk -f` new sda2 UUID → mount **sda1** (the only ESP) → sed `search.fs_uuid` in `EFI/UBUNTU-SSD/grub.cfg` → `efibootmgr -v` confirms `ubuntu-ssd` still anchors sda1 PARTUUID `b160562b` → installer used manual partitioning and did NOT repartition sda1. Grub-rescue recovery sheet printed and beside the machine |
| **G9** | Runner/console | KVM-dongle or monitor+keyboard staged at the box (R1/R2 likely need a live-session moment); for headless: confirmed KVM + USB HID |
| **G10** | GPU firmware plan | GuC: post-install check `dmesg | grep guc` must show **70.65** (never 70.72.1); if `<70.65` → man-package/manual `bmg_guc_70.bin` + `update-initramfs` (or A/B 70.44.1 recorded); IFWI ≥775 verified NOW (`xpu-smi`/dmesg) — **if <775, stop and decide flash (physical, R4)** |
| **G11** | BIOS record | IOMMU + SecureBoot + ReBAR read out and written next to the ladder; IOMMU OFF is the Intel-validated posture unless evidence says otherwise |
| **G12** | Rollback decision | Ladder L1–L5 per runbook §4; **rollback anchor tar must NOT be deleted until L5 passes**; after L5, archive (not delete) 6 weeks |
| **G13** | Team | Window, aborts, and who-does-what confirmed (names filled for R1–R8; operator on-site phone/KVM available) |

**Hard rule:** any gate above = UNVERIFIED ⇒ **do not wipe** — resolve it with §3 output or the responsible person first. Reboot/install fires only when every gate has a PASS.

---

### UNVERIFIED list (must be checked at G1; nothing here was guessed)
1. Whether `/srv` is a separate mount (→ runbook backup destinations) — check `mount | grep srv`. **(assumed SSD → redirect needed)**
2. Exact NVMe partition map (ESP partition device, `ubuntu-vg` state, `/data` UUID) — migration scripts disagree (B5 shows p1 ESP; migrate-D shows a zap-all); §3 `lsblk` resolves.
3. GPU card firmware IFWI version (target 775) and current GuC (70.58-era per session record — verify).
4. IOMMU on/off, SecureBoot, ReBAR, RAM config — BIOS/OS read-only check.
5. `bonz` sudo + `authorized_keys` contents (count only; plan key install) + whether `tailscale ssh` was the access path.
6. Whether a LAN IP/route exists (ip -br a) — would soften R2.
7. Current free space on `/data` (firstlight says ≈36 GB; re-measure) — decides where the tar and any INT4 go.
8. `b70top` provenance (pip package vs built tool) — enumerate + `pip download` if needed.
9. Which image tag is the true rollback anchor (v24g vs v24f) and actual `docker save` size.
