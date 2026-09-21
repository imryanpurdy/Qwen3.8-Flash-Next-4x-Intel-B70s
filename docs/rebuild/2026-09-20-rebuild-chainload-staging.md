# Rebuild staging — chainload autoinstall (2026-09-20)

Status: STAGED + READ-BACK VERIFIED. One-shot armed (`grubenv: next_entry=rebuild`).
Execution: Ryan-authorized ("Go on the rebuild"). Recovery operator: Alex (keyboard+monitor, next morning).
Seed key material lives ONLY local (scout/rebuild/) + rig (/data/rebuild-install/seed, inside seed-initrd.gz) — never committed.

## Staging manifest (all hashes read back from the box)

| Artifact | Location (rig) | Hash |
|---|---|---|
| ubuntu-24.04.5-live-server-amd64.iso (3.79 GB) | /data/rebuild-install/ | sha256 **OK** vs releases.ubuntu.com SHA256SUMS |
| casper-vmlinuz | /data/rebuild-install/ + /boot/rebuild/ | sha256 0066409132868538… · md5 6300ea7c87a85dd9557fbc7e4355c8de (both copies match) |
| casper-initrd | /data/rebuild-install/ + /boot/rebuild/ | sha256 62b1e77d0ff5ddeb… · md5 e0c32bc19f0430afecbcfa2c8295f5d4 (both copies match) |
| seed-initrd.gz (seed + full casper-initrd concatenated) | /data/rebuild-install/seed/ + /boot/rebuild/ | md5 0d2807611017dec579692c7e72369c7a (both copies match); contents verified by cpio listing + extracted user-data read-back |
| autoinstall.yaml | /data/rebuild-install/seed/ (+ inside initrd at autoinstall/user-data AND autoinstall.yaml) | md5 3ceb2217d592614f1ba94f73934aa240 |
| stage-v24h2-rollback.tar.gz | /data/ | sha256 3fc1d174…c3a706 (re-verified at staging; off-box copy verified earlier) |

## Boot mechanism (variant C — remastered ISO, supercedes original seed-in-initrd design)

GRUB entry (id `rebuild`, /etc/grub.d/42-rebuild → /boot/grub/grub.cfg):
`linux /rebuild/casper-vmlinuz iso-scan/filename=/rebuild-install/autoinstall-remaster.iso autoinstall "ds=nocloud;s=/cdrom/autoinstall/" ---`
`initrd /rebuild/casper-initrd`

- Seed rides INSIDE the remastered ISO at /autoinstall/{user-data,meta-data}; subiquity reads it
  from the iso-scan-mounted ISO at /cdrom/autoinstall/. Stock casper initrd untouched —
  **attempt 1 failed fatally by concatenating a gzip seed segment onto casper's RAW-CPIO initrd**
  (kernel 6.8.0-139 panicked: no root, zero block devices). Never concat foreign segments onto
  casper's uncompressed cpio.
- Remaster = xorriso extract of the official ISO + seed dir + rebuild with the ORIGINAL's
  `-report_el_torito as_mkisofs` options (El Torito BIOS + UEFI entries preserved; verified by
  re-reporting boot structures from the remaster). --modification-date = 16 digits.
- Validated END-TO-END in QEMU/KVM on the rig BEFORE re-arming jobe: replica disk with jobe's
  exact geometry (1G ESP + 2G ext4 + LVM PV), UEFI OVMF + -machine q35 (OVMF needs q35's AHCI;
  SeaBIOS/i440FX runs IDE but subiquity then plans BIOS bootloader and rejects the UEFI-targeted
  config with "did not create needed bootloader partition"), same -kernel/-initrd/-append handoff
  as the GRUB entry. Seed fixes validated there: grub.reorder_uefi=false, no `flag: boot` on the
  preserved ESP (preserve-matching rejects flag mismatch), explicit partition sizes (sda1
  1073741824 / sda2 2147483648 / sda3 996982595072). curtin "configuring disk: sda" +
  "curtin command install" PASSED on the replica.
- One-shot: `grub-reboot rebuild` → boots installer this boot only. If the casper boot hangs and
  the box is hard-reset, the one-shot is consumed → old OS boots → staging intact → re-arm and retry.
- `interactive-sections: []` + full identity/storage config = fully unattended.
- Tailscale sequencing (Ryan directive): the OLD node is logged out (`tailscale logout`) chained
  into the SAME command as the reboot — freeing the `jobe` MagicDNS name so the fresh install
  registers AS `jobe`, never `jobe-1`. The QEMU test VM never registered (died pre-late-commands);
  if a validation boot of the replica registers a node, it is logged out BEFORE the real box fires.

## Storage contract (Ryan-confirmed layout, encoded verbatim)

- Partition table NEVER repartitioned — same boundaries, same PARTUUIDs.
- sda1: ESP preserved + not formatted (`preserve: true` on partition AND format), `grub_device: true`
  (firmware boot entry is pinned to PARTUUID b160562b-4c03-4455-b86c-66438834ffb8).
- sda2 (2G): reformatted ext4 → /boot. sda3 (928G): reformatted ext4 → / (LVM layer dropped:
  early-commands lvremove ssd-vg/root → vgremove → pvremove /dev/sda3 BEFORE storage actions).
- NVMe (/data, a87f17fd-bba7-4970-a797-7916f19bd50d): absent from storage config = untouched;
  re-mounted via fstab append in late-commands. Swap 72G file (kit floor ≥64G).

## Seed contents (autoinstall.yaml — key lines masked)

identity: jobe / bonz / crypted pw (plaintext only in ~/.rig.cred2, 0600) · ssh: openssh-server +
authorized_keys (jobe-rebuild ed25519, private key on Hermes host only) · packages: cpio curl jq docker.io ·
network: DHCP on en* · timezone Etc/UTC.
late-commands: sudoers.d/bonz-nopasswd (440) · authorized_keys (700/600, 1000:1000) · /data fstab ·
swapfile · **tailscale install + ExecStartPost `tailscale up --authkey=[REDACTED-tskey] --hostname=jobe
--accept-dns=false`** (key file: C:\Users\imrya\.ts-authkey 0600; valid ~1 day from 2026-09-20) ·
bonz → docker group · seed copy to /var/lib/cloud/seed/nocloud.

## What dies / what survives

DIES with sda3+sda2 wipe: old OS, /home/bonz/hf-int4 (weights — re-download pinned rev 4c67bf686b7f…),
engine dir, /tmp harness, docker container layer, Tailscale node identity, SSH host keys.
SURVIVES on NVMe /data: rollback tarball, ISO, seed, casper pair. Off-box: all evidence, configs,
rig-scripts, rollback image copy (sha-verified), weights identity manifests.

## Post-install sequence (runbook §2.1.4 onward, unchanged)

1. Weights download FIRST (needs `pip3 install huggingface_hub` on fresh host python3 first) —
   `dl-int4.sh`, pinned rev, manifest verify vs evidence/2026-09-20/weights-identity/.
2. OMIX 0.4 per §2.3–2.4 (purge gate → single-source gate), torch 2.13.0+xpu, oneAPI 2026.0, vxk 0.1.14+.
3. Container reconstruction §3 (all earned patches re-based), rollback fallback = `docker load`
   stage-v24h2-rollback.tar.gz + /data configs.
4. scripts/rebuild-verify.sh L1→L5. Decisive: MTP1 + graphs capture clean.

## Watch procedure (from the moment of reboot)

Hermes-side loop polls SSH + `cat /etc/os-release` + `uname -r` every 60 s for up to 60 min.
Expected: SSH down 15–30 min (installer env has no sshd), then new boot answers with
VERSION_ID=24.04 + 6.17-intel kernel + tailscale up. If 60 min pass silent → Alex recovery
(kb+monitor; installer logs at /var/log/installer or tty1).
