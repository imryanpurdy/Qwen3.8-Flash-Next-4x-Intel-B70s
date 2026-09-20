#!/bin/bash
# enumerate-pre-wipe.sh — pre-rebuild state capture, READ-ONLY (audit §3, verbatim)
# Run: bash enumerate-pre-wipe.sh   (root or bonz+sudo)
# Output: /tmp/preflight-<ts>/ + tarball; NOTHING is modified.
set -uo pipefail
TS=$(date +%Y%m%d-%H%M%S); OUT=/tmp/preflight-$TS; mkdir -p "$OUT"
SUDO=""; sudo -n true 2>/dev/null && SUDO="sudo"
log(){ echo "== $*" | tee -a "$OUT/run.log" >&2; }
c(){ echo "--- $*" >> "$OUT/capture.log"; "$@" >> "$OUT/capture.log" 2>&1; }

log "HOST"; c uname -a; c hostname; c date -u
log "END OF STATE == disks"
lsblk -o NAME,SIZE,FSTYPE,FSVER,LABEL,UUID,PARTUUID,MOUNTPOINT
c blkid
c pvs; c vgs; c lvs;
c df -hT; c swapon --show
log "mounts incl. /srv question"
mount | tee "$OUT/mounts.txt"
log "== boot / EFI"
c efibootmgr -v
ls -la /dev/disk/by-id/ 2>/dev/null | head -20
ESP=$(lsblk -o NAME,MOUNTPOINT | awk '/nvme.*1.*boot|EFI/ {print $1}' | head -1)
ESP_DEV=${ESP:-nvme0n1p1}
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
c "$SUDO" -n true 2>&1 || echo "sudo needs password"
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
echo ">>> pull off-box now: pscp bonz@rig:/tmp/preflight-$TS.tar.gz <dest>"
