#!/usr/bin/env bash
# ============================================================================
# host-setup.sh — one-time host provisioning for the Qwen3.8-Flash-Next rig
#
# Idempotent: every step is verified after being applied (read-back), and a
# step already satisfied is skipped. Run again any time; it converges.
#
# Installs (2026-09-23 platform of record):
#   - intel-omix 0.4 (DLE 2026.1 userspace: oneCCL 2022.1.2, level-zero)
#   - kernel 6.17.0-1010-intel + grub default (iommu=off)
#   - GuC firmware 70.65 (linux-firmware commit fb0889c0, blob
#     sha256 70d74627e395347ea04c37168d92f01c9e940f4b32e0743b6350ca808fdb67bb,
#     377,664 bytes)
#   - docker nofile limit (systemd LimitNOFILE=infinity)
#
# REBOOT REQUIRED at the end for kernel + iommu + GuC to take effect.
# ============================================================================
set -euo pipefail

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }
warn() { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
err()  { echo -e "\033[1;31m[ERR ]\033[0m  $*"; exit 1; }

SUDO=""
if [[ $EUID -ne 0 ]]; then SUDO="sudo -n"; $SUDO true 2>/dev/null || SUDO="sudo"; fi

GUC_SHA="70d74627e395347ea04c37168d92f01c9e940f4b32e0743b6350ca808fdb67bb"
GUC_BYTES=377664
FW_URL="https://raw.githubusercontent.com/torvalds/linux/fb0889c0/firmware/xe/bmg_guc_70.bin"
FW_DIR="/lib/firmware/xe"
FW="$FW_DIR/bmg_guc_70.bin"
KERNEL="6.17.0-1010-intel"
REBOOT_NEEDED=0

# ---------------------------------------------------------------------------
# 1. OMIX 0.4 (DLE 2026.1 userspace)
# ---------------------------------------------------------------------------
if dpkg -l intel-omix 2>/dev/null | grep -q '^ii'; then
    v=$(dpkg-query -W -f='${Version}' intel-omix)
    case "$v" in 0.4*) ok "OMIX already installed: intel-omix $v" ;;
        *) err "intel-omix $v installed but 0.4.x required";; esac
else
    info "Installing intel-omix 0.4 (Intel GPU runtime repo required)"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq intel-omix intel-omix-dev \
        || err "apt install intel-omix failed — enable the Intel GPU software repository first (https://dgpu-docs.intel.com)"
    v=$(dpkg-query -W -f='${Version}' intel-omix)
    case "$v" in 0.4*) ok "OMIX installed: $v" ;; *) err "OMIX $v != 0.4.x" ;; esac
fi

# ---------------------------------------------------------------------------
# 2. Kernel 6.17.0-1010-intel + grub (iommu=off)
# ---------------------------------------------------------------------------
if dpkg -l "linux-image-$KERNEL" 2>/dev/null | grep -q '^ii'; then
    ok "Kernel $KERNEL installed"
else
    info "Installing kernel $KERNEL"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq "linux-image-$KERNEL" "linux-headers-$KERNEL" \
        || err "kernel install failed — the intel-oak/lkdc PPA hosting 6.17.0-1010-intel must be enabled"
fi
KPATH=$(ls "/boot/vmlinuz-$KERNEL" 2>/dev/null || true)
[[ -n "$KPATH" ]] && ok "Kernel image present: $KPATH" || err "vmlinuz-$KERNEL missing after install"

GRUB="/etc/default/grub"
if grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB" | grep -q 'iommu=off'; then
    ok "grub: iommu=off already set"
else
    info "Setting GRUB_CMDLINE_LINUX_DEFAULT=\"iommu=off\""
    $SUDO sed -i -E 's/^GRUB_CMDLINE_LINUX_DEFAULT=.*/GRUB_CMDLINE_LINUX_DEFAULT="iommu=off"/' "$GRUB"
    grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=' "$GRUB" | grep -q 'iommu=off' || err "grub edit failed"
    REBOOT_NEEDED=1
fi
# Make the intel kernel the default boot entry
$SUDO sed -i -E 's|^GRUB_DEFAULT=.*|GRUB_DEFAULT="Advanced options for Ubuntu>Ubuntu, with Linux '"$KERNEL"'"|' "$GRUB"
if command -v grub-editenv >/dev/null 2>&1; then
    $SUDO grub-editenv /boot/grub/grubenv unset next_entry || true
    $SUDO grub-editenv /boot/grub/grubenv set saved_entry="$KERNEL" || true
fi
$SUDO update-grub 2>/dev/null | tail -1 || true
grep -q "$KERNEL" /boot/grub/grub.cfg && ok "grub: $KERNEL menu entry present" || err "grub.cfg missing $KERNEL entry"

# ---------------------------------------------------------------------------
# 3. GuC 70.65 firmware (linux-firmware fb0889c0)
# ---------------------------------------------------------------------------
current_sha=$([[ -f "$FW" ]] && sha256sum "$FW" | cut -d' ' -f1 || echo absent)
if [[ "$current_sha" == "$GUC_SHA" ]]; then
    ok "GuC firmware already 70.65 ($GUC_SHA)"
else
    info "Installing GuC 70.65 blob (from $FW_URL)"
    $SUDO mkdir -p "$FW_DIR"
    if [[ -f "$FW_DIR/bmg_guc_70.bin.v7065.bak" ]]; then
        bak_sha=$(sha256sum "$FW_DIR/bmg_guc_70.bin.v7065.bak" | cut -d' ' -f1)
        if [[ "$bak_sha" == "$GUC_SHA" ]]; then
            info "restoring from existing verified backup .v7065.bak"
            $SUDO cp "$FW_DIR/bmg_guc_70.bin.v7065.bak" "$FW"
        fi
    fi
    current_sha=$([[ -f "$FW" ]] && sha256sum "$FW" | cut -d' ' -f1 || echo absent)
    if [[ "$current_sha" != "$GUC_SHA" ]]; then
        $SUDO curl -fsSL -o /tmp/bmg_guc_70.bin.fb0889c0 "$FW_URL" \
            || err "GuC blob download failed"
        got_sha=$(sha256sum /tmp/bmg_guc_70.bin.fb0889c0 | cut -d' ' -f1)
        got_size=$(stat -c%s /tmp/bmg_guc_70.bin.fb0889c0)
        [[ "$got_sha" == "$GUC_SHA" ]] || { echo "got $got_sha ($got_size B)"; err "GuC blob hash mismatch vs fb0889c0"; }
        [[ "$got_size" -eq "$GUC_BYTES" ]] || err "GuC blob size $got_size != $GUC_BYTES"
        $SUDO cp "$FW" "$FW_DIR/bmg_guc_70.bin.pre-v7065.bak" 2>/dev/null || true
        $SUDO cp /tmp/bmg_guc_70.bin.fb0889c0 "$FW"
        $SUDO cp /tmp/bmg_guc_70.bin.fb0889c0 "$FW_DIR/bmg_guc_70.bin.v7065.bak"
        rm -f /tmp/bmg_guc_70.bin.fb0889c0
    fi
    final_sha=$(sha256sum "$FW" | cut -d' ' -f1)
    [[ "$final_sha" == "$GUC_SHA" ]] || err "post-install GuC hash mismatch"
    ok "GuC 70.65 installed + verified ($final_sha)"
    REBOOT_NEEDED=1
fi

# ---------------------------------------------------------------------------
# 4. docker nofile ulimit
# ---------------------------------------------------------------------------
if systemctl show docker -p LimitNOFILE 2>/dev/null | grep -qE 'infinity|1048576|65536'; then
    ok "docker LimitNOFILE: $(systemctl show docker -p LimitNOFILE --value)"
else
    info "Setting docker nofile (LimitNOFILE=infinity)"
    $SUDO mkdir -p /etc/systemd/system/docker.service.d
    printf '[Service]\nLimitNOFILE=infinity\n' | $SUDO tee /etc/systemd/system/docker.service.d/nofile.conf >/dev/null
    $SUDO systemctl daemon-reload
    $SUDO systemctl restart docker
    sleep 2
    systemctl show docker -p LimitNOFILE | grep -qE 'infinity|1048576|65536' \
        || err "docker LimitNOFILE still low"
    ok "docker LimitNOFILE: $(systemctl show docker -p LimitNOFILE --value)"
fi

# ---------------------------------------------------------------------------
# 5. xe GuC job timeout — DEVICE_LOST mitigation (2026-09-23)
# ---------------------------------------------------------------------------
# 98K-class prefill workloads can hold a GuC job longer than the default
# 5000 ms on the batch-copy (bcs) and compute (ccs) engines; the resulting
# engine reset is the DEVICE_LOST (error-20) crash seen during long-context
# needles. The driver hard-caps this knob at 10000 ms (writes above fail with
# EINVAL), so the shippable maximum is 10000 — 2x the default.
JOB_TIMEOUT_SCRIPT=/usr/local/sbin/set-xe-job-timeout.sh
JOB_TIMEOUT_UNIT=/etc/systemd/system/xe-job-timeout.service
cur=$(cat /sys/class/drm/card1/device/tile0/gt0/engines/bcs/job_timeout_ms 2>/dev/null || echo 0)
if [[ "$cur" == "10000" && -x "$JOB_TIMEOUT_SCRIPT" && -f "$JOB_TIMEOUT_UNIT" ]]; then
    ok "xe job timeout already installed (current: $cur ms)"
else
    info "Installing xe job-timeout raise (bcs+ccs -> 10000 ms)"
    $SUDO tee "$JOB_TIMEOUT_SCRIPT" >/dev/null <<'JTSCRIPT'
#!/bin/bash
# Raise xe GuC job timeout on bcs+ccs engines of every GPU card.
# Driver hard cap = 10000 ms (writes above fail with EINVAL). Default = 5000.
TARGET=10000
for i in $(seq 1 60); do
  FOUND=0
  for c in /sys/class/drm/card*; do
    [ -d "$c/device/tile0" ] || continue
    for g in "$c"/device/tile0/gt*/; do
      for e in bcs ccs; do
        f="$g/engines/$e/job_timeout_ms"
        [ -e "$f" ] && FOUND=$((FOUND+1)) && { cur=$(cat "$f" 2>/dev/null); [ "$cur" != "$TARGET" ] && echo "$TARGET" > "$f" 2>/dev/null; }
      done
    done
  done
  [ "$FOUND" -ge 8 ] && break
  sleep 1
done
echo "xe-job-timeout: wrote $TARGET ms to $FOUND engine files (bcs+ccs, all cards)"
JTSCRIPT
    $SUDO chmod +x "$JOB_TIMEOUT_SCRIPT"
    $SUDO tee "$JOB_TIMEOUT_UNIT" >/dev/null <<'JTUNIT'
[Unit]
Description=Raise xe GuC job timeout to 10000ms on BCS/CCS engines of all GPU cards (B70 DEVICE_LOST mitigation; driver cap 10000, default 5000)
After=systemd-modules-load.service multi-user.target

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/set-xe-job-timeout.sh

[Install]
WantedBy=multi-user.target
JTUNIT
    $SUDO systemctl daemon-reload
    $SUDO systemctl enable --now xe-job-timeout.service >/dev/null 2>&1
fi
sleep 2
bad=0; found=0
for c in /sys/class/drm/card*; do
    [ -d "$c/device/tile0" ] || continue
    for e in bcs ccs; do
        f="$c/device/tile0/gt0/engines/$e/job_timeout_ms"
        [ -r "$f" ] || continue
        found=$((found+1))
        [[ "$(cat "$f")" == "10000" ]] || bad=$((bad+1))
    done
done
[[ "$found" -ge 8 ]] || err "xe job_timeout_ms nodes not found (xe driver not bound?)"
[[ "$bad" -eq 0 ]] || err "$bad of $found bcs/ccs engines NOT at 10000 ms"
ok "xe job timeout: 10000 ms on all bcs/ccs engines ($found files, systemd unit active)"

# ---------------------------------------------------------------------------
# Read-back summary
# ---------------------------------------------------------------------------
info "=== Read-back ==="
echo "  intel-omix : $(dpkg-query -W -f='${Version}' intel-omix 2>/dev/null || echo ABSENT)"
echo "  kernel pkg : $(dpkg-query -W -f='${Status}' "linux-image-$KERNEL" 2>/dev/null || echo ABSENT)"
echo "  grub cmd   : $(grep -E '^GRUB_CMDLINE_LINUX_DEFAULT=' /etc/default/grub)"
echo "  saved_entry: $(grub-editenv /boot/grub/grubenv list 2>/dev/null | grep saved_entry || echo unset)"
echo "  GuC sha256 : $(sha256sum "$FW" 2>/dev/null | cut -d' ' -f1 || echo ABSENT)"
echo "  LimitNOFILE: $(systemctl show docker -p LimitNOFILE --value 2>/dev/null || echo unknown)"
echo "  xe job t/o : $(cat /sys/class/drm/card1/device/tile0/gt0/engines/bcs/job_timeout_ms 2>/dev/null || echo unknown) ms (bcs; ccs must match)"
echo "  running    : $(uname -r)  (target: $KERNEL)"

if [[ "$REBOOT_NEEDED" -eq 1 || "$(uname -r)" != "$KERNEL" ]]; then
    warn ""
    warn "REBOOT REQUIRED — kernel/iommu/GuC apply on next boot:"
    warn "  sudo reboot"
else
    ok "Host is at the platform of record. No reboot needed."
fi
