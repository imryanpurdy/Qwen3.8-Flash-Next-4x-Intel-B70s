#!/usr/bin/env bash
# ============================================================================
# install-systemd.sh — install/refresh the boot-persistence systemd unit for
# the production lane.
#
# What it does:
#   1. Renders deploy/systemd/b70-lumnus-prod.service.template with the real
#      repo path and invoking user, writes it to /etc/systemd/system/.
#   2. Runs `systemctl daemon-reload` and `systemctl enable` (boot persistence).
#   3. Optionally starts the lane now (default: start; --no-start to skip —
#      e.g. when the lane is already up and you only want the unit armed).
#
# Requires sudo (NOPASSWD on the rig). Safe to re-run: idempotent re-render
# + daemon-reload; enable is a no-op when already enabled.
#
# Reboot test (the proof the unit works):
#   sudo systemctl reboot
#   ... after boot:
#   systemctl status b70-lumnus-prod          # active (exited), enabled
#   scripts/status.sh                          # container up, API READY,
#                                              # watchdog armed
#   journalctl -u b70-lumnus-prod -b | tail    # start.sh's gate log
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMPLATE="$REPO_ROOT/deploy/systemd/b70-lumnus-prod.service.template"
UNIT_NAME="b70-lumnus-prod.service"
UNIT_DEST="/etc/systemd/system/$UNIT_NAME"
RUN_USER="${SUDO_USER:-$(id -un)}"

START_LANE=1
for arg in "$@"; do
    case "$arg" in
        --no-start) START_LANE=0 ;;
        -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown argument: $arg (try --help)" >&2; exit 1 ;;
    esac
done

[[ -f "$TEMPLATE" ]] || { echo "template missing: $TEMPLATE" >&2; exit 1; }
[[ -x "$REPO_ROOT/scripts/start.sh" ]] || { echo "scripts/start.sh missing/not executable" >&2; exit 1; }
[[ -x "$REPO_ROOT/scripts/stop.sh" ]] || { echo "scripts/stop.sh missing/not executable" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "systemctl not found (not a systemd host?)" >&2; exit 1; }
if [[ "$(id -u)" -ne 0 ]]; then
    echo "This script installs to /etc/systemd/system — re-run with sudo." >&2
    exit 1
fi

info() { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()   { echo -e "\033[1;32m[ OK ]\033[0m  $*"; }

# Render: substitute the repo path and the invoking user.
sed -e "s|%REPO_ROOT%|$REPO_ROOT|g" -e "s|%USER%|$RUN_USER|g" "$TEMPLATE" > "/tmp/$UNIT_NAME.$$"
# Sanity: no unsubstituted placeholders may survive.
if grep -q '%REPO_ROOT%\|%USER%' "/tmp/$UNIT_NAME.$$"; then
    echo "ERROR: unsubstituted placeholder in rendered unit" >&2
    rm -f "/tmp/$UNIT_NAME.$$"
    exit 1
fi

install -m 0644 "/tmp/$UNIT_NAME.$$" "$UNIT_DEST"
rm -f "/tmp/$UNIT_NAME.$$"
ok "Unit written: $UNIT_DEST (user=$RUN_USER, repo=$REPO_ROOT)"

systemctl daemon-reload
ok "daemon-reload done"

systemctl enable "$UNIT_NAME"
ok "Enabled at boot (WantedBy=multi-user.target, After=docker.service)"

if [[ "$START_LANE" == "1" ]]; then
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$(grep -E '^CONTAINER_NAME=' "$REPO_ROOT/.env" 2>/dev/null | cut -d= -f2 | tr -d '\"' || echo b70-lumnus-prod)"; then
        info "Lane already running — not starting via systemd (use --no-start to silence this)."
        info "The unit is armed for next boot: systemctl status $UNIT_NAME"
    else
        info "Starting lane via systemd (full gate path, ~4-5 min to READY)..."
        systemctl start "$UNIT_NAME"
        ok "Started: $(systemctl is-active "$UNIT_NAME") — $(systemctl is-enabled "$UNIT_NAME")"
    fi
else
    ok "--no-start: unit armed only. Start with: systemctl start $UNIT_NAME"
fi

cat <<'EOF'

Reboot test (proof):
  sudo systemctl reboot
  # after boot:
  systemctl status b70-lumnus-prod     # active (exited) = lane launched+READY
  scripts/status.sh                    # container up, API READY, watchdog armed
  journalctl -u b70-lumnus-prod -b | tail
EOF
