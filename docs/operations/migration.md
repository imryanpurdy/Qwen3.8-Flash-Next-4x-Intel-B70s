# Migrating the production lane to a persistent directory

The fresh-clone production launch of 2026-10-05 (outage window
05:59:01Z → 06:13Z UTC) put the serving lane in `/tmp/fresh-launch` — a
valid deploy of record for the launch, but `/tmp` does not survive a reboot
and must not be the permanent home. This is the planned-window procedure to
move the lane into a persistent directory (`~/qwen-prod` in the examples).

Everything below is a same-recipe move: fresh clone of `main`, real `.env`,
real `lumnus.env`, real cache — then the standard stop → start → verify
sequence. No new gates, no new flags.

## What moves and what doesn't

| Item | Moves? | Why |
|---|---|---|
| Repo checkout (fresh clone of `main`) | Yes — new clone IS the new lane | `/tmp` is wiped on reboot |
| `.env` (lane config of record) | Copy from old lane | Lane identity: MODEL_PATH, PLE dirs, image pin |
| `lumnus.env` (engine env) | Copy from old lane | Engine env file (`--env-file`): UR_L0 flag, B70_* knobs |
| `cache/` (LANE_CACHE → /cache) | Copy | torch.compile cache reuse is why reboots are ~4-5 min; without it a reboot costs a fresh compile |
| `.run/` (watchdog pid/log, manifest) | **Do NOT copy** | Stale pidfiles point at dead watchdogs; `start.sh` recreates it |
| Weights, PLE tables, image | No — untouched | System paths (`/data/awq-snapshot-trial`, `/srv/hf-devan`, `/data/int8-ple`) and the local image are lane-independent |

## Procedure (planned window; expect ~15 min of API downtime)

Run everything ON the rig, from the NEW lane dir once it exists.

```bash
# 0. Preconditions (before the window): children drained; no verify/soak
#    jobs in flight; you have ~15 min of acceptable downtime.

# 1. Fresh clone (persistent home, NOT /tmp):
git clone <repo-url> ~/qwen-prod
cd ~/qwen-prod
git checkout main && git log --oneline -1        # confirm tip

# 2. Copy lane identity + cache from the old lane (adjust OLD= as needed):
OLD=/tmp/fresh-launch
cp "$OLD/.env" "$OLD/lumnus.env" .
cp -a "$OLD/cache" ./cache

# 3. Re-point LANE_CACHE at the new path (it embeds the lane dir):
sed -i "s|^LANE_CACHE=.*|LANE_CACHE=$HOME/qwen-prod/cache|" .env
grep -n LANE_CACHE .env                          # confirm

# 4. Stop the OLD lane (watchdog first, then container — stop.sh order):
"$OLD/scripts/stop.sh"
docker ps --format '{{.Names}}' | grep -E 'b70-lumnus-prod|es-lane'   # expect: none
pgrep -fa wedge-watchdog.sh                                            # expect: none

# 5. Start from the NEW lane (full gate path; ~4-5 min to READY):
cd ~/qwen-prod && ./scripts/start.sh

# 6. Verify (full gate):
./tests/verify.sh

# 7. Confirm the watchdog's restart command points at the NEW path:
grep PROD_RESTART_CMD .run/watchdog.log          # expect: .../qwen-prod/scripts/start.sh --launch --replace
#    (start.sh exports PROD_RESTART_CMD="$SCRIPT_DIR/start.sh --launch" with
#    SCRIPT_DIR = the new lane; the watchdog log echoes it on startup.)

# 8. Retire the old lane dir:
rm -rf "$OLD"
```

## Boot persistence

Install the systemd unit so the lane comes back on its own after a reboot:

```bash
sudo ./scripts/install-systemd.sh          # renders, installs, enables;
                                           # starts now only if lane is down
systemctl status b70-lumnus-prod
```

The unit (`deploy/systemd/b70-lumnus-prod.service.template`) runs the full
`scripts/start.sh` gate path `After=docker.service`, `Restart=on-failure`
with `StartLimitBurst=3` (mirrors the watchdog's bounded-retry philosophy),
and `ExecStop=scripts/stop.sh` (watchdog first, then container). The
watchdog's own restart path (`--launch`, gate-skipping) is unchanged — the
unit only covers start-time failure and boot sequencing; runtime wedge
recovery stays the watchdog's job.

Reboot-test proof (when the maintenance window allows):

```bash
sudo systemctl reboot
# after boot:
systemctl status b70-lumnus-prod      # active (exited) = lane launched
./scripts/status.sh                   # container up, API READY, watchdog armed
journalctl -u b70-lumnus-prod -b | tail
```

`install-systemd.sh` is idempotent (re-render + daemon-reload + enable).
`--no-start` arms the unit without starting the lane (use when the lane is
already serving and you only want boot persistence).
