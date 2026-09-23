#!/bin/bash
# riglib.sh — shared helpers for every future watcher/battery on this box.
# Directive (Ryan, 2026-09-22): source this file; do NOT re-implement these inline.
# No inline heredocs through SSH anywhere. Usage: source /home/bonz/riglib.sh
# Assumes: fdsample2.py (heartbeat sampler) at /home/bonz/fdsample2.py.

# --- 1) rcount: clean integer match count, never empty, never doubled -------
# rcount <pattern> <file> [offset_bytes]
#   grep -c prints 0 and exits 1 on zero matches — `|| echo 0` DOUBLE-FIRES
#   ("0\n0" -> arithmetic error). Here: capture without ||, default empty->0.
#   With offset: counts only bytes AFTER that offset (log fired at offset O ->
#   stale text from prior boots is invisible, permanently).
#   -E so alternation patterns work (census uses "a|b|c").
rcount() {
  local pat="$1" file="$2" off="${3:-0}" n
  if [ "$off" -gt 0 ] && [ -s "$file" ]; then
    n=$(tail -c +"$((off + 1))" "$file" 2>/dev/null | grep -acE "$pat")
  else
    n=$(grep -acE "$pat" "$file" 2>/dev/null)
  fi
  [ -z "$n" ] && n=0
  echo "$n"
}

# --- 2) logmark: byte offset of a log right now (0 if absent) ---------------
# Usage at fire time:  OFF=$(logmark /home/bonz/rollback-unit/server.log)
#   Save it (e.g. echo "$OFF" > .run/<boot>/log.offset); pass to rcount and
#   poll_verdict for the whole boot.
logmark() {
  local f="$1" s
  s=$(stat -c %s "$f" 2>/dev/null)
  [ -z "$s" ] && s=0
  echo "$s"
}

# --- 3) sampler_start: heartbeat sampler + 10-second standalone gate --------
# sampler_start <output_log>
#   Kills any old instance, starts fdsample2.py fresh (SNAP heartbeat every
#   pass), verifies >=2 new lines in 10 s. rc 0 = proven live; rc 1 = ABORT.
#   The heartbeat exists because a /proc fdinfo sampler writes NOTHING on an
#   idle host (no /dev/dri fds open) — file-exists is a false pass.
sampler_start() {
  local out="$1" l0 l1
  sudo -n pkill -f fdsample2.py 2>/dev/null
  sleep 1
  rm -f "$out"
  sudo -n setsid nohup python3 /home/bonz/fdsample2.py "$out" >/dev/null 2>&1 &
  l0=$(wc -l < "$out" 2>/dev/null); [ -z "$l0" ] && l0=0
  sleep 10
  l1=$(wc -l < "$out" 2>/dev/null); [ -z "$l1" ] && l1=0
  [ $((l1 - l0)) -ge 2 ] || return 1
  return 0
}

# --- 4) poll_verdict: READY / OOM / stall / container-death / timeout -------
# poll_verdict <offset> <deadline_epoch> <container> <models_url> <model_substr> <server_log> [grace_s]
#   Greps ONLY past <offset> (fire-time logmark) — stale-text immune.
#   Order: READY first (served = success), then OOM occurrence delta, then
#   shm_broadcast delta (>base+2), then container gone (seen-then-gone, or
#   never-seen past grace — start.sh preflight takes ~1 min before docker run).
#   Echoes one of: READY | TORCH_OOM | WORKER_ATTACH_STALL | CONTAINER_GONE | TIMEOUT
poll_verdict() {
  local off="$1" dl="$2" c="$3" url="$4" sub="$5" log="$6" grace="${7:-300}"
  local t0 run rd om seen=no base_oom base_shm
  t0=$(date +%s)
  base_oom=$(rcount "OutOfMemoryError" "$log" "$off")
  base_shm=$(rcount "shm_broadcast" "$log" "$off")
  while :; do
    run=$(docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$c" && echo yes || echo no)
    [ "$run" = yes ] && seen=yes
    rd=$(curl -fsS -m 5 "$url" 2>/dev/null | grep -c "$sub" || true)
    [ "${rd:-0}" -ge 1 ] && { echo READY; return 0; }
    om=$(rcount "OutOfMemoryError" "$log" "$off")
    [ "$om" -gt "$base_oom" ] && { echo TORCH_OOM; return 0; }
    [ "$(rcount "shm_broadcast" "$log" "$off")" -gt $((base_shm + 2)) ] && { echo WORKER_ATTACH_STALL; return 0; }
    if [ "$run" = no ]; then
      if [ "$seen" = yes ] || [ "$(date +%s)" -gt $((t0 + grace)) ]; then
        echo CONTAINER_GONE; return 0
      fi
    fi
    [ "$(date +%s)" -gt "$dl" ] && { echo TIMEOUT; return 0; }
    sleep 30
  done
}

# --- 5) census: THE fixed three-pattern reset census (directive 2026-09-23) --
# census [logfile] [offset_bytes]
#   No args: counts in the CURRENT BOOT's kernel ring (sudo -n dmesg).
#   With logfile(+offset): counts in that file after offset (server-side
#   evidence; pairs with logmark).
#   Patterns (fixed, by directive): Engine reset | guc_exec_queue_timedout_job
#   | DEVICE_LOST — a DEVICE_LOST-only census missed the actual GuC reset
#   storm (2026-09-23, needle B @ MML 98304). Inline grep versions of this
#   census FAILED THREE TIMES on shell quoting through SSH; this function is
#   the only census now. Prints a bare integer; prints CENSUS_UNKNOWN (rc 3)
#   when dmesg is unavailable — never a silent 0.
census() {
  local pat="Engine reset|guc_exec_queue_timedout_job|DEVICE_LOST"
  local log="${1:-}" off="${2:-0}" n
  if [ -n "$log" ]; then
    rcount "$pat" "$log" "$off"
    return 0
  fi
  if ! sudo -n dmesg >/dev/null 2>&1; then
    echo CENSUS_UNKNOWN
    return 3
  fi
  n=$(sudo -n dmesg 2>/dev/null | grep -acE "$pat")
  [ -z "$n" ] && n=0
  echo "$n"
  return 0
}
