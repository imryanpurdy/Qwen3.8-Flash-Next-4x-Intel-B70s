#!/usr/bin/env bash
# rebuild-verify.sh — runbook L1-L5 verification ladder as ONE gate script (docs/rebuild/2026-09-19-platform-rebuild-runbook.md sec 4).
# Usage: ./rebuild-verify.sh (env: PORT= MODEL= MNS= CONTAINER= BURSTS=). Windows-authored: if bash errors, sed -i 's/\r$//' $0
set -uo pipefail   # no set -e: probes/restarts handled explicitly below
PORT=${PORT:-8021}; MODEL=${MODEL:-qwen3.8-flash-next}       # SERVED_MODEL_NAME
MNS=${MNS:-16}; CONTAINER=${CONTAINER:-qwen38-flash-next}; BURSTS=${BURSTS:-15}
CONC=${CONC:-8}; CONC3=${CONC3:-8}; CONC4=${CONC4:-16}   # L3 burst width / L4 campaign width (overnight plan: 8 / 16)
LADDER=${LADDER:-full}   # full | L1 | L1,L2,L3 | L4 | L3,L4 — comma list of gates to run (L5 always report-only)
READY_TIMEOUT=${READY_TIMEOUT:-1800}; ENGINE_LOG=${ENGINE_LOG:-.run/server.log}  # L1 poll cap (s); engine-log path VERIFIED live 2026-09-20 (fn-recipe-int4/.run/server.log)
STALLSPY_CYCLES=${STALLSPY_CYCLES:-75}; STALLSPY_CADENCE=${STALLSPY_CADENCE:-5}; STALLSPY_MAX=${STALLSPY_MAX:-300}; STALLSPY_TIMEOUT=${STALLSPY_TIMEOUT:-900}; STALL_MIN=200  # L3 loop/dumper cap (s); gate floor (runbook targets >=300)
VLLM_SP=/opt/venv/lib/python3.12/site-packages/vllm
API=http://localhost:${PORT}; DUMP_DIR=/tmp/rebuild-verify-dumps; TEE_LOG=${TEE_LOG:-/tmp/rebuild-verify.log}
exec > >(tee -a "$TEE_LOG") 2>&1; echo "$(date -u +%FT%TZ) === rebuild-verify start (PORT=$PORT MODEL=$MODEL MNS=$MNS BURSTS=$BURSTS) ==="
say(){ printf '%s\n' "$*"; }; warn(){ say "WARN: $*"; }
DOCKER_OK=0; command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1 && DOCKER_OK=1
[[ $DOCKER_OK -eq 1 ]] || warn "no docker/daemon — container-local steps will SKIP"
CE(){ [[ $DOCKER_OK -eq 1 ]] && docker exec "$CONTAINER" sh -c "$1" 2>/dev/null; }
elog(){ if [[ -s "$ENGINE_LOG" ]]; then cat "$ENGINE_LOG"; elif [[ $DOCKER_OK -eq 1 ]]; then docker logs "$CONTAINER" 2>/dev/null; fi; }
gen_code(){ curl -s -o /dev/null -w '%{http_code}' -m 30 -X POST "$API/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"hi\",\"max_tokens\":1,\"temperature\":0}" 2>/dev/null || echo 000; }
gen_probe(){ local i c; for i in 1 2 3; do c=$(gen_code); [[ "$c" == "200" ]] && return 0; say "  gen-probe #$i: HTTP $c (engine busy/restarting?)"; sleep 10; done; return 1; }  # engine-true health (runbook sec 6)
complete(){ curl -s -m 90 -X POST "$API/v1/completions" -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"prompt\":\"$1\",\"max_tokens\":$2,\"temperature\":0}" 2>/dev/null; }
SOAK=/tmp/rebuild-verify-soakfix.py   # reuse /tmp/soakfix.py when URLs match, else parameterized inline copy (host-side; runbook 3.0.1 ok)
soak_setup(){
  if [[ -f /tmp/soakfix.py ]] && grep -q "localhost:$PORT" /tmp/soakfix.py && grep -q "\"$MODEL\"" /tmp/soakfix.py; then cp /tmp/soakfix.py "$SOAK"; else
    cat > "$SOAK" <<'PYEOF'
import json,os,sys,time
import urllib.request,concurrent.futures as cf
URL=os.environ.get("SOAK_URL","http://localhost:8021/v1/completions");BASE=URL.rsplit("/v1/completions",1)[0]
MODEL=os.environ.get("SOAK_MODEL","qwen3.8-flash-next")
P=json.dumps({"model":MODEL,"prompt":"The history of computing began when humans first learned to count. Write a detailed technical essay about the development of computing machinery.","max_tokens":600,"temperature":0}).encode()
def one(_):
 try:
  with urllib.request.urlopen(urllib.request.Request(URL,data=P,headers={"Content-Type":"application/json"}),timeout=500) as r:return json.loads(r.read())["usage"]["completion_tokens"],0
 except Exception:return 0,1
n=int(sys.argv[1]);ags=[];sus=[]
for t in ("r1","r2","r3","r4"):
 t0=time.time()
 with cf.ThreadPoolExecutor(max_workers=n) as ex:res=list(ex.map(one,range(n)))
 w=time.time()-t0;toks=sum(x[0] for x in res);e=sum(x[1] for x in res);a=toks/w if w else 0.0
 print("%s round: tok=%d errs=%d wall=%.1fs agg=%.1f"%(t,toks,e,w,a));ags.append(a)
 if t!="r1":sus.append(a)
try:
 with urllib.request.urlopen(BASE+"/v1/models",timeout=8) as x:post=x.status
except Exception:post=0
print("summary: sustained_agg=%.1f (r2-r4) mean_agg=%.1f min_agg=%.1f max_agg=%.1f post_models=%d"%(sum(sus)/len(sus) if sus else 0,sum(ags)/len(ags),min(ags),max(ags),post))
PYEOF
  fi
}
soak_run(){ SOAK_URL="$API/v1/completions" SOAK_MODEL="$MODEL" python3 "$SOAK" "$1" 2>&1; }
l1(){ # BOOT: READY + boot provenance + in-container canaries + capture-size coverage
  say "=== L1 BOOT ==="
  local dl=$(( $(date +%s) + READY_TIMEOUT ))
  while :; do
    if [[ $DOCKER_OK -eq 1 ]] && ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; then say "  note: container not listed (restart window?) — probe loop continues"; fi
    if curl -fsS -m 10 "$API/v1/models" 2>/dev/null | grep -q "$MODEL"; then break; fi
    (( $(date +%s) > dl )) && { say "FAIL[L1]: no /v1/models READY within ${READY_TIMEOUT}s (log: $ENGINE_LOG)"; return 1; }
    sleep 15
  done
  say "L1: /v1/models READY ($MODEL)"
  local st; st=$(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null || echo UNKNOWN)
  say "L1 provenance: StartedAt=$st (boot epoch) | boot_clock=$(grep -m1 -oE '\"boot_id\"[^,}]*' .run/boot_clock.jsonl 2>/dev/null || echo 'no boot_clock.jsonl in this engine setup (verified 2026-09-20 — wd-decisions.jsonl is the only jsonl)') | uptime=$(cut -d' ' -f1 /proc/uptime)s"
  gen_probe || { say "FAIL[L1]: /v1/models 200 but 1-token gen failed (EngineCore dead — runbook blindspot)"; return 1; }
  if [[ $DOCKER_OK -ne 1 ]]; then say "L1 canaries+capture: SKIP (no docker)"; return 0; fi
  # v25 canary (v3 RETIRED — upstream 4e8b849b8d97 event-pool fix is the base):
  # static: event pool in the installed connector, single-slot queue GONE.
  # dynamic: per-rank connector registration lines in the engine log (want >=4 on TP4).
  c1=$(CE "grep -c '_d2h_event_pool' $VLLM_SP/v1/ple_offload/connector.py"); c2=$(CE "grep -c 'maxsize=1' $VLLM_SP/v1/ple_offload/connector.py")
  say "  canaries: _d2h_event_pool=$c1 (want >=5) maxsize=1=$c2 (want 0) [4e8b849b8d97 structure]"
  { [[ "$c1" -ge 5 && "$c2" == "0" ]]; } || { say "FAIL[L1]: connector canary mismatch — wrong fork tree or 4e8b849b8d97 missing"; return 1; }
  regn=$(elog | grep -c 'PleOffload: registered'); say "  canaries: PleOffload registered lines=$regn (want >=4 on TP4)"
  [[ "$regn" -ge 4 ]] || { say "FAIL[L1]: connector did not register on all ranks (PleOffload registration < 4)"; return 1; }
  # BOOT PROFILE (misread guard): L4 throughput baselines are profile-specific.
  # The 163-sustained / 80 tok/s program numbers are GRAPHS-boot numbers; eager
  # boots are CPU-launch-bound (old-stack eager reference: 4.6-4.9 batched
  # steps/s) and MUST NOT be scored against graphs baselines.
  local mmode mMTP
  mmode=$(grep -oE '"graph_mode": *"[a-z]+"' .run/manifest.json 2>/dev/null | grep -oE '[a-z]+' | tail -1); mmode=${mmode:-unknown}
  mMTP=$(grep -oE '"mtp_num_speculative_tokens": *"[0-9]+"' .run/manifest.json 2>/dev/null | grep -oE '[0-9]+' | tail -1); mMTP=${mMTP:-?}
  BOOT_PROFILE="graphs=$mmode MTP=$mMTP"
  elog | grep -q 'Capturing CUDA graphs' && BOOT_PROFILE="graphs=on MTP=$mMTP"
  say "  boot profile: $BOOT_PROFILE (from manifest+engine log; graphs evidence = capture lines present)"
  local cap cap_n cap_last cap_cnt
  if [[ "$BOOT_PROFILE" == graphs=on* ]]; then
  cap=$(elog | grep -oE 'cudagraph_capture_sizes[^]]*\]' | tail -1)
  [[ -n "$cap" ]] || { say "FAIL[L1]: 'cudagraph_capture_sizes' not in engine log"; return 1; }
  cap_n=$(printf '%s' "$cap" | grep -oE '[0-9]+' | wc -l | tr -d ' '); cap_last=$(printf '%s' "$cap" | grep -oE '[0-9]+' | tail -1)
  # Engine emits ONE tqdm completion line with N/N denominator (live-verified 2026-09-20: 'Capturing CUDA graphs (FULL): 100%|...| 10/10'),
  # not per-size lines - gate on the final line's denominator == list length (CR progress fragments make raw line-count wrong).
  cap_cnt=$(elog | grep 'Capturing CUDA graphs (FULL)' | tail -1 | grep -oE '[0-9]+/[0-9]+' | tail -1 | cut -d/ -f2)
  say "  capsizes: '$cap' n=$cap_n last=$cap_last MNS=$MNS FULL-denominator=$cap_cnt"
  [[ "$cap_n" -ge 1 && "$cap_last" == "$MNS" && "$cap_cnt" == "$cap_n" ]] || { say "FAIL[L1]: capture list does not cover MNS or FULL denominator ($cap_cnt) != list length ($cap_n)"; return 1; }
  else
  say "  capsizes: SKIP — eager boot emits no capture lines (expected on this profile, NOT a regression; capture checks activate on the graphs-on boot)"
  fi
  say "L1 PASS: READY [$BOOT_PROFILE], connector canary OK (_d2h_event_pool=$c1, maxsize1=$c2, registered=$regn), captures $( [[ "$BOOT_PROFILE" == graphs=on* ]] && echo "$cap_cnt/$cap_n (per CAP_SIZES_LIST)" || echo SKIP-eager )"
}
l2(){ # KNOWN-ANSWER: Paris + alphabet + engine alive post-probe
  say "=== L2 KNOWN-ANSWER ==="
  local body o; body=$(complete 'The capital of France is' 12)
  printf '%s' "$body" | grep -q 'Paris' || { say "FAIL[L2]: 'Paris' absent: $(printf '%s' "$body" | head -c 200)"; return 1; }
  say "  L2: France -> Paris OK"
  body=$(complete 'The alphabet in order: ' 12)
  o=$(printf '%s' "$body" | grep -oE '"text": *"[^"]*"' | head -1 | cut -d'"' -f4 | tr '[:upper:]' '[:lower:]')
  printf '%s' "$o" | grep -Eq 'a[^a-z]*b[^b-z]*c' || { say "FAIL[L2]: alphabet order wrong: '$o'"; return 1; }
  say "  L2: alphabet ordered OK ('$o')"
  gen_probe || { say "FAIL[L2]: engine dead after known-answer"; return 1; }
  say "L2 PASS"
}
l3(){ # L0-ABSENCE: py-spy proof — >=200 dumps, 0 x appendUSMMemcpy
  say "=== L3 L0-ABSENCE ==="
  if [[ $DOCKER_OK -ne 1 ]]; then say "L3 SKIP: no docker — cannot py-spy inside the container"; L3_MODE=SKIP; return 0; fi
  if ! CE '/opt/venv/bin/py-spy --version' | grep -q 'py-spy'; then
    local wheel; wheel=$(ls /tmp/pyspywheel/py_spy*.whl 2>/dev/null | head -1)
    if [[ -z "$wheel" ]]; then
      say "  WARN: py-spy missing, no wheel — downloading py-spy==0.4.2 (runbook 2.5 #6)"
      mkdir -p /tmp/pyspywheel
      ( python3 -m pip download -q -d /tmp/pyspywheel py-spy==0.4.2 || pip3 download -q -d /tmp/pyspywheel py-spy==0.4.2 ) >/dev/null 2>&1 || true
      wheel=$(ls /tmp/pyspywheel/py_spy*.whl 2>/dev/null | head -1)
    fi
    [[ -n "$wheel" ]] || { say "FAIL[L3]: cannot obtain py-spy wheel (pip download failed)"; return 1; }
    docker cp "$wheel" "$CONTAINER:/tmp/pyspywheel.whl" >/dev/null 2>&1 || { say "FAIL[L3]: docker cp of py-spy wheel"; return 1; }
    CE '/opt/venv/bin/pip install --quiet --force-reinstall /tmp/pyspywheel.whl >/dev/null 2>&1' || { say "FAIL[L3]: pip install py-spy in container"; return 1; }
    say "  L3: py-spy installed from $wheel"
  else say "  L3: py-spy already in container"; fi
  local pids; pids=$(CE "ps -eo pid=,args= | grep -iE 'vllm|python' | grep -v grep | awk '{print \$1}'")
  [[ -n "$pids" ]] || { say "FAIL[L3]: no worker/engine pids in container"; return 1; }
  say "  L3: dump loop over pids $(printf '%s' "$pids" | tr '\n' ' ') (${STALLSPY_CYCLES} cycles x${STALLSPY_CADENCE}s, gate >= $STALL_MIN dumps)"
  rm -rf "$DUMP_DIR"; mkdir -p "$DUMP_DIR"
  ( # stallspy-style dumper (bg; py-spy via docker exec, output lands on host)
    local n=0 t0=$SECONDS p
    while (( n < STALLSPY_MAX && SECONDS - t0 < STALLSPY_TIMEOUT )); do
      for p in $pids; do timeout 20 docker exec "$CONTAINER" /opt/venv/bin/py-spy dump --pid "$p" > "$DUMP_DIR/d_${n}_p${p}.dump" 2>/dev/null || true; n=$((n+1)); done
      sleep "$STALLSPY_CADENCE"
    done
  ) &
  local dp=$!; sleep 2
  say "  L3: dumper armed — firing 1 burst (${CONC3}-way x 4 rounds x 600 tok)..."
  soak_run "$CONC3" >/dev/null 2>&1 || true
  local have=0 i=0; while (( i < 120 )); do have=$(ls "$DUMP_DIR"/*.dump 2>/dev/null | wc -l | tr -d ' '); (( have >= STALL_MIN )) && break; sleep 5; i=$((i+1)); done
  kill "$dp" 2>/dev/null; wait "$dp" 2>/dev/null
  have=$(ls "$DUMP_DIR"/*.dump 2>/dev/null | wc -l | tr -d ' ')
  (( have >= STALL_MIN )) || { say "FAIL[L3]: only ${have} dumps (< ${STALL_MIN}, dumper cap ${STALLSPY_TIMEOUT}s)"; return 1; }
  local hit
  hit=$(grep -lE 'appendUSMMemcpy' "$DUMP_DIR"/*.dump 2>/dev/null | tr '\n' ' ')   # also covers ur_command_list_manager::appendUSMMemcpy
  if [[ -n "$hit" ]]; then
    say "FAIL[L3]: appendUSMMemcpy in $(printf '%s' "$hit" | tr ' ' '\n' | wc -l) dumps: $hit"
    for d in $hit; do say "  --- thread context: $d ---"; grep -B6 'appendUSMMemcpy' "$d" | head -12; done
    say "  verdict input: connector-thread hits = the class v3 targeted (re-derive v3 on the per-batch-event structure); model/engine-thread hits = new class, do NOT blind-patch"
    return 1
  fi
  hit=$(grep -lE 'libur_adapter_level_zero_v2' "$DUMP_DIR"/*.dump 2>/dev/null | tr '\n' ' ')
  [[ -n "$hit" ]] && { say "FAIL[L3]: libur_adapter_level_zero_v2 in: $hit"; return 1; }
  say "L3 PASS: ${have} dumps, 0 appendUSMMemcpy / 0 libur_adapter_level_zero_v2 (ref 0/170 v24f)"
}
l4(){ # CAMPAIGN: BURSTS bursts; zero errs, zero non-200 post-probes, zero stalls (round wall < 400s)
  local profile; profile="graphs=? MTP=?"
  [[ -f .run/manifest.json ]] && profile=$(python3 -c "import json; m=json.load(open('.run/manifest.json')); print('graphs=%s MTP=%s'%(m.get('graph_mode','?'),m.get('mtp_num_speculative_tokens','?')))" 2>/dev/null || echo "graphs=? MTP=?")
  say "=== L4 CAMPAIGN (${BURSTS} bursts x ${CONC}-way x 4 rounds x 600 tok) [boot: $profile] ==="
  case "$profile" in
    *graphs=eager*|*"MTP=0"*)
      say "  L4 baseline scope: STRUCTURAL ONLY (stalls/errs/post-probes) — eager/MTP0 boot. The 163-sustained and 80 tok/s program baselines are GRAPHS-boot numbers, NOT comparable on this profile; scoring against them would misread eager CPU-launch cost (old-stack eager ref: 4.6-4.9 batched steps/s) as a regression. Baseline gate applies on the graphs-on boot."
      ;;
    *graphs=?*|*MTP=?*)
      say "  L4 baseline scope: UNKNOWN — manifest not found; do not compare throughput to any baseline."
      ;;
    *)
      say "  L4 baseline scope: FULL — profile comparable to program baselines (163-sustained graphs lane; 80 tok/s MTP1 single-stream)."
      ;;
  esac
  pgrep -f 'wedge-watchdog' >/dev/null 2>&1 && say "  L4: wedge-watchdog process present (runbook wants v2.5)" || warn "no wedge-watchdog process — runbook L4 requires v2.5 live; verify manually (continuing)"
  local i out errs bad sust post ed_now ed_base stalls=0 tot_errs=0 aggs="" k
  ed_base=$(elog | grep -cE 'EngineDead|TimeoutError' || true)
  for ((i=1;i<=BURSTS;i++)); do
    say "  --- burst $i/$BURSTS ---"
    out=$(soak_run "$CONC4")
    errs=$(printf '%s' "$out" | grep -oE 'errs=[0-9]+' | awk -F= '{s+=$2} END{print s+0}')
    bad=$(printf '%s' "$out" | grep -oE 'wall=[0-9.]+s' | awk -F'[=s]' '$2+0>=400' | wc -l | tr -d ' ')
    sust=$(printf '%s' "$out" | grep -oE 'sustained_agg=[0-9.]+' | tail -1 | cut -d= -f2); [[ -z "$sust" ]] && sust=0
    post=000
    for k in 1 2 3 4 5 6; do post=$(gen_code); [[ "$post" == "200" ]] && break; sleep 15; done
    ed_now=$(elog | grep -cE 'EngineDead|TimeoutError' || true)
    if (( errs > 0 || bad > 0 || post != 200 || ed_now > ed_base )); then
      stalls=$((stalls+1)); say "  burst $i: STALL (errs=$errs walls>=400s=$bad post=$post EngineDead/TO +$((ed_now-ed_base)))"
    else say "  burst $i: sustained=$sust errs=0 post=200"; fi
    tot_errs=$((tot_errs+errs)); aggs="$aggs $sust"; ed_base=$ed_now
  done
  local overall
  overall=$(awk -v a="$aggs" 'BEGIN{n=split(a,x," "); s=0; for(i=1;i<=n;i++) s+=x[i]; if(n) printf "%.1f", s/n; else print "n/a"}')
  say "  L4 overall: sustained per burst=$aggs | mean=$overall | total_errs=$tot_errs | stalls=$stalls/${BURSTS} | decision rule: 0 stalls=PASS-SHIP | 1=INVESTIGATE | >=2=FAIL"
  (( stalls == 0 )) && { say "L4 PASS-SHIP"; return 0; }
  (( stalls == 1 )) && { say "FAIL[L4]: 1 stall — INVESTIGATE before shipping (runbook 0-1 caveat noted)"; return 1; }
  say "FAIL[L4]: $stalls stalls >= 2 — fix ineffective, lock diagnostics"; return 1
}
l5(){ # MTP LADDER: report-only (no gate) — .env flips + boots are manual
  say "=== L5 MTP LADDER (report-only) ==="
  if elog | grep -q 'SpecDecoding metrics'; then
    say "  SpecDecoding metrics (vLLM 0.26.1 lineage, SpecDecodingLogging.log), latest:"
    elog | grep 'SpecDecoding metrics' | tail -1
  else
    say "  SKIP: no 'SpecDecoding metrics' in engine log — need (a) MTP1 speculative-config on, (b) --disable-log-stats off, (c) engine active during the stats window (VLLM_LOG_STATS_INTERVAL, default 10s). Scheduler-side counts from generated_token_ids — valid across runner A/Bs."
  fi
  cat <<'EOF'
  Manual steps (runbook sec 4 L5 — NOT attempted here; each = .env edit + reboot):
    1) MTP0 baseline: 1x soakfix 8-way x4x600 vs anchor boot rel0027 (v24h2 MNS16:
       sustained 314.5/310.4/320.1, mean 275.9, 0 stalls).
    2) MTP1+graphs capture-fault check: .env VLLM_XPU_ENABLE_XPU_GRAPH=1 AND
       SYCL_UR_USE_LEVEL_ZERO_V2=0; reboot (2.2), start, 1 burst; grep 'SpecDecoding
       metrics' for mean acceptance length + avg draft acceptance rate. Capture must NOT
       wedge (boot 4010612 must not recur — P4/P5 removed mid-capture D2H).
    3) Record numbers with boot ID + stack (sec 6); no MTP1+graphed default until capture OK.
EOF
}
# ------------------------------- ladder driver + verdict ------------------------
L1S=NOT-RUN; L2S=NOT-RUN; L3S=NOT-RUN; L4S=NOT-RUN; L5S=PASS; L3_MODE=
soak_setup
WANT(){ printf '%s' ",$LADDER," | grep -qi ",$1,"; }
if WANT L1; then if l1; then L1S=PASS; else L1S=FAIL; fi; else L1S=SKIP-SEL; fi
if [[ "$L1S" == "PASS" ]] && WANT L2; then if l2; then L2S=PASS; else L2S=FAIL; fi; else [[ "$L2S" == "NOT-RUN" ]] && L2S=SKIP-SEL; fi
if [[ "$L2S" == "PASS" ]] && WANT L3; then if l3; then [[ "$L3_MODE" == "SKIP" ]] && L3S=SKIP || L3S=PASS; else L3S=FAIL; fi; else [[ "$L3S" == "NOT-RUN" ]] && L3S=SKIP-SEL; fi
if [[ "$L3S" == "PASS" || "$L3S" == "SKIP" || "$L3S" == "SKIP-SEL" ]] && WANT L4; then if l4; then L4S=PASS; else L4S=FAIL; fi; else [[ "$L4S" == "NOT-RUN" ]] && L4S=SKIP-SEL; fi
l5   # report-only — always printed
say "======================== FINAL VERDICT ========================"
say "  L1 BOOT (ready+canaries+capture coverage): $L1S"
say "  L2 KNOWN-ANSWER:                           $L2S"
say "  L3 L0-ABSENCE (py-spy):                    $L3S"
say "  L4 CAMPAIGN:                               $L4S"
say "  L5 MTP LADDER:                             $L5S (report-only)"
say "  boot_id (docker StartedAt): $(docker inspect -f '{{.State.StartedAt}}' "$CONTAINER" 2>/dev/null || echo UNKNOWN)"
say "  boot_clock.jsonl: $(grep -m1 -oE '\"boot_id\"[^,}]*' .run/boot_clock.jsonl 2>/dev/null || echo 'n/a (no boot_clock.jsonl in this engine setup — docker StartedAt is the boot provenance)')"
say "  kernel:           $(uname -r)"
local_vx=$(CE 'pip show vllm-xpu-kernels 2>/dev/null | grep -iE "^Version"' | head -1); [[ -n "$local_vx" ]] || local_vx=UNKNOWN
say "  vllm-xpu-kernels: $local_vx"
local_guc=$( (dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null) | grep -iE 'guc' | tail -2 | tr '\n' ' '); [[ -n "$local_guc" ]] || local_guc="unreadable (run: sudo dmesg | grep -i guc)"
say "  GuC/firmware:     $local_guc"
say "  result log:       $TEE_LOG"
say "==================================================================="
rc=0; for s in "$L1S" "$L2S" "$L3S" "$L4S"; do [[ "$s" == "PASS" || "$s" == "SKIP" || "$s" == "SKIP-SEL" ]] || rc=1; done   # SKIP = warned non-verification; SKIP-SEL = not selected via LADDER; neither is a failure
exit $rc
