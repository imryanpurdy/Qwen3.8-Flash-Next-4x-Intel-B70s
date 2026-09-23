# 2026-09-23 — Six directives from Ryan's 16:4x message — execution record

## 1. Discriminator accepted (verbatim: "Discriminator accepted: the trigger is prefix-cache block reuse on the hybrid model, not length.")
- Result: MML 98304, prefix caching OFF, prepend-salted ×3 @ 97,785 tok → 3/3 CORRECT=YES (253.8/205.6/205.3 s full prefills), zero engine resets (fixed census: all 9 boot resets in the 15:03–15:12 PC-ON window). Same MML PC-ON = engine death on needle 2.
- Trigger = hybrid prefix-cache block reuse, not length.

## 2. Interim production (Ryan's call: 81920 + PC ON)
- Verified serving: `max_model_len 81920`, PC-ON, watchdog ×3, container Up.
- Risk ledgered (16:32Z): crash class needs HOST REBOOT; production fallback if it fires = prefix-caching OFF at 98304 + notify Ryan.

## 3. Disk report (df -h, after the 179.8 GB download landed)
- `/` (sda3, 913 G): 332 G used, 535 G free, 39% — production, Docker root (`/var/lib/docker`), and logs all live here. Healthy.
- `/data` (nvme0n1p1, 234 G): **204 G used, 18 G free, 92% — 7.7% free < 10%**. Docker does NOT use /data. The download consumed 168 G of it.
- Nothing shares /data with production/Docker/logs; nothing deleted (directive: delete nothing).
- If /data free stays under 10%: candidates to free (Ryan's call, none touched): /data/rebuild-install (31 G), /data/stage-v24h2-rollback.tar.gz (6.0 G), /data/hf (6.7 M, trivial). Warning threshold: /data has ~18 G slack over the 110% requirement that gated the download; wheels/pip cache growth eats into that.

## 4. Side-lane profile additions (sidelane-verify.sh, D's harness)
- Gate 4b: prepend-salted ×3 at ~97K on their PC-ON default launch — the exact trigger size on their newer base; fails the run and blocks 250K gates if the engine dies. needle_probe v2.1 unchanged (400 self-correction handles MML).
- Gate 5 (already in D's harness): salted ×3 at ~250K.
- Validated: bash -n clean; census.sh caller tests ALL PASS (file CENSUS=3; no-sudo → CENSUS_UNKNOWN rc3).
- Served model id confirmed from their launcher: `qwen-256k` (start-qwen-256k-vllm.sh, --served-model-name "$ALIAS").

## 5. Fix hunt (deleg_ac7d4ad0, read-only subagent) — IN FLIGHT
- Scope: upstream commits since our base 76cfe1cd touching hybrid/Mamba prefix caching (#53505/#53919/#47861/#53912/#45238 leads), D15 (MikeCaldera mamba_utils.py fix), applicability at base 76cfe1cd, ranked candidate patches, nothing applied.
- Report due next turn.

## 6. Census → riglib (tested function, only census)
- Repo `abf9b40` (pushed): `census()` + `rcount` now `-E` for alternation; tests (local git-bash): 3-match, file-mode, offset, zero-single-fire, CENSUS_UNKNOWN rc3 — ALL PASS.
- Box: riglib.sh updated (atomic rename, md5 387f3a7e… both ends); on-box tests: FILE_MODE=3, BOOT_MODE=9 (independently matches the manual census); one more inline-census failure occurred during testing, confirming the directive.
- census.sh (sidelane) now a thin riglib caller; same output contract (CENSUS=<n> | UNKNOWN rc3).

## 7. C (build) — fired with the new ledger directive
- Directive: when the image is built, ledger the image's compute-runtime (NEO) + Level Zero versions next to their ops-guide values (oneAPI 2026.1.0, their "L0 20.2.0") so any gap from 53.4 traces to runtime vs stack.
- Base-image claim checked: intel/omix:0.4.0-devel has NO torch (A's claim falsified; build installs torch 2.13.0+xpu explicitly).
- Dockerfile staged (md5 f042a8c6 both ends), their verified pins: fork xpu-qwen4exp @ a69fba21, torch==2.13.0+xpu, vxk 0.1.12 GitHub-release wheel, auto_round_lib==0.14.2, numba==0.65.0, setuptools<81 + their b70-worker-affinity.patch + smoke import (Qwen4Exp archs + vxk int4 op) + /opt/stack-versions.txt (NEO/L0 capture).
- Build running (tag es-lane:qwen4exp-a69fba21, watcher proc_1474d58c19f7).
