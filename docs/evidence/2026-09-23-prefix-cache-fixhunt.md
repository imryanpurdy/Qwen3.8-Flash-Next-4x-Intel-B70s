# Hybrid-Mamba Prefix-Cache Crash — Candidate Fix Report (read-only)

Scope: hunt upstream vLLM commits (base `76cfe1cd`, 2026-08-26) and MikeCaldera poster fixes for the
Qwen4Exp (GDN/Mamba + QSA + MoE) prefix-cache crash on 4x Intel Arc Pro B70 (GuC engine-reset storm,
3/3 with PC-ON, 3/3 clean PC-OFF). Nothing applied; nothing modified.

Verified base facts (raw.githubusercontent.com/vllm-project/vllm/76cfe1cd/...):
- All relevant cache-surface files exist at base: `vllm/v1/worker/mamba_utils.py` (66 KB),
  `vllm/v1/core/single_type_kv_cache_manager.py` (84 KB, has `class MambaManager`),
  `vllm/v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py` (has `MambaSpec`, `eagle_group_ids`,
  `drop_eagle_block = use_eagle and idx not in eagle_verified`), `kv_cache_utils.py`,
  `vllm/v1/kv_cache_interface.py`, `vllm/v1/worker/gpu_model_runner.py` (V1 race symbols present:
  `num_accepted_tokens_cpu_tensor`, `prev_positions`, `num_accepted_tokens_event`), and BOTH V1 and V2
  runner trees exist at base (V2 default is auto/opt-in; `use_v2_model_runner` auto logic in
  vllm/config/vllm.py).
- `MambaManager.find_longest_cache_hit` is **byte-identical base → HEAD (c4f6ce4)** — the upstream
  `drop_eagle_block` hole (poster's vllm#48375/#53912) is still unfixed in master today.
- Kernel bodies `_copy_mamba_state_block`, `preprocess_mamba_align_fused_kernel`,
  `postprocess_mamba_fused_kernel` are **identical base → HEAD** (only comments/types changed) — the
  upstream #53505 backward-copy hole is still unfixed in master today.
- `prefix_cacheable` on `KVCacheSpec` does NOT exist at base; it exists at HEAD (interface, 12 uses;
  `kv_cache_utils.py` 2 uses). Upstream PR #53896 introduced it (CircularBufferSpec → False).

Our fork tree (base + 16-patch overlay) state — from the local patch series
`C:/Users/imrya/flashnext-recipe/files/overlay/vllm/`:
- `0001` (merge of PR head `02f2b4c15dd987d9436e125aab29604447c77405` into base) is the ONLY patch
  touching the cache surface. It brings `CircularBufferSpec` + `CircularBufferManager` (ring group:
  `find_longest_cache_hit` returns empty, `cache_blocks` = no-op → **ring group already excluded from
  local prefix caching**) and the coordinator `prefix_cacheable` filters (group_block_sizes,
  unsupported_partial_hit_managers, attention_groups) + `resolve_kv_cache_block_sizes` hashing_sizes.
- `0001` does **NOT** touch `vllm/v1/core/kv_cache_manager.py` → our manager file is pristine base.
  Upstream's final #53896 merge adds to it the CSA-ring skip in `truncate_computed_blocks`
  (`if not group.kv_cache_spec.prefix_cacheable: assert not group_blocks; truncated.append([]);
  continue`) and the boundary-state offload rename — we have the older `_partial_tail_pins` design.
- Our tree has NO backward-copy guard (`src_col > dst_col` = 0 occurrences in 0001) and no
  `_num_retired_blocks`/`_checkpoints`/`use_eagle_block_drop` machinery (all added post-base).
- Patches 0009/0014 (QSA cache API) touch only `vllm/models/qwen4_exp/common/qsa_cache.py` + tests —
  no cache-manager or prefix-cache surface.

---

## (a) Candidates table

| id | source (commit / PR) | files touched | exists at base? | crash-fit | risk to apply |
|----|----------------------|---------------|-----------------|-----------|----------------|
| U1 | `fadfe1c7d4df` — [Bugfix][Core] Retire Mamba states across null gaps (#55450), 2026-09-11 — https://github.com/vllm-project/vllm/commit/fadfe1c7d4df | `vllm/v1/core/single_type_kv_cache_manager.py` | Y (MambaManager yes; override/`_num_retired_blocks` absent) | **HIGH** — align-mode `_remove_blocks_in_range` (generic impl breaks at first null block; no per-req retired counter) mis-accounts state blocks freed/skipped under prefix caching, so reused blocks can serve stale/un-retired state or double-free; directly on the PC skipped-block path | low (self-contained manager change; ~1 context line `_checkpoints.pop(...)` must be dropped — no checkpoints at base) |
| U2 | `263c4ff95fad` — Cache Mamba state at block-grid position of EAGLE resume (#53945), 2026-09-08 + rename follow-up `49ee12d742b4` (#57382) | config/cache.py, engine/arg_utils.py, kv_cache_manager.py, kv_cache_coordinator.py, single_type manager, sched/scheduler.py | Y (all files; but `use_eagle_block_drop`, `shared_prefix_boundary`, `_checkpoints` absent at base) | MED — opt-in flag `--enable-mamba-fine-grained-prefix-cache` (default False), requires EAGLE on mamba group + partial-hash units; **dormant with MTP=0/no EAGLE** | med/high (wide signature change: adds keyword arg `replay_boundary` to `cache_blocks` across manager classes + coordinator + scheduler) |
| U3 | `f547c23ec971` — Keep transient checkpoints out of prefix-cache eviction (#56794), 2026-09-15 | single_type manager | Y | MED (same checkpoint machinery, EAGLE-resume spec-decode; dormant MTP=0) | low–med (depends on U2-era `_checkpoints`; needs porting order) |
| U4 | `58d45fd767ff` — MooncakeStore: exclude non-prefix-cacheable (QSA ring) groups; fix align mode (#55027), 2026-09-14 | kv_cache_manager.py, kv_cache_interface.py, mooncake connector files | Y (interface Y; `prefix_cacheable` property at HEAD only, absent at base) | MED-LOW for us — connector-only change; our tree excludes the ring locally already (CircularBufferManager no-op); relevant as evidence that QSA-ring groups must never be hash-served, and our `truncate_computed_blocks` lacks the CSA-ring skip upstream ended up with | low |
| U5 | `91752b7a3e0c` — Fix Kimi-K3 RecoverSSM startup failure (#54634), 2026-08-31 | `vllm/v1/worker/mamba_utils.py` (`validate_mamba_state_copy_funcs`: `==` → `0 < len <= len(shapes)`) | Y (function present) | LOW — startup validation only; our engine starts (crash is runtime), so mismatch not hit; harmless hardening | negligible |
| U6 | `481839ad9e5e` — Disable trailing prefix-cache block dropping (#53388), 2026-09-01 | kv_cache_utils.py, sched/scheduler.py, single_type manager, spec config, offload manager | Y | LOW-MED — spec-code feature flag; changes trailing-block retention generally | low |
| U7 | `93ab92be0cde` (#52047) + `0bce411a073e` (#55390) — Annotate draft KV cache groups positionally on hybrid grouping path | `kv_cache_utils.py` | Y | **LOW / dormant** — MTP/EAGLE draft-group annotation; sender-received fixes for hybrid draft annotation; MTP=0 (upstream quoted it as the perf defect "0% mamba prefix reuse" when draft groups can't be identified) | low |
| U8 | `144e79c8106d` — Kimi K3 internal prefix checkpoints + partial prefix caching + spec decode (#53614), 2026-09-06 | block_pool.py, kv_cache_coordinator.py, sched/scheduler.py, single_type manager, kv_cache_interface.py | Y | MED (introduced the `_checkpoints`/`use_eagle_block_drop` machinery; spec-decode-flavored) | med |
| U9 | `f30a195bbb15` — Incorrect Mamba block allocation estimate prevents admission (#57050), 2026-09-16 | single_type manager | Y | LOW (admission estimation only) | low |
| U10 | `b28c3e1568bf` (#547xx, retain both replay boundaries) / MRV2+MTP fixes (`85c1f58d50b8`), MRV2 DBO (`3f41d102c5b8`), DeepSeek-V4.1 (#56227) | coordinator/manager/model_runner | Y | LOW / model-specific or V2-only | varies |
| U11 | `e126687a9a82` — final upstream merge of [Model] Support Qwen3.8-Flash-Next (#53896), 2026-08-31 — deltas vs OUR merged PR head `02f2b4c15d` | kv_cache_manager.py (CSA-ring skip in `truncate_computed_blocks`, boundary-offload rename), plus final versions of mamba_utils/mamba_hybrid (dict-keyed `MambaStateCopyFuncsByType`) | N for kv_cache_manager.py (0001 never touches it — file = pristine base) | MED — our overlay is an EARLIER PR head (pre review); upstream-final QA/state-handling deltas missing; likely interplay fixes for the same surface | med (re-targeting overlay 0001 = invasive) |
| P1 | Poster `patches/patch_fix_backward_copy.py` — guard against BACKWARD Mamba state copies (umbrella vllm#53505; issue https://github.com/vllm-project/vllm/issues/53505; NOT fixed upstream — kernel bodies identical base→HEAD) | `vllm/v1/worker/mamba_utils.py` (3 sites: `postprocess_mamba_fused_kernel` l.478, `precopy_mamba_align_fused_kernel` l.609, `collect_mamba_copy_meta` l.1344 of the poster's tree) | Y — every anchor present at our base (verified) | **HIGH** — the only existing guard for `dest < src` state copies; a backward copy writes a LATER position's state into an EARLIER block column that (1) in-flight decode reads as initial state and (2) the prefix cache has already published; poisoned page is served to every later request sharing the prefix — exact PC-ON signature; XPU then escalates (silent corruption → NaN → GuC reset storm) | low (3 strictly-greater guards, fail-closed marker + env gate; text-only) |
| P2 | Poster `patches/patch_fix_accepted_sync.py` — port of OPEN upstream PR https://github.com/vllm-project/vllm/pull/53919 (await accepted-token copy before moving batch rows) | `vllm/v1/worker/gpu_model_runner.py` (V1 runner only) | Y — all three race symbols at base | MED — silent-corruption race (step-N D2H row order vs step N+1 `_update_states`/`condense()` permutation, `prev_positions` double-gather); **requires async scheduling; spec-decode-oriented (num_accepted_tokens); V1 only; V2 keeps counters GPU-resident (race impossible by design)** | low (2 hunks; ports cleanly to base; env-gated) |
| P3 | Poster `patches/patch_fix_eagle_drop.py` — vllm#48375/#53912/#43559/#43650: `MambaManager.find_longest_cache_hit` ignores `drop_eagle_block` | `vllm/v1/core/single_type_kv_cache_manager.py` | Y (function identical base→HEAD; still unfixed upstream) | MED — stale recurrent-state reachability after EAGLE/MTP rejection; **dormant with MTP=0 (drop_eagle_block = use_eagle and …; no draft → False)**; but if production ever re-enables MTP with PC, mandatory | low (lowers coarse-branch ceiling by 1 block / fine branch by B70_EAGLE_DROP_FINE_UNITS) |

Note: no literal "D15" marker exists anywhere in `MikeCaldera/intel-arc-pro-b70-qwen38-vllm`
(all 136 text blobs grepped; `D15` = 0 hits). The task's "D15 mamba_utils.py prefix-caching fix"
matches **P1 (`patch_fix_backward_copy.py`)**, whose REL path is exactly `v1/worker/mamba_utils.py`
(P2 targets `gpu_model_runner.py`, P3 targets `single_type_kv_cache_manager.py`). All three live in
`patches/` of https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm; their
`docs/RELIABILITY-REPORT.md` calls the umbrella "Correctness gap — hybrid MTP + prefix caching"
and gives A/B evidence "0/288 vs 16/288" (PC-off vs PC-on corruption) plus mitigation
`--no-async-scheduling` suffices upstream.

## (b) Top candidates — concrete patch content

### P1 (try first): backward-copy guards (poster patch_fix_backward_copy.py, vllm#53505)

Three identical guards, one per copy site of `vllm/v1/worker/mamba_utils.py` (all anchors verified at
our base 76cfe1cd; kernel bodies unchanged upstream even at HEAD c4f6ce4):

```python
# SITE 1 — inside/jafter postprocess_mamba_fused_kernel (poster l.478; base file has the
# self-copy guard "if src_block_idx == dest_block_idx and accept_token_bias == 0:" twice: postprocess + collect_mamba_copy_meta)
    # B70_BACKWARD_COPY_GUARD_POSTPROCESS (vllm#53505): a copy with
    # dest_block_idx < src_block_idx would overwrite an earlier column (the in-flight
    # initial-state slot and/or a page already published in the prefix cache) with state
    # from a later position. Only src == dest (intra-block slide) is legitimate besides
    # the normal forward advance src < dest.
    if src_block_idx > dest_block_idx:
        return

    # Skip no-op self-copy.
    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return

# SITE 2 — precopy_mamba_align_fused_kernel (poster l.609; base guard: "if src_col < 0 or src_col == dst_col: return")
    # B70_BACKWARD_COPY_GUARD_PRECOPY (vllm#53505): original guard only rejects the brand-new
    # state (src_col < 0) and no-op (src_col == dst_col); a source column BEHIND the
    # destination (num_computed retracted after retract/preemption/resume) would copy
    # backward and corrupt an earlier, possibly already-cached, boundary's state.
    if src_col > dst_col:
        return

    if src_col < 0 or src_col == dst_col:
        return

# SITE 3 — collect_mamba_copy_meta (host path, poster l.1344)
    forward_context: dict[str, Any],
) -> None:
    # B70_BACKWARD_COPY_GUARD_COPY_META (vllm#53505): same criterion as in the kernels;
    # cleanup_mamba_state_idx documents that a stale entry may point past the new
    # (smaller) allocation, i.e. src > dest.
    if src_block_idx > dest_block_idx:
        return

    if src_block_idx == dest_block_idx and accept_token_bias == 0:
        return
```

The poster script also ships fail-closed AST/pattern invariants (exactly 2 `_copy_mamba_state_block(`
calls, 2 self-copy guards, 1 precopy guard, 1 host memcpy dest line) so a NEW upstream copy site can't
slip past the guards, plus env gates (`B70_FIX_BACKWARD_COPY=0` A/B, `B70_VLLM_ROOTS`). Porting the
3 insertions alone is enough; the script lives at
https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm/blob/HEAD/patches/patch_fix_backward_copy.py.

### U1 (try alongside P1): #55450 retire Mamba states across null gaps — full core hunk

`vllm/v1/core/single_type_kv_cache_manager.py` (only non-test hunk; patch:
https://github.com/vllm-project/vllm/commit/fadfe1c7d4df.patch):

```diff
@@ __init__ (MambaManager) @@
             self.last_state_block_idx: dict[str, int] = {}
+            self._num_retired_blocks: dict[str, int] = {}
...
+    def _remove_blocks_in_range(
+        self, request_id: str, first_block: int, last_block: int
+    ) -> None:
+        if self.mamba_cache_mode != "align":
+            return super()._remove_blocks_in_range(request_id, first_block, last_block)
+        blocks = self.req_to_blocks.get(request_id, [])
+        first_block = max(first_block, self._num_retired_blocks.get(request_id, 0))
+        last_block = min(last_block, len(blocks))
+        if first_block >= last_block:
+            return
+        freed: list[KVCacheBlock] = []
+        # Mamba prefill leaves null gaps between states awaiting retirement.
+        for i in range(last_block - 1, first_block - 1, -1):
+            if blocks[i].is_null:
+                continue
+            freed.append(blocks[i])
+            blocks[i] = self._null_block
+        if freed:
+            self.block_pool.free_blocks(freed)
+        self._num_retired_blocks[request_id] = last_block
...
@@ pop_blocks_for_free @@
         if self.mamba_cache_mode == "align":
             self._allocated_block_reqs.discard(request_id)
             self.last_state_block_idx.pop(request_id, None)
+            self._num_retired_blocks.pop(request_id, None)
             self._checkpoints.pop(request_id, None)      # ← absent at our base: DROP this context line
             self._producer_partial_tail_reqs.pop(request_id, None)
```

Port adaptation: our base MambaManager has `remove_skipped_blocks` (calls generic
`_remove_blocks_in_range`, which `break`s at the first null from the right — the misaccounting) and
`pop_blocks_for_free`, but no `_checkpoints` and no `is_null`-typed block accessor (`blocks[i].is_null`
→ in our tree the generic impl compares `blocks[i] == self._null_block`; #55450 uses `.is_null` on the
block object — check `KVCacheBlock.is_null` at base: present (5 hits) — OK). The `_checkpoints` context
line must be omitted. Straightforward hand-port (~25 lines).

### U2/U3 (only if MTP/EAGLE returns): #53945 family

State at block-grid position + fine-grained flag (config/cache.py new default-False field
`enable_mamba_fine_grained_prefix_cache`, arg, `replay_boundary` threaded through coordinator →
manager `cache_blocks(..., *, replay_boundary)`), and #56794 keeps transient checkpoints out of
eviction. All gated: `fine_grained_prefix_cache = flag and eagle_group_ids and
enable_partial_hash_hits and num_reprefillable_tokens == 0` — dormant with MTP=0. See
https://github.com/vllm-project/vllm/commit/263c4ff95fad.patch (49 KB) and
https://github.com/vllm-project/vllm/commit/f547c23ec971.patch.

## (c) Bottom line

1. **Yes — our crash is consistent with these being missing-fix situations.** Base
   `76cfe1cd` (2026-08-26) predates ALL of them (Sept 2026), and three of the relevant defects
   (#53505 backward copy, #48375/#53912 drop_eagle_block, #53919 accepted-sync) were **never merged
   upstream** — the poster patched them itself; master still has the identical kernels and logger for
   `find_longest_cache_hit` today. PC-ON vs PC-OFF clean/dirty split points squarely at block-reuse
   state handling on the hybrid (align-mode) cache: every candidate above is exactly that surface,
   and none is present in base or in our 16-patch overlay (verified: no guard code in 0001; 0009/0014
   only touch the model-side QSA cache).
2. **Try first (cheapest, highest fit, no draft needed):**
   - **P1** backward-copy guard (3 insertions, text-only, env-gated A/B) — also do a one-off
     PC-ON A/B with `B70_FIX_BACKWARD_COPY=1`.
   - **U1** #55450 retire-states-across-null-gaps (one-method hand-port; drop the `_checkpoints` line).
   - Cheap diagnostic before patching: run the same 3x prefill with PC-ON **plus
     `--no-async-scheduling`** (poster's own upstream-verified mitigation) to test the #53919 race
     participation; and confirm production really runs `mamba_cache_mode="align"` (base default is
     "none"; our 0001 does NOT force align on prefix caching, while the poster's builds do — if
     production runs PC+hybrid in "none", the state never checkpointed-then-served issue worsens).
3. **MTP-only / dormant for MTP=0** (note but don't block the crash): U2 (#53945 + #57382 rename),
   U3 (#56794), P3 (#48375/#53912 — required the moment MTP is re-enabled with PC), P2 (#53919 —
   V1+async+spec only; V2 runner race-impossible), U7 (#52047/#55390 draft-group annotation), U8
   (#53614), U6 (#53388), U9, U10.
4. **Our-tree divergence worth one look:** 0001 merges PR head `02f2b4c15d` (pre-review); upstream's
   final #53896 (`e126687a9a82`) adds the CSA-ring skip in `kv_cache_manager.truncate_computed_blocks`
   + renames to boundary-state offloads — the file is pristine base in our tree. If any external
   computed-blocks/connector path is ever used, backport the skip
   (https://github.com/vllm-project/vllm/commit/e126687a9a82.patch).

## Sources
- Base/HEAD trees + per-path commit lists: GitHub API
  (api.github.com/repos/vllm-project/vllm/compare/76cfe1cd...c4f6ce4 — 1389 commits ahead;
  commits?path= for each file, since=2026-08-26T21:08:02Z).
- Commits: fadfe1c7d4df (#55450), 263c4ff95fad (#53945), 49ee12d742b4 (#57382),
  f547c23ec971 (#56794), 58d45fd767ff (#55027), 91752b7a3e0c (#54634), 481839ad9e5e (#53388),
  93ab92be0cde (#52047), 0bce411a073e (#55390), 144e79c8106d (#53614), f30a195bbb15 (#57050),
  b28c3e1568bf, e126687a9a82 (#53896), 3f41d102c5b8 (#50945), 85c1f58d50b8, 1dc2d854c120 (#56227).
- Issues/PRs: #53505 (closed), #53912 (open), #45238 (open), #48375/#43650/#43559,
  PR #53919 (open, not merged). Poster patches + docs:
  https://github.com/MikeCaldera/intel-arc-pro-b70-qwen38-vllm
  (`patches/patch_fix_backward_copy.py`, `patch_fix_accepted_sync.py`, `patch_fix_eagle_drop.py`,
  `docs/RELIABILITY-REPORT.md`).
- Local: C:/Users/imrya/flashnext-recipe/docs/evidence/laneB_flashnext_scaffold.md,
  docs/rig-skill-snapshot-20260920.md, files/overlay/vllm/*.patch (read-only).

Scratch downloads kept under C:/Users/imrya/fixhunt_cache/ (not a repo).
