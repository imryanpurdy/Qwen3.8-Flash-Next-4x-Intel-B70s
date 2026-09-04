# files/overlay — VENDORED overlay artifacts (deploy-kit ships them, 2026-09-04)

This directory is where the Docker-first build consumes the lab's overlay
artifacts. **As of this kit's authorship the artifacts are lab staging, not
vendored** — the repo ships the *contract* (Dockerfile expects them), and the
build fails fast with a clear message while they are missing. Nothing here may
be silently skipped.

## Expected layout (the DOCKERFILE reads exactly this)

```
files/overlay/vllm/<NUM>-<name>.patch     # vLLM production patch series
files/overlay/qsa/qsa_ops.py              # merged QSA Triton kernel extract
```

### vLLM overlay patches (16 production patches)

- The lab's vLLM source overlay = base `76cfe1cd88` + the **production series
  `0001`–`0010`, `0012`, `0014`–`0018`** → head
  `1372c62d975c554f4b465c8299bc5f3295301ceb` (tree `31ebb778…`).
- `0011` and `0013` are **opt-in diagnostics — NEVER on a serving/timing tree**.
- `0019`–`0033` are post-series candidates — NOT in the production series.
  (Docs: `docs/lanes/00-shared-context.md` §Runtime identity,
  `docs/evidence/laneB_flashnext_scaffold.md` §2B.)
- **Source of the artifacts (lab, read-only on the rig):**
  `/tmp/b70lab/patches/qwen38-flash-next-fp8-b70/vllm/` (0001–0033).

### QSA Triton kernels

- `qsa_ops.py` = the merged QSA Triton kernel extract from vLLM PR #53896
  (weight-free QSA path; 128-MiB chunked logits workspace; narrow-tiles-favor-
  decode note). Docs: `docs/lanes/00-shared-context.md` (evidence map).
- **Source:** `/tmp/qsa_ops.py` (same bytes, sha-pinned at the rig).
- **Destination in the overlay tree is an ASSUMPTION:** the Dockerfile places
  it at `vllm/model_executor/layers/qwen_sparse/qsa_ops.py` (configurable via
  `ARG QSA_OPS_DEST`). The exact in-tree path is not named in the lane specs —
  confirm against the merged PR at first rig staging. (flagged in summary)

## Vendoring plan

Copy the 16 production patches (described above) in as
`files/overlay/vllm/0001-<name>.patch` … `0018-<name>.patch`, and the kernel
module as `files/overlay/qsa/qsa_ops.py`, then commit them as part of the
deploy-kit push. Until then the build message points here.

## Build-time behaviour (fail-fast, no silent no-op)

`./build-image.sh` refuses to run when:
- any production patch (0001–0010, 0012, 0014–0018) is missing, or
- `files/overlay/qsa/qsa_ops.py` is missing/empty.

The Dockerfile additionally:
- fails if any staged patch does not apply on base `76cfe1cd88` (patch-apply
  gate), and prints the base/HEAD identity receipt;
- fails if the runtime-stage placeholders (`RUNTIME_STAGE_URL`,
  `RUNTIME_STAGE_SHA256`) are unfilled — the stage tar must be sha256-verified
  (prefix `6bf1b547…` is all the lane specs freeze).

## Runtime-stage (XPU kernels) — separate download, not vendored here

The kernel binaries are NOT overlay patches: the certified rebuild series
reconstructs tree `d8c4318a…` → loaded stage
`2f829747503c77d4814834dffd0840fb1dd9f75a` (base `0fd18a7c`), packaged as an
18-file hybrid runtime tar and hosted publicly in the
`steveseguin/b70-optimization-lab` prerelease
`qwen38-flash-next-runtime-2f829747-20260827`. The Dockerfile downloads and
sha256-verifies that tar. `ad25aa9f…` is source, NOT the loaded stage — never
substitute.
