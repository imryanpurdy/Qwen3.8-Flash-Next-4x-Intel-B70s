# 2026-09-22 — PLE staging window 2 s → 120 s (the t120 image)

**Artifacts:** in-image `vllm/v1/ple_offload/connector.py` md5 `61d64725…`; repo `files/overlay/patches/t120-connector-staging-120s.patch`; image `stage-v24h2:rollback-t120` ID `0ca8598598ba…`.

## The defect

The PLE connector's first-request staging had a hard 2-second deadline. On the 98K-needle probe and other cold-table paths (first tool-call request after boot; PLE table page-in), host staging of the n-gram table exceeded 2 s → `TimeoutError: staging did not complete in 2.0 seconds` → request 500s. It is a **first-request** defect: cold PLE page-in is slow exactly once; every subsequent request hits warm tables.

## The fix (one line)

`files/overlay/patches/t120-connector-staging-120s.patch` — connector.py staging timeout `+ 2.0` → `+ 120.0`. Chosen as an image-level patch (t120 tag) because the connector runs inside the engine process; env-only overrides don't touch it.

```diff
-        deadline = time.monotonic() + 2.0
+        deadline = time.monotonic() + 120.0
```

## Verification

- Extracted in-image connector md5 `61d64725…` matches the B4T-IMAGE ledger entry; the one-line diff is the only change vs the pre-patch file (`87b25f2d…`, repo `files/overlay/conn-v24h2.py`).
- 98K needle post-t120: `CORRECT=YES`, `STAGING_NEW=0` (zero staging timeouts at 98,288 tokens).
- Warm-up request after boot remains in the launcher (PLE cold page-in); with t120 the first request no longer tears/garbles if the warm-up is missed.

**Do not reduce below 120 s** without re-running the 98K needle: the page-in wall scales with the table, not the request.
