# 2026-09-22 — MTP1 output corruption (temperature-0 differential)

**Artifacts:** ledger B5-DISCRIMINATOR; temp-0 A/B request transcripts (box `.run/`).

## Test

Same prompt, temperature 0 (fully deterministic), MTP1 vs MTP0 boots of the same image:

- **MTP1 boot:** corrupted/garbled output on otherwise-identical requests.
- **MTP0 boot:** clean output, same prompts, same seeds.

At temperature 0 a token-level difference is engine-caused by construction — no sampling explanation exists. The corruption reproduces across requests on the MTP1 boot and disappears on the MTP0 boot.

## Verdict

**MTP speculative decode (k=1) produces corrupted output on this stack (v24h2 image + XPU backend). The validated production line is MTP0.** The launcher refuses `MTP_NUM_SPECULATIVE_TOKENS != 0` (hard gate) — override attempts die at knob validation, not at first garbled answer.

Not root-caused upstream (draft/target KV interplay suspected; no upstream fix tracked). Re-testing requires a new image AND the temp-0 discriminator — do not re-enable on this image.
