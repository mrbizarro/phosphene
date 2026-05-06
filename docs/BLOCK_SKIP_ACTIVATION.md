# Block-Skip Caching — activation guide

## What it does

Adds DeepCache-style block-output caching to the joint-AV DiT. On scheduled
"skip" steps the middle 24 of 48 transformer blocks are bypassed, reusing the
residuals captured on the previous "compute" step. Compounds with TeaCache /
Turbo (which already skips full denoise steps) for additional wall-time
savings.

See [PERF_PLAN_2026-05-05.md](PERF_PLAN_2026-05-05.md) for full design + benchmarks.

## How to enable

Set environment variables BEFORE launching the panel or helper:

```bash
export LTX_BLOCK_SKIP=1                  # main on/off
export LTX_BLOCK_SKIP_SCHEDULE=3of5      # alternate | 3of5 | aggressive (default 3of5)
export LTX_BLOCK_SKIP_EDGE=12            # blocks at each end always run (default 12)
export LTX_BLOCK_SKIP_VERBOSE=0          # 1 to log per-step decisions

./run_panel.sh                           # or python mlx_ltx_panel.py
```

## How to verify it's active

When the helper subprocess starts, it logs to stderr (visible in panel logs):

```
[block-skip] enabled  schedule=3of5  edge=12
```

If it says `[block-skip] not loaded: ...` the patch couldn't import; check
`patch_block_skip.py` lives at the repo root.

## How to turn it off

```bash
unset LTX_BLOCK_SKIP
# OR
export LTX_BLOCK_SKIP=0
```

Restart panel/helper. Patch goes back to full-fidelity (no schedule, no skip).

## Schedules — what they do

In a 5-Turbo-full-step render:

| Schedule | Pattern (full=F, skip=S) | Compute steps | Block forwards | Speedup on denoise |
|---|---|---|---|---|
| `alternate` | F S F S F | 3 of 5 | 192 | 1.25× |
| `3of5` (default) | F S S F S | 2 of 5 | 168 | 1.43× |
| `aggressive` | F S S S S | 1 of 5 | 144 | 1.67× |

`3of5` is the recommended default — measured best speed-vs-quality tradeoff.

## Edge — preserve early & late blocks

`edge=12` means blocks `[0..11]` and `[36..47]` ALWAYS run (early features and
final fine-detail blocks). Only the middle blocks `[12..35]` are skip-eligible.

- Larger edge (e.g., 16) — more conservative, fewer skips, less speedup, higher
  quality preservation.
- Smaller edge (e.g., 8) — more aggressive, more skips, more speedup, more risk
  of detail loss.

## Caveats / known limits

- **First step is always compute** even if scheduled to skip — there's nothing
  to replay yet.
- **CFG / STG passes (uncond, ptb, mod) each have their own schedule cycle** —
  managed automatically by the controller.
- **TeaCache full-step skips win priority** — when TeaCache says "skip whole
  step" via `block_stack_override`, the patch falls through to the original
  path. Block-skip and TeaCache compose multiplicatively on the steps that
  TeaCache keeps.
- **Per-job state reset** — the controller is reset at the start of every
  helper job (`generate`, `extend`, `generate_hq`, `generate_keyframe`,
  `enhance_prompt`).

## Where to look if it breaks

- Patch source: [`patch_block_skip.py`](../patch_block_skip.py) at repo root.
- Helper integration: [`mlx_warm_helper.py`](../mlx_warm_helper.py) lines ~54-79
  (auto-enable + reset).
- Sampler injection: [`utils/samplers.py`](../ltx-2-mlx/packages/ltx-pipelines-mlx/src/ltx_pipelines_mlx/utils/samplers.py)
  function `_block_skip_kwargs` and call sites in `_run_pass` / `_stage_run_pass`.
- Smoke tests: `/tmp/phosphene_bench/test_block_skip_patch.py` and `test_patched_call.py`.
