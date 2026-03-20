# Stage 1 / Single non-GT Audit (2026-03-21)

## Purpose

This note records the main overfitting / self-deception risk found while auditing the
current best `single full-stream non-GT` password-recovery result.

It exists so we do not accidentally treat a promising prototype milestone as a clean
final result.

## Current best reported single non-GT result

Run:
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_v4_gap13/report.json`

Metrics:
- `top1 = 87.50%`
- `top3 = 95.83%`
- `top5 = 95.83%`
- `CER = 12.50%`

Pipeline:
- `full stream`
- `mixed-aware coarse detector`
- `energy + classifier-guided anchor selection`
- fixed-window classifier recovery

## Main audit finding

The main risk is **Stage 1 coarse detector optimism**.

The mixed-aware coarse detector was trained from:
- `/Users/shiyi/备份（mac_vs专用）/data/processed/password_segment_mixed_dataset.npz`

That dataset includes the same new domains we later evaluated on:
- `mixed_single_training`
- `mixed_retry_training`

For the specific single sessions:
- `p01_mixed_single_training_trial000_20260321_014800`
- `p01_mixed_single_training_trial001_20260321_014800`
- `p01_mixed_single_training_trial002_20260321_014800`

Current `session_split(seed=42)` places them as:
- `trial002` -> train
- `trial000`, `trial001` -> val

So the current best `single non-GT` result is **not** a strict “completely unseen single-stream session” evaluation for Stage 1.

## Stronger sanity check

We reran the same downstream `single non-GT` chain, but replaced the coarse detector
with the older detector that was **not** trained on `mixed_single` / `mixed_retry`:

Output:
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_olddet_gap13/report.json`

Result:
- `num_episodes = 0`
- `char_top1 = 0`
- `char_top5 = 0`
- `CER = 1.0`

This confirms that the strongest current single-stream result depends heavily on the
mixed-aware Stage 1 detector.

## Risk-reduced rerun

We then removed the direct Stage 1 exposure and reran the same single non-GT chain
with a stricter coarse detector:

- training dataset:
  `/Users/shiyi/备份（mac_vs专用）/data/processed/password_segment_mixed_nosingle_retry_dataset.npz`
- detector:
  `/Users/shiyi/备份（mac_vs专用）/results/password_segment_mixed_nosingle_retry_detector.pt`

This stricter detector excludes:
- `mixed_single_training`
- `mixed_retry_training`

So the evaluated single sessions are no longer part of the Stage 1 training dataset.

### Important observation

If we keep using the old threshold (`segment_threshold = 0.30`), the stricter detector
looks very weak:

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_gap13/report.json`
- `num_episodes = 2`
- `top1 = 0`
- `top5 = 0.0625`
- `CER = 1.0`

But after a small threshold sweep, the same stricter detector becomes usable again.

Best strict rerun:
- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_sweep_0.10/report.json`

Metrics:
- fixed-window baseline:
  - `top1 = 83.33%`
  - `top3 = 95.83%`
  - `top5 = 95.83%`
  - `CER = 16.67%`
- overlap refine:
  - `top1 = 75.00%`
  - `top3 = 95.83%`
  - `top5 = 100.0%`
  - `CER = 25.00%`

### Interpretation

This changes the audit conclusion in an important way:

1. The previous `95.83% / 12.50% CER` result was still optimistic because Stage 1 had
   seen the same new domain and even some of the same new sessions.
2. However, **once we fully remove `mixed_single` / `mixed_retry` from Stage 1 training
   and retune the detector threshold**, the main single non-GT chain is still strong.
3. So the problem is **not** that the whole single non-GT story disappears without leakage.
   The more accurate conclusion is:

> the old best result was too optimistic,
> but the underlying full-stream single-password pipeline remains genuinely promising
> even under a stricter Stage 1 protocol.

## What still looks real

Even with this caveat, the following conclusions still look credible:

1. `stage2_episode` is not the best current anchor source for single full-stream.
2. `coarse region -> energy peaks -> classifier-guided subset selection` is the right current structure.
3. Removing the hard-coded `200Hz` assumption is safe.
4. Automatic key-count inference is still not solved.
5. Retry / multi-password remains bottlenecked by second-password episode detection.

## Honest interpretation

The current best single-stream result should be treated as:
- a **prototype milestone**
- a **strong positive direction signal**

It should **not yet** be treated as:
- a clean final claim of full-stream generalization
- a publication-ready “arbitrary unseen data stream” result

## What would make this clean

At least one of the following is needed:

1. Evaluate on new single / retry sessions that were **not included** in the mixed-aware Stage 1 training dataset.
2. Rebuild Stage 1 with a stricter held-out protocol, then rerun the whole single non-GT chain.
