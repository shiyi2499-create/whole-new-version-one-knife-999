# Beam Sweep And Hard-Negative Follow-Up (2026-03-25)

## Scope
This note records two follow-up experiments on top of the current best automatic pipeline:

- Stage1 dense-labeling + complete-hit-first
- mixed-adapted Stage3
- peak-keyness anchors (`threshold=0.7`)
- re-adapted overlap checkpoint

Reference best auto result before these follow-ups:
- result dir: `results/stage123_end_to_end_strongstage2_fair6_mixedadapt_keyness_newoverlap_20260325`
- `stage1_bestpred_overlap.char_top1 = 0.6180`
- `stage1_bestpred_overlap.cer = 0.3371`
- `stage1_bestpred_overlap.sequence_top100_hit = 0.30`

## 1. Beam Width Sweep
Updated script:
- `onset_detection/stage2_segmental/scripts/eval_stage123_end_to_end_strongstage2.py`

New support:
- `--sequence_hit_cutoff`
- dynamic report keys such as `sequence_top500_hit`, `sequence_top1000_hit`
- stores enough `top_sequence_candidates` for wider-beam diagnostics

### Fair6 results
Result dirs:
- `results/stage123_end_to_end_strongstage2_fair6_mixedadapt_keyness_newoverlap_beam100_20260325`
- `results/stage123_end_to_end_strongstage2_fair6_mixedadapt_keyness_newoverlap_beam500_20260325`
- `results/stage123_end_to_end_strongstage2_fair6_mixedadapt_keyness_newoverlap_beam1000_20260325`

For `stage1_bestpred_overlap`:
- beam 100: `top1=0.6180`, `CER=0.3371`, `sequence_top100_hit=0.30`
- beam 500: `top1=0.6180`, `CER=0.3371`, `sequence_top500_hit=0.60`
- beam 1000: `top1=0.6180`, `CER=0.3371`, `sequence_top1000_hit=0.60`

For `gt_keyframes_fixed`:
- beam 100: `sequence_top100_hit=0.50`
- beam 500: `sequence_top500_hit=0.80`
- beam 1000: `sequence_top1000_hit=0.90`

### Interpretation
- Wider beam substantially improves **sequence candidate coverage**.
- On the automatic closed loop, the correct full string is often just outside top100 but usually not beyond top500.
- However, wider beam alone does **not** improve `top1` or `CER`; this confirms that search width is a useful diagnostic, not a direct fix by itself.
- For the current automatic path, going from 500 to 1000 does not buy extra sequence-hit coverage (`0.60 -> 0.60`).

## 2. Stage3 Hard-Negative Oversampling (B2)
Updated script:
- `adapt_password_mixed_inception.py`

New options:
- `--hard-char-group`
- `--hard-oversample-factor`

Implementation used here:
- start from already mixed-adapted Stage3 checkpoint
- oversample only **mixed-train** windows for confusion-prone characters
- no model architecture changes, no loss changes

Run configuration:
- factor: `3`
- groups:
  - `xcz`
  - `sa`
  - `poi`
  - `09`
  - `32`
  - `kl`
  - `1q`
  - `er`
  - `56`

Artifacts:
- stage3 report: `results/password_len8_len9_len10_mixedadapt_hardos3_20260325.json`
- checkpoint: `results/inception_password_len8_len9_len10_mixedadapt_hardos3_20260325.pt`
- scaler: `results/inception_password_len8_len9_len10_mixedadapt_hardos3_20260325_scaler.npz`
- fair6 eval: `results/stage123_end_to_end_strongstage2_fair6_mixedadapt_hardos3_keyness_newoverlap_20260325`

### Mixed holdout (Stage3 direct) impact
Compared with the previous mixed-adapted Stage3:
- `gt_keyframes_fixed.char_top1: 0.7416 -> 0.7303`
- `CER: 0.2584 -> 0.2697`

So on the isolated Stage3 holdout metric, this oversampling pass is slightly worse.

### Fair6 closed-loop impact
For `stage1_bestpred_overlap`:
- previous best: `top1=0.6180`, `CER=0.3371`
- hardos3: `top1=0.6404`, `CER=0.3034`

This is a real automatic closed-loop gain.

### Episode-level changes (best auto overlap)
Improved:
- `0xc8pugot`: `0.6667 -> 0.7778`, CER `0.3333 -> 0.2222`
- `b15bp8ws`: `0.7500 -> 0.8750`, CER `0.2500 -> 0.1250`
- `npd33wdez`: `0.4444 -> 0.5556`, CER `0.5556 -> 0.4444`
- `xsowvo7qh`: top1 unchanged at `0.0`, but CER `0.5556 -> 0.4444`

Worse:
- `5ftt5brux9`: `0.8000 -> 0.7000`, CER `0.2000 -> 0.3000`

Unchanged or mixed:
- `1kfxksa8`, `ijtplv3am8`, `r8s2yyl10`
- `exjdc0q95` remains difficult
- `kodtpoxk` unchanged in aggregate but final pair still confuses near-neighbor keys

### Interpretation
- The simple B2 oversampling is **not** a clean win at isolated Stage3 level.
- But it **does** help the end-to-end automatic pipeline, likely because it improves some hard confusion cases that matter disproportionately once Stage1/Stage2 noise is present.
- This is a useful signal that closed-loop value and isolated Stage3 value are not perfectly aligned.

## Current takeaways
1. `beam width` is partly limiting sequence generation, but not enough to improve top1 without a better reranking or stronger per-position logits.
2. A simple neighbor-key oversampling pass can improve the closed loop even if isolated Stage3 top1 dips slightly.
3. The most valuable remaining work is still around residual Stage3 confusion handling, but it should be evaluated in the **closed loop**, not only in `gt_keyframes_fixed` isolation.
