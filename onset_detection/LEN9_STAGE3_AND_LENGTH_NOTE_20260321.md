# LEN9 Stage3 And Length Note (2026-03-21)

## Data quality

Directory:

- `/Users/shiyi/备份（mac_vs专用）/data/raw/password/len9`

Validated sessions:

- `p01_free_type_password_part1_20260321_195652`
- `p01_free_type_password_part2_20260321_200050`
- `p01_free_type_password_part3_20260321_200517`
- `p01_free_type_password_part4_20260321_203037`
- `p01_free_type_password_part5_20260321_203405`

All five groups are usable:

- each session has complete `sensor/events/prompts/attempts/meta`
- each session has `10` prompts and `10` matched attempts
- prompt length is consistently `9`
- typed length is consistently `9`
- each session has `100` press events = `90` chars + `10` enter

## Stage3 incremental adaptation

New reusable script:

- `/Users/shiyi/备份（mac_vs专用）/adapt_password_multilen_inception.py`

Run used:

- base checkpoint: `/Users/shiyi/备份（mac_vs专用）/results/inception_password_final.pt`
- base scaler: `/Users/shiyi/备份（mac_vs专用）/results/inception_password_scaler.npz`
- new checkpoint: `/Users/shiyi/备份（mac_vs专用）/results/inception_password_len8_len9.pt`
- new scaler: `/Users/shiyi/备份（mac_vs专用）/results/inception_password_len8_len9_scaler.npz`
- report: `/Users/shiyi/备份（mac_vs专用）/results/password_len8_len9_adaptation.json`

### Held-out password results

`len8` test (`parts 17-20`):

- old `len8-only` adapted classifier:
  - `top1 = 73.44%`
  - `top3 = 97.50%`
  - `top5 = 99.38%`
  - `CER = 26.56%`
- new `len8+len9` adapted classifier:
  - `top1 = 76.56%`
  - `top3 = 98.13%`
  - `top5 = 99.38%`
  - `CER = 23.44%`

`len9` test (`part 5`):

- old classifier zero-shot:
  - `top1 = 38.89%`
  - `top3 = 74.44%`
  - `top5 = 84.44%`
  - `CER = 61.11%`
- new `len8+len9` adapted classifier:
  - `top1 = 68.89%`
  - `top3 = 96.67%`
  - `top5 = 100.00%`
  - `CER = 31.11%`

### Interpretation

- Adding `len9` clearly helps Stage3 on held-out `len9`.
- It also slightly improves held-out standalone `len8`.
- So at the standalone password level, multi-length adaptation is a real positive signal.

## Pipeline impact on current `mixed_single_training` (`len8`)

### GT-assisted (`GT episode + GT key timestamps`)

Report:

- `/Users/shiyi/备份（mac_vs专用）/results/stage3_gt_single_len8_vs_len8len9.json`

Old classifier:

- `top1 = 66.67%`
- `top3 = 91.67%`
- `top5 = 100.00%`
- `CER = 33.33%`
- `exact_match = 33.33%`

New `len8+len9` classifier:

- `top1 = 62.50%`
- `top3 = 95.83%`
- `top5 = 95.83%`
- `CER = 37.50%`
- `exact_match = 0.00%`

Interpretation:

- on the current tiny `mixed_single_training` GT-assisted len8 check, the new classifier is slightly worse overall
- so the new multi-length classifier should **not** automatically replace the current len8-specialized classifier for the existing len8 demo path

### Non-GT single full-stream len8

Reports:

- old classifier:
  - `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_sweep_0.10_oldcls_recheck/report.json`
- new classifier:
  - `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_noleak_sweep_0.10_len8len9cls/report.json`

Old classifier:

- fixed-window path:
  - `top1 = 62.50%`
  - `top3 = 91.67%`
  - `top5 = 95.83%`
  - `CER = 37.50%`
- overlap-refine path:
  - `top1 = 83.33%`
  - `top3 = 95.83%`
  - `top5 = 95.83%`
  - `CER = 16.67%`

New `len8+len9` classifier:

- fixed-window path:
  - `top1 = 58.33%`
  - `top3 = 87.50%`
  - `top5 = 91.67%`
  - `CER = 41.67%`
- overlap-refine path:
  - `top1 = 75.00%`
  - `top3 = 91.67%`
  - `top5 = 95.83%`
  - `CER = 25.00%`

Interpretation:

- on the current audited len8 single non-GT pipeline, replacing the classifier with the `len8+len9` version hurts
- this means the Stage3 multi-length gain does **not** automatically translate into a better len8 mixed-stream pipeline

## Length inference (`len8` vs `len9`)

### Anchor-search based length inference

Scripts:

- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_password_length_inference.py`
- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy_cls.py`

Results:

- old classifier:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_inference_len8_len9_oldcls.json`
- new classifier:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_inference_len8_len9_newcls.json`

Both give:

- `length_accuracy = 82.4%`

But confusion shows this is misleading:

- all `200/200` len8 attempts are predicted as `8`
- only `6/50` len9 attempts are predicted as `9`
- most len9 attempts are still predicted as `8`

So current anchor-count search is **not** really solving length inference yet.

### Lightweight dedicated length head

Quick diagnostic report:

- `/Users/shiyi/备份（mac_vs专用）/results/length_classifier_len8_len9_logreg.json`

This simple length classifier uses password-attempt region features only:

- duration
- energy statistics
- several peak-count features

Held-out result:

- `accuracy = 98.0%`

Interpretation:

- length is learnable from the signal
- the current main pipeline just is not using the right count model yet

## Practical conclusion for tomorrow

1. Keep two Stage3 tracks for now:
   - `len8-specialized` classifier for the current len8 mixed-stream pipeline
   - `len8+len9` classifier as the new multi-length branch

2. Do not claim that `len8+len9` classifier already improves the current len8 end-to-end pipeline.

3. The strongest positive signal is:
   - Stage3 multi-length adaptation works on held-out password data
   - length itself is learnable (`98%` on a simple diagnostic head)

4. Tomorrow's `len10/11` collection is valuable because it should let us:
   - test whether this dedicated count/length signal scales beyond `8 vs 9`
   - decide whether length inference should become an explicit side head in the pipeline

## len10 extension and mixed-aware length integration

### len10 data quality

Directory:

- `/Users/shiyi/备份（mac_vs专用）/data/raw/password/len10`

Status:

- `part1 / part3 / part4 / part5` are clean
- `part2` contains one failed retry row, but after loader cleanup it correctly resolves to `10` final successful `len10` attempts

This means the current `len10` pilot is valid and can be used directly for Stage3 and length-learning experiments.

### Stage3 with `len8 + len9 + len10`

Quick multi-length adaptation:

- checkpoint:
  - `/Users/shiyi/备份（mac_vs专用）/results/inception_password_len8_len9_len10_quick.pt`
- report:
  - `/Users/shiyi/备份（mac_vs专用）/results/password_len8_len9_len10_quick_adaptation.json`

Combined held-out result:

- `top1 = 81.37%`
- `top3 = 97.65%`
- `top5 = 99.61%`
- `CER = 18.63%`

Per length:

- `len8`: `top1 = 81.88%`, `CER = 18.13%`
- `len9`: `top1 = 87.78%`, `CER = 12.22%`
- `len10`: `top1 = 74.00%`, `CER = 26.00%`

Interpretation:

- adding `len10` does **not** break the multi-length Stage3 branch
- Stage3 remains a strong positive signal for the multi-length direction

### Explicit length head on `8 / 9 / 10`

Engineered length model:

- code:
  - `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/length_model.py`
  - `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/train_length_model.py`
- model:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10.pkl`
- report:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_report.json`

Held-out result:

- `accuracy = 96.67%`

Confusion highlights:

- `len8`: `40 / 40`
- `len9`: `10 / 10`
- `len10`: `8 / 10`

Interpretation:

- length is clearly learnable from the signal
- the remaining problem is **how to connect the length head to the mixed-stream main line**

### Direct plug-in failure: whole coarse region is the wrong interface

We first plugged the length model directly into:

- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy_cls.py`

using the whole coarse region as length input.

Result:

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_lenmodel_v1/report.json`

This failed badly because the standalone-trained length head predicted all three `mixed_single len8` sessions as `9`.

Interpretation:

- the issue is **not** that length is unlearnable
- the issue is **domain/interface mismatch**
- whole coarse regions include too much extra context

### Mixed-aware GT-context length head

We then trained a mixed-aware/context-aware length head:

- model:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_gtcontext.pkl`
- report:
  - `/Users/shiyi/备份（mac_vs专用）/results/length_model_len8_len9_len10_gtcontext_report.json`

Results:

- standalone test accuracy: `95%`
- `mixed_single` GT-context test: `3 / 3` correct

This showed that the direction is right: a mixed-aware length head can work.

### Key breakthrough: infer length from a peak-cluster subregion, not the whole coarse region

The fix that finally worked was:

1. detect the coarse region from the strict no-leak Stage1 detector
2. inside that region, find raw energy peaks
3. cluster peaks by temporal proximity
4. choose the strongest cluster
5. feed the length head a padded crop around that cluster instead of the whole coarse region

This is now implemented in:

- `/Users/shiyi/备份（mac_vs专用）/onset_detection/stage2_segmental/scripts/eval_overlap_single_coarse_energy_cls.py`

Strict no-leak non-GT rerun with:

- no GT episode
- no GT key timestamps
- no fixed `expected_keys = 8`
- mixed-aware GT-context length model
- cluster-subregion length inference

Result:

- `/Users/shiyi/备份（mac_vs专用）/results/stage2_overlap_single_fullstream_energy_cls_lenmodel_v2_clusterregion/report.json`

Fixed-window path:

- `top1 = 62.50%`
- `top3 = 91.67%`
- `top5 = 95.83%`
- `CER = 37.50%`

Overlap-refine path:

- `top1 = 83.33%`
- `top3 = 95.83%`
- `top5 = 95.83%`
- `CER = 16.67%`

Debug confirms:

- all three `mixed_single len8` sessions are now predicted as `8`
- confidence is high (`~0.99`)

Practical meaning:

- we now have a strict non-GT `single` pipeline that no longer needs the hard-coded `8`
- the main blocker was not windowing, but the missing length interface
- the correct interface is a compact peak-cluster subregion

### Collector support for future mixed multi-length data

To avoid another tooling bottleneck, `onset_detection/onset_collector.py` now supports:

- `--mode mixed_single_training --password-length {8,9,10,11}`
- `--mode mixed_retry_training --password-length {8,9,10,11}`

Behavior:

- generated password prompts now respect the requested length
- default output directories are:
  - `data/raw/mixed_single_training` for `len8`
  - `data/raw/mixed_single_len9`, `data/raw/mixed_single_len10`, `data/raw/mixed_single_len11`
  - `data/raw/mixed_retry_training` for `len8`
  - `data/raw/mixed_retry_len9`, `data/raw/mixed_retry_len10`, `data/raw/mixed_retry_len11`

This means the current non-GT main line is now ready for real mixed multi-length collection as soon as those sessions are recorded.
