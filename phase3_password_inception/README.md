# Phase 3 Password / Continuous-String Route

This folder isolates a password-style closure path that is closer to
continuous string recovery than sentence reconstruction.

## Why this route exists

The previous free-type route in the main training workspace mixed together
several choices that are not ideal for the attack story we want to tell:

1. it used a `Transformer` backbone by default
2. it still carried `space` / `enter` in the training label space
3. it optimized for sentence-style decoding rather than continuous-string
   recovery

For the current paper story, the cleaner task is:

`non-root IMU access -> isolated-key baseline -> password-style continuous-string closure`

That is a better fit for password-like inputs and keeps the contribution focused
on attack feasibility rather than language reconstruction.

## Baseline choice

The server-side Phase 2 result snapshot shows the strongest visible baseline is
`InceptionTime`, not `Transformer`.

Source:
- [results_phase2.json](/Users/shiyi/备份（mac_vs专用）/results/服务器results/results_phase2.json)

Key accuracies from that snapshot:
- `dl_InceptionTime = 0.8592`
- `dl_Transformer = 0.8095`

So this route intentionally uses `InceptionTime` as the baseline backbone.

Related notes:
- [STATUS.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/STATUS.md)
- [ROADMAP.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/ROADMAP.md)
- [SERVER_SYNC.md](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/SERVER_SYNC.md)

## Main script

- [run_password_closure_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py)

What it does:

1. load `merged_dataset.npz` and train a final `InceptionTime` baseline
2. filter the classifier target space to `[a-z0-9]` only
3. ignore `space`, `enter`, `backspace`, and other non-password keys
4. evaluate password-style continuous strings
5. evaluate:
   - exact sequence match
   - character top-1 / top-3 / top-5 accuracy
   - sequence top-10 / top-50 / top-100 hit rate
   - CER

## Expected data layout

Current v1 password collection protocol:

- charset: `a-z0-9`
- length: `8`
- total strings: `100`
- grouping: `10 × 10`
- raw path: `data/raw/password/len_8`

Defaults assume the current main workspace layout:

- `data/processed/merged_dataset.npz`
- `data/raw/password/len_8/`

Override paths with CLI flags if needed.

## Important separation of roles

- `merged_dataset.npz` is still the main baseline training set
- it should come from `single_key + boost`
- `data/raw/password/len_8` is not the baseline training source
- it is used for password-style closure evaluation

In other words, this route does **not** train the baseline on free_type first.
It trains on isolated-key data, then checks whether that baseline can recover
continuous strings.

## Typical server run

```bash
python phase3_password_inception/run_password_closure_inception.py \
  --device cuda \
  --merged-path data/processed/merged_dataset.npz \
  --free-type-dirs data/raw/password/len_8 \
  --checkpoint-path results/inception_password_final.pt \
  --scaler-path results/inception_password_scaler.npz \
  --report-path results/password_closure_inception.json \
  --force-train
```

## Local smoke test status

This route has already been verified to run end-to-end in a local smoke test:

- checkpoint written successfully
- scaler written successfully
- report written successfully

Artifacts:
- [inception_password_final.pt](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/results/inception_password_final.pt)
- [inception_password_scaler.npz](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/results/inception_password_scaler.npz)
- [password_closure_inception.json](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/results/password_closure_inception.json)

The local metrics are intentionally not used as a scientific result because the
smoke test was only:
- `1` epoch
- `256` training samples
- `2` evaluation sequences

Its purpose was just to verify that the code path runs to completion.

## How to interpret the new metrics

- `char_top1_accuracy`: standard per-position top-1 accuracy
- `char_top3_accuracy` / `char_top5_accuracy`: whether the true character is
  present in the model's top-3 or top-5 local candidates
- `sequence_top10_hit_rate` / `top50` / `top100`: whether the full reference
  string appears in the top-N beam candidates generated from per-position
  probabilities
- `unsupported_ref_char_rate`: fraction of reference characters that are not
  present in the current classifier vocabulary at all

For password-like attacks, these top-k and candidate-hit metrics are often more
informative than exact top-1 string match alone.
