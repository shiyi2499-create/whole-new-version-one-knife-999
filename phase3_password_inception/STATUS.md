# Current Status

## Scope

This folder is a Phase 3 experiment track for:

- `InceptionTime` baseline
- password-style continuous-string inference
- password-like threat model
- top-k / top-N attack-facing evaluation

It exists to avoid changing the main workspace under
[备份（mac_vs专用）](/Users/shiyi/备份（mac_vs专用）).

## What is already confirmed

1. The strongest visible Phase 2 server baseline is `InceptionTime`
   - [results_phase2.json](/Users/shiyi/备份（mac_vs专用）/results/服务器results/results_phase2.json)
   - `dl_InceptionTime = 0.8592`
   - `dl_Transformer = 0.8095`

2. The old sentence-style free_type route is not the right first target for the
   current attack story
   - it keeps `space/enter` in the label space
   - it uses a weaker backbone than the current best visible baseline
   - it evaluates sentence reconstruction rather than continuous-string recovery

3. The new no-space route runs end-to-end
   - [run_password_closure_inception.py](/Users/shiyi/备份（mac_vs专用）/phase3_password_inception/run_password_closure_inception.py)
   - local smoke test completed successfully

4. The current main story is now more focused than before
   - `single_key + boost` remains the main baseline training source
   - `password` prompt profile is the main continuous-input target
   - sentence-style free_type is kept, but not the current headline route
   - `continuous` bridge prompts are optional rather than required

## What has not been claimed yet

1. We are not claiming the local smoke-test accuracy is meaningful
2. We are not claiming sentence-level natural-language recovery is solved
3. We are not claiming fast-overlap typing is solved
4. We are not claiming blind onset detection is solved yet

## Immediate next step

Run the InceptionTime password closure on the real dataset:

- baseline training source: `single_key + boost`
- closure evaluation source: `data/raw/password/len_8`

## Expected output on the server

- `results/inception_password_final.pt`
- `results/inception_password_scaler.npz`
- `results/password_closure_inception.json`

The report should be interpreted using:

- `char_top1/top3/top5`
- `sequence_top10/top50/top100`
- `CER`
