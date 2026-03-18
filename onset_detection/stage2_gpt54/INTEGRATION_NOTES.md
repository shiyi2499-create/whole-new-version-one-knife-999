# Integration Notes for Existing `onset_detection`

## 1. `onset_model.py`
Either:
- import `PasswordStage2TCN` and expose it through `build_onset_model`, or
- keep it isolated and load it directly from `password_stage2_model.py`

Recommended minimal diff:

```python
from password_stage2_model import PasswordStage2TCN

if name == "password_stage2_tcn":
    return PasswordStage2TCN(n_channels=n_channels, **kwargs)
```

## 2. `train_onset.py`
Add a new task:

```python
parser.add_argument("--task", choices=["onset", "activity", "password_boundary", "password_segment", "password_stage2"], default="onset")
```

Then branch training logic:
- legacy tasks -> current window dataset path
- `password_stage2` -> sequence dataset + collate + multi-head loss

Suggested loss:

```python
loss = (
    bce(key_logits, key_target) +
    1.5 * bce(boundary_logits, boundary_target) +
    0.5 * bce(inside_logits, inside_target) +
    0.15 * temporal_smoothing_loss(key_logits, mask) +
    0.20 * temporal_smoothing_loss(boundary_logits, mask)
)
```

## 3. `password_segment_detector.py`
Add a new option:

```python
p.add_argument("--stage2-method", choices=["energy_valley", "iki_heuristic", "dense_structured"], default="dense_structured")
```

New path should:
1. run Stage 1 and get the coarse region
2. patchify that region using `build_patch_views(...)`
3. run the dense Stage 2 model
4. call `decode_stage2_dense(...)`
5. convert decoded key patch centers to onset timestamps
6. pass the 5 groups into Stage 3 classifier

## 4. Keep protocol prior explicit
Do not pull `n_passwords` from GT.
Pass it as configuration:

```python
prior = Stage2ProtocolPrior(
    expected_password_count=args.expected_password_count,
    expected_password_len=args.expected_password_len,
)
```

## 5. Debug outputs worth saving
For each session, save:
- patch times
- key_score curve
- boundary_score curve
- chosen boundaries
- chosen 8-slot key centers for each predicted password

This is the fastest way to see whether the new Stage 2 is failing because of:
- model score quality
- bad boundary DP
- bad per-segment exact-8 decode
