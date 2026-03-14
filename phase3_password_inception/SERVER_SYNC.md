# Server Sync Guide

## What to copy to the server

Copy the whole folder:

- [phase3_password_inception](/Users/shiyi/权限问题/phase3_password_inception)

If you only want the minimum required files, these are enough:

1. [run_password_closure_inception.py](/Users/shiyi/权限问题/phase3_password_inception/run_password_closure_inception.py)
2. [README.md](/Users/shiyi/权限问题/phase3_password_inception/README.md)
3. [STATUS.md](/Users/shiyi/权限问题/phase3_password_inception/STATUS.md)

## Where it should live on the server

Place the folder inside the server-side non-root trial workspace, for example:

`~/备份（mac_vs专用）/phase3_password_inception/`

## Server-side assumptions

The server workspace should already contain:

- `data/processed/merged_dataset.npz`
- `data/raw/trial_nonroot_free_type_refill/`
- a working Python environment with `torch`, `numpy`, `scipy`, and
  `scikit-learn`

## Exact server commands

From the server-side non-root trial workspace:

```bash
cd '~/备份（mac_vs专用）'
source .venv/bin/activate

python phase3_password_inception/run_password_closure_inception.py \
  --device cuda \
  --merged-path data/processed/merged_dataset.npz \
  --free-type-dirs data/raw/trial_nonroot_free_type_refill \
  --checkpoint-path results/inception_password_final.pt \
  --scaler-path results/inception_password_scaler.npz \
  --report-path results/password_closure_inception.json \
  --force-train
```

## Optional quick sanity checks

Check that the checkpoint exists:

```bash
ls -lh results/inception_password_final.pt \
      results/inception_password_scaler.npz \
      results/password_closure_inception.json
```

Pretty-print the report:

```bash
python -m json.tool results/password_closure_inception.json | sed -n '1,200p'
```

Inspect the top-level metrics only:

```bash
python - <<'PY'
import json
with open('results/password_closure_inception.json', 'r') as f:
    obj = json.load(f)
print(json.dumps(obj.get('metrics', {}), indent=2))
PY
```

## Why we are using this route

This route is intentionally different from the old sentence-style phase3:

- it uses `InceptionTime`, which is the best visible Phase 2 baseline in the
  recorded server results
- it removes `space/enter/backspace` from the target space
- it targets password-like continuous strings first
- it is a better fit for the current security story
