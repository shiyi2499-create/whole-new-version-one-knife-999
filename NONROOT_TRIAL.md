# Non-root SPU Trial

This repository is an isolated trial workspace for validating a non-root
collection path while keeping the collector-visible sensor stream close to the
legacy ~200 Hz fused data shape.

For the permission distinction between IMU access and keyboard-label capture,
see [PERMISSION_MODEL.md](./PERMISSION_MODEL.md).

## Goal

- Keep `collector.py` behavior, outputs, and logs as close as possible to the
  original workflow
- Replace the root-only `macimu` dependency with a non-root direct
  `AppleSPUHIDDevice` path
- Emit one fused 6-axis row per accel/gyro pair so downstream training keeps
  seeing a familiar ~200 Hz stream instead of the raw ~400 rows/sec callback
  stream

## What changed

- `sensor_reader.py`
  - default backend is now direct SPU
  - falls back to `macimu` only if needed
  - pairs one accel callback with one gyro callback before emitting a row
- `collector.py`
  - root-only exit removed
  - added `--keys` for quick single-key trials such as only `a`
  - in `single_key` mode, only the current target key is written to `events.csv`
    to keep labels clean
- `spu_backend.py`
  - direct IOKit backend used by the trial

## Recommended commands

Run these in Terminal on the validated machine, not inside a sandboxed host.

## Required macOS permissions

For the current Terminal-hosted `pynput` label path, treat keyboard permissions
conservatively:

- `Terminal.app` should be allowed in:
  - `Privacy & Security -> Input Monitoring`
- `Accessibility` may also matter on some setups, but it is **not sufficient by
  itself** in our current tests
- After changing either permission, fully quit and relaunch Terminal

### 1. Sensor smoke test

```bash
cd '/Users/shiyi/备份（mac_vs专用）'
.venv/bin/python3 - <<'PY'
import time
from sensor_reader import SensorReader
sr = SensorReader()
sr.start()
time.sleep(2.0)
samples = sr.drain()
backend = sr.backend_name
sr.stop()
print('backend=', backend)
print('samples=', len(samples))
if len(samples) >= 2:
    dt = (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e9
    hz = (len(samples) - 1) / dt if dt > 0 else 0
    print('effective_hz=', hz)
PY
```

### 2. Single-key trial: collect 100 presses of `a`

```bash
cd '/Users/shiyi/备份（mac_vs专用）'
.venv/bin/python3 collector.py \
  --mode single_key \
  --raw-subdir trial_nonroot_single_key_a \
  --keys a \
  --repeats 100 \
  --single-gate-rate 190 \
  --precheck-sec 5
```

### 3. Free-type trial: collect one 13-sentence group

```bash
cd '/Users/shiyi/备份（mac_vs专用）'
.venv/bin/python3 collector.py \
  --mode free_type \
  --raw-subdir trial_nonroot_free_type_part1 \
  --part 1 \
  --free-groups 16 \
  --free-gate-rate 150 \
  --precheck-sec 5
```

## What to look for

- `sensor.csv`, `events.csv`, `meta.txt` should still be produced
- free_type should still create `*_prompts.csv` and `*_attempts.csv`
- the collector should still show live rate monitoring and precheck gate output
- the reported rate should stay around the familiar high-rate regime

## Keyboard permission sanity check

If the progress bar stays at `0/100` while typed characters still appear in the
terminal, that usually means Terminal is echoing your input but `pynput` did
not receive global key events.

This repo now prints the current Accessibility-trust signal, but that signal is
not enough to guarantee key capture. If progress stays at `0/N`, check `Input
Monitoring` first.

In our observed runs:
- `Input Monitoring OFF` -> IMU worked, but keypress counting did not
- `Input Monitoring ON` -> single-key collection completed and wrote the label files

## Important note

This trial aims to preserve the old downstream data assumptions better than the
raw callback-driven 400 rows/sec variant. It is still a new acquisition path,
so the right validation is:

1. collect a small non-root single-key session
2. collect a small non-root free-type session
3. run the same preprocessing/training steps you already trust
4. compare the resulting rate stats and window quality against legacy data
