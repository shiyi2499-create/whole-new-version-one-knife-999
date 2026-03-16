# Non-root SPU Callback Trial (400Hz collector-visible stream)

This trial area is isolated from the main project.
It tests a non-root direct `AppleSPUHIDDevice` path that exposes one CSV row per real accel/gyro callback.

## What is different from the original collector?

- Original collector-visible stream: about 190-200 rows/sec total
- This trial collector-visible stream: about 400 rows/sec total on Tahoe
- Why: accel and gyro callbacks are both emitted as rows
- Each row is still a real hardware callback
- But only one sensor is fresh on each row; the other half of the 6-axis vector is the last known value

In other words, this is a denser callback-driven stream, not the same 6-axis sampling semantics at a higher quality.

## What stays the same?

- CSV schema stays the same:
  `timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z`
- Sensor and keyboard timestamps stay in the same `perf_counter_ns()` monotonic clock domain
- Collector flow, session files, rate monitor, prompts log, attempts log, and meta output remain in place

## Suggested validation commands

### 1. Sensor-only smoke test

```bash
cd '/Users/shiyi/备份（mac_vs专用）/trial_nonroot_spu_callback_400hz'
python3 - <<'PY'
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
    dt=(samples[-1].timestamp_ns-samples[0].timestamp_ns)/1e9
    hz=(len(samples)-1)/dt if dt>0 else 0
    print('effective_hz=', hz)
PY
```

### 2. Single-key collection trial (non-root)

```bash
cd '/Users/shiyi/备份（mac_vs专用）/trial_nonroot_spu_callback_400hz'
python3 collector.py \
  --mode single_key \
  --raw-subdir trial_nonroot_spu_callback_400hz \
  --group 1 --repeats 10 \
  --single-gate-rate 190 --precheck-sec 5
```

### 3. Stricter gate suggestion for this callback-driven stream

If you want a gate closer to the old "high-rate only" intent, try:

```bash
cd '/Users/shiyi/备份（mac_vs专用）/trial_nonroot_spu_callback_400hz'
python3 collector.py \
  --mode single_key \
  --raw-subdir trial_nonroot_spu_callback_400hz_strict \
  --group 1 --repeats 10 \
  --single-gate-rate 380 --precheck-sec 5
```

## Recommended interpretation

- Use this trial to test whether denser callback-driven data improves window quality
- Do not mix this data blindly with the legacy ~200Hz dataset
- Expect downstream scripts that compute or gate on "actual rate" to show about 400Hz on Tahoe for this stream
