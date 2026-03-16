# Root vs Non-root Single-Key Comparison

This note compares:

- legacy root-path session:
  - `/Users/shiyi/备份（mac_vs专用）/data/raw/legacy_round4_ro/p01_single_key_g1_20260312_001904`
- new non-root trial session:
  - `data/raw/trial_nonroot_single_key_a/p01_single_key_20260314_184003`

## Headline

The new non-root trial is **very close on the IMU side** and is already
compatible with the existing preprocessing pipeline. The latest filtered trial
run is also clean enough to use as a single-key session.

Note:
- that specific mismatch came from a pre-filter trial run
- the collector has since been updated so `single_key` mode writes only the
  current target key to `events.csv`

In short:

- sensor stream quality: effectively equivalent
- preprocessing compatibility: confirmed
- label cleanliness: good enough for single-key use

## Side-by-side summary

| Metric | Legacy root session | New non-root trial | Interpretation |
|---|---:|---:|---|
| Sensor CSV header | identical | identical | compatible |
| Events CSV header | identical | identical | compatible |
| Sensor rows | 85119 | 19052 | different session duration, not a problem |
| Duration | 425.75s | 95.23s | different collection length |
| Effective sampling rate | 199.93Hz | 199.98Hz | effectively the same |
| Median sample interval | 5.08ms | 5.00ms | effectively the same |
| Target resample rate | 190Hz | 190Hz | same downstream target |
| Preprocessor output window size | 57 | 57 | same |
| Raw samples per 300ms window | avg 59.9 | avg 60.0 | effectively the same |

## What is already confirmed

### 1. Sensor stream format matches

The new non-root trial produced:

- `timestamp_ns,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z`

This matches the legacy collector schema exactly.

### 2. Sampling behavior matches the legacy high-rate regime

Legacy root session:
- effective rate: `199.93Hz`

New non-root trial:
- effective rate: `199.98Hz`

This is strong evidence that the new paired direct-SPU path preserves the
collector-visible rate regime expected by the downstream pipeline.

### 3. Existing preprocessing works without modification

The existing `preprocessor.py` successfully processed both sessions.

Legacy root session:
- `600` valid windows
- raw window size avg `59.9`
- output shape `(600, 57, 6)`

New non-root trial:
- `100` valid windows
- raw window size avg `59.9`
- output shape `(100, 57, 6)`

This means the new non-root data can already enter the existing windowing and
resampling pipeline.

## Current label status

For the latest filtered trial session:

- `a press`: `100`
- `a release`: `99`
- unique labeled keys: only `a`

This is good enough for the current single-key pipeline, because downstream
window extraction uses `press` events as labels and successfully produced:

- `100` valid `a` windows
- `Unique keys: 1`

The missing final release is a small bookkeeping imperfection, not a training
blocker for the current single-key preprocessing path.

## Practical conclusion

### Can the old single-key dataset continue to be used?

Yes.

Nothing in this comparison suggests the old root-path single-key data should be
discarded. It remains valid and compatible.

### Can the new non-root single-key path be used for future collection?

Yes.

The IMU path is good, preprocessing compatibility is confirmed, and the latest
filtered `a`-only trial produced a clean single-key label set for practical use.

### Is this current trial file interchangeable with a legacy single-key file?

For the current single-key training path: effectively yes.

The one caveat is a missing final release event (`100` press vs `99` release),
but this does not prevent correct single-key window extraction.

## Bottom line

If we focus on the core question "does non-root collection preserve the old
200Hz training-facing IMU data shape?", the answer is:

**Yes.**

If we ask "is this exact run already clean enough to replace future root-based
single-key collection for this path?", the answer is:

**Yes.**
