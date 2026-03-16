# Finger Boost Plan for `r` and `t`

This note is only for the isolated non-root trial workspace.

## Why no collector code change is needed

The current trial collector already supports:

- explicit single-key collection via `--keys`
- clean single-key labels via target-key filtering
- the same rate monitor, precheck gate, and CSV outputs as the main trial path

So for `r/t` finger-variant supplementation, we only need separate collection
runs, not a collector rewrite.

## Recommended plan

Treat these as **boost** samples, not a replacement for the main single-key
set.

### Key `r`

- baseline/common: `left_index`
- supplement: `left_middle`

Recommendation:

- `left_index`: 80 presses
- `left_middle`: 80 presses

### Key `t`

- old single-key habit you mentioned: `right_index`
- free-type variant you observed: `left_index`

Recommendation:

- `right_index`: 80 presses
- `left_index`: 80 presses

## Why 80 per variant

- enough to give the model a meaningful view of the alternate finger pattern
- cheaper than rebuilding the whole single-key set
- still balanced across the two variants of the same key

If you want a slightly stronger boost set, `100` per variant is also reasonable.

## Commands

Run these in Terminal:

```bash
cd '/Users/shiyi/备份（mac_vs专用）'

./run_single_key_boost_finger.sh r left_index 80
./run_single_key_boost_finger.sh r left_middle 80

./run_single_key_boost_finger.sh t right_index 80
./run_single_key_boost_finger.sh t left_index 80
```

If you only want to supplement the **newly observed alternate finger patterns**
instead of rebuilding both variants, start with just:

```bash
cd '/Users/shiyi/备份（mac_vs专用）'

./run_single_key_boost_finger.sh r left_middle 100
./run_single_key_boost_finger.sh t left_index 100
```

Each run writes to its own raw subdirectory:

- `data/raw/trial_nonroot_single_key_boost_r_left_index/`
- `data/raw/trial_nonroot_single_key_boost_r_left_middle/`
- `data/raw/trial_nonroot_single_key_boost_t_right_index/`
- `data/raw/trial_nonroot_single_key_boost_t_left_index/`

## Suggested recording rule

- keep only one intended finger style per run
- if a run feels messy, keep it separate and decide later whether to keep or drop
- do not mix two finger styles in the same run
