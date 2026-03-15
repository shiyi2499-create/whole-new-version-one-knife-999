# Collection Profiles And Model Routes

This note records the currently discussed training/testing combinations and the
new collection profiles available in `collector.py`.

## 1. Model / Data Route Summary

These are the main combinations we have discussed.

### Route A: `single_key` only

Training:
- `single_key + boost`

Testing:
- isolated-key validation only

Purpose:
- clean baseline
- strongest way to show the model has learned per-key signal

### Route B: `single_key + free_type(sentence)`

Training:
- baseline from `single_key + boost`
- optional adaptation using sentence-style `free_type`

Testing:
- sentence-style free_type with spaces

Purpose:
- measure transfer from isolated keys to natural sentences

### Route C: `single_key + free_type(continuous no-space)`

Training:
- baseline from `single_key + boost`
- optional adaptation using no-space continuous strings

Testing:
- no-space continuous strings

Purpose:
- optional bridge route between isolated keys and password-like strings
- currently lower priority than the direct password route

### Route D: `single_key + password`

Training:
- baseline from `single_key + boost`

Testing:
- held-out password strings

Purpose:
- cleanest attack story for continuous string / password recovery

### Route E: pure `free_type`

Training:
- free_type only

Testing:
- held-out free_type parts

Purpose:
- domain-native comparison or upper-bound style auxiliary experiment

Risk:
- easier leakage if the split is not strict
- weaker main story if used as the only primary model

## 2. Current Recommended Priority

Recommended order:

1. `single_key + boost` as the main baseline
2. password held-out test set
3. optional password-style adaptation
4. sentence-style free_type kept as archived/secondary evidence
5. pure free_type only as an auxiliary comparison
6. `continuous` bridge prompts only if needed later

## 3. New Collector Prompt Profiles

The collector now supports three guided text profiles under `--mode free_type`:

1. `sentence`
   - the original prompt set with spaces
2. `continuous`
   - the same prompts with spaces removed
   - optional bridge profile, not the current main priority
3. `password`
   - fixed lowercase+digit password strings
   - current protocol: length 8, 200 strings, 20 groups
   - the first 100 strings (`part 1-10`) were used for the initial zero-shot and adaptation result

This keeps collection inside one collector entrypoint instead of branching into
multiple separate scripts.

## 4. Example Collection Commands

### Sentence-style free_type

```bash
.venv/bin/python3 collector.py \
  --mode free_type \
  --prompt-profile sentence \
  --raw-subdir free_type_sentence_v1 \
  --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5
```

### Continuous no-space bridge strings

```bash
.venv/bin/python3 collector.py \
  --mode free_type \
  --prompt-profile continuous \
  --raw-subdir free_type_continuous_v1 \
  --part 1 --free-groups 16 \
  --free-gate-rate 150 --precheck-sec 5
```

### Password strings (`len=8`, `a-z0-9`, `200` total)

```bash
.venv/bin/python3 collector.py \
  --mode free_type \
  --prompt-profile password \
  --raw-subdir password/len_8 \
  --part 11 --free-groups 20 \
  --free-gate-rate 150 --precheck-sec 5
```

Helper:

```bash
./run_password_len8_part.sh 11
```

## 5. Recommended Testing Logic

If the goal is a password-style attack story, the cleanest setup is:

1. train baseline on `single_key + boost`
2. use `password` prompts as the held-out attack-style test set
3. use `continuous` prompts only if we later need a bridge/adaptation dataset

This gives a cleaner interpretation than taking sentence prompts and only
removing spaces after the fact.
