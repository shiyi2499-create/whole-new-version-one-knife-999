# Password Route Roadmap

This file records the currently discussed next-step plans for the password /
continuous-string attack route so they do not get lost while we iterate.

## Current Position

We already have:

1. non-root IMU collection confirmed on macOS
2. a usable non-root data collection workflow for `single_key` and `free_type`
3. a password-style Phase 3 route that:
   - uses `InceptionTime`
   - removes `space / enter / backspace` from the target class space
   - evaluates no-space continuous strings
   - reports top-k and candidate-hit metrics
4. a collector that can now record `sentence`, `continuous`, and `password`
   prompt profiles through the same entrypoint
5. a concrete password v1 collection protocol:
   - charset: `a-z0-9`
   - length: `8`
   - total strings: `100`
   - grouping: `10 × 10`
   - raw path: `data/raw/password/len_8`

What we do **not** have yet:

1. a full Phase 2 "best recipe" InceptionTime training path inside the password route
2. a free_type adaptation / fine-tuning stage
3. a final top-N password attack story with held-out evaluation and clean ablations
4. a continuous-stream keystroke onset detector

## Immediate Diagnostic Rule

Before over-interpreting a low password zero-shot result, we should first run a
held-out single-key diagnostic using the same current Phase 3 Inception
trainer.

Reason:

- this separates "trainer/recipe weakness" from "password domain gap"
- it is the fastest way to explain a counterintuitively low password result

## Why The Current Low Result Is Not Final

The current password-style result should be treated as an early prototype result,
not the final attack capability.

Main reasons:

1. the current password route still uses a simplified training recipe compared to
   the strongest recorded Phase 2 `InceptionTime`
2. the model is trained on `single_key + boost` and tested zero-shot on
   continuous password strings
3. the current evaluation should be moved from reused no-space free_type strings
   to the dedicated password dataset
4. exact top-1 whole-string match is too harsh to be the only security metric

## Planned Experiment Tracks

### Track A: Zero-Shot Password Closure

Goal:

- train on `single_key + boost`
- test directly on held-out password strings

Why:

- this is the cleanest way to show whether isolated-key learning transfers to
  continuous input
- this is the current main story

Metrics to keep:

- `char_top1_accuracy`
- `char_top3_accuracy`
- `char_top5_accuracy`
- `sequence_top10_hit_rate`
- `sequence_top50_hit_rate`
- `sequence_top100_hit_rate`
- `CER`

Status:

- implemented
- should remain the first baseline for the password route

### Track B: Full-Recipe InceptionTime

Goal:

- replace the simplified current training loop with a version much closer to the
  original best Phase 2 `InceptionTime` recipe

Why:

- the current password route uses the right backbone but not yet the strongest
  known training recipe
- this is the first thing to improve before making strong claims about low
  password-route accuracy

Planned upgrades:

1. closer alignment with original augmentation
2. closer alignment with original scheduler / patience / epochs
3. reuse of the original feature and split assumptions where appropriate

Status:

- not implemented yet
- high priority

### Track C: Password / Free-Type Adaptation

Goal:

- keep `single_key + boost` as the main baseline training set
- use part of the password dataset for adaptation / fine-tuning
- reserve held-out password strings for final evaluation

Why:

- zero-shot transfer may be too strict
- adaptation is a realistic second-stage experiment
- this helps separate "is there signal?" from "how much domain shift remains?"

Suggested split style:

1. `single_key + boost` -> base training
2. `password strings 1-80` -> adaptation
3. `password strings 81-100` -> held-out evaluation

Status:

- not implemented yet
- medium-high priority after Track B

### Track D: Keystroke Onset Detection

Goal:

- continuously monitor IMU
- detect when a keypress happens
- cut windows automatically instead of relying on label timestamps

Why:

- this is the missing link between "continuous sensor access" and a more
  automatic real attack
- it should appear in the paper either as a controlled prototype or as an
  explicit next-stage task

Status:

- not implemented yet
- high value

### Track E: Pure Password / Free-Type Model

Goal:

- train and test directly on password/free_type only

Why:

- useful as a domain-native comparison
- may show the upper bound if we fully specialize to continuous strings

Risks:

1. class imbalance
2. easier leakage if splits are not strict
3. weaker paper story if used as the only main model

Recommended role:

- auxiliary experiment
- not the primary headline result

Status:

- not implemented
- lower priority than Tracks A/B/C

## Vocabulary Coverage Plan

We already observed that some characters used in free_type references are not in
the current classifier label set.

This can create an artificial ceiling on evaluation.

Planned follow-up:

1. audit training vocabulary against free_type references
2. identify missing keys such as `w` or `7` if they are absent
3. decide whether to:
   - add them through existing single_key / boost data
   - or exclude evaluation samples containing unsupported characters

Status:

- identified as an issue
- partial diagnostics already added

## Attack-Facing Metrics We Intend To Keep

Because this is a password-style attack, final reporting should not depend only
on exact top-1 string match.

Preferred attack-facing metrics:

1. per-character top-1 / top-3 / top-5
2. sequence top-10 / top-50 / top-100 hit rate
3. CER / edit distance
4. candidate-space reduction, if we add that later

Why:

- password attacks are naturally budgeted by number of guesses
- many prior works report top-k style results rather than only exact top-1

## Recommended Execution Order

If we continue this route, the most sensible order is:

1. keep Track A as the zero-shot baseline
2. implement Track B so the backbone training is not artificially weak
3. rerun password-route evaluation on `data/raw/password/len_8`
4. if still needed, add Track C adaptation
5. add Track D onset detection
6. only then decide whether Track E is worth doing

## Current Practical Interpretation

The current password-route result should be read as:

- the no-space evaluation pipeline works
- top-k / top-N reporting is now in place
- low current top-1 accuracy does not yet falsify the attack story
- we still need stronger baseline training and possibly adaptation before making
  a final scientific claim

## Current Main Story

The current best story is:

1. Apple internal IMU is exposed on macOS
2. that path is accessible from a non-root process
3. `single_key + boost` gives a strong isolated-key baseline
4. held-out password strings (`a-z0-9`, len=8) are the main attack target
5. onset detection is the next major technical step toward a more automatic attack
