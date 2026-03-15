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
   - current pool size: `200`
   - grouping: `20 × 10`
   - raw path: `data/raw/password/len_8`
   - the first reported result uses `part 1-10`

What we do **not** have yet:

1. a final multi-length password story (`len_8`, `len_10`, `len_12`)
2. symbol-inclusive password evaluation
3. a continuous-stream keystroke onset detector
4. a final clean ablation table for zero-shot vs adapted password inference

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

### Track B: Strong InceptionTime Password Route

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

- implemented enough to reproduce a strong isolated-key diagnostic
- no longer the main blocker

### Track C: Password Adaptation

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

- implemented
- first `len_8` result is positive:
  - `char_top1 = 62.5%`
  - `char_top3 = 87.5%`
  - `char_top5 = 96.2%`
  - `sequence_top100 = 35.0%`
  - `CER = 37.5%`
- now promoted to the main experimental route

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

### Track E: Pure Password Model

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

The current password-route result should now be read as:

- the no-space password evaluation pipeline works
- the trainer is strong on isolated-key held-out evaluation
- direct zero-shot transfer to continuous password input is weak
- password-style adaptation substantially improves performance
- the present bottleneck is domain shift, not obviously broken collection or
  model training
- we still need stronger baseline training and possibly adaptation before making
  a final scientific claim

## Why Strong Single-Key Accuracy Does Not Automatically Transfer

Even though password evaluation also cuts per-key windows, the resulting windows
are not drawn from the same distribution as isolated `single_key` samples.

Main reasons:

1. the finger and hand state before each key is different in continuous input
2. neighboring keys change the local motion context
3. continuous typing introduces preparation and recovery motion that is absent
   in isolated-key collection
4. the same nominal key can therefore occupy a meaningfully different feature
   distribution in password mode

In short:

- the model is still asked to classify one key window at a time
- but the password-mode windows are not i.i.d. copies of isolated-key windows
- this explains why strong isolated-key top-k accuracy can coexist with weak
  zero-shot password performance

## ROI Priority List For Improving `len_8`

### Tier 1: Highest ROI

1. add more password-style adaptation data
   - this is the only thing already proven to move the main metric sharply
   - the first `80/20` result is strongly positive

2. repeat `len_8` collection once more for stability
   - confirms the current result is not a one-off split artifact
   - gives us more password-domain data without changing task definition

3. run ablations on adaptation size
   - e.g. `20 / 40 / 60 / 80` password strings
   - tells us how much password data is actually needed

### Tier 2: Medium ROI

4. targeted single-key boost for password-confused characters
   - only worth doing after we inspect password confusion patterns
   - useful if a small set of characters dominates the residual error

5. per-position / per-character error audit on `len_8`
   - helps identify whether errors cluster at early vs late characters or at
     specific keys

6. test a slightly lighter adaptation schedule vs a stronger one
   - useful for checking whether current gains are training-limited

### Tier 3: Lower ROI Right Now

7. broadly collecting many more generic single-key sessions
   - likely improves isolated-key accuracy more than password-domain accuracy
   - unlikely to close the main domain gap by itself

8. collecting both single-key and password in lockstep without analysis
   - risks increasing both datasets while leaving the main gap unchanged
   - should be avoided unless guided by ablation results

## Current Recommended Plan For `len_8`

1. keep the current `single_key + boost` baseline fixed
2. treat password-domain data as the main lever for improvement
3. first replicate `len_8`
4. then run adaptation-size ablations
5. only after that decide whether targeted extra single-key collection is worth
   the cost

## Model Comparison Policy

Current decision:

- keep `InceptionTime` fixed as the backbone for the next password-route
  experiments
- do **not** mix in model changes while we are still validating:
  - multi-split stability
  - `single_key + password adaptation`
  - `password only`

Why:

1. `InceptionTime` is the strongest visible Phase 2 baseline so far
2. we first need a clean comparison of **data/training protocol choices**
3. changing the model now would confound:
   - domain-shift diagnosis
   - adaptation benefit
   - password-only upper-bound estimation

Planned later comparison:

- once the password-route protocol is stable, run a smaller route-specific model
  comparison under the same split protocol
- likely candidates:
  - `InceptionTime`
  - `Transformer`
  - `1D CNN` or `CNN_BiLSTM`

Important note:

- the best model for `single_key` is **not guaranteed** to be the best model
  for `single_key + password adaptation` or `password only`
- this is a later paper-quality ablation, not the current blocking step

## Current Main Story

The current best story is:

1. Apple internal IMU is exposed on macOS
2. that path is accessible from a non-root process
3. `single_key + boost` gives a strong isolated-key baseline
4. held-out password strings (`a-z0-9`, len=8) are the main attack target
5. onset detection is the next major technical step toward a more automatic attack
