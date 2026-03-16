# Paper Outline

## Working Title

Apple Internal IMU as a Non-Root Keystroke Side Channel on macOS:
From Sensor Exposure to Password-Like Continuous-String Inference

## 1. Introduction

This section should establish:

1. Apple devices expose an internal IMU path that is not commonly treated as a
   keyboard side-channel surface
2. on macOS, this sensor path can be reached by a non-root process
3. if that signal is strong enough to recover keystrokes, the result is a
   system-level security problem rather than only a modeling curiosity

The introduction should end with the main message:

- we show a previously under-discussed Apple internal IMU exposure path
- we demonstrate non-root access under real macOS conditions
- we establish a controlled end-to-end password-style inference result

## 2. Threat Model

Clarify the attacker assumptions:

1. attacker code runs as a non-root process on macOS
2. attacker can continuously read IMU data from the internal Apple sensor path
3. attacker does not rely on keyboard ground-truth during the actual attack
4. current controlled evaluation uses slow, fixed-finger, password-like input as
   an upper-bound closure setting

Also distinguish:

- research data collection permissions
- real attack requirements

Important clarification:

- Input Monitoring / Accessibility may be needed for label collection
- they are not the same as the IMU read permission path used by the attack

## 3. Attack Surface Analysis

This section should summarize the permission work:

1. Apple internal SPU IMU devices are present and enumerable
2. the `IOHIDManager` path is affected by Input Monitoring
3. the direct `AppleSPUHIDDevice` path is the important attack path
4. on the validated systems, non-root direct IMU reads are possible

This is one of the strongest novelty sections.

## 4. Data Collection Methodology

Explain the collection design:

1. `single_key + boost` for isolated-key baseline
2. `password` prompt profile for no-space continuous-string evaluation
3. controlled slow typing and fixed finger assignment
4. same monotonic clock domain for sensor and labels
5. strict rate gating around ~190-200 Hz

Why this matters:

- this is not a random data dump
- it is a deliberately controlled protocol to close the loop under realistic but
  bounded conditions

## 5. Signal Processing And Windowing

Explain:

1. 100ms pre-trigger + 200ms post-trigger
2. uniform resampling to 190 Hz -> 57 samples
3. six-channel IMU window representation
4. why this fixed window protocol is used across training and evaluation

Also explain current limitation:

- current main evaluation still uses labeled event boundaries
- blind onset detection is treated as a dedicated next-stage challenge

## 6. Baseline Models

This section should be modest and honest:

1. the core novelty is not a new deep model
2. we use established time-series baselines
3. strongest recorded baseline in current results is `InceptionTime`

Say clearly:

- model novelty is limited
- system exposure + attack feasibility is the main contribution

## 7. Password-Style Continuous-String Inference

This is the core closure section.

Explain the evaluation protocol:

1. train on `single_key + boost`
2. test on held-out password-like no-space strings
3. report:
   - `char top-1`
   - `char top-3`
   - `char top-5`
   - `sequence top-10`
   - `sequence top-50`
   - `sequence top-100`
   - `CER`

Important framing:

- exact full-string top-1 match is too strict to be the only security metric
- top-k and candidate-space style reporting is the right attack-facing view

## 8. Continuous Stream Segmentation / Onset Detection

This section should address the missing link:

1. once IMU streaming is possible, the attacker must decide when keystrokes
   occur
2. current paper should either:
   - present a controlled onset detector, or
   - clearly mark this as a bounded limitation and future extension

If implemented, report:

- onset precision / recall
- timing offset distribution
- impact on downstream top-k password inference

## 9. Discussion Of Limits

Be explicit:

1. current closure setting is slow and controlled
2. fixed finger assignment reduces one source of variation
3. natural-language sentence reconstruction is not the current headline
4. high-speed overlap remains harder and is not claimed solved here

This section protects the paper from over-claiming.

## 10. Related Work

Structure related work by source modality:

1. wearables and arm/hand inertial sensing
2. audio / thermal / hybrid side channels
3. password-focused top-k or candidate-reduction attacks

Then explain the gap:

- prior work studies other sensors or other devices
- our contribution is the Apple internal IMU + macOS non-root exposure + same-device password-like closure

## 11. Mitigations

Suggested mitigation directions:

1. restrict or gate internal IMU access more strictly
2. move the IMU path behind stronger privilege or TCC policy
3. reduce sampling precision or add noise for untrusted consumers
4. detect suspicious long-lived IMU consumers

## 12. Conclusion

Close on the main story:

1. Apple internal IMU exposure is a real attack surface
2. non-root access makes the issue systemically important
3. controlled password-style inference is enough to show end-to-end attack feasibility
4. this motivates platform-level mitigation
