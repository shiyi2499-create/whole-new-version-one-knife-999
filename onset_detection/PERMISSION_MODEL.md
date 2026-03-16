# Permission Model: IMU Access vs Keyboard Ground Truth

This note explains an important distinction in the project:

- non-root IMU acquisition
- keyboard ground-truth collection for training data

These are related during research collection, but they are not the same
permission problem on macOS.

## Short version

- `AppleSPUHIDDevice` IMU access:
  - this is the sensor-data path
  - on the validated setup, it works in `non-root`
  - it does **not** rely on keyboard-monitoring privileges

- `pynput` keyboard event capture:
  - this is the ground-truth label path used to produce `events.csv`
  - on macOS, it should be treated as a keyboard-monitoring permission path
    rather than part of IMU acquisition itself
  - without it, the collector can still read IMU, but it cannot count or label
    key presses

So:

- attack-side IMU collection and
- research-side labeled data collection

must be analyzed separately.

## Two different pipelines

### 1. IMU data pipeline

Purpose:
- collect raw accelerometer and gyroscope data
- produce `sensor.csv`

Path in this trial:
- `collector.py`
- `sensor_reader.py`
- `spu_backend.py`
- direct `AppleSPUHIDDevice` access

Permission meaning:
- this is the core sensor access question
- this is the path we validated for `non-root`

What it does not require:
- keyboard labels
- `pynput`
- `Accessibility` for key hooks

### 2. Keyboard ground-truth pipeline

Purpose:
- collect the actual pressed key and its timestamp
- produce `events.csv`

Path in this trial:
- `collector.py`
- `keyboard_listener.py`
- `pynput`

Permission meaning:
- this is a research-labeling requirement
- it is not the same thing as IMU access

What it requires on macOS:
- `Terminal.app` enabled in:
  - `System Settings -> Privacy & Security -> Input Monitoring`
- `Accessibility` may also matter depending on host/tooling, but current
  evidence says it is not sufficient by itself for this `pynput` path
Without the required keyboard-monitoring permission:
- Terminal still echoes typed characters on screen
- IMU collection can still run
- but `pynput` does not receive global key events
- so progress bars that depend on keypress counts will stay at `0/N`
- and `events.csv` labeling will fail or remain empty

## Input Monitoring vs Accessibility

These two permissions are easy to confuse, but they gate different things.

| Permission | What it means here | Needed for direct IMU path? | Needed for `pynput` labels? |
|---|---|---:|---:|
| `Input Monitoring` | lets an app monitor keyboard input in other apps; Apple engineers also describe Quartz event taps as belonging to this bucket | No for the validated direct `AppleSPUHIDDevice` path | Likely yes on current macOS for the `pynput` event-tap path |
| `Accessibility` | lets an app act as a trusted assistive client and use global keyboard/UI hooks | No | May still matter, but not sufficient by itself in our current Terminal+pynput tests |

In this project:

- the direct IMU path should be reasoned about independently
- the `pynput` label path should be treated as a separate research-only
  requirement

### Important correction for this trial

Our earlier explanation was too narrow: we described the `pynput` label path as
if it were controlled only by `Accessibility`.

After checking the installed `pynput` Darwin backend, we confirmed that it uses
Quartz event taps (`CGEventTapCreate`). Apple engineering guidance describes
event taps as belonging to the `Input Monitoring` permission bucket, while
`NSEvent` global monitors are associated with `Accessibility`.

So for this collector:

- `Accessibility = ON` is **not enough** to conclude key capture will work
- if Terminal still echoes typed characters but the collector stays at `0/N`,
  `Input Monitoring` is now the leading suspect
- the safest current wording is:
  - direct IMU path: independent of keyboard permissions
  - `pynput` label path: may require `Input Monitoring`, and possibly also
    `Accessibility`, depending on host and macOS version

## Why this matters for the threat model

If we think like an attacker:

- the attacker wants to read IMU data
- the attacker does **not** need the victim's true key labels
- the attacker should not be assumed to have `Accessibility` permission for
  keyboard capture

Therefore:

- lack of `Accessibility` does **not** refute the non-root IMU access result
- it only limits our ability to create labeled training data with this
  collector

This is the key distinction:

- `sensor.csv` capability speaks to the attack surface
- `events.csv` capability speaks to the research data-collection workflow

## Practical interpretation for this trial

### If IMU precheck passes but single-key progress stays at `0/100`

Interpretation:
- IMU is working
- keyboard labels are not working

Most likely reason:
- `Input Monitoring` is missing for `Terminal.app`
- or macOS still treats the current Terminal session as not trusted for the
  event-tap path

### Observed result in this trial workspace

We now have a direct empirical result from this collector:

- with `Input Monitoring` not granted to `Terminal.app`:
  - IMU precheck still passed
  - but single-key progress stayed at `0/N`
  - `events.csv` labeling was effectively unusable

- after granting `Input Monitoring` to `Terminal.app` and restarting Terminal:
  - single-key collection completed successfully
  - `sensor.csv`, `events.csv`, and `meta.txt` were all produced

So for the current Terminal-hosted research collector, `Input Monitoring` is an
important practical requirement for keyboard ground-truth collection.

### If IMU precheck fails

Interpretation:
- this is a sensor path problem
- it is unrelated to keyboard label permissions

## Recommended wording for notes or papers

English:

> On macOS, our research collector uses a `pynput`-based global keyboard hook to
> record ground-truth keystrokes. In our current Terminal-hosted setup,
> `pynput` relies on the macOS event-tap path, which should be treated as a
> keyboard-monitoring permission requirement separate from IMU acquisition
> itself. This requirement applies only to label collection (`events.csv`) and
> should not be conflated with the non-root IMU access result, which is based
> on the direct `AppleSPUHIDDevice` path.

Chinese:

> 在 macOS 上，我们用于采集按键真值标签的研究采集器依赖 `pynput`
> 全局键盘监听。在当前 Terminal 宿主方案下，这条路径应视为“键盘监听权限”
> 问题，和 IMU 本身的读取能力分开讨论。该权限需求仅影响 `events.csv`
> 的标签采集，不影响 IMU 本身的读取能力。我们证明的 non-root IMU 访问能力
> 来自 direct `AppleSPUHIDDevice` 路径，与键盘监听权限是两回事。

## Bottom line

- direct IMU collection:
  - security / attack-surface question
- keyboard label capture:
  - research data-collection question

Do not merge these into a single permission claim.
