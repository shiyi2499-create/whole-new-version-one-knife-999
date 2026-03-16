# Onset Detection Claude Pack

This folder is a curated snapshot from the main workspace for designing and implementing
continuous-stream keyboard onset detection.

## What Claude should read first

1. `workspace/README.md`
2. `workspace/CODE_MAP.md`
3. `ONSET_DETECTION_BRIEF.md`
4. `CLAUDE_PROMPT.md`

## Why this pack exists

The main workspace contains a lot of historical code and data. This folder keeps the code,
README files, and design notes Claude needs for the next task:

- detect when keyboard keystrokes happen in a continuous IMU stream
- distinguish keyboard onset from common nuisance motion
- preserve compatibility with the current non-root SPU collection route
- later hand candidate windows to the existing password/key classifier

## Important current context

- Current main workspace: `workspace/`
- Current IMU access path: direct non-root `AppleSPUHIDDevice`
- Current strongest route: `single_key + boost` baseline + `password` adaptation
- Current next big missing piece: onset detection / continuous-stream segmentation
