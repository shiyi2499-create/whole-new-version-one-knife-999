Please help implement a first controlled keyboard onset detector for this project.

Read these files first:
1. `workspace/README.md`
2. `workspace/CODE_MAP.md`
3. `ONSET_DETECTION_BRIEF.md`
4. `workspace/spu_backend.py`
5. `workspace/sensor_reader.py`
6. `workspace/collector.py`
7. `workspace/preprocessor.py`
8. `workspace/keyboard_listener.py`

Project context:
- We already have non-root IMU collection on Apple Silicon Macs through the direct AppleSPUHIDDevice path.
- We already have strong per-keystroke classification and password-style adaptation results.
- The missing piece is onset detection in a continuous IMU stream.

Task:
- Design and implement a first controlled onset-detection pipeline.
- Prefer minimal, practical changes that fit the existing codebase.
- Keep the current non-root SPU collection path intact.

What I want from you:
1. Propose the cleanest code entrypoint for onset-data collection and onset-model training.
2. Decide whether to extend `collector.py` or add a new focused onset collector.
3. Add code for nuisance-motion collection modes such as:
   - idle
   - trackpad_move
   - trackpad_tap
   - shake
4. Reuse existing keyboard-labeled data where appropriate for positive samples.
5. Add a first training/evaluation script for binary onset detection.
6. Define outputs and metrics:
   - precision / recall / F1
   - timing tolerance
   - false alarms per minute
7. Keep the implementation compatible with the current workspace conventions.

Constraints:
- Do not redesign the whole repository.
- Prefer simple additions over large refactors.
- Explain proposed file layout and why it fits the current project.
- If assumptions are needed, state them explicitly.
