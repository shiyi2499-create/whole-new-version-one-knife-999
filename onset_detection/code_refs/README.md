# Onset Reference Code

This folder contains the minimum non-data code context that the onset module
needs to understand the current mainline repository.

Files included:
- ROOT_README.md: top-level project status and current paper direction
- CODE_MAP.md: active code map for the repository
- sensor_reader.py / spu_backend.py: non-root SPU IMU acquisition path
- keyboard_listener.py: keyboard event timestamps / labels
- collector.py / typing_prompt_profiles.py / config.py: current collection stack
- preprocessor.py: existing time-window extraction conventions
- adapt_password_len8_inception.py: current password adaptation route
- phase3_password_inception/run_password_closure_inception.py: password classifier loading and scoring path
- ONSET_DETECTION_PLAN.md: prior onset plan, if present
