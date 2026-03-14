#!/bin/zsh
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"

"$PYTHON_BIN" - <<'PY'
import time
from sensor_reader import SensorReader

sr = SensorReader()
sr.start()
time.sleep(2.0)
samples = sr.drain()
backend = sr.backend_name
sr.stop()

print(f"backend={backend}")
print(f"samples={len(samples)}")
if len(samples) >= 2:
    dt = (samples[-1].timestamp_ns - samples[0].timestamp_ns) / 1e9
    hz = (len(samples) - 1) / dt if dt > 0 else 0
    print(f"duration={dt:.6f}")
    print(f"effective_hz={hz:.2f}")
PY
