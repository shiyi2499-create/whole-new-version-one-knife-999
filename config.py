"""
Keystroke Vibration Data Collector - Configuration
===================================================
All tunable parameters for data collection.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class CollectorConfig:
    # ── Sampling & Rate Monitor ──────────────────────────────
    # MacBook Air M-series SPU actual rate is ~130 Hz
    EXPECTED_SAMPLE_RATE_HZ: int = 130
    MIN_ACCEPTABLE_RATE_HZ: int = 120        # ~92% of 130; stop & alert if drops below
    RATE_CHECK_INTERVAL_SEC: float = 1.0     # how often to check rate
    RATE_WINDOW_SEC: float = 3.0             # sliding window for rate calc

    # ── Keystroke Window (for post-processing) ───────────────
    PRE_TRIGGER_MS: int = 100                # ms before keypress to capture
    POST_TRIGGER_MS: int = 200               # ms after keypress to capture
    # total window = 300ms ≈ 39 samples @ 130Hz

    # ── Single-Key Mode ──────────────────────────────────────
    REPEATS_PER_KEY: int = 50                # presses per key in single-key mode
    PAUSE_BETWEEN_KEYS_SEC: float = 2.0      # pause before switching to next key

    # 36 alphanumeric keys in 6 groups + 1 special key group + 1 hard-key boost group
    KEY_GROUPS: dict = field(default_factory=lambda: {
        1: {"name": "Group 1 (a-f)",       "keys": list("abcdef")},
        2: {"name": "Group 2 (g-l)",       "keys": list("ghijkl")},
        3: {"name": "Group 3 (m-r)",       "keys": list("mnopqr")},
        4: {"name": "Group 4 (s-x)",       "keys": list("stuvwx")},
        5: {"name": "Group 5 (y-z,0-3)",   "keys": list("yz0123")},
        6: {"name": "Group 6 (4-9)",       "keys": list("456789")},
        7: {"name": "Group 7 (special)",   "keys": ["space", "enter", "backspace",
                                                      ",", ".", "shift"]},
        # Group 8: centre-keyboard hard keys with low per-key accuracy.
        # Collect 100 extra presses (round 3) to boost their sample count to ~200.
        8: {"name": "Group 8 (hard keys)", "keys": ["d", "f", "h", "j", "m", "n"]},
    })

    # Will be set at runtime based on group selection
    KEY_LIST: list = field(default_factory=list)

    # ── Data Paths ───────────────────────────────────────────
    DATA_ROOT: str = "data"
    RAW_DIR: str = "data/raw"
    PROCESSED_DIR: str = "data/processed"
    ROUND: int = 1                           # legacy round index (used if RAW_SUBDIR empty)
    RAW_SUBDIR: str = ""                     # preferred raw-data folder under data/raw (e.g. single_key)

    # ── Participant ──────────────────────────────────────────
    PARTICIPANT_ID: str = "p01"

    def __post_init__(self):
        # Prefer named folders (single_key / boost / free_type). Keep round{N}
        # behavior for backward compatibility.
        if self.RAW_SUBDIR:
            self.RAW_DIR = os.path.join(self.DATA_ROOT, "raw", self.RAW_SUBDIR)
        else:
            self.RAW_DIR = os.path.join(self.DATA_ROOT, "raw", f"round{self.ROUND}")
        os.makedirs(self.RAW_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DIR, exist_ok=True)

    def session_prefix(self, mode: str, group: int = 0, part: int = 0) -> str:
        """Generate a filename prefix like: p01_single_key_g1_20260306_143022"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if group > 0:
            return f"{self.PARTICIPANT_ID}_{mode}_g{group}_{ts}"
        if part > 0:
            return f"{self.PARTICIPANT_ID}_{mode}_part{part}_{ts}"
        return f"{self.PARTICIPANT_ID}_{mode}_{ts}"
