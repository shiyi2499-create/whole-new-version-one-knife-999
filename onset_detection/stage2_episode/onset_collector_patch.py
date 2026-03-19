"""
onset_collector_patch.py
========================

This file documents the changes needed to onset_collector.py to support
variable-length password collection for the episode-based Stage 2.

The changes are minimal — the existing mixed_training mode already works
for episode-based training. The main enhancement is:

1. Support variable password lengths (not just 8 chars)
2. Support variable password count per trial
3. Record per-attempt timestamps more precisely

These are OPTIONAL improvements. The existing collector already produces
data that the episode-based pipeline can consume, because:
  - Enter-separated password groups already give us episode boundaries
  - The 2-class frame model doesn't care about password length
  - The episode decoder uses temporal gaps, not password count assumptions

CHANGES TO APPLY:
"""

# ── Change 1: Variable password length generation ──
#
# In _generate_fresh_passwords(), replace fixed k=8 with variable length:
#
# OLD:
#   candidate = "".join(rng.choices(PASSWORD_CHARS, k=8))
#
# NEW:
#   if password_len is None:
#       password_len = rng.randint(4, 13)  # 4-12 chars
#   candidate = "".join(rng.choices(PASSWORD_CHARS, k=password_len))
#
# And update the caller to pass a list of lengths.


# ── Change 2: Variable password count per trial ──
#
# In run_mixed_training_mode(), support a range:
#
# OLD:
#   n_passwords=5 (fixed)
#
# NEW:
#   n_passwords_min=3, n_passwords_max=7
#   n_passwords = rng.randint(n_passwords_min, n_passwords_max + 1)


# ── Change 3: Add per-attempt CSV alongside activity_log ──
#
# During the password stage, write an attempts.csv with:
#   attempt_start_ns, submit_ns, prompt_text, typed_text
#
# This is already partially done (events.csv has all keystrokes).
# The main addition is recording the start_ns of each attempt
# (= the timestamp right after the previous Enter press or block start).


# ════════════════════════════════════════════════════════════════
# CONCRETE REPLACEMENT for generate_structured_protocol()
# ════════════════════════════════════════════════════════════════

import random
from typing import Optional


PASSWORD_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789"


def generate_fresh_passwords_variable(
    n_passwords: int,
    rng: random.Random,
    used_prompts: Optional[set] = None,
    min_len: int = 4,
    max_len: int = 12,
) -> list:
    """Generate n_passwords with variable lengths."""
    used = set(used_prompts or set())
    out = []
    for _ in range(n_passwords):
        pw_len = rng.randint(min_len, max_len)
        for _try in range(10000):
            candidate = "".join(rng.choices(PASSWORD_CHARS, k=pw_len))
            if candidate not in used:
                out.append(candidate)
                used.add(candidate)
                break
        else:
            raise RuntimeError("Failed to generate unique password")
    return out


def generate_structured_protocol_variable(
    n_passwords_min: int = 3,
    n_passwords_max: int = 7,
    pw_len_min: int = 4,
    pw_len_max: int = 12,
    seed: int = 42,
    duration_jitter_pct: float = 0.15,
    used_prompts: Optional[set] = None,
) -> list:
    """
    Generate a structured protocol with variable password count and lengths.

    This is a drop-in replacement for generate_structured_protocol().
    """
    rng = random.Random(seed)

    n_passwords = rng.randint(n_passwords_min, n_passwords_max)

    # Base protocol is the same, just typing_2 duration scales with n_passwords
    base_pw_duration = 8.0 * n_passwords  # ~8s per password is generous

    protocol = [
        {"activity": "idle", "duration_s": 12.0, "label": "idle_1"},
        {"activity": "trackpad_move", "duration_s": 18.0, "label": "trackpad_move_1"},
        {"activity": "keyboard", "duration_s": 35.0, "label": "typing_1",
         "typing_style": "free",
         "prompt_instructions": "Type whatever you want – random words, sentences, etc."},
        {"activity": "trackpad_click", "duration_s": 18.0, "label": "trackpad_click_1"},
        {"activity": "idle", "duration_s": 12.0, "label": "idle_2"},
        {"activity": "keyboard", "duration_s": base_pw_duration, "label": "typing_2",
         "typing_style": "password",
         "prompt_instructions": ""},
        {"activity": "shake", "duration_s": 12.0, "label": "shake_1"},
    ]

    # Apply jitter to non-password segments
    for entry in protocol:
        if entry.get("typing_style") != "password":
            scale = 1.0 + rng.uniform(-duration_jitter_pct, duration_jitter_pct)
            entry["duration_s"] = max(3.0, round(entry["duration_s"] * scale, 1))

    # Generate variable-length passwords for the password stage
    passwords = generate_fresh_passwords_variable(
        n_passwords=n_passwords,
        rng=rng,
        used_prompts=used_prompts,
        min_len=pw_len_min,
        max_len=pw_len_max,
    )

    for entry in protocol:
        if entry.get("typing_style") == "password":
            entry["prompts"] = passwords
            entry["prompt_instructions"] = (
                f"请慢速、仔细地输入下面每一条密码；每输入完一条后按一次 Enter。\n"
                f"这一阶段不设硬性倒计时，输完第 {n_passwords} 条并按下 Enter 后自动进入下一阶段：\n"
                + "\n".join(f"  {i+1}. {pw} ({len(pw)}字符)"
                           for i, pw in enumerate(passwords))
            )

    return protocol
