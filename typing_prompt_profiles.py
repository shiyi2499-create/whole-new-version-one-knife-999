"""
Prompt profiles for guided text collection.

Profiles:
- sentence: existing natural-language free_type prompts with spaces
- continuous: the same prompts with spaces removed, useful as a no-space bridge
- password: fixed password-like lowercase+digit strings (len configurable, 200 prompts)
"""

from __future__ import annotations

import random
import re

from typing_prompts import PROMPTS, PROMPTS_PER_PART, TOTAL_PARTS


VALID_RE = re.compile(r"^[a-z0-9 ]+$")


def _normalize_continuous(prompt: str) -> str:
    return prompt.replace(" ", "")


def _build_password_prompts(total: int = 200, length: int = 8) -> list[str]:
    rng = random.Random(20260316 + int(length) * 1009)
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    prompts = []
    seen = set()
    while len(prompts) < total:
        s = "".join(rng.choice(alphabet) for _ in range(length))
        # Keep the password-like strings realistic enough: at least one letter
        # and one digit, all lowercase, no separators.
        if not any(ch.isalpha() for ch in s):
            continue
        if not any(ch.isdigit() for ch in s):
            continue
        if s in seen:
            continue
        seen.add(s)
        prompts.append(s)
    return prompts


CONTINUOUS_PROMPTS = [_normalize_continuous(p) for p in PROMPTS]
PASSWORD_PROMPTS = _build_password_prompts(length=8)


PROMPT_PROFILES = {
    "sentence": {
        "name": "Sentence",
        "description": "Natural-language guided free typing with spaces.",
        "prompts": PROMPTS,
        "default_groups": TOTAL_PARTS,
        "unit_name": "sentence",
        "unit_name_plural": "sentences",
    },
    "continuous": {
        "name": "Continuous",
        "description": "No-space bridge strings derived from the sentence prompts.",
        "prompts": CONTINUOUS_PROMPTS,
        "default_groups": TOTAL_PARTS,
        "unit_name": "string",
        "unit_name_plural": "strings",
    },
    "password": {
        "name": "Password",
        "description": "Password-like lowercase+digit strings (len=8, 200 total, 20 groups).",
        "prompts": PASSWORD_PROMPTS,
        "default_groups": 20,
        "unit_name": "password",
        "unit_name_plural": "passwords",
        "password_length": 8,
    },
}


def validate_profiles() -> None:
    for key, info in PROMPT_PROFILES.items():
        prompts = info["prompts"]
        expected = info.get("default_groups")
        if key == "password":
            if len(prompts) != 200:
                raise ValueError(f"{key} prompt count must be 200, got {len(prompts)}")
        else:
            expected_total = PROMPTS_PER_PART * TOTAL_PARTS
            if len(prompts) != expected_total:
                raise ValueError(f"{key} prompt count must be {expected_total}, got {len(prompts)}")
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"{key} prompts contain duplicates")
        for p in prompts:
            if not VALID_RE.fullmatch(p):
                raise ValueError(f"{key} prompt contains invalid chars: {p!r}")
        if key == "password":
            pw_len = int(info.get("password_length", 8))
            if not all(len(p) == pw_len for p in prompts):
                raise ValueError(f"password prompts must all be length {pw_len}")
        if not isinstance(expected, int) or expected <= 0:
            raise ValueError(f"{key} must define a positive default_groups")


def get_prompt_profile(name: str, password_length: int = 8) -> dict:
    name = (name or "sentence").strip().lower()
    if name not in PROMPT_PROFILES:
        raise KeyError(f"Unknown prompt profile: {name}")
    if name != "password":
        return PROMPT_PROFILES[name]

    pw_len = int(password_length)
    if pw_len < 4:
        raise ValueError("password_length must be >= 4")
    prompts = _build_password_prompts(length=pw_len)
    return {
        "name": f"Password (len={pw_len})",
        "description": f"Password-like lowercase+digit strings (len={pw_len}, 200 total, 20 groups).",
        "prompts": prompts,
        "default_groups": 20,
        "unit_name": "password",
        "unit_name_plural": "passwords",
        "password_length": pw_len,
    }


validate_profiles()
