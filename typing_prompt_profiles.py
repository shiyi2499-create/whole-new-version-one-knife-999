"""
Prompt profiles for guided text collection.

Profiles:
- sentence: existing natural-language free_type prompts with spaces
- continuous: the same prompts with spaces removed, useful as a no-space bridge
- password: fixed password-like lowercase+digit strings
"""

from __future__ import annotations

import re

from typing_prompts import PROMPTS, PROMPTS_PER_PART, TOTAL_PARTS


VALID_RE = re.compile(r"^[a-z0-9 ]+$")


def _normalize_continuous(prompt: str) -> str:
    return prompt.replace(" ", "")


def _build_password_prompts() -> list[str]:
    prefix = [
        "alpha", "amber", "aster", "atlas", "basil", "cinder", "cobalt", "comet",
        "delta", "ember", "falcon", "forest", "galaxy", "harbor", "ivory", "juniper",
        "kepler", "lantern", "magnet", "matrix", "nebula", "onyx", "orbit", "phoenix",
        "quartz", "raven", "rocket", "shadow", "signal", "silver", "solar", "summit",
    ]
    suffix = [
        "bridge", "clock", "drift", "echo", "flame", "garden", "glow", "haven",
        "jolt", "lattice", "meadow", "nova", "oasis", "pixel", "prism", "quest",
        "ridge", "river", "spark", "stone", "trail", "vector", "vista", "whisper",
        "window", "zenith",
    ]
    numbers = [
        "07", "11", "14", "18", "21", "24", "27", "30", "33", "36", "42", "48",
        "53", "57", "61", "64", "68", "72", "75", "79", "82", "86", "90", "94",
        "105", "117", "128", "203", "217", "304", "315", "426",
    ]
    prompts = []
    seen = set()
    for i in range(PROMPTS_PER_PART * TOTAL_PARTS):
        a = prefix[i % len(prefix)]
        b = suffix[(i * 3) % len(suffix)]
        n = numbers[(i * 5) % len(numbers)]
        if i % 4 == 0:
            s = f"{a}{b}{n}"
        elif i % 4 == 1:
            s = f"{a}{n}{b}"
        elif i % 4 == 2:
            s = f"{n}{a}{b}"
        else:
            s = f"{a}{b}{i % 10}{n[-1]}"
        if s in seen:
            s = f"{a}{b}{n}{i % 10}"
        seen.add(s)
        prompts.append(s)
    return prompts


CONTINUOUS_PROMPTS = [_normalize_continuous(p) for p in PROMPTS]
PASSWORD_PROMPTS = _build_password_prompts()


PROMPT_PROFILES = {
    "sentence": {
        "name": "Sentence",
        "description": "Natural-language guided free typing with spaces.",
        "prompts": PROMPTS,
    },
    "continuous": {
        "name": "Continuous",
        "description": "No-space bridge strings derived from the sentence prompts.",
        "prompts": CONTINUOUS_PROMPTS,
    },
    "password": {
        "name": "Password",
        "description": "Password-like lowercase+digit strings with no spaces.",
        "prompts": PASSWORD_PROMPTS,
    },
}


def validate_profiles() -> None:
    expected = PROMPTS_PER_PART * TOTAL_PARTS
    for key, info in PROMPT_PROFILES.items():
        prompts = info["prompts"]
        if len(prompts) != expected:
            raise ValueError(f"{key} prompt count must be {expected}, got {len(prompts)}")
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"{key} prompts contain duplicates")
        for p in prompts:
            if not VALID_RE.fullmatch(p):
                raise ValueError(f"{key} prompt contains invalid chars: {p!r}")


def get_prompt_profile(name: str) -> dict:
    name = (name or "sentence").strip().lower()
    if name not in PROMPT_PROFILES:
        raise KeyError(f"Unknown prompt profile: {name}")
    return PROMPT_PROFILES[name]


validate_profiles()
