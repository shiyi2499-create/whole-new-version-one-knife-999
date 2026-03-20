from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class EpisodeEval:
    episode_id: str
    session_id: str
    reference: str
    prediction: str
    char_top1: float
    char_top3: float
    char_top5: float
    cer: float


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def cer(a: str, b: str) -> float:
    return levenshtein(a, b) / max(len(a), 1)


def char_topk_from_logits(logits: np.ndarray, labels: np.ndarray, ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    if len(labels) == 0:
        return {f"char_top{k}": 0.0 for k in ks}
    ranks = np.argsort(logits, axis=1)[:, ::-1]
    out = {}
    for k in ks:
        hits = 0
        for row, y in zip(ranks, labels):
            if int(y) in row[:k]:
                hits += 1
        out[f"char_top{k}"] = hits / max(len(labels), 1)
    return out


def aggregate_episode_results(results: Iterable[dict]) -> dict:
    results = list(results)
    if not results:
        return {
            "num_episodes": 0,
            "num_chars": 0,
            "char_top1": 0.0,
            "char_top3": 0.0,
            "char_top5": 0.0,
            "cer": 1.0,
            "exact_match": 0.0,
        }

    num_chars = int(sum(len(r["reference"]) for r in results))
    top1 = 0.0
    top3 = 0.0
    top5 = 0.0
    total_edits = 0
    exact = 0
    for r in results:
        n = max(len(r["reference"]), 1)
        top1 += float(r["char_top1"]) * n
        top3 += float(r["char_top3"]) * n
        top5 += float(r["char_top5"]) * n
        total_edits += int(levenshtein(r["reference"], r["prediction"]))
        exact += int(r["reference"] == r["prediction"])
    return {
        "num_episodes": len(results),
        "num_chars": num_chars,
        "char_top1": top1 / max(num_chars, 1),
        "char_top3": top3 / max(num_chars, 1),
        "char_top5": top5 / max(num_chars, 1),
        "cer": total_edits / max(num_chars, 1),
        "exact_match": exact / max(len(results), 1),
    }
