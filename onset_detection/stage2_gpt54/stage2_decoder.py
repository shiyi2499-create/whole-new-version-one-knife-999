"""
Structured decoder for dense Stage 2 outputs.
=============================================

This file now supports:
- exact-N boundary decoding with k-best hypotheses
- exact-K key-slot decoding with k-best hypotheses per segment
- full-segmentation hypothesis generation for global classifier reranking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Stage2ProtocolPrior:
    expected_password_count: int = 5
    expected_password_len: int = 8
    min_segment_s: float = 0.8
    max_segment_s: float = 8.0
    min_iki_s: float = 0.08
    max_iki_s: float = 1.8


@dataclass
class Stage2DecodeConfig:
    min_boundary_gap_s: float = 0.5
    boundary_candidate_topk: int = 96
    key_candidate_topk: int = 64
    boundary_score_weight: float = 1.0
    inside_mass_weight: float = 0.25
    duration_penalty_weight: float = 0.75
    key_score_weight: float = 1.0
    gap_penalty_weight: float = 0.3
    coverage_weight: float = 0.2
    local_max_radius: int = 1
    boundary_beam_size: int = 8
    key_beam_size: int = 6
    n_best_boundaries: int = 8
    n_best_keys_per_segment: int = 3
    n_best_hypotheses: int = 12


@dataclass
class DecodedPasswordSegment:
    start_idx: int
    end_idx: int
    key_indices: list[int]
    score: float


@dataclass
class DenseStage2DecodeResult:
    boundary_indices: list[int]
    segments: list[DecodedPasswordSegment]
    total_score: float
    boundary_score: float = 0.0
    hypothesis_rank: int = 0

    @property
    def password_groups_s(self) -> list[list[float]]:
        out: list[list[float]] = []
        for seg in self.segments:
            times = [float(seg._times_s[i]) for i in seg.key_indices] if hasattr(seg, "_times_s") else []
            out.append(times)
        return out


def _topk_candidates(score: np.ndarray, topk: int) -> np.ndarray:
    if len(score) == 0:
        return np.zeros((0,), dtype=np.int64)
    topk = min(max(1, topk), len(score))
    idx = np.argpartition(score, -topk)[-topk:]
    return np.unique(np.sort(idx.astype(np.int64)))


def _take_topk(items: list[tuple], k: int) -> list[tuple]:
    if not items:
        return []
    items.sort(key=lambda x: float(x[0]), reverse=True)
    return items[: max(1, int(k))]


def _duration_penalty(duration_s: float, prior: Stage2ProtocolPrior) -> float:
    if duration_s < prior.min_segment_s:
        return -((prior.min_segment_s - duration_s) / max(prior.min_segment_s, 1e-6)) ** 2
    if duration_s > prior.max_segment_s:
        return -((duration_s - prior.max_segment_s) / max(prior.max_segment_s, 1e-6)) ** 2
    mid = 0.5 * (prior.min_segment_s + prior.max_segment_s)
    span = max(0.5 * (prior.max_segment_s - prior.min_segment_s), 1e-6)
    return -0.1 * ((duration_s - mid) / span) ** 2


def _segment_score(
    start_idx: int,
    end_idx: int,
    times_s: np.ndarray,
    boundary_score: np.ndarray,
    inside_score: Optional[np.ndarray],
    prior: Stage2ProtocolPrior,
    cfg: Stage2DecodeConfig,
) -> float:
    if end_idx <= start_idx:
        return -1e9
    duration_s = float(times_s[end_idx] - times_s[start_idx])
    score = 0.0
    score += cfg.boundary_score_weight * float(boundary_score[end_idx])
    score += cfg.duration_penalty_weight * _duration_penalty(duration_s, prior)
    if inside_score is not None and end_idx > start_idx + 1:
        score += cfg.inside_mass_weight * float(np.mean(inside_score[start_idx:end_idx]))
    return score


def _reconstruct_boundary_chain(
    states: list[list[list[tuple[float, int, int]]]],
    boundary_candidates: np.ndarray,
    end_rank: int,
    n_seg: int,
) -> tuple[list[int], float]:
    end_c = len(boundary_candidates) - 1
    if end_rank >= len(states[n_seg][end_c]):
        return [], -1e18
    chain = [end_c]
    cur_c = end_c
    cur_rank = end_rank
    score = float(states[n_seg][end_c][end_rank][0])
    for k in range(n_seg, 0, -1):
        _score, prev_c, prev_rank = states[k][cur_c][cur_rank]
        if prev_c < 0:
            return [], -1e18
        chain.append(int(prev_c))
        cur_c = int(prev_c)
        cur_rank = int(prev_rank)
    chain = chain[::-1]
    boundaries = [int(boundary_candidates[c]) for c in chain[1:-1] if c >= 0]
    return boundaries, score


def decode_boundaries_topk(
    times_s: np.ndarray,
    boundary_score: np.ndarray,
    inside_score: Optional[np.ndarray],
    prior: Stage2ProtocolPrior,
    cfg: Stage2DecodeConfig,
) -> list[tuple[list[int], float]]:
    """Choose k-best exact boundary chains with beam DP."""
    T = len(times_s)
    n_seg = prior.expected_password_count
    if T < n_seg + 2:
        return []

    boundary_candidates = _topk_candidates(boundary_score, cfg.boundary_candidate_topk)
    boundary_candidates = np.unique(np.concatenate([
        np.asarray([0, T - 1], dtype=np.int64),
        boundary_candidates,
    ]))
    boundary_candidates.sort()
    C = len(boundary_candidates)

    states: list[list[list[tuple[float, int, int]]]] = [
        [[] for _ in range(C)] for _ in range(n_seg + 1)
    ]

    start_c = 0
    for j in range(1, C):
        seg_start = int(boundary_candidates[start_c])
        seg_end = int(boundary_candidates[j])
        if times_s[seg_end] - times_s[seg_start] < prior.min_segment_s:
            continue
        s = _segment_score(seg_start, seg_end, times_s, boundary_score, inside_score, prior, cfg)
        states[1][j] = [(float(s), start_c, -1)]

    for k in range(2, n_seg + 1):
        for j in range(k, C):
            end_idx = int(boundary_candidates[j])
            hyps: list[tuple[float, int, int]] = []
            for i in range(k - 1, j):
                start_idx = int(boundary_candidates[i])
                if times_s[end_idx] - times_s[start_idx] < max(prior.min_segment_s, cfg.min_boundary_gap_s):
                    continue
                if not states[k - 1][i]:
                    continue
                seg_score = _segment_score(start_idx, end_idx, times_s, boundary_score, inside_score, prior, cfg)
                for prev_rank, (prev_score, _pc, _pr) in enumerate(states[k - 1][i]):
                    hyps.append((float(prev_score + seg_score), int(i), int(prev_rank)))
            states[k][j] = _take_topk(hyps, cfg.boundary_beam_size)

    end_c = C - 1
    if not states[n_seg][end_c]:
        return []

    out: list[tuple[list[int], float]] = []
    for end_rank in range(min(len(states[n_seg][end_c]), cfg.n_best_boundaries)):
        boundaries, score = _reconstruct_boundary_chain(states, boundary_candidates, end_rank, n_seg)
        if boundaries or n_seg <= 1:
            out.append((boundaries, float(score)))
    out = _take_topk([(float(score), boundaries) for boundaries, score in out], cfg.n_best_boundaries)
    return [(boundaries, float(score)) for score, boundaries in out]


def decode_boundaries_dp(
    times_s: np.ndarray,
    boundary_score: np.ndarray,
    inside_score: Optional[np.ndarray],
    prior: Stage2ProtocolPrior,
    cfg: Stage2DecodeConfig,
) -> list[int]:
    hyps = decode_boundaries_topk(times_s, boundary_score, inside_score, prior, cfg)
    return hyps[0][0] if hyps else []


def _extract_key_candidates(key_score: np.ndarray, start_idx: int, end_idx: int, cfg: Stage2DecodeConfig) -> np.ndarray:
    if end_idx <= start_idx:
        return np.zeros((0,), dtype=np.int64)
    segment_score = key_score[start_idx:end_idx]
    if len(segment_score) == 0:
        return np.zeros((0,), dtype=np.int64)
    local = []
    r = max(1, int(cfg.local_max_radius))
    if len(segment_score) <= (2 * r + 1):
        local = list(range(start_idx, end_idx))
    else:
        for i in range(r, len(segment_score) - r):
            v = segment_score[i]
            if np.all(v >= segment_score[i - r:i + r + 1]):
                local.append(start_idx + i)
    if not local:
        local = list(range(start_idx, end_idx))
    local = np.asarray(local, dtype=np.int64)
    scores = key_score[local]
    if len(local) > cfg.key_candidate_topk:
        keep = np.argpartition(scores, -cfg.key_candidate_topk)[-cfg.key_candidate_topk:]
        local = np.sort(local[keep])
    return local


def _reconstruct_key_chain(
    states: list[list[list[tuple[float, int, int]]]],
    cand: np.ndarray,
    end_j: int,
    end_rank: int,
    K: int,
) -> tuple[list[int], float]:
    if end_rank >= len(states[K - 1][end_j]):
        return [], -1e18
    picked = []
    cur_j = int(end_j)
    cur_rank = int(end_rank)
    raw_score = float(states[K - 1][end_j][end_rank][0])
    for k in range(K - 1, -1, -1):
        picked.append(int(cand[cur_j]))
        _score, prev_j, prev_rank = states[k][cur_j][cur_rank]
        if k > 0 and prev_j < 0:
            return [], -1e18
        cur_j = int(prev_j)
        cur_rank = int(prev_rank)
    picked.reverse()
    return picked, raw_score


def decode_exact_k_keys_topk(
    key_score: np.ndarray,
    times_s: np.ndarray,
    start_idx: int,
    end_idx: int,
    prior: Stage2ProtocolPrior,
    cfg: Stage2DecodeConfig,
) -> list[tuple[list[int], float]]:
    """k-best DP to pick exactly expected_password_len key slots inside one segment."""
    K = prior.expected_password_len
    cand = _extract_key_candidates(key_score, start_idx, end_idx, cfg)
    if len(cand) < K:
        dense = np.arange(start_idx, end_idx, dtype=np.int64)
        cand = np.unique(np.concatenate([cand, dense]))
    if len(cand) < K:
        return []

    seg_start_s = float(times_s[start_idx])
    seg_end_s = float(times_s[end_idx - 1]) if end_idx - 1 < len(times_s) else float(times_s[-1])
    target_gap = max((seg_end_s - seg_start_s) / max(K - 1, 1), prior.min_iki_s)

    C = len(cand)
    states: list[list[list[tuple[float, int, int]]]] = [
        [[] for _ in range(C)] for _ in range(K)
    ]

    for j in range(C):
        states[0][j] = [(cfg.key_score_weight * float(key_score[cand[j]]), -1, -1)]

    for k in range(1, K):
        for j in range(k, C):
            tj = float(times_s[cand[j]])
            hyps: list[tuple[float, int, int]] = []
            for i in range(k - 1, j):
                ti = float(times_s[cand[i]])
                gap = tj - ti
                if gap < prior.min_iki_s or gap > prior.max_iki_s:
                    continue
                if not states[k - 1][i]:
                    continue
                gap_pen = -((gap - target_gap) / max(target_gap, 1e-6)) ** 2
                emit = cfg.key_score_weight * float(key_score[cand[j]]) + cfg.gap_penalty_weight * gap_pen
                for prev_rank, (prev_score, _pj, _pr) in enumerate(states[k - 1][i]):
                    hyps.append((float(prev_score + emit), int(i), int(prev_rank)))
            states[k][j] = _take_topk(hyps, cfg.key_beam_size)

    finals: list[tuple[float, list[int]]] = []
    for end_j in range(K - 1, C):
        for end_rank in range(min(len(states[K - 1][end_j]), cfg.n_best_keys_per_segment)):
            picked, raw_score = _reconstruct_key_chain(states, cand, end_j, end_rank, K)
            if not picked:
                continue
            span = float(times_s[picked[-1]] - times_s[picked[0]]) if len(picked) >= 2 else 0.0
            seg_span = max(float(seg_end_s - seg_start_s), 1e-6)
            coverage = span / seg_span
            total_score = float(raw_score) + cfg.coverage_weight * coverage
            finals.append((total_score, picked))
    finals = _take_topk(finals, cfg.n_best_keys_per_segment)
    return [(picked, float(score)) for score, picked in finals]


def decode_exact_k_keys(
    key_score: np.ndarray,
    times_s: np.ndarray,
    start_idx: int,
    end_idx: int,
    prior: Stage2ProtocolPrior,
    cfg: Stage2DecodeConfig,
) -> tuple[list[int], float]:
    hyps = decode_exact_k_keys_topk(key_score, times_s, start_idx, end_idx, prior, cfg)
    return hyps[0] if hyps else ([], -1e9)


def _combine_segment_hypotheses(
    segment_ranges: list[tuple[int, int]],
    segment_key_hypotheses: list[list[tuple[list[int], float]]],
    boundary_score: float,
    times_s: np.ndarray,
    max_results: int,
) -> list[DenseStage2DecodeResult]:
    partials: list[tuple[float, list[DecodedPasswordSegment]]] = [(float(boundary_score), [])]
    for (start_idx, end_idx), key_hyps in zip(segment_ranges, segment_key_hypotheses):
        if not key_hyps:
            key_hyps = [([], -1e9)]
        new_partials: list[tuple[float, list[DecodedPasswordSegment]]] = []
        for prev_score, prev_segments in partials:
            for key_indices, seg_score in key_hyps:
                seg = DecodedPasswordSegment(
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                    key_indices=[int(i) for i in key_indices],
                    score=float(seg_score),
                )
                setattr(seg, "_times_s", times_s)
                new_partials.append((float(prev_score + seg_score), prev_segments + [seg]))
        new_partials = _take_topk(new_partials, max_results)
        partials = new_partials

    results: list[DenseStage2DecodeResult] = []
    for rank, (score, segments) in enumerate(partials):
        boundaries = [int(seg.end_idx) for seg in segments[:-1]]
        results.append(
            DenseStage2DecodeResult(
                boundary_indices=boundaries,
                segments=segments,
                total_score=float(score),
                boundary_score=float(boundary_score),
                hypothesis_rank=int(rank),
            )
        )
    return results


def decode_stage2_dense_topk(
    times_s: np.ndarray,
    key_score: np.ndarray,
    boundary_score: np.ndarray,
    inside_score: Optional[np.ndarray] = None,
    prior: Stage2ProtocolPrior = Stage2ProtocolPrior(),
    cfg: Stage2DecodeConfig = Stage2DecodeConfig(),
) -> list[DenseStage2DecodeResult]:
    times_s = np.asarray(times_s, dtype=np.float64)
    key_score = np.asarray(key_score, dtype=np.float64)
    boundary_score = np.asarray(boundary_score, dtype=np.float64)
    inside_score = None if inside_score is None else np.asarray(inside_score, dtype=np.float64)

    boundary_hyps = decode_boundaries_topk(times_s, boundary_score, inside_score, prior, cfg)
    if not boundary_hyps:
        return []

    all_results: list[DenseStage2DecodeResult] = []
    for boundaries, bdry_score in boundary_hyps[: max(1, cfg.n_best_boundaries)]:
        idx_chain = [0] + list(boundaries) + [len(times_s) - 1]
        segment_ranges = [(int(a), int(b)) for a, b in zip(idx_chain[:-1], idx_chain[1:])]
        segment_key_hypotheses: list[list[tuple[list[int], float]]] = []
        for a, b in segment_ranges:
            key_hyps = decode_exact_k_keys_topk(key_score, times_s, a, b + 1, prior, cfg)
            segment_key_hypotheses.append(key_hyps)
        full_hyps = _combine_segment_hypotheses(
            segment_ranges,
            segment_key_hypotheses,
            boundary_score=float(bdry_score),
            times_s=times_s,
            max_results=cfg.n_best_hypotheses,
        )
        all_results.extend(full_hyps)

    all_results.sort(key=lambda r: float(r.total_score), reverse=True)
    out = all_results[: max(1, cfg.n_best_hypotheses)]
    for rank, hyp in enumerate(out):
        hyp.hypothesis_rank = int(rank)
    return out


def decode_stage2_dense(
    times_s: np.ndarray,
    key_score: np.ndarray,
    boundary_score: np.ndarray,
    inside_score: Optional[np.ndarray] = None,
    prior: Stage2ProtocolPrior = Stage2ProtocolPrior(),
    cfg: Stage2DecodeConfig = Stage2DecodeConfig(),
) -> DenseStage2DecodeResult:
    hyps = decode_stage2_dense_topk(times_s, key_score, boundary_score, inside_score, prior=prior, cfg=cfg)
    if hyps:
        return hyps[0]
    return DenseStage2DecodeResult(boundary_indices=[], segments=[], total_score=-1e9, boundary_score=-1e9, hypothesis_rank=0)
