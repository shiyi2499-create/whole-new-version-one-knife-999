from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from scipy.signal import resample

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "phase3_password_inception")):
    if p not in sys.path:
        sys.path.insert(0, p)

from onset_detection.stage2_episode.data.loaders import SessionLoader, discover_sessions
from phase3_password_inception.run_password_closure_inception import supported_key


DEFAULT_PRE_PAD_MS = 250.0
DEFAULT_POST_PAD_MS = 350.0
DEFAULT_PRE_TRIGGER_MS = 100.0
DEFAULT_POST_TRIGGER_MS = 200.0


@dataclass
class PasswordEpisode:
    session_path: str
    session_id: str
    episode_index: int
    episode_id: str
    prompt: str
    password: str
    imu: np.ndarray
    timestamps_ns: np.ndarray
    key_frames: np.ndarray
    key_timestamps_ns: np.ndarray
    chars: list[str]
    sample_rate_hz: float

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["imu"] = self.imu.tolist()
        d["timestamps_ns"] = self.timestamps_ns.tolist()
        d["key_frames"] = self.key_frames.tolist()
        d["key_timestamps_ns"] = self.key_timestamps_ns.tolist()
        return d


def estimate_sample_rate_hz(timestamps_ns: np.ndarray) -> float:
    if len(timestamps_ns) < 3:
        return 200.0
    diffs = np.diff(timestamps_ns.astype(np.int64))
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 200.0
    median_dt_ns = float(np.median(diffs))
    return float(1e9 / max(median_dt_ns, 1.0))


def _safe_prompt(prompts: list[str], index: int) -> str:
    if 0 <= index < len(prompts):
        return str(prompts[index])
    return ""


def _filter_supported_keys(keys: list[dict]) -> list[dict]:
    out = []
    for item in keys:
        key = str(item["key"]).lower().strip()
        if supported_key(key):
            out.append({"ts": int(item["ts"]), "key": key})
    return out


def build_password_episodes(
    input_dir: str,
    pre_pad_ms: float = DEFAULT_PRE_PAD_MS,
    post_pad_ms: float = DEFAULT_POST_PAD_MS,
    min_len: int = 4,
) -> list[PasswordEpisode]:
    episodes: list[PasswordEpisode] = []
    for session_path in discover_sessions(input_dir):
        loader = SessionLoader(session_path)
        ts_all, imu_all = loader.get_imu()
        if len(ts_all) == 0:
            continue

        block = loader.get_password_block() or {}
        prompts = block.get("prompts", []) or []
        groups = loader.split_password_groups_from_enters()
        session_id = Path(session_path).name
        sample_rate_hz = estimate_sample_rate_hz(ts_all)

        pre_pad_ns = int(round(pre_pad_ms * 1e6))
        post_pad_ns = int(round(post_pad_ms * 1e6))

        for episode_index, group in enumerate(groups):
            keys = _filter_supported_keys(group.get("keys", []))
            if len(keys) < min_len:
                continue
            start_ns = int(keys[0]["ts"]) - pre_pad_ns
            end_ns = int(keys[-1]["ts"]) + post_pad_ns
            mask = (ts_all >= start_ns) & (ts_all <= end_ns)
            idx = np.where(mask)[0]
            if len(idx) < 8:
                continue

            ep_ts = ts_all[idx].astype(np.int64)
            ep_imu = imu_all[idx].astype(np.float32)
            local_frames = []
            local_ts = []
            chars: list[str] = []
            for item in keys:
                evt_ts = int(item["ts"])
                frame = int(np.searchsorted(ep_ts, evt_ts, side="left"))
                frame = min(max(frame, 0), len(ep_ts) - 1)
                local_frames.append(frame)
                local_ts.append(evt_ts)
                chars.append(str(item["key"]))

            password = "".join(chars)
            episodes.append(
                PasswordEpisode(
                    session_path=str(session_path),
                    session_id=session_id,
                    episode_index=episode_index,
                    episode_id=f"{session_id}::ep{episode_index:02d}",
                    prompt=_safe_prompt(prompts, episode_index),
                    password=password,
                    imu=ep_imu,
                    timestamps_ns=ep_ts,
                    key_frames=np.asarray(local_frames, dtype=np.int64),
                    key_timestamps_ns=np.asarray(local_ts, dtype=np.int64),
                    chars=chars,
                    sample_rate_hz=sample_rate_hz,
                )
            )
    return episodes


class EpisodeListDataset:
    def __init__(self, episodes: list[PasswordEpisode]):
        self.episodes = episodes

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: int) -> PasswordEpisode:
        return self.episodes[index]


ALL_CLASSES = np.array(list("0123456789abcdefghijklmnopqrstuvwxyz"), dtype="<U1")


def split_by_session(
    episodes: list[PasswordEpisode],
    val_ratio: float = 0.25,
    seed: int = 42,
    holdout_session_ids: Optional[Iterable[str]] = None,
) -> tuple[list[PasswordEpisode], list[PasswordEpisode], list[str], list[str]]:
    session_ids = sorted({ep.session_id for ep in episodes})
    if not session_ids:
        return [], [], [], []

    if holdout_session_ids:
        requested = [str(x) for x in holdout_session_ids]
        val_sessions = sorted({
            s for s in session_ids
            if any((req == s) or (req in s) for req in requested)
        })
        if not val_sessions:
            raise ValueError(f"None of holdout_session_ids found: {list(holdout_session_ids)}")
    else:
        rng = random.Random(seed)
        session_ids_shuf = session_ids[:]
        rng.shuffle(session_ids_shuf)
        n_val = max(1, int(round(len(session_ids_shuf) * val_ratio)))
        val_sessions = sorted(session_ids_shuf[:n_val])

    train_sessions = sorted([s for s in session_ids if s not in set(val_sessions)])
    if not train_sessions:
        train_sessions = val_sessions[:-1]
        val_sessions = val_sessions[-1:]

    train_eps = [ep for ep in episodes if ep.session_id in set(train_sessions)]
    val_eps = [ep for ep in episodes if ep.session_id in set(val_sessions)]
    return train_eps, val_eps, train_sessions, val_sessions


def _resample_numpy(values: np.ndarray, target_len: int) -> np.ndarray:
    out = resample(values, target_len, axis=0)
    if np.iscomplexobj(out):
        out = np.real(out)
    return np.asarray(out, dtype=np.float32)


def extract_fixed_window(
    episode: PasswordEpisode,
    center_frame: int,
    pre_ms: float = DEFAULT_PRE_TRIGGER_MS,
    post_ms: float = DEFAULT_POST_TRIGGER_MS,
    target_len: int = 57,
) -> Optional[np.ndarray]:
    if len(episode.imu) < 2:
        return None
    pre_frames = int(round(pre_ms / 1000.0 * episode.sample_rate_hz))
    post_frames = int(round(post_ms / 1000.0 * episode.sample_rate_hz))
    lo = max(0, int(center_frame) - pre_frames)
    hi = min(len(episode.imu), int(center_frame) + post_frames)
    if hi - lo < 3:
        return None
    return _resample_numpy(episode.imu[lo:hi], target_len)


def windows_from_episodes(
    episodes: list[PasswordEpisode],
    class_to_idx: dict[str, int],
    target_len: int = 57,
    pre_ms: float = DEFAULT_PRE_TRIGGER_MS,
    post_ms: float = DEFAULT_POST_TRIGGER_MS,
) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    labels = []
    for ep in episodes:
        for frame, char in zip(ep.key_frames.tolist(), ep.chars):
            if char not in class_to_idx:
                continue
            win = extract_fixed_window(ep, int(frame), pre_ms=pre_ms, post_ms=post_ms, target_len=target_len)
            if win is None:
                continue
            windows.append(win.astype(np.float32))
            labels.append(class_to_idx[char])
    if not windows:
        raise RuntimeError("No fixed windows could be extracted from the provided episodes.")
    return np.stack(windows).astype(np.float32), np.asarray(labels, dtype=np.int64)


def compute_channel_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    means = windows.mean(axis=(0, 1)).astype(np.float32)
    stds = windows.std(axis=(0, 1)).astype(np.float32)
    stds = np.maximum(stds, 1e-6)
    return means, stds


def describe_episodes(episodes: list[PasswordEpisode]) -> dict:
    lengths = [len(ep.chars) for ep in episodes]
    return {
        "num_episodes": len(episodes),
        "num_sessions": len({ep.session_id for ep in episodes}),
        "num_keys": int(sum(lengths)),
        "episode_lengths": lengths,
        "sessions": sorted({ep.session_id for ep in episodes}),
    }


def save_split_manifest(path: str, train_sessions: list[str], val_sessions: list[str], episodes: list[PasswordEpisode]):
    payload = {
        "train_sessions": train_sessions,
        "val_sessions": val_sessions,
        "episodes": [
            {
                "episode_id": ep.episode_id,
                "session_id": ep.session_id,
                "password": ep.password,
                "prompt": ep.prompt,
                "num_keys": len(ep.chars),
            }
            for ep in episodes
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
