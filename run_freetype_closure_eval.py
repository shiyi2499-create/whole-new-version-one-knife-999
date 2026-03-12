"""
run_freetype_closure_eval.py
============================
Minimal free_type closure bring-up/evaluation pipeline:

1) Data quality check on raw free_type sessions
2) Build an independent free_type dataset (kept separate from merged_dataset.npz)
3) Run existing single-key Transformer on free_type keystrokes (full probs/logits)
4) Evaluate raw argmax vs phase3 decoder
5) Add lightweight calibration (temperature + prior + special-key bias)
6) Report open-vocab and prompt-constrained decoder results

Run:
  python3 run_freetype_closure_eval.py
  python3 run_freetype_closure_eval.py --rounds free_type
"""

import os
import re
import csv
import json
import math
import argparse
import platform
from typing import Optional
from dataclasses import dataclass
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn

from preprocessor import Preprocessor, WindowConfig, resolve_data_dirs
from phase3_decoder import NgramLanguageModel, WordDecoder, SentenceDecoder
from typing_prompts import PROMPTS as ALL_PROMPTS


DEVICE = torch.device("cpu")

MODEL_PATH = "results/transformer_final.pt"
SCALER_PATH = "results/transformer_scaler.npz"
MERGED_SINGLE_KEY_PATH = "data/processed/merged_dataset.npz"
FREE_TYPE_DATASET_PATH = "data/processed/free_type_dataset.npz"
REPORT_PATH = "results/free_type_closure_report.json"
BASELINE_REPORT_27_PATH = "results/free_type_closure_report_27sent.json"

WORD_BOUNDARY_KEYS = {"space"}
SENTENCE_BOUNDARY_KEYS = {"enter", "return"}
CONTROL_SKIP_KEYS = {
    "shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc", "delete",
    "left", "right", "up", "down",
    "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}


def resolve_torch_device(device: str = "auto") -> torch.device:
    req = (device or "auto").lower()
    if req == "auto":
        if platform.system() == "Darwin":
            req = "cpu"
        elif torch.cuda.is_available():
            req = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            req = "mps"
        else:
            req = "cpu"

    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda but CUDA is not available.")
    elif req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Requested device=mps but MPS is not available.")
    elif req != "cpu":
        raise ValueError(f"Unsupported device: {device}. Use one of auto/cpu/mps/cuda.")

    return torch.device(req)


def set_device(device: str = "auto") -> torch.device:
    global DEVICE
    DEVICE = resolve_torch_device(device)
    return DEVICE


class TransformerClassifier(nn.Module):
    def __init__(self, n_timesteps=39, n_channels=6, n_classes=42,
                 d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, n_timesteps, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.35),
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)


@dataclass
class SessionQuality:
    session_prefix: str
    is_valid: bool
    issues: list[str]
    sensor_samples: int
    effective_hz: float
    median_hz: float
    hz_p01: float
    hz_p99: float
    dt_nonpositive: int
    gaps_20ms: int
    press_count: int
    prompt_rows: int
    prompt_yes_rows: int
    prompt_no_rows: int
    reconstructed_sentences: int
    align_prompt: int
    align_typed: int
    spaces: int
    enters: int
    backspaces: int
    digit_presses: int
    alpha_presses: int
    windows_extracted: int
    valid_sample_rate: float
    prompt_texts: list[str]
    typed_texts: list[str]
    prompt_matches: list[str]
    reconstructed_texts: list[str]


def discover_freetype_sessions(rounds: list[str]) -> list[str]:
    sessions = []
    data_dirs = resolve_data_dirs(rounds, data_root="data/raw")
    for rd in data_dirs:
        if not os.path.isdir(rd):
            continue
        for f in sorted(os.listdir(rd)):
            if "_free_type_" in f and f.endswith("_sensor.csv"):
                prefix = os.path.join(rd, f.replace("_sensor.csv", ""))
                if os.path.exists(prefix + "_events.csv"):
                    sessions.append(prefix)
    return sorted(sessions)


def parse_press_events(events_path: str) -> list[dict]:
    rows = []
    with open(events_path, "r") as f:
        for row in csv.DictReader(f):
            if row["event_type"] != "press":
                continue
            rows.append({
                "timestamp_ns": int(row["timestamp_ns"]),
                "key": row["key"].lower(),
            })
    return rows


def normalize_text(s: str) -> str:
    return " ".join(s.lower().split())


def keys_to_text(keys: list[str]) -> str:
    buf = []
    for k in keys:
        if k in SENTENCE_BOUNDARY_KEYS:
            break
        if k == "backspace":
            if buf:
                buf.pop()
            continue
        if k == "space":
            buf.append(" ")
            continue
        if len(k) == 1:
            buf.append(k)
    return normalize_text("".join(buf))


def reconstruct_sentences_from_events(press_events: list[dict]) -> list[str]:
    sentences = []
    cur_keys = []
    for evt in press_events:
        key = evt["key"]
        cur_keys.append(key)
        if key in SENTENCE_BOUNDARY_KEYS:
            sentences.append(keys_to_text(cur_keys))
            cur_keys = []
    tail = keys_to_text(cur_keys)
    if tail:
        sentences.append(tail)
    return sentences


def sensor_rate_stats(sensor_path: str) -> dict:
    ts = []
    with open(sensor_path, "r") as f:
        for row in csv.DictReader(f):
            ts.append(int(row["timestamp_ns"]))
    ts = np.array(ts, dtype=np.int64)
    dt = np.diff(ts)
    pos_dt = dt[dt > 0]
    effective_hz = (len(ts) - 1) / max((ts[-1] - ts[0]) / 1e9, 1e-9)
    if len(pos_dt):
        hz = 1e9 / pos_dt
        median_hz = float(np.median(hz))
        hz_p01 = float(np.percentile(hz, 1))
        hz_p99 = float(np.percentile(hz, 99))
    else:
        median_hz = hz_p01 = hz_p99 = 0.0
    return {
        "sensor_samples": int(len(ts)),
        "effective_hz": float(effective_hz),
        "median_hz": median_hz,
        "hz_p01": hz_p01,
        "hz_p99": hz_p99,
        "dt_nonpositive": int(np.sum(dt <= 0)),
        "gaps_20ms": int(np.sum(dt > 20_000_000)),
    }


def load_prompts_rows(prompts_path: str) -> list[dict]:
    if not os.path.exists(prompts_path):
        return []
    with open(prompts_path, "r") as f:
        return list(csv.DictReader(f))


def assess_session(session_prefix: str, wcfg: WindowConfig) -> SessionQuality:
    sensor_path = session_prefix + "_sensor.csv"
    events_path = session_prefix + "_events.csv"
    prompts_path = session_prefix + "_prompts.csv"

    rate = sensor_rate_stats(sensor_path)
    press_events = parse_press_events(events_path)
    keys = [e["key"] for e in press_events]
    reconstructed = reconstruct_sentences_from_events(press_events)

    prompt_rows = load_prompts_rows(prompts_path)
    prompt_texts = [normalize_text(r["prompt_text"]) for r in prompt_rows]
    typed_texts = [normalize_text(r["typed_text"]) for r in prompt_rows]
    prompt_matches = [str(r.get("match", "")).strip().upper() for r in prompt_rows]
    prompt_yes_rows = sum(1 for m in prompt_matches if m == "YES")
    prompt_no_rows = sum(1 for m in prompt_matches if m == "NO")

    align_prompt = sum(1 for a, b in zip(reconstructed, prompt_texts) if a == b)
    align_typed = sum(1 for a, b in zip(reconstructed, typed_texts) if a == b)

    proc = Preprocessor(session_prefix=session_prefix, output_dir="data/processed/freetype_tmp", window_cfg=wcfg)
    proc.load()
    proc.extract_windows()
    windows_extracted = len(proc.windows)

    issues = []
    if rate["effective_hz"] < 100.0:
        issues.append(f"low_effective_rate={rate['effective_hz']:.1f}Hz")
    if rate["dt_nonpositive"] > 0:
        issues.append(f"nonpositive_dt={rate['dt_nonpositive']}")
    if len(prompt_rows) == 0:
        issues.append("missing_prompts_csv")
    if len(reconstructed) != len(prompt_rows):
        issues.append(f"sentence_count_mismatch reconstructed={len(reconstructed)} prompts={len(prompt_rows)}")
    if len(prompt_rows) > 0 and align_typed / len(prompt_rows) < 0.9:
        issues.append(f"low_alignment_to_typed={align_typed}/{len(prompt_rows)}")
    if windows_extracted < len(press_events) * 0.95:
        issues.append(f"low_window_yield={windows_extracted}/{len(press_events)}")

    is_valid = len(issues) == 0

    key_counter = Counter(keys)
    return SessionQuality(
        session_prefix=session_prefix,
        is_valid=is_valid,
        issues=issues,
        sensor_samples=rate["sensor_samples"],
        effective_hz=rate["effective_hz"],
        median_hz=rate["median_hz"],
        hz_p01=rate["hz_p01"],
        hz_p99=rate["hz_p99"],
        dt_nonpositive=rate["dt_nonpositive"],
        gaps_20ms=rate["gaps_20ms"],
        press_count=len(press_events),
        prompt_rows=len(prompt_rows),
        prompt_yes_rows=prompt_yes_rows,
        prompt_no_rows=prompt_no_rows,
        reconstructed_sentences=len(reconstructed),
        align_prompt=align_prompt,
        align_typed=align_typed,
        spaces=key_counter["space"],
        enters=key_counter["enter"] + key_counter["return"],
        backspaces=key_counter["backspace"],
        digit_presses=sum(v for k, v in key_counter.items() if len(k) == 1 and k.isdigit()),
        alpha_presses=sum(v for k, v in key_counter.items() if len(k) == 1 and k.isalpha()),
        windows_extracted=windows_extracted,
        valid_sample_rate=windows_extracted / max(1, len(press_events)),
        prompt_texts=prompt_texts,
        typed_texts=typed_texts,
        prompt_matches=prompt_matches,
        reconstructed_texts=reconstructed,
    )


def load_model_and_scaler(model_path: str = MODEL_PATH,
                          scaler_path: str = SCALER_PATH,
                          base_ckpt_path: str = MODEL_PATH):
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError("Missing transformer model/scaler. Run run_real_freetype.py or train first.")

    loaded = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if isinstance(loaded, dict) and "model_state" in loaded:
        ckpt = loaded
        model_state = loaded["model_state"]
    elif isinstance(loaded, dict):
        # Support state_dict-only fine-tuned weights by borrowing architecture metadata
        # from a full checkpoint (default: transformer_final.pt).
        if not os.path.exists(base_ckpt_path):
            raise FileNotFoundError(f"State-dict model given but missing base checkpoint: {base_ckpt_path}")
        ckpt = torch.load(base_ckpt_path, map_location=DEVICE, weights_only=False)
        if not (isinstance(ckpt, dict) and "model_state" in ckpt):
            raise RuntimeError("Base checkpoint must contain model metadata and model_state.")
        model_state = loaded
    else:
        raise RuntimeError("Unsupported model file format.")

    model = TransformerClassifier(
        n_timesteps=ckpt["n_timesteps"],
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
    ).to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()

    scaler = np.load(scaler_path)
    means = scaler["means"].astype(np.float32)
    stds = scaler["stds"].astype(np.float32)
    classes = np.array(ckpt["classes"])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return model, classes, class_to_idx, means, stds


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=1, keepdims=True)


def infer_logits_probs(model: nn.Module, X: np.ndarray, means: np.ndarray, stds: np.ndarray,
                       batch_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    Xn = X.astype(np.float32).copy()
    for ch in range(Xn.shape[2]):
        Xn[:, :, ch] = (Xn[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)

    logits_list = []
    with torch.no_grad():
        for i in range(0, len(Xn), batch_size):
            xb = torch.from_numpy(Xn[i:i + batch_size]).to(DEVICE)
            out = model(xb).cpu().numpy()
            logits_list.append(out)
    logits = np.concatenate(logits_list, axis=0)
    probs = softmax_np(logits)
    return logits, probs


def levenshtein(seq1, seq2) -> int:
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )
    return int(dp[n, m])


def compute_text_metrics(refs: list[str], hyps: list[str]) -> dict:
    char_dist = 0
    char_total = 0
    word_dist = 0
    word_total = 0
    exact = 0
    n = len(refs)
    for r, h in zip(refs, hyps):
        r_n = normalize_text(r)
        h_n = normalize_text(h)
        if r_n == h_n:
            exact += 1
        char_dist += levenshtein(list(r_n), list(h_n))
        char_total += max(1, len(r_n))
        r_w = r_n.split()
        h_w = h_n.split()
        word_dist += levenshtein(r_w, h_w)
        word_total += max(1, len(r_w))
    return {
        "sentence_count": n,
        "sentence_exact_match": exact / max(1, n),
        "cer": char_dist / max(1, char_total),
        "wer": word_dist / max(1, word_total),
        "char_edits": char_dist,
        "char_ref_len": char_total,
        "word_edits": word_dist,
        "word_ref_len": word_total,
    }


def compute_topk_metrics(probs: np.ndarray, true_idx: np.ndarray) -> dict:
    keep = true_idx >= 0
    if keep.sum() == 0:
        return {"n_eval": 0, "top1": 0.0, "top3": 0.0, "top5": 0.0}

    p = probs[keep]
    y = true_idx[keep]
    top1 = np.mean(np.argmax(p, axis=1) == y)
    top3_idx = np.argpartition(-p, kth=2, axis=1)[:, :3]
    top5_idx = np.argpartition(-p, kth=4, axis=1)[:, :5]
    top3 = np.mean([yy in row for yy, row in zip(y, top3_idx)])
    top5 = np.mean([yy in row for yy, row in zip(y, top5_idx)])
    return {
        "n_eval": int(len(y)),
        "top1": float(top1),
        "top3": float(top3),
        "top5": float(top5),
    }


def fit_temperature(logits: np.ndarray, true_idx: np.ndarray) -> tuple[float, float]:
    keep = true_idx >= 0
    l = logits[keep]
    y = true_idx[keep]
    best_t = 1.0
    best_nll = float("inf")
    for t in np.arange(0.5, 3.01, 0.05):
        p = softmax_np(l / t)
        nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, float(best_nll)


def class_prior_from_labels(labels: np.ndarray, class_to_idx: dict, n_classes: int, smooth: float = 1.0) -> np.ndarray:
    cnt = np.full(n_classes, smooth, dtype=np.float64)
    for k in labels.tolist():
        if k in class_to_idx:
            cnt[class_to_idx[k]] += 1.0
    cnt /= cnt.sum()
    return cnt


def apply_calibration(
    logits: np.ndarray,
    true_idx: np.ndarray,
    classes: np.ndarray,
    train_prior: np.ndarray,
    free_prior: np.ndarray,
    temperature: float,
    prior_weight: float = 1.0,
    special_weight: float = 0.7,
) -> tuple[np.ndarray, dict]:
    n_classes = len(classes)
    log_train = np.log(train_prior + 1e-12)
    log_free = np.log(free_prior + 1e-12)
    prior_bias = prior_weight * (log_free - log_train)

    logits_t = logits / temperature + prior_bias.reshape(1, n_classes)
    probs_t = softmax_np(logits_t)

    special_keys = ["space", "backspace"] + [str(i) for i in range(10)]
    special_bias = np.zeros(n_classes, dtype=np.float64)
    keep = true_idx >= 0

    for key in special_keys:
        idx = np.where(classes == key)[0]
        if len(idx) == 0:
            continue
        j = int(idx[0])
        true_freq = float(np.mean(true_idx[keep] == j)) if keep.any() else 0.0
        pred_freq = float(np.mean(probs_t[:, j]))
        if true_freq > 0.0 and pred_freq > 0.0:
            special_bias[j] = math.log((true_freq + 1e-8) / (pred_freq + 1e-8))

    logits_cal = logits_t + special_weight * special_bias.reshape(1, n_classes)
    probs_cal = softmax_np(logits_cal)

    calib_info = {
        "temperature": float(temperature),
        "prior_weight": float(prior_weight),
        "special_weight": float(special_weight),
        "special_bias": {
            classes[i]: float(special_weight * special_bias[i])
            for i in range(n_classes) if abs(special_bias[i]) > 1e-9
        },
    }
    return probs_cal, calib_info


class PromptConstrainedLM:
    """
    Lightweight prompt-domain LM used for prompt-constrained upper-bound decoding.
    """
    def __init__(self, prompts: list[str], smoothing: float = 1.0, bigram_weight: float = 0.7):
        self.smoothing = smoothing
        self.bigram_weight = bigram_weight
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.vocab = set()
        for sent in prompts:
            toks = re.findall(r"[a-z0-9]+", sent.lower())
            for i, tok in enumerate(toks):
                self.unigram[tok] += 1
                self.vocab.add(tok)
                if i > 0:
                    self.bigram[toks[i - 1]][tok] += 1

    def is_valid_word(self, word: str) -> bool:
        return word.lower() in self.vocab

    def word_log_prob(self, word: str, prev_word: Optional[str] = None,
                      prev_prev_word: Optional[str] = None) -> float:
        word = word.lower()
        V = max(1, len(self.vocab))
        uni_count = self.unigram.get(word, 0) + self.smoothing
        uni_total = sum(self.unigram.values()) + self.smoothing * V
        log_uni = math.log(uni_count / uni_total)

        if prev_word is None:
            return log_uni
        prev = prev_word.lower()
        if prev not in self.bigram:
            return log_uni
        bi_count = self.bigram[prev].get(word, 0) + self.smoothing
        bi_total = sum(self.bigram[prev].values()) + self.smoothing * V
        log_bi = math.log(bi_count / bi_total)

        w = self.bigram_weight
        return math.log(w * math.exp(log_bi) + (1 - w) * math.exp(log_uni) + 1e-300)


def decode_one_sentence(event_rows: list[dict], probs: np.ndarray, classes: np.ndarray,
                        lm, beam_width: int, alpha: float) -> str:
    wd = WordDecoder(lm, beam_width=beam_width, top_chars=6, alpha=alpha)
    sd = SentenceDecoder(wd, lm, beam_sentences=20)
    sd.set_classes(classes)

    has_sentence_end = False
    for row in event_rows:
        key = row["key"]
        pvec = probs[row["global_index"]]
        if key in WORD_BOUNDARY_KEYS:
            sd.word_boundary(top_k=10)
        elif key in SENTENCE_BOUNDARY_KEYS:
            has_sentence_end = True
            return normalize_text(sd.sentence_end())
        elif key == "backspace":
            if sd._current_word_probs:
                sd._current_word_probs.pop()
            continue
        elif key in CONTROL_SKIP_KEYS:
            continue
        else:
            sd.push_keystroke(pvec)

    if not has_sentence_end:
        return normalize_text(sd.sentence_end())
    return ""


def raw_argmax_sentence(event_rows: list[dict], probs: np.ndarray, classes: np.ndarray) -> str:
    pred_keys = []
    for row in event_rows:
        pred = classes[int(np.argmax(probs[row["global_index"]]))]
        pred_keys.append(pred)
    return keys_to_text(pred_keys)


def decode_corpus(sentence_records: list[dict], probs: np.ndarray, classes: np.ndarray,
                  lm, beam_width: int, alpha: float) -> tuple[list[str], list[str]]:
    refs, hyps = [], []
    for sent in sentence_records:
        refs.append(sent["reference"])
        hyps.append(decode_one_sentence(sent["events"], probs, classes, lm, beam_width, alpha))
    return refs, hyps


def raw_argmax_corpus(sentence_records: list[dict], probs: np.ndarray, classes: np.ndarray) -> tuple[list[str], list[str]]:
    refs, hyps = [], []
    for sent in sentence_records:
        refs.append(sent["reference"])
        hyps.append(raw_argmax_sentence(sent["events"], probs, classes))
    return refs, hyps


def _nearest_window_fallback(ts: int,
                             ts_sorted: np.ndarray,
                             ts_to_window: dict[int, np.ndarray],
                             max_gap_ns: int = 80_000_000) -> Optional[np.ndarray]:
    """
    Find nearest extracted window by timestamp, used as a fallback when one
    keystroke window is missing. This preserves sequence length alignment.
    """
    if ts_sorted.size == 0:
        return None
    pos = int(np.searchsorted(ts_sorted, ts))
    cand = []
    if pos < len(ts_sorted):
        cand.append(int(ts_sorted[pos]))
    if pos > 0:
        cand.append(int(ts_sorted[pos - 1]))
    if not cand:
        return None
    best_ts = min(cand, key=lambda x: abs(x - ts))
    if abs(best_ts - ts) > max_gap_ns:
        return None
    return ts_to_window.get(best_ts)


def build_events_and_dataset(
    valid_sessions: list[SessionQuality],
    wcfg: WindowConfig,
    dataset_yes_only: bool = True,
    drop_iki_overlap: bool = True,
    iki_overlap_ms: float = 200.0,
    max_imputed_ratio: float = 0.03,
) -> tuple[np.ndarray, dict]:
    X_all = []
    y_all = []
    ts_all = []
    sess_all = []
    sent_idx_all = []
    role_all = []
    event_rows = []
    sentence_records = []
    imputed_window_count = 0
    overlap_keystroke_count = 0
    overlap_dropped_count = 0
    dropped_sessions_imputed = []
    session_build_stats = []

    overlap_ns = int(max(0.0, float(iki_overlap_ms)) * 1_000_000)
    max_imputed_ratio = max(0.0, float(max_imputed_ratio))

    global_idx = 0
    for sq in valid_sessions:
        session = sq.session_prefix
        proc = Preprocessor(session_prefix=session, output_dir="data/processed/freetype_tmp", window_cfg=wcfg)
        proc.load()
        proc.extract_windows()
        ts_to_window = {w["timestamp_ns"]: w["window"] for w in proc.windows}
        ts_sorted = np.array(sorted(ts_to_window.keys()), dtype=np.int64)

        press_events = parse_press_events(session + "_events.csv")
        cur_sent_events = []
        sent_idx = 0
        typed_texts = sq.typed_texts
        yes_idx = {i for i, m in enumerate(sq.prompt_matches) if m == "YES"} if dataset_yes_only else None
        prev_keystroke_row = None

        session_units = []
        session_sentence_records = []
        session_imputed = 0

        for evt in press_events:
            ts = evt["timestamp_ns"]
            key = evt["key"]
            if key in SENTENCE_BOUNDARY_KEYS:
                role = "sentence_boundary"
            elif key in WORD_BOUNDARY_KEYS:
                role = "word_boundary"
            elif key == "backspace":
                role = "backspace"
            elif key in CONTROL_SKIP_KEYS:
                role = "control"
            else:
                role = "keystroke"

            sentence_allowed = (yes_idx is None) or (sent_idx in yes_idx)
            if not sentence_allowed:
                if key in SENTENCE_BOUNDARY_KEYS:
                    sent_idx += 1
                    prev_keystroke_row = None
                continue

            window = ts_to_window.get(ts)
            imputed_window = False
            if window is None:
                window = _nearest_window_fallback(ts, ts_sorted, ts_to_window)
                imputed_window = window is not None
            if window is None:
                continue
            if imputed_window:
                session_imputed += 1

            row = {
                "session": os.path.basename(session),
                "timestamp_ns": ts,
                "sentence_idx": sent_idx,
                "key": key,
                "role": role,
                "global_index": -1,
                "imputed_window": imputed_window,
                "is_overlap": False,
                "iki_prev_ms": None,
            }

            if role == "keystroke" and overlap_ns > 0 and prev_keystroke_row is not None:
                dt_ns = ts - int(prev_keystroke_row["timestamp_ns"])
                if 0 < dt_ns < overlap_ns:
                    row["is_overlap"] = True
                    row["iki_prev_ms"] = float(dt_ns / 1_000_000.0)
                    prev_keystroke_row["is_overlap"] = True

            if role == "keystroke":
                prev_keystroke_row = row
            elif role == "sentence_boundary":
                prev_keystroke_row = None

            session_units.append({
                "row": row,
                "window": window,
                "key": key,
                "timestamp_ns": ts,
                "role": role,
            })
            cur_sent_events.append(row)

            if key in SENTENCE_BOUNDARY_KEYS:
                if sent_idx < len(typed_texts):
                    reference = typed_texts[sent_idx]
                else:
                    reference = keys_to_text([e["key"] for e in cur_sent_events])
                session_sentence_records.append({
                    "session": os.path.basename(session),
                    "sentence_idx": sent_idx,
                    "reference": normalize_text(reference),
                    "events": cur_sent_events.copy(),
                })
                cur_sent_events = []
                sent_idx += 1

        if cur_sent_events:
            reference = typed_texts[sent_idx] if sent_idx < len(typed_texts) else keys_to_text([e["key"] for e in cur_sent_events])
            session_sentence_records.append({
                "session": os.path.basename(session),
                "sentence_idx": sent_idx,
                "reference": normalize_text(reference),
                "events": cur_sent_events.copy(),
            })

        session_overlap_count = sum(
            1
            for u in session_units
            if u["row"]["role"] == "keystroke" and bool(u["row"].get("is_overlap", False))
        )
        overlap_keystroke_count += session_overlap_count

        session_overlap_dropped = 0
        if drop_iki_overlap and overlap_ns > 0 and session_overlap_count > 0:
            keep_ids = {
                id(u["row"])
                for u in session_units
                if not (u["row"]["role"] == "keystroke" and bool(u["row"].get("is_overlap", False)))
            }
            session_overlap_dropped = len(session_units) - len(keep_ids)
            overlap_dropped_count += session_overlap_dropped
            session_units = [u for u in session_units if id(u["row"]) in keep_ids]
            for rec in session_sentence_records:
                rec["events"] = [e for e in rec["events"] if id(e) in keep_ids]
            session_sentence_records = [rec for rec in session_sentence_records if rec["events"]]

        kept_rows = len(session_units)
        imputed_ratio = session_imputed / max(1, kept_rows)
        dropped_by_imputed = bool(kept_rows > 0 and imputed_ratio > max_imputed_ratio)
        if dropped_by_imputed:
            dropped_sessions_imputed.append(os.path.basename(session))
            session_build_stats.append({
                "session": os.path.basename(session),
                "rows_kept": kept_rows,
                "imputed_count": int(session_imputed),
                "imputed_ratio": float(imputed_ratio),
                "overlap_keystroke_count": int(session_overlap_count),
                "overlap_dropped_count": int(session_overlap_dropped),
                "dropped_by_imputed_ratio": True,
            })
            continue

        for u in session_units:
            row = u["row"]
            row["global_index"] = global_idx
            global_idx += 1

            event_rows.append(row)
            X_all.append(u["window"])
            y_all.append(u["key"])
            ts_all.append(u["timestamp_ns"])
            sess_all.append(row["session"])
            sent_idx_all.append(int(row["sentence_idx"]))
            role_all.append(u["role"])
            imputed_window_count += int(bool(row["imputed_window"]))

        sentence_records.extend(session_sentence_records)
        session_build_stats.append({
            "session": os.path.basename(session),
            "rows_kept": kept_rows,
            "imputed_count": int(session_imputed),
            "imputed_ratio": float(imputed_ratio),
            "overlap_keystroke_count": int(session_overlap_count),
            "overlap_dropped_count": int(session_overlap_dropped),
            "dropped_by_imputed_ratio": False,
        })

    X_np = np.array(X_all, dtype=np.float32)
    y_np = np.array(y_all)
    ts_np = np.array(ts_all, dtype=np.int64)
    sess_np = np.array(sess_all)
    sent_np = np.array(sent_idx_all, dtype=np.int32)
    role_np = np.array(role_all)

    os.makedirs("data/processed", exist_ok=True)
    np.savez_compressed(
        FREE_TYPE_DATASET_PATH,
        X=X_np,
        y=y_np,
        timestamps=ts_np,
        session=sess_np,
        sentence_idx=sent_np,
        role=role_np,
        target_rate_hz=wcfg.target_rate_hz,
        window_len=wcfg.target_window_len,
        channels=["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    )

    payload = {
        "X": X_np,
        "y": y_np,
        "timestamps": ts_np,
        "session": sess_np,
        "sentence_idx": sent_np,
        "role": role_np,
        "event_rows": event_rows,
        "sentence_records": sentence_records,
        "imputed_window_count": imputed_window_count,
        "overlap_keystroke_count": int(overlap_keystroke_count),
        "overlap_dropped_count": int(overlap_dropped_count),
        "dataset_yes_only": bool(dataset_yes_only),
        "drop_iki_overlap": bool(drop_iki_overlap),
        "iki_overlap_ms": float(iki_overlap_ms),
        "max_imputed_ratio": float(max_imputed_ratio),
        "dropped_sessions_imputed": dropped_sessions_imputed,
        "session_build_stats": session_build_stats,
    }
    return X_np, payload


def pick_examples(sentence_records: list[dict], refs: list[str], raw_hyps: list[str],
                  open_hyps: list[str], prompt_hyps: list[str]) -> list[dict]:
    items = []
    for i, sent in enumerate(sentence_records):
        r = refs[i]
        w_open = levenshtein(r.split(), open_hyps[i].split())
        items.append((w_open, i, sent))

    items.sort(reverse=True)
    chosen = []
    used = set()
    for _, idx, sent in items[:3]:
        chosen.append(idx)
        used.add(idx)

    for i, sent in enumerate(sentence_records):
        if any(ch.isdigit() for ch in sent["reference"]) and i not in used:
            chosen.append(i)
            used.add(i)
            break

    for i in range(min(2, len(sentence_records))):
        if i not in used:
            chosen.append(i)
            used.add(i)

    chosen = chosen[:5]
    out = []
    for i in chosen:
        out.append({
            "session": sentence_records[i]["session"],
            "sentence_idx": int(sentence_records[i]["sentence_idx"]),
            "reference": refs[i],
            "raw_argmax": raw_hyps[i],
            "decoder_open": open_hyps[i],
            "decoder_prompt_constrained": prompt_hyps[i],
        })
    return out


def nearest_prompt_hypotheses(hyps: list[str], prompt_pool: list[str]) -> list[str]:
    norm_prompts = [normalize_text(p) for p in prompt_pool]
    out = []
    for h in hyps:
        h_n = normalize_text(h)
        best = min(
            norm_prompts,
            key=lambda p: (
                levenshtein(h_n.split(), p.split()),
                levenshtein(list(h_n), list(p)),
            ),
        )
        out.append(best)
    return out


def filter_sentence_records_yes_only(sentence_records: list[dict],
                                     valid_sessions: list[SessionQuality]) -> list[dict]:
    yes_map: dict[str, set[int]] = {}
    for q in valid_sessions:
        session_name = os.path.basename(q.session_prefix)
        yes_idx = {i for i, m in enumerate(q.prompt_matches) if m == "YES"}
        yes_map[session_name] = yes_idx

    kept = []
    for rec in sentence_records:
        sid = rec["session"]
        idx = int(rec["sentence_idx"])
        if idx in yes_map.get(sid, set()):
            kept.append(rec)
    return kept


def main():
    parser = argparse.ArgumentParser(description="Free-type minimal closure evaluation")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Torch device (default: auto; macOS auto->cpu)",
    )
    parser.add_argument(
        "--rounds",
        nargs="+",
        default=["free_type"],
        help="Data selectors to scan (default: free_type). "
             "Supports folder names under data/raw, direct paths, or legacy round numbers.",
    )
    parser.add_argument("--beam-grid", nargs="+", type=int, default=[50, 100, 150], help="Beam grid for decoder check")
    parser.add_argument("--alpha-grid", nargs="+", type=float, default=[0.05, 0.15, 0.30], help="LM alpha grid")
    parser.add_argument("--model-path", default=MODEL_PATH, help="Model checkpoint path (full ckpt or state_dict)")
    parser.add_argument("--scaler-path", default=SCALER_PATH, help="Scaler npz path")
    parser.add_argument(
        "--base-ckpt-path",
        default=MODEL_PATH,
        help="Base full checkpoint for metadata when --model-path is state_dict only",
    )
    parser.add_argument(
        "--report-path",
        default=REPORT_PATH,
        help="Output report json path",
    )
    parser.add_argument(
        "--yes-only",
        action="store_true",
        help="Evaluate only YES attempts from *_prompts.csv (recommended for sentence-level reporting)",
    )
    parser.add_argument(
        "--dataset-yes-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When building free_type_dataset.npz, keep only YES attempts (default: true).",
    )
    parser.add_argument(
        "--drop-iki-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop overlapped keystroke windows with IKI below --iki-overlap-ms (default: true).",
    )
    parser.add_argument(
        "--iki-overlap-ms",
        type=float,
        default=200.0,
        help="IKI threshold in ms for overlap marking/filtering (default: 200).",
    )
    parser.add_argument(
        "--max-imputed-ratio",
        type=float,
        default=0.03,
        help="Drop sessions whose imputed-window ratio exceeds this value (default: 0.03).",
    )
    parser.add_argument(
        "--baseline-report",
        default=BASELINE_REPORT_27_PATH,
        help="Optional previous report path for before/after comparison",
    )
    args = parser.parse_args()
    device = set_device(args.device)
    print(f"Torch device: {device}")

    # Keep almost all key events aligned in free_type decoding; lower threshold
    # reduces dropped windows that can truncate word length.
    wcfg = WindowConfig(min_window_samples=2)
    sessions = discover_freetype_sessions(args.rounds)
    if not sessions:
        raise RuntimeError("No free_type sessions found")

    print(f"Found {len(sessions)} free_type sessions")
    qualities = [assess_session(s, wcfg) for s in sessions]
    valid_sessions = [q for q in qualities if q.is_valid]
    dropped_sessions = [q for q in qualities if not q.is_valid]
    print(f"Valid sessions: {len(valid_sessions)} | Dropped sessions: {len(dropped_sessions)}")

    X_free, payload = build_events_and_dataset(
        valid_sessions,
        wcfg,
        dataset_yes_only=args.dataset_yes_only,
        drop_iki_overlap=args.drop_iki_overlap,
        iki_overlap_ms=args.iki_overlap_ms,
        max_imputed_ratio=args.max_imputed_ratio,
    )
    if len(X_free) == 0:
        raise RuntimeError("No free_type events remained after dataset filtering (YES/overlap/imputed gates).")
    y_free = payload["y"]
    event_rows = payload["event_rows"]
    sentence_records = payload["sentence_records"]
    imputed_window_count = int(payload.get("imputed_window_count", 0))
    overlap_keystroke_count = int(payload.get("overlap_keystroke_count", 0))
    overlap_dropped_count = int(payload.get("overlap_dropped_count", 0))
    dropped_sessions_imputed = list(payload.get("dropped_sessions_imputed", []))
    session_build_stats = list(payload.get("session_build_stats", []))
    print(f"Saved independent free_type dataset: {FREE_TYPE_DATASET_PATH} | X={X_free.shape}")

    model, classes, class_to_idx, means, stds = load_model_and_scaler(
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        base_ckpt_path=args.base_ckpt_path,
    )
    logits, probs = infer_logits_probs(model, X_free, means, stds)

    true_idx = np.array([class_to_idx.get(k, -1) for k in y_free], dtype=np.int32)

    if not os.path.exists(MERGED_SINGLE_KEY_PATH):
        raise FileNotFoundError(f"Missing {MERGED_SINGLE_KEY_PATH}")
    merged = np.load(MERGED_SINGLE_KEY_PATH, allow_pickle=True)
    train_prior = class_prior_from_labels(merged["y"], class_to_idx, len(classes), smooth=1.0)
    free_prior = class_prior_from_labels(y_free, class_to_idx, len(classes), smooth=1.0)

    best_t, best_nll = fit_temperature(logits, true_idx)
    probs_cal, calib_info = apply_calibration(
        logits=logits,
        true_idx=true_idx,
        classes=classes,
        train_prior=train_prior,
        free_prior=free_prior,
        temperature=best_t,
        prior_weight=1.0,
        special_weight=0.7,
    )
    calib_info["temperature_fit_nll"] = best_nll

    if args.yes_only:
        sentence_records_eval = filter_sentence_records_yes_only(sentence_records, valid_sessions)
    else:
        sentence_records_eval = sentence_records
    if len(sentence_records_eval) == 0:
        raise RuntimeError("No sentence records available after filtering.")

    eval_global_indices = sorted({
        int(e["global_index"])
        for sent in sentence_records_eval
        for e in sent["events"]
    })
    eval_mask = np.zeros(len(y_free), dtype=bool)
    eval_mask[eval_global_indices] = True

    keystroke_uncal = compute_topk_metrics(probs[eval_mask], true_idx[eval_mask])
    keystroke_cal = compute_topk_metrics(probs_cal[eval_mask], true_idx[eval_mask])

    open_lm = NgramLanguageModel(smoothing=1.0, bigram_weight=0.4)
    raw_refs, raw_hyps = raw_argmax_corpus(sentence_records_eval, probs, classes)
    raw_metrics = compute_text_metrics(raw_refs, raw_hyps)

    grid_rows = []
    best_cfg = None
    best_open_metrics = None
    best_open_hyps = None
    for beam in args.beam_grid:
        for alpha in args.alpha_grid:
            refs_open, hyps_open = decode_corpus(sentence_records_eval, probs_cal, classes, open_lm, beam, alpha)
            m = compute_text_metrics(refs_open, hyps_open)
            row = {"beam": int(beam), "alpha": float(alpha), "wer": m["wer"], "cer": m["cer"], "sem": m["sentence_exact_match"]}
            grid_rows.append(row)
            if best_open_metrics is None or m["wer"] < best_open_metrics["wer"]:
                best_cfg = {"beam": int(beam), "alpha": float(alpha)}
                best_open_metrics = m
                best_open_hyps = hyps_open

    prompt_lm = PromptConstrainedLM(ALL_PROMPTS, smoothing=1.0, bigram_weight=0.7)
    refs_prompt_vocab, hyps_prompt_vocab = decode_corpus(
        sentence_records_eval, probs_cal, classes, prompt_lm,
        beam_width=best_cfg["beam"], alpha=max(0.3, best_cfg["alpha"])
    )
    prompt_vocab_metrics = compute_text_metrics(refs_prompt_vocab, hyps_prompt_vocab)

    # Prompt-aware upper bound: map each decoded hypothesis to nearest known prompt.
    # This represents a controlled prompt-set decoding mode.
    hyps_prompt_aware = nearest_prompt_hypotheses(best_open_hyps, ALL_PROMPTS)
    prompt_aware_metrics = compute_text_metrics(raw_refs, hyps_prompt_aware)

    examples = pick_examples(sentence_records_eval, raw_refs, raw_hyps, best_open_hyps, hyps_prompt_aware)

    all_reconstructed = []
    key_counter = Counter()
    total_press = 0
    total_windows = 0
    total_prompts_yes = 0
    total_prompts_no = 0
    for q in valid_sessions:
        all_reconstructed.extend(q.reconstructed_texts)
        total_press += q.press_count
        total_windows += q.windows_extracted
        total_prompts_yes += q.prompt_yes_rows
        total_prompts_no += q.prompt_no_rows
        key_counter["space"] += q.spaces
        key_counter["enter"] += q.enters
        key_counter["backspace"] += q.backspaces

    joined_text = " ".join(all_reconstructed)
    alpha_cov = sorted({c for c in joined_text if c.isalpha()})
    digit_cov = sorted({c for c in joined_text if c.isdigit()})
    sentence_attempt_count = len(all_reconstructed)
    sentence_yes_count = total_prompts_yes if total_prompts_yes > 0 else sentence_attempt_count
    sentence_no_count = total_prompts_no
    prompt_total = len(ALL_PROMPTS)

    quality_report = {
        "prompt_templates_total": prompt_total,
        "recorded_attempts": sentence_attempt_count,
        "completed_yes": sentence_yes_count,
        "retry_no": sentence_no_count,
        "missing_vs_template": max(0, prompt_total - sentence_yes_count),
        "valid_sessions": [
            {
                "session": os.path.basename(q.session_prefix),
                "effective_hz": q.effective_hz,
                "median_hz": q.median_hz,
                "hz_p01": q.hz_p01,
                "hz_p99": q.hz_p99,
                "press_count": q.press_count,
                "prompts_rows": q.prompt_rows,
                "prompts_yes_rows": q.prompt_yes_rows,
                "prompts_no_rows": q.prompt_no_rows,
                "reconstructed_sentences": q.reconstructed_sentences,
                "align_to_typed": f"{q.align_typed}/{q.prompt_rows}",
                "windows_extracted": q.windows_extracted,
                "valid_sample_rate": q.valid_sample_rate,
                "issues": q.issues,
            }
            for q in valid_sessions
        ],
        "dropped_sessions": [
            {"session": os.path.basename(q.session_prefix), "issues": q.issues}
            for q in dropped_sessions
        ],
        "dropped_sessions_imputed_ratio": dropped_sessions_imputed,
        "build_stats_by_session": session_build_stats,
        "overview": {
            "sentence_count": sentence_yes_count,
            "sentence_attempt_count": sentence_attempt_count,
            "sentence_yes_count": sentence_yes_count,
            "sentence_no_count": sentence_no_count,
            "total_presses": total_press,
            "total_windows": total_windows,
            "imputed_window_count": imputed_window_count,
            "overlap_keystroke_count": overlap_keystroke_count,
            "overlap_dropped_count": overlap_dropped_count,
            "valid_sample_rate": total_windows / max(1, total_press),
            "dataset_rows_after_filters": int(len(y_free)),
            "dataset_yes_only": bool(payload.get("dataset_yes_only", False)),
            "drop_iki_overlap": bool(payload.get("drop_iki_overlap", False)),
            "iki_overlap_ms": float(payload.get("iki_overlap_ms", 0.0)),
            "max_imputed_ratio": float(payload.get("max_imputed_ratio", 0.0)),
            "alpha_coverage_count": len(alpha_cov),
            "alpha_coverage": "".join(alpha_cov),
            "digit_coverage_count": len(digit_cov),
            "digit_coverage": "".join(digit_cov),
            "space_count": key_counter["space"],
            "enter_count": key_counter["enter"],
            "backspace_count": key_counter["backspace"],
        },
    }

    report = {
        "free_type_data_quality": quality_report,
        "artifacts": {
            "free_type_dataset": FREE_TYPE_DATASET_PATH,
            "model_path": args.model_path,
            "scaler_path": args.scaler_path,
            "report_path": args.report_path,
            "evaluation_mode": "yes_only" if args.yes_only else "all_attempts",
            "dataset_filter_mode": {
                "dataset_yes_only": bool(args.dataset_yes_only),
                "drop_iki_overlap": bool(args.drop_iki_overlap),
                "iki_overlap_ms": float(args.iki_overlap_ms),
                "max_imputed_ratio": float(args.max_imputed_ratio),
            },
            "device": str(device),
        },
        "keystroke_metrics": {
            "uncalibrated": keystroke_uncal,
            "calibrated": keystroke_cal,
        },
        "calibration": calib_info,
        "decoder": {
            "open_vocab_grid": sorted(grid_rows, key=lambda x: x["wer"]),
            "open_vocab_best_config": best_cfg,
            "open_vocab_metrics": best_open_metrics,
            "prompt_vocab_metrics": prompt_vocab_metrics,
            "prompt_constrained_metrics": prompt_aware_metrics,
            "raw_argmax_metrics": raw_metrics,
            "examples": examples,
        },
        "judgement": {
            "bring_up_with_39_or_less": "sufficient",
            "lightweight_calibration_with_current_data": "borderline_but_usable",
            "train_from_scratch_free_type_model": "insufficient",
        },
    }

    if args.baseline_report and os.path.exists(args.baseline_report):
        try:
            with open(args.baseline_report, "r") as bf:
                base = json.load(bf)

            base_keystroke = base.get("keystroke_metrics", {}).get("calibrated", {})
            curr_keystroke = report["keystroke_metrics"]["calibrated"]

            base_open = base.get("decoder", {}).get("open_vocab_metrics", {})
            curr_open = report["decoder"]["open_vocab_metrics"]

            report["comparison_with_baseline"] = {
                "baseline_report_path": args.baseline_report,
                "baseline_sentence_count": int(base.get("free_type_data_quality", {}).get("overview", {}).get("sentence_count", 0)),
                "current_sentence_count": int(report["free_type_data_quality"]["overview"]["sentence_count"]),
                "keystroke_calibrated": {
                    "top1_before": float(base_keystroke.get("top1", 0.0)),
                    "top1_after": float(curr_keystroke.get("top1", 0.0)),
                    "top1_delta": float(curr_keystroke.get("top1", 0.0) - base_keystroke.get("top1", 0.0)),
                    "top3_before": float(base_keystroke.get("top3", 0.0)),
                    "top3_after": float(curr_keystroke.get("top3", 0.0)),
                    "top3_delta": float(curr_keystroke.get("top3", 0.0) - base_keystroke.get("top3", 0.0)),
                    "top5_before": float(base_keystroke.get("top5", 0.0)),
                    "top5_after": float(curr_keystroke.get("top5", 0.0)),
                    "top5_delta": float(curr_keystroke.get("top5", 0.0) - base_keystroke.get("top5", 0.0)),
                },
                "decoder_open_vocab": {
                    "cer_before": float(base_open.get("cer", 0.0)),
                    "cer_after": float(curr_open.get("cer", 0.0)),
                    "cer_delta": float(curr_open.get("cer", 0.0) - base_open.get("cer", 0.0)),
                    "wer_before": float(base_open.get("wer", 0.0)),
                    "wer_after": float(curr_open.get("wer", 0.0)),
                    "wer_delta": float(curr_open.get("wer", 0.0) - base_open.get("wer", 0.0)),
                },
            }
        except Exception as e:
            report["comparison_with_baseline_error"] = str(e)

    os.makedirs("results", exist_ok=True)
    with open(args.report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Report saved: {args.report_path}")
    print("Done.")


if __name__ == "__main__":
    main()
