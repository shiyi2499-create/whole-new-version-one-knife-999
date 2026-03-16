"""
run_real_freetype.py — End-to-End Real Free-Type Decoder
=========================================================

Pipeline:
  1. Read real free_type _sensor.csv + _events.csv files
  2. Cut windows using IDENTICAL logic as preprocessor.py
  3. Train a final Transformer on ALL single_key data (no CV holdout)
  4. Run inference on free_type windows → per-keystroke softmax probs
  5. Feed to SentenceDecoder (from phase3_decoder.py) respecting
     word boundaries (space) and sentence boundaries (enter/return)

Run:
  .venv/bin/python3 run_real_freetype.py
  .venv/bin/python3 run_real_freetype.py --rounds free_type  # only free_type folder
  .venv/bin/python3 run_real_freetype.py --no-train          # skip retraining (use saved model)
  .venv/bin/python3 run_real_freetype.py --alpha 0.15        # LM weight

Requirements:
  - data/processed/merged_dataset.npz  (single_key training data)
  - data/raw/<source>/ with *_free_type_sensor.csv and *_free_type_events.csv
  - phase3_decoder.py in same directory
"""

import os
import sys
import csv
import copy
import time
import glob
import json
import argparse
import platform
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from collections import Counter, defaultdict
from sklearn.preprocessing import LabelEncoder

# ── PyTorch ───────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    DEVICE = torch.device("cpu")
except ImportError:
    print("❌ PyTorch not installed"); sys.exit(1)


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

# ── Phase 3 decoder (import the algorithm classes, ignore test fns) ──
try:
    from phase3_decoder import NgramLanguageModel, WordDecoder, SentenceDecoder
except ImportError:
    print("❌ phase3_decoder.py not found in current directory"); sys.exit(1)

# ── Preprocessor windowing (reuse exact same logic as training) ──
try:
    from preprocessor import Preprocessor, WindowConfig, resolve_data_dirs
except ImportError:
    print("❌ preprocessor.py not found in current directory"); sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  1. TRANSFORMER MODEL (identical to run_transformer_only.py)
# ══════════════════════════════════════════════════════════════

class TransformerClassifier(nn.Module):
    def __init__(self, n_timesteps=39, n_channels=6, n_classes=42,
                 d_model=64, nhead=4, num_layers=3):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(
            torch.randn(1, n_timesteps, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=128, dropout=0.3, batch_first=True,
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


def augment_batch(X_batch, p=0.5):
    B, T, C = X_batch.shape
    X_aug = X_batch.clone()
    for i in range(B):
        if np.random.random() > p:
            continue
        aug_type = np.random.choice(["shift", "noise", "scale", "ch_drop"])
        if aug_type == "shift":
            shift = np.random.randint(-T // 10, T // 10 + 1)
            X_aug[i] = torch.roll(X_aug[i], shifts=shift, dims=0)
        elif aug_type == "noise":
            X_aug[i] += torch.randn_like(X_aug[i]) * X_aug[i].std() * 0.01
        elif aug_type == "scale":
            X_aug[i] *= 0.8 + 0.4 * np.random.random()
        elif aug_type == "ch_drop":
            X_aug[i][:, np.random.randint(0, C)] = 0.0
    return X_aug


# ══════════════════════════════════════════════════════════════
#  2. TRAIN FINAL MODEL ON ALL SINGLE_KEY DATA
# ══════════════════════════════════════════════════════════════

MODEL_PATH = "results/transformer_final.pt"
SCALER_PATH = "results/transformer_scaler.npz"

def train_final_model(X_raw, y_keys, epochs=300, lr=5e-4,
                      batch_size=32, patience=50, force=False):
    """
    Train Transformer on ALL training data (no CV split).
    Saves model weights + per-channel normalization stats.

    Returns: (model, le, channel_means, channel_stds)
    """
    if not force and os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        print(f"  Found saved model: {MODEL_PATH}")
        return load_final_model(y_keys)

    print(f"\n{'='*60}\n  TRAINING FINAL TRANSFORMER (all data, no holdout)\n{'='*60}")
    print(f"  Samples: {len(X_raw)}  |  Epochs: {epochs}  |  Device: {DEVICE}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y_keys)
    n_classes = len(le.classes_)

    # Per-channel normalisation — save stats for inference time
    X_norm = X_raw.copy()
    ch_means = np.zeros(X_raw.shape[2])
    ch_stds  = np.zeros(X_raw.shape[2])
    for ch in range(X_raw.shape[2]):
        mu = X_norm[:, :, ch].mean()
        sd = X_norm[:, :, ch].std()
        if sd > 1e-10:
            X_norm[:, :, ch] = (X_norm[:, :, ch] - mu) / sd
            ch_means[ch] = mu
            ch_stds[ch]  = sd
        else:
            ch_stds[ch] = 1.0

    model = TransformerClassifier(
        n_timesteps=X_norm.shape[1],
        n_channels=X_norm.shape[2],
        n_classes=n_classes,
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    X_t = torch.FloatTensor(X_norm)
    y_t = torch.LongTensor(y_enc)
    loader = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)

    # Use a 10% validation split just for early-stopping monitoring
    n_val = max(1, int(0.1 * len(X_norm)))
    val_idx = np.random.choice(len(X_norm), n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(len(X_norm)), val_idx)

    X_val_t = torch.FloatTensor(X_norm[val_idx]).to(DEVICE)
    y_val_t = torch.LongTensor(y_enc[val_idx]).to(DEVICE)
    X_tr_t  = torch.FloatTensor(X_norm[train_idx])
    y_tr_t  = torch.LongTensor(y_enc[train_idx])
    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t),
                           batch_size=batch_size, shuffle=True)

    best_val_acc = 0.0
    best_state   = None
    patience_ctr = 0

    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for xb, yb in tr_loader:
            xb = augment_batch(xb, p=0.5)
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out  = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            correct += (out.argmax(1) == yb).sum().item()
            total   += len(yb)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(1) == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1:3d}: train_acc={correct/total:.3f}  "
                  f"val_acc={val_acc:.3f}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    print(f"  Best val acc: {best_val_acc:.3f}  |  Total time: {time.time()-t0:.0f}s")

    # ── Save ──────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "n_timesteps": X_norm.shape[1],
        "n_channels":  X_norm.shape[2],
        "n_classes":   n_classes,
        "classes":     le.classes_,
    }, MODEL_PATH)
    np.savez(SCALER_PATH, means=ch_means, stds=ch_stds)
    print(f"  Saved → {MODEL_PATH}")
    print(f"  Saved → {SCALER_PATH}")

    return model, le, ch_means, ch_stds


def load_final_model(y_keys=None):
    """Load saved model + scaler. Returns (model, le, ch_means, ch_stds)."""
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = TransformerClassifier(
        n_timesteps=ckpt["n_timesteps"],
        n_channels=ckpt["n_channels"],
        n_classes=ckpt["n_classes"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    le = LabelEncoder()
    le.classes_ = ckpt["classes"]

    scaler = np.load(SCALER_PATH)
    print(f"  Loaded model: {ckpt['n_classes']} classes, "
          f"n_timesteps={ckpt['n_timesteps']}")
    return model, le, scaler["means"], scaler["stds"]


# ══════════════════════════════════════════════════════════════
#  3. LOAD FREE_TYPE DATA WITH REAL TEMPORAL ORDER
# ══════════════════════════════════════════════════════════════

# Keys treated as word boundaries (trigger SentenceDecoder.word_boundary())
WORD_BOUNDARY_KEYS = {"space"}
# Keys treated as sentence boundaries (trigger SentenceDecoder.sentence_end())
SENTENCE_BOUNDARY_KEYS = {"enter", "return"}
# Keys to skip entirely (not typed characters, not boundaries)
SKIP_KEYS = {"shift", "capslock", "ctrl", "alt", "cmd", "tab", "esc",
             "backspace", "delete", "left", "right", "up", "down",
             "f1","f2","f3","f4","f5","f6","f7","f8","f9","f10","f11","f12"}


def load_freetype_sessions(round_dirs: list[str]) -> list[str]:
    """Find all free_type session prefixes across round directories."""
    sessions = []
    for rd in round_dirs:
        if not os.path.isdir(rd):
            print(f"  ⚠ Not found: {rd}")
            continue
        for f in sorted(os.listdir(rd)):
            if "_free_type_" in f and f.endswith("_sensor.csv"):
                prefix = os.path.join(rd, f.replace("_sensor.csv", ""))
                if os.path.exists(prefix + "_events.csv"):
                    sessions.append(prefix)
    return sessions


def extract_freetype_windows(session_prefix: str) -> list[dict]:
    """
    Extract keystroke windows from a free_type session,
    PRESERVING temporal order and boundary key events.

    Returns list of dicts, each with:
      {
        "type": "keystroke" | "word_boundary" | "sentence_boundary",
        "key":  str,
        "window": np.ndarray (39, 6) or None for boundary events,
        "timestamp_ns": int,
      }
    """
    # Lower min_window_samples to avoid dropping too many free_type keystrokes.
    wcfg = WindowConfig(min_window_samples=2)   # 100+200ms @ 130Hz → 39 samples

    # Use Preprocessor to load + window the session
    proc = Preprocessor(session_prefix=session_prefix,
                        output_dir="data/processed/freetype_tmp",
                        window_cfg=wcfg)
    proc.load()

    # Extract all press events (Preprocessor.extract_windows does the windowing)
    proc.extract_windows()

    # Build a timestamp→window lookup from the Preprocessor result
    ts_to_window = {w["timestamp_ns"]: w for w in proc.windows}
    ts_sorted = np.array(sorted(ts_to_window.keys()), dtype=np.int64)

    def nearest_window(ts: int, max_gap_ns: int = 80_000_000):
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

    # Now replay the events file IN ORDER to get the correct sequence
    # including boundary events that don't have windows (space, enter, etc.)
    events_path = session_prefix + "_events.csv"
    ordered_events = []
    with open(events_path) as f:
        for row in csv.DictReader(f):
            if row["event_type"] != "press":
                continue
            key = row["key"].lower()
            ts  = int(row["timestamp_ns"])
            win = None

            if key in SKIP_KEYS:
                continue
            elif key in SENTENCE_BOUNDARY_KEYS:
                ordered_events.append({
                    "type": "sentence_boundary",
                    "key": key, "window": None, "timestamp_ns": ts,
                })
            elif key in WORD_BOUNDARY_KEYS:
                ordered_events.append({
                    "type": "word_boundary",
                    "key": key, "window": None, "timestamp_ns": ts,
                })
            else:
                win = ts_to_window.get(ts)
                if win is None:
                    win = nearest_window(ts)
                if win is not None:
                    ordered_events.append({
                        "type": "keystroke",
                        "key": key,
                        "window": win["window"],
                        "timestamp_ns": ts,
                    })
            # else: press event was skipped by Preprocessor (too short window), skip here too

    return ordered_events


# ══════════════════════════════════════════════════════════════
#  4. RUN INFERENCE + DECODE
# ══════════════════════════════════════════════════════════════

def run_inference_on_window(model: nn.Module,
                            window: np.ndarray,
                            ch_means: np.ndarray,
                            ch_stds: np.ndarray) -> np.ndarray:
    """
    Normalise one window (39, 6) and run through Transformer.
    Returns softmax probability vector (n_classes,).
    """
    w = window.copy().astype(np.float32)
    for ch in range(w.shape[1]):
        w[:, ch] = (w[:, ch] - ch_means[ch]) / (ch_stds[ch] + 1e-10)

    with torch.no_grad():
        x = torch.FloatTensor(w).unsqueeze(0).to(DEVICE)   # (1, 39, 6)
        logits = model(x)                                 # (1, n_classes)
        probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return probs


def decode_session(events: list[dict],
                   model: nn.Module,
                   le: LabelEncoder,
                   ch_means: np.ndarray,
                   ch_stds: np.ndarray,
                   sentence_decoder: SentenceDecoder,
                   verbose: bool = True) -> list[dict]:
    """
    Replay a session's event stream through the model and decoder.

    Returns list of sentence results:
      { "original_keys": [...], "decoded": str, "word_count": int }
    """
    classes = le.classes_
    sentence_decoder.set_classes(classes)

    results       = []
    current_keys  = []   # raw ground-truth key sequence for this sentence
    n_keystrokes  = 0
    n_sentences   = 0
    n_words       = 0

    for evt in events:
        if evt["type"] == "keystroke":
            probs = run_inference_on_window(
                model, evt["window"], ch_means, ch_stds
            )
            sentence_decoder.push_keystroke(probs)
            current_keys.append(evt["key"])
            n_keystrokes += 1

        elif evt["type"] == "word_boundary":
            candidates = sentence_decoder.word_boundary(top_k=10)
            current_keys.append(" ")
            n_words += 1
            if verbose and candidates:
                top3 = [w for w, _ in candidates[:3]]
                print(f"    [word boundary] top-3: {top3}")

        elif evt["type"] == "sentence_boundary":
            decoded = sentence_decoder.sentence_end()
            original = "".join(current_keys).strip()

            result = {
                "original_keys": current_keys.copy(),
                "original_str":  original,
                "decoded":       decoded,
                "word_count":    len(decoded.split()),
            }
            results.append(result)
            n_sentences += 1

            print(f"\n  ── Sentence {n_sentences} ──")
            print(f"    Keys typed: {original!r}")
            print(f"    Decoded:    {decoded!r}")
            # Word-level accuracy
            orig_words = original.split()
            dec_words  = decoded.split()
            if orig_words:
                matches = sum(a == b for a, b in zip(orig_words, dec_words))
                word_acc = matches / len(orig_words)
                print(f"    Word acc:   {word_acc:.1%}  ({matches}/{len(orig_words)})")

            current_keys = []

    # Flush any remaining content (session ended without enter)
    if current_keys and sentence_decoder._current_word_probs:
        decoded = sentence_decoder.sentence_end()
        original = "".join(current_keys).strip()
        if decoded.strip():
            results.append({
                "original_keys": current_keys.copy(),
                "original_str":  original,
                "decoded":       decoded,
                "word_count":    len(decoded.split()),
            })
            print(f"\n  ── Sentence {n_sentences+1} (flushed) ──")
            print(f"    Keys typed: {original!r}")
            print(f"    Decoded:    {decoded!r}")

    print(f"\n  Session stats: {n_keystrokes} keystrokes | "
          f"{n_words} words | {n_sentences} sentences")
    return results


# ══════════════════════════════════════════════════════════════
#  5. AGGREGATE METRICS
# ══════════════════════════════════════════════════════════════

def compute_metrics(all_results: list[dict]) -> dict:
    """Compute word-level and sentence-level accuracy across all sessions."""
    total_words   = 0
    correct_words = 0
    perfect_sents = 0
    total_sents   = len(all_results)

    for r in all_results:
        orig_words = r["original_str"].split()
        dec_words  = r["decoded"].split()
        n = len(orig_words)
        if n == 0:
            continue
        matches = sum(a == b for a, b in zip(orig_words, dec_words))
        total_words   += n
        correct_words += matches
        if matches == n and len(dec_words) == n:
            perfect_sents += 1

    return {
        "total_sentences":  total_sents,
        "perfect_sentences": perfect_sents,
        "sentence_accuracy": perfect_sents / max(total_sents, 1),
        "total_words":      total_words,
        "correct_words":    correct_words,
        "word_accuracy":    correct_words / max(total_words, 1),
    }


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    # ── Disable MPS-unsafe attention kernels (same fix as run_transformer_only) ──
    torch.set_num_threads(1)
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Real free-type end-to-end decoder")
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
        help="Data selectors to search for free_type data (default: free_type). "
             "Supports folder names under data/raw, direct paths, or legacy round numbers.",
    )
    parser.add_argument("--no-train", action="store_true",
                        help="Skip training, load saved model from results/")
    parser.add_argument("--force-train", action="store_true",
                        help="Retrain even if saved model exists")
    parser.add_argument("--alpha", type=float, default=0.15,
                        help="LM weight in word decoder (default: 0.15)")
    parser.add_argument("--beam", type=int, default=100,
                        help="Beam width (default: 100)")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="Print word-boundary candidates (default: True)")
    args = parser.parse_args()
    device = set_device(args.device)

    print(f"\n{'='*60}")
    print(f"  🚀 REAL FREE-TYPE END-TO-END DECODER")
    print(f"  alpha={args.alpha}  beam={args.beam}")
    print(f"  device={device}")
    print(f"{'='*60}\n")

    # ── Step 1: Load or train the final model ─────────────────
    data_path = "data/processed/merged_dataset.npz"
    if not os.path.exists(data_path):
        print(f"❌ {data_path} not found. Run preprocessor.py first.")
        sys.exit(1)

    raw = np.load(data_path, allow_pickle=True)
    X_raw  = raw["X"].astype(np.float32)
    y_keys = raw["y"]

    # Filter spurious keys
    counts = Counter(y_keys.tolist())
    valid  = {k for k, v in counts.items() if v >= 10}
    mask   = np.array([k in valid for k in y_keys])
    X_raw, y_keys = X_raw[mask], y_keys[mask]
    print(f"  Training data: {X_raw.shape}, {len(valid)} classes")

    if args.no_train:
        if not os.path.exists(MODEL_PATH):
            print(f"❌ --no-train specified but {MODEL_PATH} not found.")
            sys.exit(1)
        model, le, ch_means, ch_stds = load_final_model()
    else:
        model, le, ch_means, ch_stds = train_final_model(
            X_raw, y_keys, force=args.force_train
        )

    model.eval()

    # ── Step 2: Build language model + decoders ───────────────
    print(f"\n  Building language model...")
    lm = NgramLanguageModel(smoothing=1.0, bigram_weight=0.4)
    word_dec = WordDecoder(lm, beam_width=args.beam,
                           top_chars=6, alpha=args.alpha)
    sent_dec = SentenceDecoder(word_dec, lm, beam_sentences=20)

    # ── Step 3: Find free_type sessions ───────────────────────
    round_dirs = resolve_data_dirs(args.rounds, data_root="data/raw")
    sessions   = load_freetype_sessions(round_dirs)

    if not sessions:
        print(f"\n❌ No free_type sessions found in: {round_dirs}")
        print("  Expected files: *_free_type_sensor.csv + *_free_type_events.csv")
        sys.exit(1)

    print(f"\n  Found {len(sessions)} free_type session(s):")
    for s in sessions:
        print(f"    {s}")

    # ── Step 4: Extract windows + decode each session ─────────
    all_results = []

    for sess in sessions:
        print(f"\n{'='*60}")
        print(f"  SESSION: {os.path.basename(sess)}")
        print(f"{'='*60}")

        events = extract_freetype_windows(sess)
        n_ks = sum(1 for e in events if e["type"] == "keystroke")
        n_wb = sum(1 for e in events if e["type"] == "word_boundary")
        n_sb = sum(1 for e in events if e["type"] == "sentence_boundary")
        print(f"  Events: {n_ks} keystrokes | {n_wb} spaces | {n_sb} enters")

        if n_ks == 0:
            print("  ⚠ No keystroke windows extracted, skipping")
            continue

        sess_results = decode_session(
            events, model, le, ch_means, ch_stds,
            sent_dec, verbose=args.verbose
        )
        all_results.extend(sess_results)

    # ── Step 5: Final metrics ──────────────────────────────────
    if not all_results:
        print("\n⚠ No sentences decoded.")
        sys.exit(0)

    print(f"\n{'='*60}\n  📊 FINAL METRICS\n{'='*60}")
    metrics = compute_metrics(all_results)
    print(f"  Sentences decoded:     {metrics['total_sentences']}")
    print(f"  Perfect sentences:     {metrics['perfect_sentences']}  "
          f"({metrics['sentence_accuracy']:.1%})")
    print(f"  Total words:           {metrics['total_words']}")
    print(f"  Correct words:         {metrics['correct_words']}  "
          f"({metrics['word_accuracy']:.1%})")

    # Save
    out = {
        "sessions": [os.path.basename(s) for s in sessions],
        "alpha":    args.alpha,
        "beam":     args.beam,
        "metrics":  metrics,
        "sentences": [
            {"original": r["original_str"], "decoded": r["decoded"]}
            for r in all_results
        ],
    }
    os.makedirs("results", exist_ok=True)
    with open("results/results_freetype.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved → results/results_freetype.json")
    print(f"\n{'='*60}\n  ✓ Done!\n{'='*60}\n")


if __name__ == "__main__":
    main()
