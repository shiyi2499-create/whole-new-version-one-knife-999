"""
run_freetype_finetune_beam.py
=============================
Pipeline:
  1) preprocess free_type sessions into an independent dataset
  2) fine-tune existing single-key Transformer (no re-design, no from-scratch)
  3) beam-search sentence decoding on free_type
  4) report Top1/Top3/Top5 + CER/WER + sentence exact match
"""

import os
import json
import math
import argparse
import csv
import glob
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from phase3_decoder import NgramLanguageModel, WordDecoder, SentenceDecoder
from run_freetype_closure_eval import (
    WindowConfig,
    discover_freetype_sessions,
    assess_session,
    build_events_and_dataset,
    set_device,
    load_model_and_scaler,
    infer_logits_probs,
    compute_topk_metrics,
    compute_text_metrics,
    normalize_text,
    fit_temperature,
    apply_calibration,
    class_prior_from_labels,
    PromptConstrainedLM,
)
from typing_prompts import PROMPTS as PROMPT_POOL


DEVICE = torch.device("cpu")
REPORT_PATH = "results/free_type_finetune_beam_report.json"
FINETUNE_MODEL_PATH = "results/transformer_freetype_finetuned.pt"

WORD_BOUNDARY_KEYS = {"space"}
SENTENCE_BOUNDARY_KEYS = {"enter", "return"}


@dataclass
class Split:
    train_sentence_ids: list[tuple[str, int]]
    eval_sentence_ids: list[tuple[str, int]]
    train_indices: np.ndarray
    eval_indices: np.ndarray


def sentence_level_split(sentence_records: list[dict], eval_ratio: float = 0.2, seed: int = 42) -> Split:
    sentence_ids = [(r["session"], int(r["sentence_idx"])) for r in sentence_records]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(sentence_ids))

    n_eval = max(1, int(round(len(sentence_ids) * eval_ratio)))
    eval_pos = set(perm[:n_eval].tolist())
    train_sentence_ids = [sid for i, sid in enumerate(sentence_ids) if i not in eval_pos]
    eval_sentence_ids = [sid for i, sid in enumerate(sentence_ids) if i in eval_pos]

    train_idx = []
    eval_idx = []
    eval_set = set(eval_sentence_ids)
    for i, r in enumerate(sentence_records):
        sid = (r["session"], int(r["sentence_idx"]))
        indices = [int(e["global_index"]) for e in r["events"]]
        if sid in eval_set:
            eval_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return Split(
        train_sentence_ids=train_sentence_ids,
        eval_sentence_ids=eval_sentence_ids,
        train_indices=np.array(sorted(train_idx), dtype=np.int64),
        eval_indices=np.array(sorted(eval_idx), dtype=np.int64),
    )


def keep_yes_attempts_only(sentence_records: list[dict]) -> list[dict]:
    """
    Keep only attempts labeled YES in *_prompts.csv.
    Also replace reference with prompt_text (ground truth target sentence).
    """
    by_session: dict[str, list[dict]] = {}
    for rec in sentence_records:
        by_session.setdefault(rec["session"], []).append(rec)

    kept = []
    for session, recs in by_session.items():
        recs = sorted(recs, key=lambda r: int(r["sentence_idx"]))
        cand = glob.glob(f"data/raw/*/{session}_prompts.csv")
        if not cand:
            continue
        with open(sorted(cand)[0], "r") as f:
            rows = list(csv.DictReader(f))

        for rec in recs:
            idx = int(rec["sentence_idx"])
            if idx >= len(rows):
                continue
            row = rows[idx]
            if row.get("match", "").upper() != "YES":
                continue
            out = dict(rec)
            out["reference"] = normalize_text(row["prompt_text"])
            kept.append(out)

    return kept


def _normalize_windows(X: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    Xn = X.astype(np.float32).copy()
    for ch in range(Xn.shape[2]):
        Xn[:, :, ch] = (Xn[:, :, ch] - means[ch]) / (stds[ch] + 1e-10)
    return Xn


def augment_batch(X_batch: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """
    Lightweight augmentation inspired by side-channel robustness practice:
    random temporal shift + tiny additive noise + amplitude scaling.
    """
    B, T, C = X_batch.shape
    X_aug = X_batch.clone()
    for i in range(B):
        if np.random.random() > p:
            continue
        aug_type = np.random.choice(["shift", "noise", "scale"])
        if aug_type == "shift":
            shift = np.random.randint(-max(1, T // 10), max(2, T // 10 + 1))
            X_aug[i] = torch.roll(X_aug[i], shifts=int(shift), dims=0)
        elif aug_type == "noise":
            X_aug[i] += torch.randn_like(X_aug[i]) * max(float(X_aug[i].std()), 1e-6) * 0.01
        elif aug_type == "scale":
            X_aug[i] *= float(0.85 + 0.3 * np.random.random())
    return X_aug


def fine_tune_model(
    model: nn.Module,
    X: np.ndarray,
    y_idx: np.ndarray,
    train_indices: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    epochs: int = 12,
    lr: float = 1e-4,
    batch_size: int = 64,
    aug_prob: float = 0.5,
) -> nn.Module:
    model = model.to(DEVICE)
    Xtr = _normalize_windows(X[train_indices], means, stds)
    ytr = y_idx[train_indices]
    keep = ytr >= 0
    Xtr = Xtr[keep]
    ytr = ytr[keep]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long()),
        batch_size=batch_size,
        shuffle=True,
    )

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            xb = augment_batch(xb, p=aug_prob)
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total += len(yb)
        print(f"  fine-tune epoch {epoch+1:02d}/{epochs} loss={total_loss/max(1,total):.4f}")

    model.eval()
    return model


def decode_sentence_with_backspace(events: list[dict], probs: np.ndarray, classes: np.ndarray,
                                   word_decoder: WordDecoder, lm: NgramLanguageModel) -> str:
    """
    Use true boundary events (space/enter/backspace) from free_type stream.
    Backspace is applied as pop on current word keystroke buffer.
    """
    sd = SentenceDecoder(word_decoder, lm, beam_sentences=20)
    sd.set_classes(classes)

    for ev in events:
        key = ev["key"]
        idx = int(ev["global_index"])
        if key in SENTENCE_BOUNDARY_KEYS:
            return normalize_text(sd.sentence_end())
        if key in WORD_BOUNDARY_KEYS:
            sd.word_boundary(top_k=10)
            continue
        if key == "backspace":
            if sd._current_word_probs:
                sd._current_word_probs.pop()
            continue
        sd.push_keystroke(probs[idx])

    return normalize_text(sd.sentence_end())


def decode_eval_sentences(sentence_records: list[dict], eval_ids: set[tuple[str, int]],
                          probs: np.ndarray, classes: np.ndarray,
                          beam_width: int, alpha: float,
                          lm: Optional[NgramLanguageModel] = None) -> tuple[list[str], list[str]]:
    if lm is None:
        lm = NgramLanguageModel(smoothing=1.0, bigram_weight=0.4)
    wd = WordDecoder(lm, beam_width=beam_width, top_chars=6, alpha=alpha)
    refs = []
    hyps = []
    for rec in sentence_records:
        sid = (rec["session"], int(rec["sentence_idx"]))
        if sid not in eval_ids:
            continue
        refs.append(normalize_text(rec["reference"]))
        hyps.append(decode_sentence_with_backspace(rec["events"], probs, classes, wd, lm))
    return refs, hyps


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description="Fine-tune single-key model on free_type + beam decoding")
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
        help="Data selectors to include (default: free_type). "
             "Supports folder names under data/raw, direct paths, or legacy round numbers.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.2, help="sentence-level eval split ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed for split")
    parser.add_argument("--epochs", type=int, default=12, help="fine-tuning epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="fine-tuning learning rate")
    parser.add_argument("--batch-size", type=int, default=64, help="fine-tuning batch size")
    parser.add_argument("--aug-prob", type=float, default=0.5, help="fine-tune augmentation probability")
    parser.add_argument("--beam", type=int, default=100, help="beam width for decoder")
    parser.add_argument("--alpha", type=float, default=0.15, help="LM weight for decoder")
    parser.add_argument(
        "--beam-grid",
        nargs="*",
        type=int,
        default=None,
        help="Optional beam-size grid, e.g. --beam-grid 50 100 150",
    )
    parser.add_argument(
        "--alpha-grid",
        nargs="*",
        type=float,
        default=None,
        help="Optional alpha grid, e.g. --alpha-grid 0.05 0.15 0.3",
    )
    parser.add_argument("--prior-weight", type=float, default=1.0, help="calibration prior correction strength")
    parser.add_argument("--special-weight", type=float, default=0.7, help="calibration special-key correction strength")
    parser.add_argument("--prompt-aware", action="store_true", default=True,
                        help="Also evaluate prompt-constrained upper-bound decoding (default: on)")
    args = parser.parse_args()
    DEVICE = set_device(args.device)
    print(f"Torch device: {DEVICE}")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("== discover free_type sessions ==")
    sessions = discover_freetype_sessions(args.rounds)
    if not sessions:
        raise RuntimeError("No free_type sessions found")

    # Keep free_type keystroke sequence as complete as possible.
    wcfg = WindowConfig(min_window_samples=2)
    qualities = [assess_session(s, wcfg) for s in sessions]
    valid = [q for q in qualities if q.is_valid]
    dropped = [q for q in qualities if not q.is_valid]
    if not valid:
        raise RuntimeError("No valid free_type sessions after quality check")

    print(f"valid sessions={len(valid)} dropped={len(dropped)}")

    X, payload = build_events_and_dataset(valid, wcfg)
    y = payload["y"]
    sentence_records_all = payload["sentence_records"]
    sentence_records = keep_yes_attempts_only(sentence_records_all)
    print(f"free_type dataset: X={X.shape}, sentences(all)={len(sentence_records_all)}, sentences(YES)={len(sentence_records)}")

    model, classes, class_to_idx, means, stds = load_model_and_scaler()
    y_idx = np.array([class_to_idx.get(k, -1) for k in y], dtype=np.int32)

    split = sentence_level_split(sentence_records, eval_ratio=args.eval_ratio, seed=args.seed)
    print(f"split: train_sent={len(split.train_sentence_ids)} eval_sent={len(split.eval_sentence_ids)}")
    print(f"split: train_keys={len(split.train_indices)} eval_keys={len(split.eval_indices)}")

    # before fine-tune (on eval key events only)
    _, probs_before = infer_logits_probs(model, X, means, stds)
    key_before = compute_topk_metrics(probs_before[split.eval_indices], y_idx[split.eval_indices])

    eval_ids = set(split.eval_sentence_ids)
    refs_before, hyps_before = decode_eval_sentences(
        sentence_records, eval_ids, probs_before, classes, beam_width=args.beam, alpha=args.alpha
    )
    text_before = compute_text_metrics(refs_before, hyps_before)

    print("== fine-tuning ==")
    model_ft = fine_tune_model(
        model=model,
        X=X,
        y_idx=y_idx,
        train_indices=split.train_indices,
        means=means,
        stds=stds,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        aug_prob=args.aug_prob,
    )

    torch.save(model_ft.state_dict(), FINETUNE_MODEL_PATH)
    print(f"saved fine-tuned weights: {FINETUNE_MODEL_PATH}")

    logits_after, probs_after_raw = infer_logits_probs(model_ft, X, means, stds)
    # Post-fine-tune calibration on train split only
    train_true = y_idx[split.train_indices]
    train_logits = logits_after[split.train_indices]
    best_t, train_nll = fit_temperature(train_logits, train_true)
    merged_path = "data/processed/merged_dataset.npz"
    if os.path.exists(merged_path):
        merged = np.load(merged_path, allow_pickle=True)
        single_prior = class_prior_from_labels(merged["y"], class_to_idx, len(classes), smooth=1.0)
    else:
        single_prior = class_prior_from_labels(y, class_to_idx, len(classes), smooth=1.0)
    free_train_prior = class_prior_from_labels(y[split.train_indices], class_to_idx, len(classes), smooth=1.0)
    probs_after, calib_info = apply_calibration(
        logits=logits_after,
        true_idx=y_idx,
        classes=classes,
        train_prior=single_prior,
        free_prior=free_train_prior,
        temperature=best_t,
        prior_weight=args.prior_weight,
        special_weight=args.special_weight,
    )
    calib_info["temperature_fit_nll_train"] = float(train_nll)
    key_after_raw = compute_topk_metrics(probs_after_raw[split.eval_indices], y_idx[split.eval_indices])
    key_after = compute_topk_metrics(probs_after[split.eval_indices], y_idx[split.eval_indices])

    refs_after, hyps_after = decode_eval_sentences(
        sentence_records, eval_ids, probs_after, classes, beam_width=args.beam, alpha=args.alpha
    )
    text_after = compute_text_metrics(refs_after, hyps_after)

    grid_rows = []
    best_grid = None
    if args.beam_grid:
        alpha_grid = args.alpha_grid if args.alpha_grid else [args.alpha]
        for beam in args.beam_grid:
            for alpha in alpha_grid:
                refs_g, hyps_g = decode_eval_sentences(
                    sentence_records, eval_ids, probs_after, classes, beam_width=beam, alpha=alpha
                )
                m = compute_text_metrics(refs_g, hyps_g)
                grid_rows.append({
                    "beam": int(beam),
                    "alpha": float(alpha),
                    "cer": float(m["cer"]),
                    "wer": float(m["wer"]),
                    "sentence_exact_match": float(m["sentence_exact_match"]),
                })
        grid_rows.sort(key=lambda r: (r["wer"], r["cer"], -r["sentence_exact_match"]))
        best_grid = grid_rows[0]

    prompt_upper = None
    if args.prompt_aware:
        prompt_lm = PromptConstrainedLM(PROMPT_POOL, smoothing=1.0, bigram_weight=0.7)
        beam_for_prompt = int(best_grid["beam"]) if best_grid else int(args.beam)
        alpha_for_prompt = float(best_grid["alpha"]) if best_grid else float(args.alpha)
        refs_p, hyps_p = decode_eval_sentences(
            sentence_records, eval_ids, probs_after, classes,
            beam_width=beam_for_prompt, alpha=alpha_for_prompt, lm=prompt_lm
        )
        m_p = compute_text_metrics(refs_p, hyps_p)
        prompt_upper = {
            "beam": beam_for_prompt,
            "alpha": alpha_for_prompt,
            "metrics": m_p,
        }

    report = {
        "config": {
            "device": str(DEVICE),
            "rounds": args.rounds,
            "eval_ratio": args.eval_ratio,
            "seed": args.seed,
            "epochs": args.epochs,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "aug_prob": args.aug_prob,
            "beam": args.beam,
            "alpha": args.alpha,
            "prior_weight": args.prior_weight,
            "special_weight": args.special_weight,
        },
        "data": {
            "session_count": len(sessions),
            "valid_session_count": len(valid),
            "dropped_session_count": len(dropped),
            "sentence_count": len(sentence_records),
            "train_sentence_count": len(split.train_sentence_ids),
            "eval_sentence_count": len(split.eval_sentence_ids),
            "train_keystroke_count": int(len(split.train_indices)),
            "eval_keystroke_count": int(len(split.eval_indices)),
        },
        "metrics_before_finetune": {
            "keystroke_topk": key_before,
            "beam_decode": text_before,
        },
        "metrics_after_finetune": {
            "keystroke_topk_raw": key_after_raw,
            "keystroke_topk": key_after,
            "beam_decode": text_after,
        },
        "calibration": calib_info,
        "beam_search_optimization": {
            "current_config": {"beam": int(args.beam), "alpha": float(args.alpha)},
            "current_metrics": {
                "cer": float(text_after["cer"]),
                "wer": float(text_after["wer"]),
                "sentence_exact_match": float(text_after["sentence_exact_match"]),
            },
            "grid_candidates": grid_rows,
            "best_config": best_grid,
            "delta_best_minus_current": (
                None if best_grid is None else {
                    "cer": float(best_grid["cer"] - text_after["cer"]),
                    "wer": float(best_grid["wer"] - text_after["wer"]),
                    "sentence_exact_match": float(best_grid["sentence_exact_match"] - text_after["sentence_exact_match"]),
                }
            ),
        },
        "prompt_aware_upper_bound": prompt_upper,
        "delta_after_minus_before": {
            "top1": key_after["top1"] - key_before["top1"],
            "top3": key_after["top3"] - key_before["top3"],
            "top5": key_after["top5"] - key_before["top5"],
            "cer": text_after["cer"] - text_before["cer"],
            "wer": text_after["wer"] - text_before["wer"],
            "sentence_exact_match": text_after["sentence_exact_match"] - text_before["sentence_exact_match"],
        },
        "examples_eval": [
            {"reference": r, "before": b, "after": a}
            for r, b, a in list(zip(refs_after, hyps_before, hyps_after))[:8]
        ],
    }

    # lightweight diagnosis if not ideal
    notes = []
    if key_after["top1"] < 0.4:
        notes.append("keystroke top1 is still low; consider expanding free_type coverage and more balanced digits/backspace samples")
    if text_after["wer"] > 0.5:
        notes.append("WER is high; decoder/LM adaptation and more natural sentence transitions are needed")
    if not notes:
        notes.append("fine-tuning + beam decoding is usable for current closure benchmark")
    report["analysis_notes"] = notes

    os.makedirs("results", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
