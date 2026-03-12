"""
Phase 3: Language Decoder
==========================
Takes per-keystroke softmax probability vectors from the Ensemble model
and recovers the most likely English words/sentences using:

  1. Beam Search   — explores top-K character paths through the prob matrix
  2. Dictionary    — filters beam candidates to real English words
  3. N-gram LM     — ranks candidates by word frequency + bigram context
  4. Viterbi       — optimal word sequence for multi-word sentences

TWO MODES
---------
A) Evaluation mode  (default)
   Loads results/ensemble_probs.npz, synthetically assembles random
   3–8 letter words from the test keystroke probs, decodes them, and
   reports: raw accuracy / beam@1 / beam@3 / LM-reranked.
   Use this to measure the decoder's benefit before you have free_type data.

B) Live / sentence mode
   WordDecoder and SentenceDecoder classes are importable for real-time use.
   Feed them one probability vector per keystroke; they emit word candidates.

Run:
  .venv/bin/python3 phase3_decoder.py              # eval mode
  .venv/bin/python3 phase3_decoder.py --demo       # demo on fixed sentences
  .venv/bin/python3 phase3_decoder.py --sentence "hello world"  # encode+decode

Requires:
  pip install nltk
  python -c "import nltk; nltk.download('words'); nltk.download('brown')"
"""

import os
import re
import sys
import json
import math
import heapq
import argparse
import numpy as np
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from typing import Optional

# ── NLTK (optional but strongly recommended) ─────────────────
try:
    import nltk
    from nltk.corpus import words as nltk_words
    from nltk.corpus import brown
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False
    print("  ⚠ nltk not installed. Run: pip install nltk")
    print("    Then: python -c \"import nltk; nltk.download('words'); nltk.download('brown')\"")


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — LANGUAGE MODEL
# ══════════════════════════════════════════════════════════════

class NgramLanguageModel:
    """
    Smoothed unigram + bigram + trigram language model built from the Brown corpus.
    Falls back to a hard-coded top-5000 word frequency list if NLTK unavailable.

    Scoring:
      log P(word)         = log unigram probability (Laplace smoothed)
      log P(word | prev)  = log bigram probability  (Kneser-Ney inspired,
                            interpolated with unigram)
      log P(word | p2,p1) = trigram probability interpolated with bigram/unigram
    """

    # Minimal fallback word list (most common English words) used when NLTK missing
    FALLBACK_WORDS = (
        "the be to of and a in that have it for not on with he as you do at "
        "this but his by from they we say her she or an will my one all would "
        "there their what so up out if about who get which go me when make can "
        "like time no just him know take people into year your good some could "
        "them see other than then now look only come its over think also back "
        "after use two how our work first well way even new want because any "
        "these give day most us great between need large often hand high place "
        "hold turn were said each tell does set three want air well also play "
        "small number off always move try kind hand again change world off "
        "every near add food keep children begin got walk example ease paper "
        "group always music those both mark book letter until mile river car "
        "feet care second enough plain girl usual young ready above ever red "
        "list though feel talk bird soon body dog family direct pose leave "
        "song measure door product black short numeral class wind question "
        "happen complete ship area half rock order fire south problem piece "
        "told knew pass since top whole king space heard best hour better true "
        "during hundred five remember step early hold west ground interest "
        "reach fast verb sing listen six table travel less morning ten simple "
        "several vowel toward war lay against pattern slow center love person "
        "money serve appear road map rain rule govern pull cold notice voice "
        "unit power town fine drive led cry dark machine note wait plan figure "
        "star box noun field rest correct able pound done beauty drive stood "
        "contain front teach week final gave green oh quick develop ocean warm "
        "free minute strong special mind behind clear tail produce fact street "
        "inch multiply nothing course stay wheel full force blue object decide "
        "surface deep moon island foot system busy test record boat common gold "
        "possible plane age wonder laugh thousand ago ran check game shape miss "
        "brought heat snow tire bring yes distant fill east paint language "
        "among grand ball yet wave drop heart present heavy dance engine "
        "position arm wide sail material fraction forest sit race window store "
        "summer train sleep prove lone leg exercise wall catch mount wish sky "
        "board joy winter sat written wild instrument kept glass grass cow job "
        "edge sign visit past soft fun bright gas weather month million bear "
        "finish happy hope flower clothe strange gone jump baby eight village "
        "meet root buy raise solve metal whether push seven paragraph third "
        "held hair describe cook floor either result burn hill safe cat century "
        "consider type law bit coast copy phrase silent tall sand soil roll "
        "temperature finger industry value fight lie beat excite natural view "
        "plain queen breath stick letter enjoy indicate character symbol die "
        "least trouble shout except wrote seed tone join suggest clean break "
        "lady yard rise bad blow oil blood touch grew cent mix team wire cost "
        "lost brown wear garden equal sent choose fell fit flow fair bank "
        "collect save control decimal gentle woman captain practice separate "
        "difficult doctor please protect noon whose locate ring character serve"
    ).split()

    def __init__(self, smoothing: float = 1.0, bigram_weight: float = 0.4,
                 trigram_weight: float = 0.2):
        """
        Args:
            smoothing:      Laplace smoothing count added to every word
            bigram_weight:  interpolation weight for bigram vs unigram (0-1)
            trigram_weight: interpolation weight for trigram (0-1)
        """
        self.smoothing = smoothing
        self.bigram_weight = bigram_weight
        self.trigram_weight = trigram_weight
        self.unigram: Counter = Counter()
        self.bigram: defaultdict = defaultdict(Counter)
        self.trigram: defaultdict = defaultdict(Counter)
        self.vocab: set = set()
        self._build()

    def _build(self):
        if HAS_NLTK:
            self._build_from_brown()
        else:
            self._build_from_fallback()
        print(f"  LM built: {len(self.vocab):,} vocab words, "
              f"{sum(self.unigram.values()):,} unigram tokens")

    def _build_from_brown(self):
        """Build from Brown corpus (balanced, ~1M words)."""
        try:
            sents = brown.sents()
        except LookupError:
            nltk.download('brown', quiet=True)
            sents = brown.sents()

        for sent in sents:
            tokens = [w.lower() for w in sent if w.isalpha()]
            for i, tok in enumerate(tokens):
                self.unigram[tok] += 1
                self.vocab.add(tok)
                if i > 0:
                    self.bigram[tokens[i-1]][tok] += 1
                if i > 1:
                    self.trigram[(tokens[i-2], tokens[i-1])][tok] += 1

        # Also load the full word list for dictionary lookup (Brown only has ~50k types)
        try:
            word_list = set(w.lower() for w in nltk_words.words() if w.isalpha())
        except LookupError:
            nltk.download('words', quiet=True)
            word_list = set(w.lower() for w in nltk_words.words() if w.isalpha())
        # Add dictionary words with smoothing count so they appear in vocab
        for w in word_list:
            self.vocab.add(w)
            if w not in self.unigram:
                self.unigram[w] = 0   # smoothing handles the count

    def _build_from_fallback(self):
        """Use the embedded minimal word list."""
        tokens = [w.lower() for w in self.FALLBACK_WORDS if w.isalpha()]
        for i, w in enumerate(tokens):
            self.unigram[w] += 1
            self.vocab.add(w)
            if i > 0:
                self.bigram[tokens[i - 1]][w] += 1
            if i > 1:
                self.trigram[(tokens[i - 2], tokens[i - 1])][w] += 1
        print("  ⚠ Using fallback word list (install nltk for better results)")

    def word_log_prob(self, word: str, prev_word: Optional[str] = None,
                      prev_prev_word: Optional[str] = None) -> float:
        """
        Log probability of word with optional bigram/trigram context.
        Interpolation:
          P = w3*P_trigram + w2*P_bigram + w1*P_unigram
        """
        word = word.lower()
        V = max(len(self.vocab), 1)

        # Unigram (Laplace smoothed)
        uni_count = self.unigram.get(word, 0) + self.smoothing
        uni_total = sum(self.unigram.values()) + self.smoothing * V
        p_uni = uni_count / uni_total

        if prev_word is None:
            return math.log(p_uni + 1e-300)

        # Bigram
        prev = prev_word.lower()
        if self.bigram[prev]:
            bi_count = self.bigram[prev].get(word, 0) + self.smoothing
            bi_total = sum(self.bigram[prev].values()) + self.smoothing * V
            p_bi = bi_count / bi_total
        else:
            p_bi = p_uni

        # Trigram (optional)
        p_tri = None
        if prev_prev_word is not None:
            prev2 = prev_prev_word.lower()
            key = (prev2, prev)
            if self.trigram[key]:
                tri_count = self.trigram[key].get(word, 0) + self.smoothing
                tri_total = sum(self.trigram[key].values()) + self.smoothing * V
                p_tri = tri_count / tri_total

        # Interpolate with safe normalized weights
        if p_tri is None:
            w_bi = min(max(self.bigram_weight, 0.0), 1.0)
            w_uni = 1.0 - w_bi
            p = w_bi * p_bi + w_uni * p_uni
            return math.log(p + 1e-300)

        w_tri = min(max(self.trigram_weight, 0.0), 1.0)
        w_bi = min(max(self.bigram_weight, 0.0), 1.0 - w_tri)
        w_uni = 1.0 - w_tri - w_bi
        p = w_tri * p_tri + w_bi * p_bi + w_uni * p_uni
        return math.log(p + 1e-300)

    def is_valid_word(self, word: str) -> bool:
        return word.lower() in self.vocab


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — BEAM SEARCH WORD DECODER
# ══════════════════════════════════════════════════════════════

@dataclass(order=True)
class BeamState:
    """A single hypothesis in the beam."""
    neg_score: float                        # negative for min-heap
    prefix: str = field(compare=False)      # characters accumulated so far
    log_prob: float = field(compare=False)  # sum of log acoustic probs

    @property
    def score(self) -> float:
        return -self.neg_score


class WordDecoder:
    """
    Decodes a sequence of per-keystroke probability vectors into the most
    likely English word using beam search + dictionary filtering.

    Algorithm:
      For each timestep t with prob vector p_t (length = n_classes):
        For each beam state (prefix, log_prob):
          For each top-M characters (by p_t):
            new_prefix = prefix + char
            new_log_prob = log_prob + log(p_t[char])
            push to next beam
      Prune beam to top-K states.
      After all timesteps:
        - if token looks alphabetic, prefer valid dictionary words and
          run illegal→legal rescue using confidence-aware lexical search.
        - if token looks alphanumeric, keep acoustic candidates (digits allowed).

    Args:
        lm:         NgramLanguageModel instance
        beam_width: number of beam hypotheses kept per step (default 100)
        top_chars:  number of top characters expanded per step (default 8)
        alpha:      LM weight in final re-ranking (0 = acoustic only)
        min_len:    minimum word length to consider (default 2)
    """

    # Backward-compatible set used by some evaluation helpers (alpha-only filters).
    # Keeps digits here so old alpha-only simulation behavior remains unchanged.
    NON_ALPHA = {"space", "enter", "backspace", "shift", "capslock",
                 "ctrl", "alt", "cmd", "tab", "esc", "delete",
                 ",", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    # Actual token exclusions for decoding; digits are allowed as token chars.
    NON_TOKEN = {"space", "enter", "backspace", "shift", "capslock",
                 "ctrl", "alt", "cmd", "tab", "esc", "delete",
                 ",", "."}

    def __init__(self, lm: NgramLanguageModel, beam_width: int = 100,
                 top_chars: int = 8, alpha: float = 0.5, min_len: int = 2,
                 lock_confidence: float = 0.97):
        self.lm = lm
        self.beam_width = beam_width
        self.top_chars = top_chars
        self.alpha = alpha
        self.min_len = min_len
        self.lock_confidence = lock_confidence
        self._alpha_vocab_by_len: Optional[dict[int, list[str]]] = None

    @staticmethod
    def _levenshtein(a: str, b: str) -> int:
        n, m = len(a), len(b)
        if n == 0:
            return m
        if m == 0:
            return n
        dp = np.zeros((n + 1, m + 1), dtype=np.int32)
        dp[:, 0] = np.arange(n + 1)
        dp[0, :] = np.arange(m + 1)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[i, j] = min(
                    dp[i - 1, j] + 1,
                    dp[i, j - 1] + 1,
                    dp[i - 1, j - 1] + cost,
                )
        return int(dp[n, m])

    @staticmethod
    def _score_fixed_length_word(word: str,
                                 log_prob_matrix: np.ndarray,
                                 class_to_idx: dict[str, int]) -> float:
        if len(word) != log_prob_matrix.shape[0]:
            return -1e18
        score = 0.0
        for i, ch in enumerate(word):
            idx = class_to_idx.get(ch)
            if idx is None:
                return -1e18
            score += float(log_prob_matrix[i, idx])
        return score

    def _looks_like_alphabetic_token(self, prob_matrix: np.ndarray,
                                     classes: np.ndarray) -> bool:
        digit_idx = [i for i, c in enumerate(classes) if len(c) == 1 and c.isdigit()]
        if not digit_idx:
            return True
        top_idx = np.argmax(prob_matrix, axis=1)
        digit_top_votes = sum(
            1 for i in top_idx if len(classes[i]) == 1 and classes[i].isdigit()
        )
        digit_mass = float(np.mean(np.sum(prob_matrix[:, digit_idx], axis=1)))
        # If digit evidence is strong, avoid forcing dictionary words.
        if digit_mass >= 0.20:
            return False
        if digit_top_votes >= max(1, int(round(0.20 * len(top_idx)))):
            return False
        return True

    def _build_alpha_vocab_index(self) -> None:
        if self._alpha_vocab_by_len is not None:
            return
        by_len: dict[int, list[str]] = defaultdict(list)
        for w in self.lm.vocab:
            wl = w.lower()
            if wl.isalpha():
                by_len[len(wl)].append(wl)
        self._alpha_vocab_by_len = dict(by_len)

    def _legal_word_rescue(self,
                           prob_matrix: np.ndarray,
                           log_prob_matrix: np.ndarray,
                           classes: np.ndarray,
                           class_to_idx: dict[str, int],
                           prev_word: Optional[str]) -> list[tuple[str, float]]:
        """
        Dictionary rescue for invalid alpha words.
        Keeps word length fixed to current token length to reduce
        accidental under/over-length decoding.
        """
        T = prob_matrix.shape[0]
        if T < self.min_len:
            return []

        self._build_alpha_vocab_index()
        candidates = (self._alpha_vocab_by_len or {}).get(T, [])
        if not candidates:
            return []

        top_idx = np.argmax(prob_matrix, axis=1)
        top_probs = np.max(prob_matrix, axis=1)
        top_chars = [classes[i] for i in top_idx]
        raw_top1 = "".join(ch if len(ch) == 1 and ch.isalpha() else "?" for ch in top_chars)

        locked: dict[int, str] = {}
        for i, (ch, p) in enumerate(zip(top_chars, top_probs)):
            if len(ch) == 1 and ch.isalpha() and p >= self.lock_confidence:
                locked[i] = ch

        prefix_hint = ""
        for i in range(min(2, T)):
            ch = top_chars[i]
            if len(ch) == 1 and ch.isalpha() and top_probs[i] >= 0.80:
                prefix_hint += ch
            else:
                break

        k = max(self.top_chars, 4)
        pos_top_chars: list[set[str]] = []
        for t in range(T):
            idxs = np.argsort(prob_matrix[t])[::-1][:k]
            chset = {
                classes[i] for i in idxs
                if len(classes[i]) == 1 and classes[i].isalpha()
            }
            pos_top_chars.append(chset)

        rescued = []
        for w in candidates:
            if prefix_hint and not w.startswith(prefix_hint):
                continue
            bad_lock = False
            for i, ch in locked.items():
                if i >= len(w) or w[i] != ch:
                    bad_lock = True
                    break
            if bad_lock:
                continue

            mismatch_positions = sum(
                1 for i, ch in enumerate(w)
                if ch not in pos_top_chars[i]
            )
            if mismatch_positions > max(1, T // 3):
                continue

            ac_score = self._score_fixed_length_word(w, log_prob_matrix, class_to_idx)
            if ac_score < -1e17:
                continue
            lm_score = self.lm.word_log_prob(w, prev_word)
            edit_penalty = self._levenshtein(w, raw_top1)
            score = ac_score + (self.alpha * 0.9) * lm_score - 0.30 * edit_penalty
            rescued.append((w, score))

        rescued.sort(key=lambda x: -x[1])
        return rescued[:self.beam_width]

    def decode_word(self, prob_matrix: np.ndarray, classes: np.ndarray,
                    prev_word: Optional[str] = None) -> list[tuple[str, float]]:
        """
        Decode one word from a prob_matrix of shape (T, n_classes).

        Args:
            prob_matrix: (T, n_classes) — one row per keystroke in this word
            classes:     (n_classes,)  — string label for each class index
            prev_word:   previous decoded word (for bigram LM scoring)

        Returns:
            List of (word, combined_score) sorted by descending score.
            combined_score = acoustic_log_prob + alpha * lm_log_prob
        """
        T, n_classes = prob_matrix.shape
        log_prob_matrix = np.log(prob_matrix + 1e-12)

        # Build char→index lookup (letters + digits, skip control symbols)
        token_indices = [(i, classes[i]) for i in range(n_classes)
                         if classes[i] not in self.NON_TOKEN and len(classes[i]) == 1]
        if not token_indices:
            return []
        class_to_idx = {ch: i for i, ch in token_indices}

        # Beam: list of (neg_score, prefix, log_prob)
        beam: list[BeamState] = [BeamState(0.0, "", 0.0)]

        for t in range(T):
            # Top-M character indices by probability at this timestep
            step_probs = [(log_prob_matrix[t, i], ch) for i, ch in token_indices]
            step_probs.sort(reverse=True)
            top_chars = step_probs[:self.top_chars]

            next_candidates = []
            for state in beam:
                for log_p, ch in top_chars:
                    new_log_prob = state.log_prob + log_p
                    next_candidates.append(
                        BeamState(-new_log_prob, state.prefix + ch, new_log_prob)
                    )

            # Keep top-K by acoustic score
            next_candidates.sort()
            beam = next_candidates[:self.beam_width]

        # Deduplicate by surface form
        ranked = []
        seen = set()
        for state in beam:
            word = state.prefix.lower()
            if len(word) < self.min_len:
                continue
            if word in seen:
                continue
            seen.add(word)
            ranked.append((word, state.log_prob))

        if not ranked:
            return []

        # If token looks numeric/alphanumeric, do not force dictionary words.
        if not self._looks_like_alphabetic_token(prob_matrix, classes):
            mixed = []
            for word, ac_score in ranked:
                bonus = 0.0
                if word.isalpha() and self.lm.is_valid_word(word):
                    bonus = (self.alpha * 0.05) * self.lm.word_log_prob(word, prev_word)
                mixed.append((word, ac_score + bonus))
            mixed.sort(key=lambda x: -x[1])
            return mixed

        # Alpha mode: prioritize legal words + explicit illegal→legal rescue.
        valid_results = []
        invalid_results = []
        for word, ac_score in ranked:
            if word.isalpha() and self.lm.is_valid_word(word):
                valid_results.append((word, ac_score))
            else:
                invalid_results.append((word, ac_score))

        valid_reranked = []
        lm_tiebreak = self.alpha * 0.2
        for word, ac_score in valid_results:
            lm_score = self.lm.word_log_prob(word, prev_word)
            valid_reranked.append((word, ac_score + lm_tiebreak * lm_score))
        valid_reranked.sort(key=lambda x: -x[1])

        rescued_legal = self._legal_word_rescue(
            prob_matrix=prob_matrix,
            log_prob_matrix=log_prob_matrix,
            classes=classes,
            class_to_idx=class_to_idx,
            prev_word=prev_word,
        )

        legal_map: dict[str, float] = {}
        for w, s in valid_reranked + rescued_legal:
            if w not in legal_map or s > legal_map[w]:
                legal_map[w] = s
        legal_sorted = sorted(legal_map.items(), key=lambda x: -x[1])

        if legal_sorted:
            invalid_scored = []
            for w, s in invalid_results:
                lm_score = self.lm.word_log_prob(w, prev_word)
                invalid_scored.append((w, s + self.alpha * 0.05 * lm_score))
            invalid_scored.sort(key=lambda x: -x[1])
            return legal_sorted + invalid_scored

        # No legal word found: return LM-weighted invalids.
        rescued = []
        for word, ac_score in invalid_results:
            lm_score = self.lm.word_log_prob(word, prev_word)
            rescued.append((word, ac_score + self.alpha * lm_score))
        rescued.sort(key=lambda x: -x[1])
        return rescued

    def top1(self, prob_matrix: np.ndarray, classes: np.ndarray,
             prev_word: Optional[str] = None) -> str:
        """Return the single best word."""
        results = self.decode_word(prob_matrix, classes, prev_word)
        if results:
            return results[0][0]
        # Fallback: argmax at each timestep
        return "".join(classes[np.argmax(prob_matrix[t])]
                       for t in range(len(prob_matrix))
                       if classes[np.argmax(prob_matrix[t])] not in self.NON_TOKEN)


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — SENTENCE DECODER (Viterbi over word sequence)
# ══════════════════════════════════════════════════════════════

class SentenceDecoder:
    """
    Decodes a full sentence from a sequence of words (each with candidate lists).

    Uses Viterbi algorithm over the word sequence:
      - States:     candidate words for each position
      - Emission:   acoustic score of the candidate word
      - Transition: bigram LM log-prob P(word_i | word_{i-1})

    For real-time use:
      1. Feed keystrokes one by one via push_keystroke(prob_vec, char_class)
      2. Call word_boundary() when space is detected → decodes current word
      3. Call finalize() at sentence end (enter key) → returns best sentence
    """

    def __init__(self, word_decoder: WordDecoder, lm: NgramLanguageModel,
                 beam_sentences: int = 20):
        self.wd = word_decoder
        self.lm = lm
        self.beam_sentences = beam_sentences

        # Real-time state
        self._current_word_probs: list[np.ndarray] = []  # prob vectors for current word
        self._sentence_candidates: list[list[tuple[str, float]]] = []
        self._classes: Optional[np.ndarray] = None

    # ── Real-time interface ───────────────────────────────────

    def set_classes(self, classes: np.ndarray):
        """Must be called once before push_keystroke."""
        self._classes = classes

    def push_keystroke(self, prob_vec: np.ndarray):
        """
        Add a new keystroke probability vector to the current word buffer.
        Call word_boundary() or sentence_end() instead of this for space/enter.
        """
        self._current_word_probs.append(prob_vec)

    def word_boundary(self, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Space detected: decode the accumulated word buffer and store candidates.
        Returns the top-k word candidates for the just-finished word.
        """
        if not self._current_word_probs or self._classes is None:
            return []

        mat = np.stack(self._current_word_probs, axis=0)   # (T, n_classes)
        prev_word_hint = None
        if self._sentence_candidates and self._sentence_candidates[-1]:
            prev_word_hint = self._sentence_candidates[-1][0][0]
        candidates = self.wd.decode_word(mat, self._classes, prev_word=prev_word_hint)[:top_k]
        self._sentence_candidates.append(candidates)
        self._current_word_probs = []
        return candidates

    def sentence_end(self) -> str:
        """Enter detected: flush last word and run Viterbi over all words."""
        # Flush remaining buffer as last word
        if self._current_word_probs:
            self.word_boundary()
        sentence = self.viterbi(self._sentence_candidates)
        # Reset
        self._sentence_candidates = []
        self._current_word_probs = []
        return sentence

    # ── Batch / offline interface ─────────────────────────────

    def decode_sentence_from_word_probs(
            self,
            word_prob_matrices: list[np.ndarray],
            classes: np.ndarray,
            top_k: int = 10) -> str:
        """
        Decode a full sentence from a list of per-word prob matrices.

        Args:
            word_prob_matrices: list of (T_i, n_classes) arrays, one per word
            classes:            class label array
            top_k:              beam width for word candidates

        Returns:
            Best decoded sentence string.
        """
        all_candidates = []
        prev_word_hint = None
        for mat in word_prob_matrices:
            candidates = self.wd.decode_word(mat, classes, prev_word=prev_word_hint)[:top_k]
            all_candidates.append(candidates)
            if candidates:
                prev_word_hint = candidates[0][0]
        return self.viterbi(all_candidates)

    def viterbi(self, word_candidates: list[list[tuple[str, float]]]) -> str:
        """
        Viterbi over word sequence.

        State at position i = one candidate word.
        Emission = acoustic+LM score from WordDecoder.
        Transition = bigram LM log-prob P(curr | prev).
        """
        if not word_candidates:
            return ""

        # beam: list of (total_score, [word_sequence])
        # Initialise with first word candidates
        beam = []
        for word, score in word_candidates[0][:self.beam_sentences]:
            heapq.heappush(beam, (-score, [word]))

        for pos in range(1, len(word_candidates)):
            next_beam = []
            for neg_score, seq in beam:
                prev_word = seq[-1]
                prev_prev_word = seq[-2] if len(seq) >= 2 else None
                for word, acoustic_score in word_candidates[pos][:self.beam_sentences]:
                    transition = self.lm.word_log_prob(word, prev_word, prev_prev_word)
                    total = -neg_score + acoustic_score + transition
                    heapq.heappush(next_beam, (-total, seq + [word]))
            # Prune
            next_beam.sort()
            beam = next_beam[:self.beam_sentences]

        if not beam:
            return " ".join(cands[0][0] if cands else "?" for cands in word_candidates)

        _, best_seq = beam[0]
        return " ".join(best_seq)


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — EVALUATION ENGINE
# ══════════════════════════════════════════════════════════════

def simulate_word_decoding(
        probs: np.ndarray,       # (N, n_classes) — all test keystroke probs
        y_true: np.ndarray,      # (N,)           — true encoded labels
        classes: np.ndarray,     # (n_classes,)   — class strings
        lm: NgramLanguageModel,
        word_decoder: WordDecoder,
        n_words: int = 500,
        min_word_len: int = 3,
        max_word_len: int = 8,
        seed: int = 42,
) -> dict:
    """
    Simulate word-level decoding on isolated keystroke data.

    Since ensemble_probs.npz contains individual isolated keystrokes (not
    sentence sequences), this function synthetically groups them into words:
      1. Filter to only alphabetic keystroke samples.
      2. Group consecutive samples whose true labels spell a valid English word.
      3. Run beam decoder on each group and measure accuracy.

    This is a conservative lower bound on real-world performance because:
      - Real typed words have more consistent rhythm than assembled chunks.
      - The assembler uses random consecutive samples, not actual typed words.

    Returns dict with accuracy metrics.
    """
    rng = np.random.default_rng(seed)

    # Build alpha-only index
    alpha_mask = np.array([c not in WordDecoder.NON_ALPHA and len(c) == 1
                           for c in classes])
    alpha_class_set = set(c for c in classes if c not in WordDecoder.NON_ALPHA and len(c) == 1)

    # Filter samples to alpha-only keystrokes
    true_labels = classes[y_true]
    alpha_sample_idx = np.where(np.array([c in alpha_class_set for c in true_labels]))[0]

    if len(alpha_sample_idx) < max_word_len:
        print("  ⚠ Not enough alpha samples for word simulation")
        return {}

    # Build per-character pools
    char_pool: dict[str, list[int]] = defaultdict(list)
    for idx in alpha_sample_idx:
        char_pool[true_labels[idx]].append(idx)

    # Load a word list to sample target words from
    if HAS_NLTK:
        try:
            word_list = [w.lower() for w in nltk_words.words()
                         if w.isalpha() and min_word_len <= len(w) <= max_word_len
                         and all(c in char_pool for c in w.lower())]
        except LookupError:
            nltk.download('words', quiet=True)
            word_list = [w.lower() for w in nltk_words.words()
                         if w.isalpha() and min_word_len <= len(w) <= max_word_len
                         and all(c in char_pool for c in w.lower())]
    else:
        word_list = [w for w in lm.vocab
                     if min_word_len <= len(w) <= max_word_len
                     and all(c in char_pool for c in w)]

    if not word_list:
        print("  ⚠ No valid target words found for simulation")
        return {}

    # Sample target words
    target_words = rng.choice(word_list,
                              size=min(n_words, len(word_list)),
                              replace=False).tolist()

    # Metrics
    raw_correct = 0        # argmax at each step, then concat
    beam1_correct = 0      # beam top-1 (acoustic only, no LM)
    lm_correct = 0         # beam top-1 with LM reranking
    top3_correct = 0       # answer in top-3 beam candidates
    top5_correct = 0       # answer in top-5 beam candidates
    n_valid = 0

    for word in target_words:
        # Build prob matrix by sampling one keystroke per character from the pool
        try:
            word_matrix = np.stack(
                [probs[rng.choice(char_pool[c])] for c in word], axis=0
            )   # (len(word), n_classes)
        except (ValueError, KeyError):
            continue

        # Raw argmax prediction (no beam, no LM)
        raw_pred = "".join(classes[np.argmax(word_matrix[t])]
                           for t in range(len(word))
                           if classes[np.argmax(word_matrix[t])] in alpha_class_set)
        if raw_pred == word:
            raw_correct += 1

        # Beam decode (acoustic only: alpha=0)
        wd_acoustic = WordDecoder(lm, beam_width=100, top_chars=6, alpha=0.0)
        beam_candidates_acoustic = wd_acoustic.decode_word(word_matrix, classes)
        beam1_acoustic = beam_candidates_acoustic[0][0] if beam_candidates_acoustic else ""

        # Beam decode with LM reranking (alpha=0.5)
        beam_candidates_lm = word_decoder.decode_word(word_matrix, classes)
        beam1_lm = beam_candidates_lm[0][0] if beam_candidates_lm else ""
        top3_words = [w for w, _ in beam_candidates_lm[:3]]
        top5_words = [w for w, _ in beam_candidates_lm[:5]]

        if beam1_acoustic == word:
            beam1_correct += 1
        if beam1_lm == word:
            lm_correct += 1
        if word in top3_words:
            top3_correct += 1
        if word in top5_words:
            top5_correct += 1

        n_valid += 1

    if n_valid == 0:
        return {}

    results = {
        "evaluation_mode": "synthetic_word_simulation_from_isolated_keystrokes",
        "n_words":      n_valid,
        "raw_acc":      raw_correct / n_valid,
        "beam1_acc":    beam1_correct / n_valid,
        "lm_acc":       lm_correct / n_valid,
        "top3_acc":     top3_correct / n_valid,
        "top5_acc":     top5_correct / n_valid,
        "lm_gain":      (lm_correct - raw_correct) / n_valid,
    }
    return results


def run_eval_mode(probs_path: str = "results/ensemble_probs.npz"):
    """Full evaluation pipeline."""
    print(f"\n{'='*60}\n  📊 PHASE 3 — EVALUATION MODE\n{'='*60}")

    # ── Load probs ───────────────────────────────────────────
    if not os.path.exists(probs_path):
        print(f"  ❌ {probs_path} not found.")
        print("  Run run_transformer_only.py first to generate it.")
        sys.exit(1)

    data = np.load(probs_path, allow_pickle=True)
    probs   = data["probs"]       # (N, n_classes)
    y_true  = data["y_true"]      # (N,) encoded
    classes = data["classes"]     # (n_classes,) strings

    print(f"  Loaded: {probs.shape[0]} samples, {len(classes)} classes")
    print(f"  Classes: {' '.join(sorted(classes))}\n")

    # ── Build LM ─────────────────────────────────────────────
    print("  Building language model...")
    lm = NgramLanguageModel(smoothing=1.0, bigram_weight=0.4)

    # ── Build decoders ───────────────────────────────────────
    word_decoder   = WordDecoder(lm, beam_width=100, top_chars=6, alpha=0.5)
    sentence_dec   = SentenceDecoder(word_decoder, lm, beam_sentences=20)

    # ── Simulate word decoding ────────────────────────────────
    print(f"\n{'='*60}\n  WORD DECODING SIMULATION (N=500 words)\n{'='*60}")
    print("  ⚠ This is a synthetic protocol (not real free_type end-to-end decoding).")
    results = simulate_word_decoding(probs, y_true, classes, lm, word_decoder,
                                     n_words=500, min_word_len=3, max_word_len=8)

    if results:
        print(f"\n  Words evaluated:      {results['n_words']}")
        print(f"  Raw argmax accuracy:  {results['raw_acc']:.1%}  "
              f"(argmax each keystroke, concat)")
        print(f"  Beam@1 (acoustic):    {results['beam1_acc']:.1%}  "
              f"(beam search, no LM)")
        print(f"  Beam@1 + LM:          {results['lm_acc']:.1%}  "
              f"(beam + bigram LM reranking)  ← main result")
        print(f"  Top-3 coverage:       {results['top3_acc']:.1%}  "
              f"(correct word in top-3 candidates)")
        print(f"  Top-5 coverage:       {results['top5_acc']:.1%}  "
              f"(correct word in top-5 candidates)")
        print(f"\n  LM gain over raw:     {results['lm_gain']:+.1%}")
        print("  ⚠ Do not use this number as free_type headline accuracy.")

    # ── Save results ─────────────────────────────────────────
    out_path = "results/results_phase3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\n  Results saved → {out_path}")

    # ── Phase summary ─────────────────────────────────────────
    print(f"\n{'='*60}\n  📈 FULL PIPELINE SUMMARY\n{'='*60}")

    phase_files = {
        "Phase 1": "results/results_phase1.json",
        "Phase 2": "results/results_phase2.json",
    }
    for phase, path in phase_files.items():
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            print(f"\n  {phase}:")
            for k, v in d.items():
                if isinstance(v, dict) and "accuracy" in v:
                    acc = v["accuracy"]
                    t3  = v.get("top3_accuracy", 0)
                    line = f"    {k:35s}  acc={acc:.1%}"
                    if t3 > 0:
                        line += f"  top3={t3:.1%}"
                    print(line)

    if results:
        print(f"\n  Phase 3 (Language Decoder):")
        print(f"    {'word_raw_argmax':35s}  acc={results['raw_acc']:.1%}")
        print(f"    {'word_beam_acoustic':35s}  acc={results['beam1_acc']:.1%}")
        print(f"    {'word_beam_lm':35s}  acc={results['lm_acc']:.1%}")
        print(f"    {'word_top3_coverage':35s}  acc={results['top3_acc']:.1%}")

    print(f"\n{'='*60}\n  ✓ Phase 3 complete!\n{'='*60}\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — DEMO MODE
# ══════════════════════════════════════════════════════════════

def run_demo_mode():
    """
    Demo: show decoder working on a few well-known phrases.
    Uses uniform keystroke probs perturbed with noise to simulate
    near-correct acoustic model output.
    """
    print(f"\n{'='*60}\n  🎮 DEMO MODE\n{'='*60}")

    # Build a minimal class set (26 letters only for demo)
    import string
    classes = np.array(list(string.ascii_lowercase))
    n_classes = len(classes)

    lm = NgramLanguageModel(smoothing=1.0, bigram_weight=0.4)
    wd = WordDecoder(lm, beam_width=200, top_chars=8, alpha=0.6)
    sd = SentenceDecoder(wd, lm, beam_sentences=20)

    rng = np.random.default_rng(0)

    test_cases = [
        ("hello", 0.85),
        ("world", 0.80),
        ("python", 0.78),
        ("keyboard", 0.82),
        ("machine", 0.75),
        ("learning", 0.70),
        ("vibration", 0.72),
        ("acoustic", 0.68),
    ]

    print(f"\n  {'Target':15s} {'p_correct':10s} {'Predicted':15s} {'Top-3'}")
    print(f"  {'-'*65}")

    for target, p_correct in test_cases:
        # Build noisy prob matrix
        mat = np.zeros((len(target), n_classes))
        for i, ch in enumerate(target):
            idx = np.where(classes == ch)[0][0]
            # Correct key gets p_correct, rest split remaining probability
            noise = rng.dirichlet(np.ones(n_classes - 1) * 0.5) * (1 - p_correct)
            mat[i] = noise
            mat[i, idx] = p_correct
            # Add small perturbation to adjacent keys
            for delta in [-1, 1]:
                adj = idx + delta
                if 0 <= adj < n_classes:
                    mat[i, adj] += 0.03
            mat[i] /= mat[i].sum()  # re-normalise

        candidates = wd.decode_word(mat, classes)
        top1 = candidates[0][0] if candidates else "?"
        top3 = [w for w, _ in candidates[:3]]
        correct = "✓" if top1 == target else "✗"
        print(f"  {target:15s} {p_correct:.0%}        "
              f"{top1:15s} {correct}  {top3}")

    # Sentence demo
    print(f"\n  Sentence demo — 'hello world':")
    word_mats = []
    for word in ["hello", "world"]:
        mat = np.zeros((len(word), n_classes))
        for i, ch in enumerate(word):
            idx = np.where(classes == ch)[0][0]
            noise = rng.dirichlet(np.ones(n_classes - 1) * 0.5) * 0.18
            mat[i] = noise
            mat[i, idx] = 0.82
            mat[i] /= mat[i].sum()
        word_mats.append(mat)

    decoded = sd.decode_sentence_from_word_probs(word_mats, classes, top_k=5)
    print(f"    Decoded: '{decoded}'")
    print()


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — SENTENCE ENCODE+DECODE TEST
# ══════════════════════════════════════════════════════════════

def run_sentence_test(sentence: str,
                      probs_path: str = "results/ensemble_probs.npz",
                      p_correct: float = 0.82):
    """
    Encode a sentence → simulate acoustic model probs → decode → compare.
    Uses per-character probs sampled from ensemble_probs.npz for realism.
    """
    print(f"\n{'='*60}\n  🔤 SENTENCE TEST\n  Input: '{sentence}'\n{'='*60}")

    data = np.load(probs_path, allow_pickle=True)
    probs   = data["probs"]
    y_true  = data["y_true"]
    classes = data["classes"]

    lm = NgramLanguageModel()
    wd = WordDecoder(lm, beam_width=100, top_chars=6, alpha=0.5)
    sd = SentenceDecoder(wd, lm)

    # Build per-char sample pools from real model output
    true_labels = classes[y_true]
    char_pool = defaultdict(list)
    for idx, ch in enumerate(true_labels):
        if len(ch) == 1 and ch.isalpha():
            char_pool[ch.lower()].append(idx)

    rng = np.random.default_rng(42)
    words = sentence.lower().split()
    word_mats = []
    missing = []

    for word in words:
        if not all(c in char_pool for c in word):
            missing.append(word)
            continue
        mat = np.stack([probs[rng.choice(char_pool[c])] for c in word], axis=0)
        word_mats.append(mat)

    if missing:
        print(f"  ⚠ Skipped words (characters not in training set): {missing}")

    if not word_mats:
        print("  ❌ No decodable words found.")
        return

    decoded = sd.decode_sentence_from_word_probs(word_mats, classes)
    decodable_words = [w for w in words if w not in missing]

    print(f"  Original:  '{' '.join(decodable_words)}'")
    print(f"  Decoded:   '{decoded}'")
    match = decodable_words == decoded.split()
    print(f"  Match:     {'✓ Perfect' if match else '✗ Partial'}")
    if not match:
        word_acc = sum(a == b for a, b in zip(decodable_words, decoded.split())) / max(len(decodable_words), 1)
        print(f"  Word acc:  {word_acc:.1%}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Phase 3 Language Decoder")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo mode (no data files needed)")
    parser.add_argument("--sentence", type=str, default=None,
                        help="Test decoding of a specific sentence")
    parser.add_argument("--probs", type=str, default="results/ensemble_probs.npz",
                        help="Path to ensemble_probs.npz (default: results/ensemble_probs.npz)")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="LM weight in reranking, 0=acoustic only (default: 0.5)")
    parser.add_argument("--beam", type=int, default=100,
                        help="Beam width (default: 100)")
    parser.add_argument("--words", type=int, default=500,
                        help="Number of words to simulate in eval mode (default: 500)")
    args = parser.parse_args()

    if args.demo:
        run_demo_mode()
    elif args.sentence:
        run_sentence_test(args.sentence, probs_path=args.probs)
    else:
        run_eval_mode(probs_path=args.probs)


if __name__ == "__main__":
    main()
