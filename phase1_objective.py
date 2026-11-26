"""Phase 1: sentence-boundary-aware prose objective.

The diagnosis (DIAGNOSIS_AND_PLAN.md, Diagnosis 3) is that `continuation_nll`
averages NLL over a 128-byte continuation with no notion of where a sentence
ends. A model can therefore reach low loss by padding the span with locally
frequent function words ("and the", "of the") without ever producing a
well-formed clause. The greedy samples from run wiki-20261126-055649-yrlg5b4l
show the deeper symptom: the model continues the *frame* rather than advancing
the content ("the ship was able to return to the ship , and the ship was able
to return to the ship").

This module adds a sentence-level discrimination term:

  For each sampled sentence boundary we take the true next sentence plus K-1
  foils drawn from other paragraphs, and ask the model to rank the true
  continuation highest. Every candidate is a real sentence that follows a real
  boundary, so the task is "which sentence completes this paragraph", not
  "which bytes are locally probable".

The paragraph-level objective is kept because it stabilises the byte-level
distribution; the two are combined with fixed weights.

Byte layout
-----------
A window is `left + right + 1` bytes:

    bytes[0 : left]        context, immediately before the true boundary
    bytes[left : left+right+1]   the candidate sentence

Following the notebook's `windows()` convention, the input is `bytes[:-1]` and
the target is `bytes[1:]`. The scored region is `target[left:]`, which covers
the candidate sentence.

Every candidate in a group shares the SAME context; only the candidate bytes
differ. The model's choice therefore depends on which sentence it prefers, not
on which context it happened to see.
"""

import numpy as np
import torch
import torch.nn.functional as F

# NOTE: the tunable constants (SENTENCE_CANDIDATES, SENTENCE_LEFT_BYTES,
# SENTENCE_RIGHT_BYTES, SENTENCE_LOSS_WEIGHT, PARAGRAPH_LOSS_WEIGHT,
# SELECTION_SENTENCE_WEIGHT) live in the notebook's "Run settings" cell so they
# can be changed without editing this file. The defaults below are only used
# when this module is imported directly by the tests.

# Number of candidate sentences per discrimination group (1 true + K-1 foils).
SENTENCE_CANDIDATES = 8

# Byte budget for the context shown before the candidate sentence.
SENTENCE_LEFT_BYTES = 192

# Byte budget for one candidate sentence.
SENTENCE_RIGHT_BYTES = 96

# Weights of the two terms in the combined prose objective.
SENTENCE_LOSS_WEIGHT = 0.4
PARAGRAPH_LOSS_WEIGHT = 0.6

# Weight of sentence accuracy in the checkpoint selection score.
SELECTION_SENTENCE_WEIGHT = 0.30


def sentence_anchor_cuts(paragraph, left, right, cuts_fn):
    """All usable sentence boundaries in a paragraph, not just the first.

    `sentence_boundary` (in the corpus cell) returns only the first cut that
    fits. For the sentence-level task we want every boundary, so that different
    sentences of the same paragraph can act as foils for one another.
    """
    return [cut for cut in cuts_fn(paragraph) if left <= cut <= len(paragraph) - right]


def build_sentence_anchors(raw, paragraph_spans, left, right, cuts_fn):
    """Collect sentence-level anchor offsets.

    `paragraph_spans` is a sequence of (offset, length) spans of paragraphs in
    the split. Returns an int64 array of byte offsets where a sentence starts.
    """
    anchors = []
    for offset, length in paragraph_spans:
        paragraph = bytes(raw[offset:offset + length])
        for cut in sentence_anchor_cuts(paragraph, left, right, cuts_fn):
            anchors.append(offset + cut)
    return np.asarray(anchors, dtype=np.int64)


def choose_sentence_foils(rng, count, size, groups):
    """Pick the true anchor and the foil anchors for each group.

    Returns (real, foils):
      real  - (G,) indices of the true anchor for each group
      foils - (G, K-1) indices of the foil anchors

    Every foil in a group is distinct from the true anchor and from the other
    foils in that group. The true and foil indices are returned together so
    that the caller can never end up pairing a true anchor with foils that
    were chosen relative to a different one.
    """
    real = rng.integers(count, size=size)
    foils = np.empty((size, groups - 1), dtype=np.int64)
    for row in range(size):
        true_index = int(real[row])
        picks = []
        while len(picks) < groups - 1:
            candidate = int(rng.integers(count))
            if candidate != true_index and candidate not in picks:
                picks.append(candidate)
        foils[row] = picks
    return real, foils


def sentence_group_batch(raw, anchors, real, foils, left, right, device):
    """Build multi-candidate sentence-completion groups.

    Args:
      raw     - byte storage (numpy memmap or torch tensor)
      anchors - int64 array of sentence start offsets
      real    - (G,) indices of the true anchor for each group
      foils   - (G, K-1) indices of the foil anchors
      left    - context bytes before the boundary
      right   - candidate sentence bytes

    Returns:
      ids     - (G, K, left+right) input bytes
      targets - (G, K, left+right) target bytes
    """
    groups = len(real)
    k = 1 + foils.shape[1]
    width = left + right

    def read(start, n):
        if isinstance(raw, torch.Tensor):
            chunk = raw[start:start + n]
            if chunk.numel() < n:
                chunk = F.pad(chunk, (0, n - chunk.numel()))
            return chunk.detach().cpu().numpy().astype(np.uint8)
        block = raw[start:start + n]
        # `raw` may be a memmap (uint8) or a bytes-like object.
        block = np.frombuffer(bytes(block), dtype=np.uint8) if not isinstance(block, np.ndarray) \
            else np.asarray(block, dtype=np.uint8)
        if block.size < n:
            block = np.pad(block, (0, n - block.size))
        return block

    ids = np.zeros((groups, k, width + 1), dtype=np.uint8)
    for row in range(groups):
        anchor = int(anchors[real[row]])
        context = read(anchor - left, left)
        for slot in range(k):
            source = anchor if slot == 0 else int(anchors[foils[row, slot - 1]])
            candidate = read(source, right + 1)
            ids[row, slot, :] = np.concatenate((context, candidate))

    ids_t = torch.from_numpy(ids).to(device).long()
    targets = torch.from_numpy(ids[:, :, 1:].astype(np.int64)).to(device)
    return ids_t[:, :, :-1], targets


def sentence_choice_loss(logits, targets, left, temperature=1.0):
    """Score candidates within each group.

    logits  - (G, K, T, 256) per-candidate next-byte logits
    targets - (G, K, T)    per-candidate target bytes
    left    - number of context bytes to skip (the context region)

    Target position j predicts byte j+1, so the first candidate byte
    (bytes[left]) is predicted by target position left-1. The scored region is
    therefore target positions [left-1:], which is the candidate sentence.

    The candidate score is the NEGATED mean NLL, so a higher score means a more
    likely sentence. The loss is the cross-entropy of the softmax over
    candidates against the true candidate, which is the standard multi-candidate
    formulation and falls to 0 as the true candidate becomes clearly preferred.

    Returns (loss, accuracy). Accuracy is the fraction of groups whose true
    candidate has the lowest mean NLL over the sentence region.
    """
    g, k, t, v = logits.shape
    flat_logits = logits.reshape(g * k, t, v).float()
    flat_targets = targets.reshape(g * k, t)
    nll = F.cross_entropy(flat_logits.transpose(1, 2), flat_targets, reduction='none')
    nll = nll.reshape(g, k, t)[:, :, max(0, left - 1):].mean(-1)     # (G, K)
    score = -nll / max(1e-6, temperature)
    true_index = torch.zeros(g, dtype=torch.long, device=nll.device)
    loss = F.cross_entropy(score, true_index)
    accuracy = (nll.argmin(-1) == true_index).float().mean()
    return loss, accuracy


def selection_score(continuation_matched_bpb, sentence_accuracy,
                     sentence_weight=SELECTION_SENTENCE_WEIGHT):
    """Checkpoint selection score: lower is better.

    Combines the paragraph-level continuation BPB with the sentence-level
    discrimination accuracy. With the default weight of 0.30, a 0.1 improvement
    in sentence accuracy is worth 0.03 BPB.
    """
    if continuation_matched_bpb is None:
        return None
    if sentence_accuracy is None:
        return float(continuation_matched_bpb)
    return float(continuation_matched_bpb) + sentence_weight * (1.0 - float(sentence_accuracy))


def warmup_steps(resume_path, initial_weights, fresh=50, seeded=5):
    """Length of the linear LR warm-up ramp.

    A fresh model needs the full ramp. A model that resumes with its optimizer
    state restored must not be re-warmed at all. A model seeded from saved
    weights starts a fresh optimizer, so it gets a short ramp to let AdamW's
    moments settle without knocking the weights out of their basin.

    Observed failure this prevents: restarting the full 50-step ramp on an
    already-trained model raised validation BPB from 1.417 to 1.518 over 19k
    steps in run wiki-20261126-055649-yrlg5b4l.
    """
    if resume_path:
        return 1
    if initial_weights:
        return seeded
    return fresh
