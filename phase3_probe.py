"""Phase 3 (cheap half): long-range context probes and coherence metrics.

The expensive half of Phase 3 -- retraining at CONTEXT=1024 -- was rejected on
evidence; see PHASE3_DECISION.md. The BPB-vs-context curve there shows the
memory system saturating (512->1024 buys only ~+0.03 BPB, with the
per-doubling gain already halving at 512).

What remains useful and free is measuring the model's long-range behaviour
from ANY checkpoint, so we can tell whether a future bottleneck really is
context, without paying for a retrain to find out.
"""

import numpy as np
import torch
import torch.nn.functional as F


def context_sensitivity(model, rng, distances=(5, 20, 40, 80, 160, 320, 640),
                         trials=6, context_bytes=32):
    """Measure how much the prediction depends on context at each distance.

    For each distance D we build a random byte context, predict the final
    byte, then perturb one byte at distance D and measure how much the
    prediction changes. A model that binds long-range structure shows a
    non-trivial effect that decays smoothly with distance.

    Returns a dict mapping distance -> mean absolute NLL change.
    """
    results = {}
    for distance in distances:
        deltas = []
        for _ in range(trials):
            base = bytes(rng.integers(97, 123, size=distance + context_bytes))
            base = base + b'xyz'
            position = len(base) - 1 - distance
            if position < 0:
                continue
            modified = bytearray(base)
            modified[position] = (modified[position] + 7) % 256
            deltas.append(abs(_last_byte_nll(model, base) - _last_byte_nll(model, bytes(modified))))
        if deltas:
            results[distance] = float(np.mean(deltas))
    return results


def _last_byte_nll(model, context):
    """NLL of the final byte of `context` given the bytes before it."""
    data = np.frombuffer(context, dtype=np.uint8)
    x = torch.from_numpy(data[:-1].copy()).long().unsqueeze(0)
    target = int(data[-1])
    with torch.inference_mode():
        logits = model(x)
    return float(F.log_softmax(logits[0, -1].float(), -1)[target])


def bpb_by_context(model, data, lengths=(64, 128, 256, 512, 1024, 2048),
                   windows=40, offset=50000):
    """Held-out BPB as a function of the context window length.

    The model was trained at one context length; this shows how much BPB
    improves if it were given more or less context. A saturating curve means
    the memory system is not the bottleneck.
    """
    out = {}
    for length in lengths:
        if offset + windows * length + 1 > len(data):
            continue
        starts = np.arange(offset, offset + windows * length, length)
        blocks = np.stack([data[s:s + length + 1] for s in starts])
        x = torch.from_numpy(blocks[:, :-1]).long()
        y = torch.from_numpy(blocks[:, 1:]).long()
        with torch.inference_mode():
            logits = model(x)
            nll = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction='sum')
        out[length] = float(nll) / (windows * length * np.log(2))
    return out


def coherence_score(text):
    """A cheap syntactic-coherence score for a generated passage.

    Counts complete sentences, dangling function-word endings and the ratio of
    sentences to total words. No external parser is required, so this runs
    anywhere, including inside the notebook.

    Returns a dict with:
      words          - word count
      sentences      - number of sentence-like units
      complete       - sentences that end in a terminator
      complete_rate  - complete / sentences
      dangling       - ends on a function word (an unfinished clause)
      repetition     - fraction of repeated 4-grams (a degeneracy signal)
    """
    import re
    cleaned = text.replace('@-@', '-').replace('@,@', ',')
    words = re.findall(r"[A-Za-z']+", cleaned)
    units = [s for s in re.split(r'(?<=[.!?])\s+', cleaned.strip()) if s.strip()]
    complete = sum(1 for s in units if re.search(r'[.!?]$', s.strip()))
    dangling = bool(re.search(r'\b(and|or|of|in|on|at|to|with|the|a|an|but|that|which|for)$',
                              cleaned.strip(), re.I))
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    repeated = sum(1 for g in grams if grams.count(g) > 1) if grams else 0
    repetition = repeated / len(grams) if grams else 0.0
    return dict(words=len(words), sentences=len(units), complete=complete,
                complete_rate=(complete / len(units)) if units else 0.0,
                dangling=dangling, repetition=repetition)


def coherence_penalty(coherence):
    """Turn a coherence dict into a penalty to ADD to the selection loss.

    Lower is better, so a dangling ending or heavy repetition increases the
    penalty and makes the checkpoint less likely to be selected.
    """
    penalty = 0.0
    if coherence.get('dangling'):
        penalty += 0.15
    penalty += 0.20 * float(coherence.get('repetition', 0.0))
    penalty += 0.10 * (1.0 - float(coherence.get('complete_rate', 0.0)))
    return penalty
