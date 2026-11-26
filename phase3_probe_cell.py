"""Notebook cell: the Phase 3 context probe.

Runs after training to record whether the model is actually context-limited.
The evidence that made the CONTEXT=1024 retrain unnecessary lives in
PHASE3_DECISION.md; this cell re-derives it for whatever checkpoint is
selected, so the question can be re-answered cheaply in future runs.
"""

PHASE3_PROBE_CODE = '''
# --- Phase 3: long-range context probe ----------------------------------------
# The plan originally proposed retraining at CONTEXT=1024. PHASE3_DECISION.md
# records the measurements that rejected it: the BPB-vs-context curve
# saturates, and generation streams independently of CONTEXT anyway.
#
# This cell re-derives that evidence for the selected checkpoint so the
# question stays answerable without paying for a retrain.
@torch.inference_mode()
def probe_context(model, split='validation'):
    """BPB as a function of context length, plus a context-sensitivity curve."""
    import numpy as _np
    raw = corpus[split]
    probe_bytes = min(len(raw), 1 << 20)
    data = _np.frombuffer(bytes(raw[:probe_bytes]), dtype=_np.uint8)
    lengths = [L for L in (64, 128, 256, 512, 1024, 2048) if probe_bytes > L * 64]
    curve = {}
    for L in lengths:
        windows = max(4, min(40, (probe_bytes - 1) // L))
        starts = _np.arange(0, windows * L, L)
        blocks = _np.stack([data[s:s+L+1] for s in starts])
        x = torch.from_numpy(blocks[:, :-1].copy()).long().to(device)
        y = torch.from_numpy(blocks[:, 1:].copy()).long().to(device)
        with precision(device):
            logits = model(x)
            nll = F.cross_entropy(logits.reshape(-1, 256).float(), y.reshape(-1), reduction='sum')
        curve[L] = float(nll) / (windows * L * math.log(2))
    return curve

probe_curve = probe_context(prose_model)
print('BPB by context length:')
for L, value in probe_curve.items():
    print(f'  L={L:5d}: BPB = {value:.4f}')
gains = {L: probe_curve[L] - probe_curve[prev]
         for prev, L in zip(list(probe_curve)[:-1], list(probe_curve)[1:])}
if gains:
    print('Gain per context doubling:', {L: round(v, 4) for L, v in gains.items()})
    halving = all(gains[a] >= gains[b] for a, b in zip(list(gains)[:-1], list(gains)[1:]))
    print('Gains are shrinking (memory saturating):', halving)
save_json('context_probe.json', dict(bpb_by_context={str(k): v for k, v in probe_curve.items()},
                                      gains={str(k): v for k, v in gains.items()},
                                      saturating=bool(halving)))
'''
