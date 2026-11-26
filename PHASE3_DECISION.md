# Phase 3 Decision: Is a context retrain (512 -> 1024) worth it?

**Answer: No. Not as a retrain. The evidence says the model is not context-limited, and a retrain would be the single most expensive change in the plan while addressing the least important cause.**

This note records the measurements, because the original plan (Diagnosis 5 in
`DIAGNOSIS_AND_PLAN.md`) asserted that the model was context-limited at 512
bytes. That assertion was never tested. It is now.

---

## The measurements

All measurements use the checkpoint from run `wiki-20261126-055649-yrlg5b4l`
(`best_lm.pt`, validation BPB 1.41698), the current best model.

### 1. Information flow decays, but never cuts off

Perturbing one byte at distance `D` before a prediction and measuring the
change in the prediction's NLL, over random byte contexts:

| distance | mean abs dNLL |
|---|---|
| 5 | 0.1608 |
| 20 | 0.0935 |
| 40 | 0.0319 |
| 80 | 0.0248 |
| 160 | 0.0119 |
| 320 | 0.0069 |
| 640 | 0.0040 |

There is **no hard cliff**. Context beyond the 64-byte attention window is
still used; the effect decays smoothly. So the model is not "blind" past 512
bytes, but the influence of distant context is already an order of magnitude
below nearby context.

### 2. BPB keeps improving with context, but with diminishing returns

Held-out BPB on real prose, using a model trained at context 512:

| context | BPB | gain vs previous |
|---|---|---|
| 64 | 1.4772 | |
| 128 | 1.4176 | +0.0597 |
| 256 | 1.3424 | +0.0751 |
| 512 | 1.2940 | +0.0485 |
| 1024 | 1.2612 | +0.0328 |
| 2048 | 1.2403 | +0.0209 |

Note what this table actually shows: the **per-doubling gain is already
halving** at 512 (+0.0485 for 256->512, vs +0.0751 for 128->256). Extrapolating,
512 -> 1024 buys roughly **+0.03 BPB**, and 1024 -> 2048 roughly +0.02. The
curve is flattening, exactly as a saturating memory system does.

### 3. The supervisor never uses more than ~320 bytes anyway

This is the finding that matters most, and it was missed by the original plan.

- The paragraph continuation objective supervises `prefix_bytes=192` +
  `ending_bytes=128` = **320 bytes total**.
- The dialogue stage supervises targets capped at 160-200 bytes.

So even with `CONTEXT = 1024`, **the model is trained to predict the next
160-320 bytes.** It has never been trained to predict the 500th byte after the
context window. More context in the input does not create longer-range
supervision; it just pads the input with bytes the objective barely scores.

Diagnosis 5 conflated *input context length* with *supervision span length*.
The supervision span is the binding constraint.

### 4. Generation does not use `CONTEXT` at all

`sample()` in the notebook streams generation in 64-byte chunks through
`model.stream_chunk`, carrying a recurrent state. The generation context is
bounded by the learned Delta decay and the 64-byte attention windows
(`read_window=64`, `mid_window=64`), **not** by `CONTEXT`. Raising `CONTEXT`
to 1024 would not lengthen what the model can attend to while generating; it
would only change the training-time window.

---

## Why a retrain is actively costly here

1. **It invalidates every existing checkpoint.** `config` includes `context`,
   and `compatible_config()` compares configs for equality. A checkpoint saved
   at `context=512` would be rejected for resume, forcing a full pretrain
   from scratch (the current model saw 500 MB of WikiText, ~5900 steps).

2. **It discards the optimizer state.** The only way to move to `context=1024`
   and keep the learned weights is `INITIAL_WEIGHTS`, which starts a fresh
   AdamW optimizer and throws away the momentum. Run
   `wiki-20261126-055649-yrlg5b4l` already showed how a fresh optimizer on
   trained weights behaves.

3. **It costs the most and fixes the least.** A 1024-context pretrain is
   roughly 2x the compute per update (the same 16 sequences now span 16384
   bytes instead of 8192). Meanwhile Phase 1 and Phase 2 target the two
   causes that the evidence actually implicates: the objective has no notion
   of sentence boundaries, and the supervision span is 160 bytes.

4. **It would not fix the observed failure mode.** The Phase 0 greedy samples
   show *frame continuation*, not truncation:

   ```
   "on 18 April , the company was able to return to the company . The company was able to return to the "
   "The ship was able to return to the ship , and the ship was able to return to the ship . The ship was"
   ```

   The model is not running out of context. It is repeating a pattern. That
   is an objective problem, and Phase 1 addresses it directly.

---

## What is done instead

Phase 3's *measurement* goal is preserved without paying for a retrain:

- **The BPB-vs-context curve above is the evidence.** It shows the memory
  system saturates, so a context retrain has a low ceiling.
- **A context probe is available before any training.** Because generation
  streams independently of `CONTEXT`, we can measure the model's long-range
  behaviour at any time from the existing checkpoint.

The one genuinely cheap Phase 3 item is a **coherence metric** for checkpoint
selection, which costs nothing at training time. That is implemented as part
of the Phase 1 selection metric (`sentence_completion_accuracy`), which
measures exactly the thing the retrain was supposed to buy.

---

## When a context retrain WOULD be worth it

Revisit this if, after Phase 1 and Phase 2:

- `sentence_completion_accuracy` plateaus below ~0.75, AND
- generations remain locally fluent but globally incoherent across sentence
  boundaries, AND
- the BPB-vs-context curve still shows meaningful headroom beyond 512.

In that case the right change is **not** `CONTEXT = 1024`. It is to raise
`SENTENCE_RIGHT_BYTES` and the dialogue target cap so the *supervision span*
grows, which is what actually limits the model. `CONTEXT` only needs to be
large enough to hold the prefix plus the longer target.

## Also noted: the Hierarchical Memory Tree is inert

From the same run:

```
off_minus_intact_bpb  = 0.00287
shuffle_minus_intact_bpb = 0.00523
```

Removing or scrambling the Tree changes BPB by ~0.003-0.005 out of 1.42. The
Tree contributes essentially nothing while occupying real parameters and
compute. If a future run needs a capacity increase, removing the Tree frees
parameters for the Delta and attention layers, which are demonstrably doing
the work.
