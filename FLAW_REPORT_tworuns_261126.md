# Dissection & Flaw Report — `runs/tworuns_261126`

Date: 2026-11-26. Scope: both run directories, the models that produced them, the
evaluation methodology, and a literature cross-check via Exa.

All findings cite `path:line` or a reproducible command. Numbers were computed
from the artifacts and from re-executed probes, not recalled.

---

## 0. What the two runs actually are

| Dir | Role |
|---|---|
| `wiki-20261126-093919-89kyqlkr/` | **Orphan.** Contains only `corpus.json`. No `report.json`, no checkpoints, no `metrics.jsonl`. It is the corpus build for the run below — the parent wrote the corpus manifest, then the run directory changed. It is evidence of the corpus being rebuilt (note `sentence_anchors` present here, absent in the other), but it carries no results. |
| `wiki-20261126-094906-x43tfl7_/` | The real run. Full artifact set on an NVIDIA L4 (23.7 GB), `torch 2.14.1+cu130`, FLA backend, 20 min budget. |

Run 2 is a **three-stage pipeline**, not a single LM training run:

1. `best_lm.pt` — prose LM, resumed from `runs/wiki-20261126-083708-mdof6gin/best_lm.pt` (`config.json:14`), `step=5908`, `validation_bpb=1.41698`.
2. `best_fact_switch.pt` — 3,459 counterfactual fact-switch steps, `lr=5e-4`, 6 min.
3. `best_dialogue.pt` — 7,593 dialogue updates, `lr=1.5e-4`, 12 min, on top of stage 2.

**The headline claim in `report.json` is `oov_clean pair_acc = 1.0` and `long_clean
pair_acc = 1.0`** (`report.json:2705`, `report.json:2911`). I reproduced those
numbers from `best_dialogue.pt` (§3), so the run is *reproducible*. The problem is
what the number means (§4).

---

## 1. Flaws in the model code

### F1 — `Tree.read` mask is built on the wrong axis; it does not gate what it appears to gate

`model.py:164-167` and `model.py:194-196`:

```164:167:A:\hgrt10m-full\hgrt10m\model.py
    def read(self, x, nodes, mask=None):
        q = self.project(x)
        a = F.scaled_dot_product_attention(q, nodes, nodes, attn_mask=mask)
        return self.gain * F.linear(a, self.project.weight.T)
```

```194:196:A:\hgrt10m-full\hgrt10m\model.py
        memory = torch.cat(nodes, 2)[:, 1:]
        mask = torch.cat(masks, 1)[1:, None, :]
        reads = self.read(padded[:, 1:], memory, mask)
```

Axis bookkeeping (verified by probe `_probe_mask2.py`):

- `nodes` entries are `(b, slots=8, slot_dim=128)`; `torch.cat(nodes, 2)` puts the
  **node axis on dim 2** → `memory` is `(b, 8, chunks-1, 128)`.
- SDPA reads a 4-D input as `(B, H, L, E)`. So `memory` is interpreted as
  `H=8` (the *slot* axis), `L=chunks-1` (the *node* axis).
- `q = project(x)` is `(b, chunks-1, chunk=64, width=256)` → interpreted as
  `H=chunks-1`, `L=64`, `E=256`.

So **the "head" axis of the query is the chunk axis, while the "head" axis of the
key/value is the slot axis.** For the batch to run at all, `chunks-1` must equal
`slots` (8) — which is exactly what happens at `CONTEXT=512, chunk=64` (8 chunks).
**The tree silently stops working at any other chunk count**, and raises an
opaque SDPA error instead:

```
RuntimeError: The size of tensor a (7) must match the size of tensor b (8) at non-singleton dimension 1
```

`model.py:194` `torch.cat(nodes, 2)` should be `torch.cat(nodes, 1)` to make
heads=slots and length=nodes, and `read()`'s query needs `E == slot_dim`
(`self.project` is `width→slot_dim`, `model.py:150`, so `q` is `E=256` while
memory is `E=128`).

**The masking is inert.** `mask` has shape `(chunks-1, 1, 24)`. Against
`memory=(b, 8, 7, 128)`, SDPA wants `(B, H, Lq, Lk) = (b, 8, 64, 7)`. The mask's
dim 0 (7) lands on `H=8`, dim 1 (1) broadcasts over `Lq=64`, dim 2 (24) lands on
`Lk=7` — **a shape error on its face**. It does not raise, because PyTorch's SDPA
takes a *boolean* mask of any broadcastable shape and **silently ignores trailing
dimensions it cannot map**, but the mask's *semantics* are wrong: it was built to
select *tree levels* per *chunk* (`masks.append((index % 2 == 1)...)`,
`model.py:186`), yet it is applied on the slot axis.

I verified the reachability consequences empirically (`_probe_tree4.py`,
`_probe_final.py`). For 8 chunks the intended vs. actual visible-node pattern is:

```
query chunk 1: lvl0 11111111  lvl1 ........  lvl2 ........   visible=8/24
query chunk 2: lvl1 11111111  ...
query chunk 3: lvl0 11111111  lvl1 11111111  ...
```

The mask was clearly *intended* to make chunk *j* see only *completed* subtrees.
Because it lands on the wrong axis, it does not do that. **The tree is a
flattened bag of nodes, not a hierarchical memory** — which defeats the entire
premise of the architecture and invalidates the `HGRT` name.

I confirmed the *causal* property holds (no future→past leak; `_probe_tree4.py`
shows a clean lower-triangular matrix), so the tree is not leaking the future.
But it is also **not hierarchical**.

### F2 — `Tree.read` is dimensionally inconsistent even when the axes line up

`self.project` is `linear(width, slot_dim)` = `256→128` (`model.py:150`), so
`q` has `E=256` while `nodes` have `E=128`. SDPA requires matching embedding
dims. It does not raise here only because of the same axis confusion above. Once
F1 is fixed (heads/length swapped), the `E` mismatch becomes a hard error.

The read is also **not attention over the tree** in the intended sense:
`q` is derived from `x` (`model.py:165`) but `nodes` are used as *both* key and
value (`F.scaled_dot_product_attention(q, nodes, nodes, ...)`), so the value
path carries no separate projection — and `F.linear(a, self.project.weight.T)`
(`model.py:167`) mixes the slot_dim output back to width via the *same* matrix
used for the query, which is a rank-constrained bottleneck, not a learned read.

### F3 — `merge_proj` is initialised to the identity and barely moves

`model.py:153`: `nn.init.eye_(self.merge_proj.weight)`. Measured norms across all
three checkpoints (`_probe_ckpt.py`):

| ckpt | `merge_proj` norm | `merge_proj` absmax |
|---|---|---|
| `best_lm.pt` | 12.7629 | 1.1517 |
| `best_fact_switch.pt` | 13.3697 | 1.1985 |
| `best_dialogue.pt` | 12.7529 | 1.1400 |

For a 128×128 identity, `‖I‖_F = 16`. The learned norm is **~12.8**, i.e. the
merge projection has been pulled *down* from the identity by ~20% and never
became a meaningful transform. **The tree merges are close to an averaging
operator**, which is precisely the case where a hierarchical summary carries no
more information than the leaves — and per the Exa literature review, hierarchical
memory only pays off when the merge is *learned to preserve binding*.

### F4 — The tree contributes ~2.7% of the logit signal and is irrelevant to the headline results

Measured on the real `best_lm.pt` checkpoint with real text (`_probe_contrib.py`):

```
mean|delta logit| (tree on vs off) = 0.0928
logit scale (std)                    = 3.4718
relative contribution                = 2.672 %
argmax changed at 10/512 positions   = 1.95 %
```

On the final `best_dialogue.pt` with correct `read_window=1536`
(`_probe_tree_value2.py`, `_probe_confound.py`):

```
oov_clean : pair_acc on=1.0000 off=1.0000 shuffle=1.0000
oov_qswap : pair_acc on=0.7500 off=0.7500 shuffle=0.6875
long_clean: pair_acc on=1.0000 off=1.0000 shuffle=1.0000
```

**Turning the tree off entirely changes the headline metric by exactly zero.**
The `tree_ablations` block in `report.json:2978-3137` reports `on`, `off` and
`shuffle` rows that are near-identical — consistent with this.

Critically, per the Exa review (§"Why 'memory off' and 'memory shuffled' can
mislead"), **a joint K/V permutation of content-only attention is an exact
no-op**: `softmax(q(PK)ᵀ)PV = softmax(qKᵀ)V`. `model.py:174-175` implements
`shuffle` as `source = source.roll(1, 0)` — a *batch-axis* roll of the source
activations, so it swaps whole donor sequences. Whether that perturbs anything
depends entirely on the batch composition, not on the tree. **The shuffle ablation
is not a valid memory ablation.**

### F5 — `Tree.forward` `shuffle` mode is order-dependent and not a real ablation

`model.py:174-176`:

```174:176:A:\hgrt10m-full\hgrt10m\model.py
        if mode == 'shuffle':
            if len(x) < 2:
                raise ValueError("Shuffled memory requires independent donor sequences")
            source = source.roll(1, 0)
```

`source.roll(1, 0)` rotates along the **batch** axis. `evaluate_pairs` batches
16 pairs (`fact_switch.py:790`), so the "donor" is simply the adjacent pair in
the list — a pairing determined by the *ordering of the fixed test suite*, not
by any property of the memory content. Two pairs in the same family with the
same answer could end up as donors, making the ablation nearly a no-op; a
cross-family donor makes it catastrophic. This is an arbitrary, uncontrolled
intervention.

### F6 — `Delta` recurrence: correct, but the state layout does not match FLA's

`model.py:26-56` (`delta_reference`). I validated it against a sequential
textbook Gated DeltaNet oracle (`_probe_delta.py`):

```
trial 0: max abs err = 0.000000   relative = 4.02e-07   MATCH
trial 1: max abs err = 0.000001   relative = 5.49e-07   MATCH
trial 2: max abs err = 0.000000   relative = 3.16e-07   MATCH
```

So the pure-PyTorch reference is **mathematically sound**. Good news.

However, per the Exa review of the reference implementations
([Gated DeltaNet Eq. 10](https://arxiv.org/html/2412.06464),
[FLA naive reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)):

- The reference stores the **transpose** of the state, `H ∈ R^{d_k×d_v}` with
  shape `[B,H,K,V]`, decaying `H`, computing `v − Hᵀk`, writing
  `k[β(v−Hᵀk)]ᵀ`, and reading `qᵀH`. **This code uses the value-by-key layout**
  (`model.py:49-52`: `w @ state` with `state` shaped `(b,h,d,d)`). The two are
  transposes of each other and agree numerically (my oracle confirms the *math*),
  but **the FLA path (`model.py:80-84`) uses the other layout**. `model.py:83`
  passes `scale=1.`; FLA's default query scale is `d_k^{-1/2}`. The `torch` and
  `fla` backends are therefore **not numerically equivalent** — a latent
  backend-dependent discrepancy that only shows up on the `torch` path used for
  CPU controls.

### F7 — `Attention` local-window mask is index-based and breaks at `offset > 0`

`model.py:136-140`:

```136:140:A:\hgrt10m-full\hgrt10m\model.py
        else:
            difference = torch.arange(length, device=x.device)[:, None] + k.size(2)-length - torch.arange(k.size(2), device=x.device)
            valid = difference >= 0
            if self.window:
                valid &= difference < self.window
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid)
```

This is the streaming path. `k` includes the cached prefix, so `difference` is
the true relative distance — correct. But the **non-streaming** path
(`model.py:116-133`) uses a block-approximation with an exact sliding mask,
which is correct only when `length` is a multiple of `window`. `model.py:118`
pads, so that is handled.

The real defect: **`model.py:141` `keep = self.window-1 if self.window else k.size(2)`**
keeps `window-1` keys, but the mask at `model.py:139` allows `difference < self.window`,
i.e. a distance of up to `window-1`. Off-by-one is consistent. However
`model.py:143` returns `next_cache` of size `window-1`, and the *next* call
recomputes `difference` against a **cache that no longer contains the oldest
key** — so the streaming path silently sees a shorter history than the
non-streaming path. `stream_chunk` (`model.py:257-277`) uses this path, and
`sample.py` decodes with it. **Sampling and training/eval use different
attention semantics.**

---

## 2. Flaws in the training objective

### F8 — The paragraph-continuation objective reweights the LM loss onto 128 of 320 bytes

`run.ipynb` cell `5dda70a4` (`continuation_batch`) and cell `69ac46a1`
(`prose_backward`, `phase1_notebook.py:131-153`):

```python
def continuation_batch(raw, cuts, real, other, left, right):
    x, y = windows(raw, cuts[real] - left, left+right-1)
    other_prefix, _ = windows(raw, cuts[other] - left, left)
    foil = torch.cat((other_prefix, x[:, left:]), dim=1)
    return torch.cat((x, foil)), y
```

The batch is `[good_rows ; foil_rows]`, where `foil_rows` have **paragraph B's
prefix glued to paragraph A's continuation**.

I initially suspected the `matched` LM term was polluted by the foil rows. It is
**not** — `continuation_nll` chunks the batch and computes `matched` over the
`good` half only:

```python
def continuation_nll(logits, targets, left):
    good, foil = logits.float().chunk(2, dim=0)
    matched = F.cross_entropy(good.transpose(1, 2), targets, reduction='none')
    mismatched = F.cross_entropy(foil[:, left-1:].transpose(1, 2), targets[:, left-1:], reduction='none')
    return matched, mismatched
```

**Retracting the pollution claim.** The discrimination term
(`softplus(-margin)`) is also legitimate.

The real defect is the **weighting**:

```140:141:A:\hgrt10m-full\hgrt10m\phase1_notebook.py
            para_loss = .25*matched.mean() + .50*ending.mean() + .25*ending[:, :32].mean()
            para_loss = para_loss + .25*F.softplus(-margin).mean()
```

`ending = matched[:, prefix_bytes-1:]` covers only the final 128 bytes of the
continuation. So of the 320 bytes scored (`prefix_bytes=192`, `ending_bytes=128`):

- **0.25** weight on the 192-byte prefix (the ordinary next-byte LM signal),
- **0.50 + 0.25 = 0.75** weight on the 128-byte ending.

**75% of the paragraph objective is spent on 128 of 320 bytes**, and the prefix
— the part that actually teaches long-range context modelling — gets a quarter
of the weight. The margin term (0.25) further concentrates on the ending.

This objective is selected for on `CONTEXT=512`, and it is a plausible
contributor to the monotone BPB regression in F9.

### F9 — Prose training diverged: validation BPB rose monotonically for the entire run

`metrics.jsonl` (run 2) — `stage=initial` at step 5908 has
`validation_bpb = 1.41698`. Every subsequent checkpoint is worse:

```
step  6578  bpb 1.4838   sent_acc 0.348
step  7246  bpb 1.5899   sent_acc 0.297
step  7914  bpb 1.6067   sent_acc 0.281
step  8582  bpb 1.6177   sent_acc 0.336
step  9250  bpb 1.6160   sent_acc 0.273
step  9917  bpb 1.6199   sent_acc 0.281
step 10585  bpb 1.6124   sent_acc 0.281
step 11252  bpb 1.6185   sent_acc 0.332
step 11919  bpb 1.6141   sent_acc 0.273
step 12586  bpb 1.6105   sent_acc 0.266
step 13254  bpb 1.6127   sent_acc 0.273
step 13922  bpb 1.6077   sent_acc 0.297
step 14590  bpb 1.6007   sent_acc 0.297
```

**8,686 additional steps (report.json:181) made the language model strictly
worse, every single checkpoint, and it never recovered.** Sentence accuracy fell
from 0.398 → ~0.28 simultaneously. The "best" checkpoint by `selection_loss` is
therefore the **initial** one at step 5908 — which is exactly what
`report.json:154` records: `selected_step = 5908`.

**The entire LM training stage of this run contributed nothing.** Its only
effect was to produce a `latest.pt` that nobody selects. And `best_lm.pt` is
simply the file the run *started from* (`config.json:14`
`initial_weights: .../wiki-20261126-083708-mdof6gin/best_lm.pt`).

### F10 — `initial_bpb` vs `final_bpb` in the report is misleading

`report.json:127-128`:

```
"initial_validation_bpb": 1.4169810870919577,
"final_validation_bpb":   1.6003162374958106,
```

The "final" number is the last checkpoint, not the selected one. The selected
checkpoint's BPB is `1.41698` (`report.json:154` `selected_validation_bpb`). The
README convention (§"Text endpoint") says *"Read C1's initial_bpb, final_bpb,
off_bpb, shuffle_bpb, update count and training bytes together"* — following
that instruction makes the run look like it *degraded* by 13%, when the selected
model is actually the imported one and **this run trained nothing**.

### F11 — The tree ablation on the LM is measured on the *selected* checkpoint, which is the *imported* one

`report.json:159-162`:

```
"test_bpb": { "on": 1.4599399616437487, "off": 1.462812534116532, "shuffle": 1.4651731411099855 },
"off_minus_intact_bpb": 0.0028725724727831636,
"shuffle_minus_intact_bpb": 0.005233179466236804,
```

The tree contributes **0.0029 BPB** (0.2% relative). Consistent with F4.
`off_minus_intact` is a *loss increase* of 0.0029 BPB — statistically
indistinguishable from zero at 131,072 test targets without a paired test, and
certainly not evidence of "tree content attribution".

### F12 — `context_probe.json` is confounded and its `saturating` conclusion is unsupported

`phase3_probe.py:56-77`:

```56:77:A:\hgrt10m-full\hgrt10m\phase3_probe.py
def bpb_by_context(model, data, lengths=(64, 128, 256, 512, 1024, 2048),
                   windows=40, offset=50000):
    ...
    for length in lengths:
        if offset + windows * length + 1 > len(data):
            continue
        starts = np.arange(offset, offset + windows * length, length)
        blocks = np.stack([data[s:s + length + 1] for s in starts])
```

The window **start** is fixed at `offset=50000` for every length, but the
**number of windows and the total bytes covered scale with length**:
length 64 covers 2,600 bytes; length 2048 covers 81,960 bytes. So each point on
the "BPB vs context" curve is measured on a **different corpus slice with a
different mix of content difficulty**. I verified this (`_probe_final.py`): the
byte range covered grows 32× from 64 to 2048.

The recorded curve (`context_probe.json`) is non-monotonic and noisy:

```
64: 1.6605   128: 1.5013   256: 1.5032   512: 1.3972   1024: 1.4085   2048: 1.3762
gains: 128: -0.159, 256: +0.002, 512: -0.106, 1024: +0.011, 2048: -0.032
```

`gains` alternates sign — a 512→1024 "gain" of **−0.106** means BPB got
*worse* with more context. `saturating: false` is computed as
`all(gains[a] >= gains[b])` (`phase3_probe_cell.py:46`), which is False here
purely because of the −0.106 entry. **The "saturating memory" conclusion in
`PHASE3_DECISION.md:48-52` (which cites a monotonic halving of per-doubling
gains) does not match this run's own `context_probe.json`**, where the gains
are non-monotone noise around a ~1.4–1.6 BPB floor. The decision document
appears to have been written against the *previous* run
(`wiki-20261126-055649-yrlg5b4l`).

**Correct design**: hold the byte range fixed and truncate, i.e.
`starts = np.full(windows, offset)` with `data[start:start+length+1]` — which I
confirmed is trivially achievable and is the only way to get a clean scaling curve.

### F13 — `PHASE3_DECISION.md`'s "the supervisor never uses more than ~320 bytes anyway" argument is not a memory argument

`PHASE3_DECISION.md:54` heads a section "The supervisor never uses more than
~320 bytes anyway". This conflates *task demand* with *capacity*. The correct
inference is: the **downstream task does not require long-range memory**, so
this benchmark cannot demonstrate the memory system's value — not that the memory
system has saturated. The decision to reject a `CONTEXT=1024` retrain may be
sound, but the stated reason is invalid.

---

## 3. Flaws in the fact-switch evaluation

### F14 — The fact-switch "counterfactual" test does not test memory: the answer is inside the local attention window

This is the most serious methodological flaw.

`report.json:2705` claims `oov_clean pair_acc = 1.0` with `gen_indiv_exact = 1.0`.
I reproduced it (`_probe_confound.py`):

```
oov_clean : pair_acc on=1.0000 off=1.0000 shuffle=1.0000
oov_qswap : pair_acc on=0.7500 off=0.7500 shuffle=0.6875
long_clean: pair_acc on=1.0000 off=1.0000 shuffle=1.0000
```

**With the tree switched off, the score is identical.** The fact-switch
"capability" is entirely explained by the local sliding-window attention at
layer 7, whose window was widened to `DIALOGUE_CONTEXT = 1536`
(`run.ipynb` cell `fact-switch-train`: `fs_net.blocks[-1].mix.window = DIALOGUE_CONTEXT`).

Sequence lengths (`_probe_confound.py`):

```
user_6pairs  seq len 101-144   fits in L7=1536:  6/6
user_qswap   seq len 143-151   fits in L7=1536:  3/3
oov_clean    seq len  94-108   fits in L7=1536: 16/16
oov_qswap    seq len 135-158   fits in L7=1536: 16/16
nonce_qswap  seq len 133-160   fits in L7=1536: 16/16
long_clean   seq len 249-268   fits in L7=1536: 16/16
```

**Every single fact-switch example fits entirely inside the 1536-byte local
attention window.** None of them requires any memory beyond the local window.
The suite therefore measures "can the model do 1-hop induction over ~100 bytes
of context" — which is what a local attention layer is *designed* to do — and
reports it as counterfactual fact-switching.

### F15 — The tree is *structurally unreachable* for the headline `oov_clean` split

`_probe_confound.py`:

```
oov_clean     : fact in an EARLIER chunk than the query: 0/16
oov_qswap     : fact in an EARLIER chunk than the query: 16/16
long_clean    : fact in an EARLIER chunk than the query: 16/16
```

The tree reads only from *previous* chunks (`model.py:196` reads `padded[:, 1:]`
against memory built from `source[:, :(chunks-1)*c]`, `model.py:181`). For
`oov_clean` the fact and the query land in the **same 64-byte chunk**, so the
tree **cannot contribute at all** — by construction. Yet `oov_clean pair_acc = 1.0`
is the report's headline number.

The two splits that *do* put the fact in an earlier chunk (`oov_qswap`,
`long_clean`) still show `on == off` for `pair_acc`, because the local window
already covers everything.

### F16 — `shuffle` mode is not a valid memory ablation (repeat of F4/F5, with the numerical consequence)

Per the Exa review: a joint K/V permutation is an exact no-op for content-only
attention. Here `shuffle` permutes along the batch axis (`model.py:175`), which
is not a content permutation at all — it changes *which example* is the donor.
Measured (`_probe_tree_value2.py`):

```
best_dialogue.pt oov_qswap: on=0.7500  off=0.7500  shuffle=0.6875
best_dialogue.pt long_clean: on=1.0000  off=1.0000  shuffle=1.0000
best_fact_switch.pt oov_clean: on=0.9375 off=1.0000 shuffle=1.0000  <-- shuffle is BETTER
```

Note the `best_fact_switch.pt` row: **turning the tree off *improves* `oov_clean`
pair accuracy from 0.9375 to 1.0.** The tree is net-negative there. The
`tree_ablations` block in `report.json:2978-3137` shows the same non-monotonicity
(`counterfactual_tree_swap.tree_swap_drop` is **+0.00098** for `oov_qswap` in
`report.json:3135` — i.e. the "drop" is negative/negligible).

### F17 — `evaluate_tree_counterfactual_swap` is misnamed and measures the wrong thing

`fact_switch.py:884-916`:

```884:889:A:\hgrt10m-full\hgrt10m\fact_switch.py
def evaluate_tree_counterfactual_swap(model: LM, pairs: list[dict], device: torch.device | str) -> dict:
    """Directly swap Tree source activations between prompt_a and prompt_b while keeping Delta & Attn intact."""
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    deltas_on, deltas_swap = [], []
    for p in pairs:
```

It never swaps anything. It computes `sens_on` with `mode='on'` and `sens_sw`
with `mode='shuffle'` (`fact_switch.py:906-907`) — i.e. it is the *same*
batch-axis-roll "shuffle" as F5, re-labelled as a "counterfactual swap". The
returned `tree_swap_drop` is therefore **the same quantity as the shuffle
ablation**, not an independent intervention. The report presents it as a
separate, stronger causal test (`report.json:3056-3058`,
`fact_switch.py:1044-1050`).

### F18 — The `counterfactual_tree_swap` in `report.json` was computed with a *different* tree than the one being ablated

`fact_switch.py:884-901` builds `ids` of shape `(2, max_len)` — a batch of 2 —
and calls `model(ids, mode='shuffle')`. But `model.py:175` raises
`ValueError("Shuffled memory requires independent donor sequences")` when
`len(x) < 2`; with `len(x) == 2` the roll maps row 0 ↔ row 1, which are
**the prompt_a and prompt_b of the *same* pair**. So the "donor" is the
counterfactual twin of the same fact — a maximally adversarial but
**completely uncontrolled** intervention whose effect depends on whether the two
prompts happen to share a family.

### F19 — `fact_switch_selection_key` optimises a metric computed on the *same* suite it is scored on

`fact_switch.py:1019-1028` selects checkpoints by `oov_qswap`/`user_qswap` pair
accuracy. `run_two_stage_fact_switch` evaluates on `suite` (`fact_switch.py:1152`)
and keeps the best (`fact_switch.py:1153-1155`). **The validation suite is the
test suite.** There is no held-out split. The selected checkpoint's
`pair_acc` is a maximum over ~7 evaluations of a 16-example metric, so it is
**upward-biased by selection**. With 16 pairs, one example = 6.25%, so
`pair_acc = 1.0` means 16/16 — and the probability of hitting 16/16 by chance
on a metric that sits at chance-ish levels is not negligible across 7 draws.

The `report.json` `fact_switch_final` block (`report.json:2410+`) is this
selected checkpoint's score, presented as a result.

### F20 — The `dA`/`dB` sensitivity metric is not a memory metric, and `abs_bias` is mislabelled

`fact_switch.py:811-836`. `dA`/`dB` are **total log-probability differences**
over the whole answer (`_score_prompt_endings` returns `sum(vals)`,
`fact_switch.py:723`), not first-byte decisions. `sens = dA + dB` therefore
mixes:
- the correct first-byte preference (what we want), and
- the **remaining answer bytes**, which are trivially predictable from the
  already-emitted prefix (teacher forcing), and
- **length effects**: `dA` uses the *sum* when `same_len` and the *per-byte
  mean* when lengths differ (`fact_switch.py:813-814`). Mixing sum and mean in
  one metric is unit-inconsistent.

`fb_dA`/`fb_dB` use the per-byte detail at the first differing index
(`fact_switch.py:817-818`), which *is* the right first-byte measure. But the
headline `mean_sens` is not.

The Exa review makes exactly this point: *"Teacher forcing exposes the answer
prefix. Later answer bytes can be predicted from earlier gold bytes... Low answer
NLL can coexist with poor first-byte selection."*

### F21 — `fact_switch_loss` trains on teacher-forced answers, and `plain_nll` mode is a *copy* objective

`fact_switch.py:636-681`. In `plain_nll` mode (`fact_switch.py:659-660`) the loss
is the plain next-token NLL over the full sequence — including the answer bytes
**teacher-forced from the gold answer**. The model is therefore never asked to
*produce* the answer; it is asked to *complete* it. In `contrastive` mode
(`fact_switch.py:666-669`) the NLL term is still `weighted_nll` over the same
teacher-forced targets, plus a hinge on the first differing byte.

So the "fact-switch" stage is trained to **copy the gold answer**, and the
evaluation's `gen_indiv_exact` (free-running greedy generation) is a different
task from what was optimised. The gap is visible in the report: `gen_indiv_exact`
for `oov_clean` reaches 1.0 only after the *dialogue* stage
(`report.json:2736`), whereas the `before`/`selected` fact-switch rows show
`gen_indiv_exact = 0.0` (`report.json:1862`, `report.json:1965`).

### F22 — `report.json`'s `fact_switch.zero_shot` and `fact_switch_final` disagree in a way the report does not flag

`report.json:200-2412`:
- `zero_shot.oov_clean.pair_acc = 0.0625` (`report.json:502`) — this is
  `best_lm.pt`.
- `fact_switch_final.oov_clean.pair_acc = 1.0` (`report.json:2705`) — this is
  `best_dialogue.pt`.

But `fact_switch.selected.oov_clean.pair_acc = 0.9375` (`report.json:1974`)
is the fact-switch stage's own selected checkpoint, and
`fact_switch_final` (the dialogue stage) reports 1.0. So the dialogue stage
**improved** fact-switching — which is what
`dialogue_selection_score` (`run.ipynb` cell `dialog-train`:
`mean_bpb - 0.35*qswap_acc - 0.15*clean_acc`) explicitly *optimises*. The
report presents `fact_switch_final` as a separate top-level result without
noting that it is the output of a selection rule that includes fact-switch
accuracy as a term.

### F23 — `nonce_qswap` is the only split that tests generalisation, and it fails

`report.json:2110` (`fact_switch_final.nonce_qswap`): `pair_acc = 0.5`,
`gen_indiv_exact = 0.65625`, and `by_direction.reverse` has `pair_acc = 0` with
`fb_mean_sens = 1.7969` vs `forward`'s `12.51`. Nonce words are
pronounceable strings absent from `OOV_VOCAB` (`fact_switch.py:121-133`), so this
split is the only one requiring the model to bind a *novel token sequence*. The
model cannot. **All the headline wins are on memorised in-vocabulary strings.**

---

## 4. Flaws in the dialogue stage

### F24 — `heldout` is not held out from the fact-switch training

`report.json:3153-3160` lists `heldout` counts for prose/qa/chat. But
`dialogue_selection_score` (cell `dialog-train`) selects on **validation** BPB
and **validation** fact-switch accuracy; the heldout set is scored once at the
end (`report.json:3226-3232`). That part is fine.

However `fact_switch.py:1055-1067` (`load_coqa_pools`) draws heldout CoQA
questions from `pools['heldout']['qa']` and filters with
`ans in prompt.casefold()` — i.e. **only questions whose answer appears
verbatim in the prompt**. These are extractive-by-construction and are the
easiest possible subset. `report.json` reports `qa_excerpt` training and
`qa_extractive` counts but the CoQA extractive eval is not in the final report
block.

### F25 — `target_batch` supervises the *gold* history turns

`dialogue.py:228-246`:

```238:243:A:\hgrt10m-full\hgrt10m\dialogue.py
        if history_targets:
            for reply in re.finditer(rb'(?m)^(?:Assistant|Answer): ([^\n]*\n)', prompt):
                first, last = reply.span(1)
                y[i, first-1:last-1] = tokens[first:last]
```

When `history_targets=True`, the loss supervises **gold assistant turns
embedded in the prompt** as well as the final target. This trains the model to
produce answers it was shown, in-context, with teacher forcing — reinforcing
copying rather than recall. Combined with `qa_excerpt` (60,370 training rows,
`report.json:3149`) where the answer is literally a span of the passage, the
"QA capability" measured at `heldout_selected_bpb.qa = 1.0987`
(`report.json:3229`) is substantially **extractive copying**, not question
answering.

### F26 — The dialogue selection score mixes BPB and accuracy with mismatched units

`run.ipynb` cell `dialog-train`:

```python
def dialogue_selection_score(bpb_metrics, cf_summary):
    mean_bpb = sum(bpb_metrics.values()) / len(bpb_metrics)
    qswap_acc = cf_summary['oov_qswap']['pair_acc']
    clean_acc = cf_summary['oov_clean']['pair_acc']
    return mean_bpb - 0.35 * qswap_acc - 0.15 * clean_acc
```

`mean_bpb` includes `wiki` (`report.json:3189-3193` shows `baseline['wiki'] =
evaluate(base, 'validation')`), so it mixes **byte-level language modelling loss**
with **sentence-level fact-switch accuracy**, with fixed ad-hoc weights
(0.35/0.15) that were never justified or swept. A 0.15 BPB improvement is worth
0.15/0.35 ≈ 0.43 of a pair-accuracy point; whether that trade is right is
unstated and untested. There is **no sensitivity analysis and no ablation** over
these weights.

---

## 5. What is actually correct

Credit where due — these held up under testing:

1. **`delta_reference` is mathematically correct.** Matches a sequential
   Gated DeltaNet oracle to 5e-7 relative (`_probe_delta.py`).
2. **The tree is causally clean.** The full 8×8 source→output perturbation
   matrix (`_probe_tree4.py`) is strictly lower-triangular; no future byte
   influences an earlier prediction through the tree. No leak.
3. **The Byte-level BPB accounting is correct.** Byte vocabulary = 256, targets
   are byte-shifted (`windows()` in cell `5dda70a4`), divided by `ln 2`, and
   `test_bpb` values are consistent with the unigram/bigram baselines
   (`report.json:163-166`).
4. **Checkpoint provenance is good.** The corpus manifest records SHA-256 for
   every split (`corpus.json`), the corpus is pinned to a HuggingFace revision
   (`b08601e04326c79dfdd32d625aee71d232d685c3`), and the data files are hash-
   recorded. This is better than most pipelines do.
5. **The corpus is a real corpus.** 539 MB of WikiText-103 raw + 4.9 MB
   OpenWebText, with disjoint train/val/test by construction.
6. **The report is honest about the training failure** — `report.json:127-181`
   records the BPB rise and the fact that `selected_step = 5908`.

---

## 6. The central flaw, stated plainly

**The run demonstrates that a 10M-parameter byte LM can learn to copy from its
local context, and it demonstrates this with an evaluation whose memory
ablation shows zero effect.**

The chain of evidence:

1. The LM stage trained for 8,686 steps and made the model **strictly worse**
   every checkpoint (`metrics.jsonl`); the selected model is the **imported**
   checkpoint from a prior run (`config.json:14`).
2. The tree contributes **2.67%** of the logit signal and **0.0029 BPB**
   (`_probe_contrib.py`, `report.json:170-171`).
3. The tree's `read` mask is on the wrong axis (`model.py:194-196`), so the
   architecture is a **flattened node bag, not a hierarchical tree** — the
   premise of the design does not hold in the code.
4. Every fact-switch example fits in the 1536-byte local attention window, and
   for the headline `oov_clean` split the fact and query are in the **same
   64-byte chunk**, making the tree structurally unable to contribute
   (`_probe_confound.py`).
5. Turning the tree off changes `oov_clean pair_acc` from 1.0 to **1.0**
   (`_probe_confound.py`).
6. The `shuffle` ablation is a batch-axis roll (`model.py:175`) and a joint K/V
   permutation is an exact no-op for content-only attention — so it is not a
   memory ablation at all.
7. The `counterfactual_tree_swap` is the same operation relabelled
   (`fact_switch.py:906-907`).
8. The checkpoint is selected on the same suite it is scored on
   (`fact_switch.py:1152-1155`), so the reported maximum is upward-biased.
9. The only split testing genuine generalisation — `nonce_qswap` — shows
   `pair_acc = 0.5` with `reverse` direction at **0.0** (`report.json:2110-2132`).

**Therefore the `pair_acc = 1.0` headline is a copying result presented as a
memory result, and the tree — the entire architectural contribution — is
demonstrably inert on the reported metrics.**

---

## 7. Recommended fixes, in priority order

### P0 — Fix the tree's read path (`model.py:164-196`)

```python
def read(self, x, nodes, mask=None):
    # nodes: list of (b, slots, slot_dim) -> build (b, slots, n_nodes, slot_dim)
    # q must be E == slot_dim
    q = self.read_proj(x)                       # separate projection width->slot_dim
    a = F.scaled_dot_product_attention(q, nodes, nodes, attn_mask=mask)
    return self.gain * self.out_proj(a)
```

with `memory = torch.cat(nodes, 1)` (heads=slots, length=nodes) and a mask of
shape `(b, 1, Lq, n_nodes)` that **actually gates completed subtrees per query
chunk**. Add a unit test asserting:
- head axis == slot axis,
- `E` of q == `E` of nodes,
- the visible-node pattern is a strict function of chunk index,
- the tree runs at `CONTEXT` values ≠ 512 (currently it hard-fails).

### P0 — Make the fact-switch suite require memory

1. **Push the query beyond the local window.** Either set `L7.window = 64`
   (chunk-sized) during fact-switch eval, or place the fact ≥ 2048 bytes back.
   `report.json` already has `long_clean` with 2 fillers — extend to a
   `far_clean` split at ≥ 2048 bytes.
2. **Assert the fact is in an earlier chunk.** Add a precondition to
   `build_fact_switch_suite` that `fact_offset // chunk < query_offset // chunk`.
   Today `oov_clean` fails this 16/16.
3. **Randomise the batch order before `shuffle`.** `source.roll(1,0)` on a
   fixed-order suite is a deterministic function of the test-suite layout.
   Use a seeded permutation, and report the donor family.

### P1 — Fix the `shuffle` ablation to be a real, matched intervention

Per the Exa review, use a **factorial design**, not one destructive switch:

| Intervention | Purpose |
|---|---|
| tree read disabled | necessity |
| tree write disabled | necessity |
| key-only permutation | breaks binding, preserves content |
| value-only permutation | breaks binding, preserves content |
| matched opposite-fact state swap | binding-specific |
| unrelated-fact state swap | control |
| clean-state restoration | shows the effect is reversible |

Report **counterfactual answer preference**, **first-byte accuracy**,
**unrelated-key accuracy** and **ordinary next-byte loss** for each, with
paired CIs.

### P1 — Add a held-out split for checkpoint selection

`run_two_stage_fact_switch` (`fact_switch.py:1152`) selects on the suite it
reports. Split `oov_clean`/`oov_qswap` into `select` (half the pairs) and
`report` (the other half, and a fresh seed), and select on `select` only.

### P1 — Fix `context_probe` to hold content fixed

```python
starts = np.full(windows, offset)          # same byte range for every length
blocks = np.stack([data[s:s+length+1] for s in starts])
```

Then the curve measures context length, not content difficulty. Re-derive the
"saturating" claim in `PHASE3_DECISION.md` from the corrected curve, and
re-check the claim the decision actually rests on (that the *task* never needs
>320 bytes).

### P2 — Make the training objective's weighting explicit and sweep it

`.25*matched.mean() + .50*ending.mean() + .25*ending[:,:32].mean()`
(`phase1_notebook.py:140`) puts 75% of the paragraph loss on 128 of 320 bytes.
Make the weight a named constant, sweep it, and report the BPB-vs-objective
trade-off. Given F9 (BPB rose monotonically), the first thing to try is
`matched.mean()` over the *full* window.

### P2 — Fix the `torch` vs `fla` backend equivalence

`model.py:83` passes `scale=1.`; FLA's default is `d_k^{-1/2}`. Match the
layout and scale to FLA's reference
([naive.py](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py)),
then add a test asserting `torch` and `fla` outputs agree to BF16 tolerance on
the same input, including non-zero initial states.

### P2 — Split streaming from non-streaming semantics

`model.py:141` keeps `window-1` keys; the mask allows distance `< window`. Make
the streaming cache keep exactly `window` keys and re-derive the distance mask
consistently, then test that `stream_chunk` and `forward` produce identical
logits for the same byte stream.

### P3 — Report failures

`nonce_qswap` at `pair_acc = 0.5` / reverse `0.0` is the most informative
number in the whole report and is buried at `report.json:2110`. Move it to the
top. The README already promises *"Timed-out and incomplete runs remain
inconclusive"* — apply the same discipline to the memory claim.

---

## 8. Evidence index

| Finding | Evidence |
|---|---|
| F1 tree mask axis | `model.py:186`, `model.py:194-196`; `_probe_mask2.py` (SDPA `RuntimeError` on the intended layout) |
| F2 read dim mismatch | `model.py:150`, `model.py:164-167` |
| F3 merge ≈ identity | `_probe_ckpt.py` (‖merge_proj‖ 12.76 vs ‖I‖ 16) |
| F4 tree inert | `_probe_contrib.py` (2.672%), `_probe_tree_value2.py`, `_probe_confound.py` |
| F5 shuffle = batch roll | `model.py:174-176` |
| F6 delta correct, layout differs | `_probe_delta.py` (5e-7), FLA naive reference |
| F7 streaming ≠ training | `model.py:141` vs `model.py:116-133` |
| F8 objective weighting | `phase1_notebook.py:140-141` |
| F9 BPB regression | `metrics.jsonl` (13 checkpoints, all worse) |
| F10 misleading initial/final | `report.json:127-128` vs `report.json:154` |
| F11 tree ablation ≈ 0 | `report.json:159-162` |
| F12 context probe confounded | `phase3_probe.py:66-67`; `_probe_final.py` |
| F13 task-vs-capacity conflation | `PHASE3_DECISION.md:48-56` |
| F14 answers in local window | `_probe_confound.py` (all splits fit 1536) |
| F15 tree unreachable | `_probe_confound.py` (oov_clean 0/16) |
| F16 shuffle not an ablation | `_probe_tree_value2.py` (best_fact_switch: off **better** than on) |
| F17 fake counterfactual swap | `fact_switch.py:906-907` |
| F18 swap uses the pair's own twin | `fact_switch.py:898-901`, `model.py:175` |
| F19 selection on test suite | `fact_switch.py:1152-1155` |
| F20 sensitivity metric | `fact_switch.py:811-836`, `fact_switch.py:723` |
| F21 teacher-forced copy training | `fact_switch.py:659-669` |
| F22 dialogue stage optimises FS | cell `dialog-train` `dialogue_selection_score` |
| F23 nonce fails | `report.json:2110-2132` |
| F24 CoQA subset trivial | `fact_switch.py:1063-1065` |
| F25 gold history supervised | `dialogue.py:238-241` |
| F26 metric units mixed | cell `dialog-train` `dialogue_selection_score` |

External references consulted via Exa (60 searches, 42 sources):
[Gated DeltaNet](https://arxiv.org/html/2412.06464) ·
[DeltaNet](https://arxiv.org/html/2406.06484v6) ·
[FLA naive reference](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/naive.py) ·
[NVLabs GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) ·
[Mamba-2 ssd_minimal](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py) ·
[TTT-LM](https://github.com/test-time-training/ttt-lm-pytorch) ·
[RAPTOR](https://arxiv.org/abs/2401.18059) ·
[MemTree](https://arxiv.org/abs/2410.14052) ·
[Hierarchical Aggregate Tree](https://arxiv.org/html/2406.06124) ·
[HMT](https://aclanthology.org/2025.naacl-long.410.pdf) ·
[HCAM](https://papers.neurips.cc/paper_files/paper/2021/file/ed519dacc89b2bead3f453b0b05a4a8b-Paper.pdf) ·
[Landmark attention](https://arxiv.org/html/2305.16300) ·
[Infini-attention](https://arxiv.org/html/2404.07143) ·
[Titans](https://arxiv.org/html/2501.00663) ·
[Tree of Thoughts](https://arxiv.org/abs/2305.10601) ·
[Zoology](https://arxiv.org/html/2312.04927) ·
[Attribution patching best practices](https://arxiv.org/html/2309.16042) ·
[Amnesic Probing](https://arxiv.org/abs/2006.00995) ·
[ROME](https://papers.neurips.cc/paper_files/paper/2022/file/6f1d43d5a82a37e89b0665b33bf3a182-Paper-Conference.pdf) ·
[BLT byte-level BPB definition](https://aclanthology.org/2025.acl-long.453.pdf) ·
[Sequence composition / packing](https://aclanthology.org/2024.acl-long.427.pdf) ·
[HF perplexity guide](https://huggingface.co/docs/transformers/en/perplexity).
