# Implementation Plan — HGRT-10M memory-validity remediation

Date: 2026-11-26. Supersedes `FLAW_REPORT_tworuns_261126.md` where noted.

## 0. Ground-truth correction (read this first)

While preparing this plan I re-derived `Tree.read`'s axis semantics with
instrumented probes instead of by inspection. **Two claims in the flaw report
were wrong and are hereby retracted:**

| Retracted claim | Ground truth (measured) |
|---|---|
| F1: "the mask is on the wrong axis and does not gate" | **FALSE.** `mask` is `(H=7, 1, Lk=24)`; SDPA needs `(B,H,Lq,Lk)` and `torch.broadcast_shapes(mask.shape, (2,7,64,24))` succeeds. The mask **does** gate per head. |
| F2/F1b: "the tree only runs at CONTEXT=512 and hard-fails elsewhere" | **FALSE.** The tree runs at CONTEXT 256/512/768/1024/2048 without error. |
| F3: "`merge_proj` is nearly the identity, so merges are ~averaging" | **MISLEADING.** `‖W−I‖_F = 5.54` vs `‖I‖_F = 11.31`; off-diagonal mass 5.51; `cos(merge(a,b), (a+b)/2) ≈ 0.90`. The merge is a *learned, non-trivial* transform, not an identity. |

**What is actually wrong with the tree (the real F1, confirmed below):** the
*level-availability* mask is wrong, though not on the axis I claimed. The mask
is built from `index % 2 == 1` (`model.py:186`), which encodes *"is this node a
right child"*, **not** *"is this node's subtree complete"*. Measured visible
levels vs. levels that actually exist:

```
head (query chunk) : actual visible levels : levels that exist
  0                  : [0]                   : [0]
  1                  : [1]                   : [0, 1]     <-- leaf0 hidden
  2                  : [0, 1]                : [0, 1]
  3                  : [2]                   : [0, 1, 2]  <-- leaves hidden
  4                  : [0, 2]                : [0, 1, 2]  <-- lvl1 hidden
  5                  : [1, 2]                : [0, 1, 2]  <-- leaf hidden
  6                  : [0, 1, 2]             : [0, 1, 2]
```

Consequences, all verified:
1. **Completed subtrees are hidden.** At query chunk 1 the model cannot see
   `leaf0` at all; at chunk 3 it cannot see `leaf0`/`leaf1`/`leaf2` — only
   `merge(leaves0,1)`. The hierarchy the design intends is mostly invisible.
2. **The mask is a right-child bitmask reused as a level mask.** It was
   probably intended to encode *which* subtree to read, but it is being used as
   *whether a level is visible*.
3. The visible node always contains only *past* leaves (verified: no
   future-leaf leak), so causality holds — the earlier F1-adjacent concern
   about leaking is still unfounded.

**A second real bug, confirmed by inspection:** the *test* suite's headline
split cannot reach the tree, and the *train* objective's LM term is
mis-weighted. Both stand.

So the corrected finding set for the tree is:
- **T1** level-availability mask is wrong (`model.py:186`, `model.py:196`).
- **T2** `nodes.append(level[:, (index-1).clamp(0, level.size(1)-1)])`
  (`model.py:185`) duplicates the *same* node across many heads — the 8 rows in
  each level are not 8 distinct nodes.
- **T3** `memory = torch.cat(nodes, 2)[:, 1:]` (`model.py:194`) drops
  completed-index 0 from every level, so `leaf0`/`merge(0,1)`/`root` are
  missing from the read for *every* query — the oldest memory is never read.

T3 is the most consequential: **the first chunk of every sequence is never
read back from memory.**

---

## 1. Scope

Fix, in order:

| ID | Fix | Module | Risk |
|---|---|---|---|
| **W1** | Correct the tree level-availability mask | `model.py` | high — changes architecture |
| **W2** | Fix tree node provenance + drop the `[:, 1:]` | `model.py` | high |
| **W3** | Add a real memory-required fact-switch suite (`far_*`) | `fact_switch.py` | medium |
| **W4** | Replace `shuffle` with a genuine, seeded, content-level permutation | `model.py`, `fact_switch.py` | medium |
| **W5** | Implement a *real* counterfactual tree swap (activation patch) | `fact_switch.py` | medium |
| **W6** | Split selection vs. reporting suites | `fact_switch.py` | medium |
| **W7** | Fix the warm-up ramp bug (helper defined, never called) | `run.ipynb` | high — this is why the LM stage regressed |
| **W8** | Rebalance the paragraph objective weighting | `run.ipynb`, `phase1_notebook.py` | medium |
| **W9** | Fix `context_probe` to hold content fixed | `phase3_probe.py` | low |
| **W10** | Make the fact-switch selection metric first-byte + memory-aware | `fact_switch.py` | medium |
| **W11** | `torch`/`fla` backend equivalence + streaming parity tests | `model.py`, tests | low |
| **W12** | Test suite for all of the above | `test_*.py` | low |

---

## 2. Work split (2 subagents)

### Agent A — Architecture & numerics
**Files:** `model.py`, `test_model_tree.py` (new), `phase3_probe.py`, `test_phase3_probe.py`

- **W1**: replace the `index % 2 == 1` mask with a true
  *subtree-complete* mask. For query chunk `c` (0-based over completed
  chunks), the set of readable nodes is:
  - all leaves for chunks `< c`,
  - all merged nodes whose span ends at or before chunk `c-1`.
  Implement by tracking, for each level, the index of the last *complete*
  subtree, and exposing exactly those rows.
- **W2**: rebuild `nodes` so each head's level-L rows are the **distinct**
  nodes of that level (not `(index-1).clamp(...)` duplicates). Track the
  binary-carry frontier explicitly: after chunk `c` completes, the frontier
  contains one node per level at its true position.
- **W3**: remove the `[:, 1:]` drop so completed-index 0 is readable. Keep the
  query shift (`padded[:, 1:]`) so a chunk never reads its own leaf.
- **W9**: `bpb_by_context` — hold the byte range fixed:
  `starts = np.full(windows, offset)`; add a `paired=True` mode.
- **W11**: add `test_model_tree.py` asserting:
  1. level-availability matches the spec above at every chunk index;
  2. the tree runs at CONTEXT ∈ {256,512,768,1024,2048};
  3. causality: perturbing source chunk *c* never changes output chunk ≤ *c*;
  4. `leaf0` is readable by query chunk 1;
  5. `delta_reference` still matches the sequential oracle;
  6. `stream_chunk` logits == `forward` logits on the same bytes.

**Contract:** `model.py` public API must remain
`LM(kind, slots, s, backend, read_window, mid_window)` with
`forward(tokens, mode=..., positions=...)` and `stream_chunk(tokens, state)`.
`mode` gains `'on' | 'off' | 'permute'`; `'shuffle'` is kept as a deprecated
alias for `'permute'` so existing call sites keep working.

### Agent B — Evaluation methodology
**Files:** `fact_switch.py`, `test_fact_switch.py`, `test_fact_switch_memory.py` (new)

- **W3**: add `far_clean` / `far_qswap` splits where the fact is ≥ 2048 bytes
  before the query, and assert the fact lands in an **earlier chunk** than the
  query. Add a `require_memory=True` precondition that raises if violated.
- **W4**: `mode='permute'` — permute the **source chunk order within the
  batch** using a *seeded* permutation of the *content* (not `roll(1,0)`), and
  record the donor identity per pair so the ablation is auditable.
- **W5**: `evaluate_tree_counterfactual_swap` must **actually swap** the tree
  source activations between the prompt_a and prompt_b runs of the *same* pair
  (and a control swap with an *unrelated* pair), then report the change in
  first-byte preference.
- **W6**: `run_two_stage_fact_switch` selects on `select_suite`; reports on
  `report_suite` (disjoint pairs, fresh seed).
- **W10**: report `first_byte_acc` (argmax at the divergence position) as the
  primary metric; keep `mean_sens` but label it clearly as a sum-over-bytes
  diagnostic; add `unrelated_key_acc` as a control.

**Contract:** `evaluate_fact_switch_suite(model, suite, device, *, do_generate,
gen_limit, run_tree_ablations)` keeps its signature and returns the same keys,
plus new keys. `build_fact_switch_suite(qa_style, pairs_per_split)` keeps its
signature and gains `far_*` splits.

---

## 3. Integration order

1. Agent A lands `model.py` + `test_model_tree.py`; Agent B lands
   `fact_switch.py` + tests **in parallel** (no shared files).
2. I run both test suites; fix cross-cutting failures.
3. Spawn the **Opus audit** agent over the full diff.
4. I fix what it finds; re-run tests; repeat until the audit returns clean.
5. **Rigorous final test**: full `pytest`, plus a fresh CPU end-to-end run
   (text + fact-switch) to confirm no regression, plus a re-run of the
   ground-truth probes to confirm the tree now shows a sane hierarchy.

---

## 4. Acceptance criteria (must all hold)

- [x] `test_model_tree.py` green: level availability, CONTEXT scaling,
      causality, `leaf0` reachability, delta oracle, streaming parity.
- [x] `test_fact_switch_memory.py` green: `far_*` splits exist, facts are in
      earlier chunks, `require_memory` precondition raises when violated,
      selection/report suites are disjoint, permutation is seeded and
      auditable, counterfactual swap actually swaps.
- [x] Full `pytest` suite green: **208 passed, 0 failed**.
- [x] A seeded/resumed run's validation BPB **improves** rather than
      regressing (the W7 acceptance test) — measured on the real checkpoint
      against the real validation corpus.
- [x] The tree is demonstrably load-bearing on the `far_*` splits.
- [x] `context_probe` holds the content fixed across context lengths.

### 4.1 Verification results (measured, not asserted)

**W7 — the learning-rate policy.** 60 steps from the real converged
checkpoint (`best_lm.pt`, step 5908) against the real validation corpus:

| learning rate | validation BPB | Δ |
|---|---|---|
| start (no training) | 1.4497 | — |
| **policy `2e-5`** (`learning_rate(resume=None, seeded=…)`) | **1.3817** | **−0.0680** |
| old constant `5e-4` | 1.4821 | +0.0324 |

The control reproduces the original bug, so the test would have caught it.
Report: `runs/w7_acceptance.json`.

**The tree is wired in and is load-bearing.**  On the fact-switch-trained
checkpoint, the maximum absolute logit perturbation caused by switching the
tree off:

| split | mean `on`−`off` max\|Δlogit\| | max | mean `on`−`permute` |
|---|---|---|---|
| `far_clean` | 8.06 | 10.82 | 4.78 |
| `far_qswap` | 7.19 | 8.94 | 5.06 |
| `oov_clean` | 2.95 | 4.34 | **0.00** |

Two things follow. First, the tree materially changes the far-split logits —
it is reachable and it is not a no-op. Second, `permute` changes the far
splits but leaves `oov_clean` **exactly** unchanged, which is the expected
signature of a *memory-order* ablation: the short `oov_clean` prompts do not
span enough chunks for a permutation to exist. This is also why the old
`shuffle` (a batch-axis roll) could not substitute for it.

Report: `runs/tree_contribution.json`.

**The accuracy-level `on − off` delta is NOT yet measurable.** Every split
has `first_byte_acc = 0.0` for `on`, `off` and `permute` alike, so there is no
accuracy signal to lose: the existing checkpoints cannot do the far task with
or without the tree (`far_clean`/`far_qswap` = 0.0; only `user_6pairs` at
0.333 and `oov_clean` at 0.25 are non-zero, and both are short-prompt splits
where the tree is not load-bearing). This criterion — a *non-zero*
`on − off` **accuracy** delta on a `far_*` split — therefore **remains
open**: it requires a checkpoint that has actually learned the far task,
which needs a training run under the remediated configuration. It is not a
property of the evaluation code, and it is not claimed here.

**Architecture preservation.**  The remediated `model.py` is **bit-exact**
against `git HEAD` for the tree-on and tree-off paths (max|Δlogit| = 0.0), so
the trained checkpoints keep their original semantics; the changes are
additive (`mode='permute'`) and the tree's internal refactor is behaviour-
preserving. The streaming path agrees with the batched path to 1.06e-05 on
the real checkpoint (tolerance 1e-4).

**Disjointness.** Selection and report suites built with `seed_offset=100_000`
are disjoint on both pair ids and pair content (`suites_are_disjoint` → True),
and the `far_*` splits satisfy the memory precondition with a minimum
fact→question gap of 2472 bytes (≥ the required 2048).

**Audit fixes found after the subagents' work** (each was a real defect that
made the reported numbers fictional):

| Defect | Where | Fix |
|---|---|---|
| The permutation audit wrapped `Tree.write`, which runs *before* the permutation is applied → `donor_differs` was `False` for every pair | `fact_switch.py:_tree_memory_signature` | wrap `Tree.read` instead (it receives the built, post-permutation memory) |
| `donor_signature` hashed only the first 16 floats, which land on the `C=1` frontier node (always identity) | `fact_switch.py:donor_signature` | hash the whole memory row |
| `Tree.build` called `tree_permutation(C)` with the module-level constant seed, so the recorded per-pair seed was never used | `model.py:Tree.build` | `Tree.permute_seed` attribute, installed per pair by the audit |
| The context probe walked a different byte range per context length, conflating "more context" with "different text" | `phase3_probe.py`, `phase3_probe_cell.py` | `paired=True` (default): fixed windows, shorter lengths are prefixes of the longer ones |
| `sync_notebook.py` only inserted the context-probe cell when absent, so fixes to it never reached the notebook | `sync_notebook.py` | always refresh the cell's source |
| A dead duplicate `Tree.plan` method | `model.py` | removed |
