# Phase Implementation Status

## ✅ DONE

### Phase 0: Hypothesis Validation Experiments
**Status:** Code written and committed, but NOT YET EXECUTED.

**What's done:**
- Added `run.ipynb` cell 0 with greedy vs. sampled decode comparison
- Diagnostic analysis complete (traced every number back to source code)
- Root causes identified and documented in `DIAGNOSIS_AND_PLAN.md`

**What's NOT done:**
- Cell 0 has not been run on the checkpoint
- Phase 0 decision (greedy >> sampled? or greedy ≈ sampled?) has not been made

**Time to run:** ~30 min (inference only, no training)
**Requires:** A100 + checkpoint at `runs/best_lm.pt` (should exist from previous run)

---

## ❌ NOT DONE

### Phase 1: Fix Prose Objective (Sentence-Boundary-Aware Loss)
**Status:** Planned, NOT implemented.

**What needs to happen:**
1. Modify `run.ipynb` cell 10 (`prose_backward`):
   - Add multi-candidate sentence-level contrastive term
   - Keep paragraph-level term for stability
   - Combine as: `loss = 0.6*para_loss + 0.4*sent_loss`

2. Modify `run.ipynb` cell 8 (data preparation):
   - Extract sentence boundaries from corpus anchors
   - Prepare K=8 candidate sentences per training example
   - Build `sentence_anchors`, `sentence_foils` tensors

3. Modify selection metric:
   - Add `sentence_completion_accuracy` computation
   - Best checkpoint by: `selection_loss = 0.7*matched_bpb + 0.3*(1 - sent_acc)`

**Time required:**
- Code writing: ~2 hours
- Training: 4–6 hours on A100 (depends on new loss overhead)
- **Total: 6–8 hours**

**Decision blocker:** Phase 0 results. If greedy >> sampled, may skip Phase 1 (or simplify it).

---

### Phase 2: Raise Target Length & Add Long-Form Corpus
**Status:** Planned, NOT implemented.

**What needs to happen:**
1. Download OpenWebText via `datasets` library
   - Approx. 200K documents, ~40GB raw
   - Parse paragraphs (heuristic: `\n\n`-delimited)
   - Dedup with existing corpus

2. Raise target-length cap:
   - Keep `dialogue.py` capped at 160 B for QA/chat
   - Add uncapped `prose_owt` task for OpenWebText paragraphs
   - Targets: 500–2000 B (full paragraphs)

3. Update training schedule:
   - Add `'owt'` or `'owt_long'` to `TRAIN_STAGES`
   - Allocate ~20% of training budget to it

**Time required:**
- Corpus prep: ~2–3 hours (download + dedup, can parallelize)
- Code changes: ~1–2 hours
- Training: 6–8 hours on A100
- **Total: 9–13 hours**

---

### Phase 3: Context Window & Coherence Metrics
**Status:** Planned, NOT implemented.

**What needs to happen:**
1. Change pretraining context:
   - `CONTEXT = 512` → `CONTEXT = 1024`
   - Re-profile hardware benchmarks (re-run cell with choose_batch)

2. Add coherence scoring:
   - Implement `coherence_score(text)` using spaCy
   - Measure: ROOT-arc coverage, complete-dependency rate, dangling-arc rate
   - Add to evaluation loop

3. Update checkpoint selection:
   - Compute held-out coherence score for each validation checkpoint
   - Use coherence score + BPB for best-model selection

**Time required:**
- Code writing: ~1–2 hours
- Hardware profiling: ~1 hour
- Training: 4–6 hours on A100 (may be slower due to larger context, so longer profiling)
- **Total: 6–9 hours**

---

## Summary: What's Actually Running vs. Planned

| Phase | Status | Code Ready? | Executed? | Blockers | Hours to Complete |
|---|---|---|---|---|---|
| **0** | Committed | ✅ Yes | ❌ No | Need to run cell 0 | 0.5 (inference) |
| **1** | Planned | ❌ No | ❌ No | Phase 0 results; code not written | 6–8 (mostly training) |
| **2** | Planned | ❌ No | ❌ No | Phase 1 done; corpus not downloaded | 9–13 (mostly corpus prep + training) |
| **3** | Planned | ❌ No | ❌ No | Phase 1–2 done; code not written | 6–9 (mostly training) |

---

## Hours Clarification

**Yes, all hour estimates are TRAINING time on A100.**

Breakdown for Phase 1 (example):
- Code writing: 2 hours (non-GPU, human time)
- Training on A100: 4–6 hours (GPU time, actual wall-clock training)
- **Total elapsed: 6–8 hours** (if you start code immediately after Phase 0)

Similarly for Phase 2:
- Corpus prep: 2–3 hours (can be parallel with Phase 1 training)
- Code changes: 1–2 hours
- Training: 6–8 hours
- **Total: 9–13 hours**

If running sequentially (Phase 0 → 1 → 2 → 3):
- Phase 0: 0.5 hours
- Phase 1: 6–8 hours
- Phase 2: 9–13 hours (can overlap with Phase 1 training)
- Phase 3: 6–9 hours
- **Total: ~22–30 hours elapsed** (much of it GPU-idle if phases don't overlap)

If running optimally with parallelization:
- Phase 0: 0.5 hours (can't parallelize)
- Phase 1 + Phase 2 corpus prep: 6–8 hours + 2–3 hours = 8–11 hours total (can overlap)
- Phase 1 training happens while Phase 2 corpus preps
- Phase 2 training: 6–8 hours (happens after Phase 1 completes)
- Phase 3: 6–9 hours (happens after Phase 2 completes)
- **Total: ~20–28 hours elapsed**

---

## What Needs To Happen Right Now

**Immediate next step:** Run Phase 0 (cell 0 in notebook).

This is a **decision gate**:
- **If greedy >> sampled:** Problem is sampling, not training. Can ship a quick fix (switch to greedy decoding), defer Phase 1–3.
- **If greedy ≈ sampled:** Proceed to Phase 1. The objective is the bottleneck.

**Command to run Phase 0:**
```
cd A:/hgrt10m-full/hgrt10m
# In Jupyter or via nbconvert:
jupyter nbconvert --to notebook --execute run.ipynb --output-dir . --ExecutePreprocessor.timeout=3600
# Or manually run cell 0 in Jupyter UI
```

Expected output: comparison of greedy vs. sampled complete-sentence rates, coherence scores.

