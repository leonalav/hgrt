# HGRT-10M: Deep Analysis — Prose Coherence Diagnosis

**Conclusion:** The model is **trained to emit short, padded text**. The root causes are (1) the **training objective** (`continuation_nll` with 128-byte supervised spans) has no notion of sentence boundaries, and (2) the **dialogue stage** caps all targets at 160 bytes, teaching short-form template completion rather than long-form prose composition.

---

## Evidence Summary

### Diagnosis 1: NOT Capacity-Limited
- Test BPB: **1.46** vs. bigram baseline 3.56 → **2.10 BPB improvement** = real language structure
- Tree ablation: ±0.003–0.005 BPB (negligible)
- Parameter budget: 10.01M (correctly allocated)
- **Conclusion:** Model has sufficient capacity; better prose is possible without resizing.

### Diagnosis 2: NOT Decoding-Limited  
- Fact-switch greedy generations: `gen_indiv_exact = 1.000` on `oov_clean`
- Tokens are correct; the **structure breaks between tokens**
- Proof: "Cork"/"Lima" (single-token answers) are flawless; prose continuations are incoherent

### Diagnosis 3: **ROOT CAUSE — Training Objective**
From `run.ipynb` cell 10, `prose_backward()`:
```python
loss = .25*matched.mean() + .50*ending.mean() + .25*ending[:, :32].mean()
loss = loss + .25*F.softplus(-margin).mean()
```
The loss supervises:
- **Paragraph-level discrimination** (192 B prefix → true 128 B continuation vs. foil prefix spliced to same suffix)
- **No sentence boundaries** — a 128-byte continuation that is 1 broken sentence vs. 2 short sentences = same NLL
- **Early-token bias** (`ending[:, :32]` = first 32 bytes = ~5 words) — the loss up-weights only the opening fragment
- **No syntactic notion** — high NLL ≠ ungrammatical if bytes are locally probable

**Example failure modes this enables:**
```
"The properties of plants will also attempt to lead to the invertebrate , the results of all planets ."
       ↑ locally fluent bytes ↑                                        ↑ high-freq padding ↑
```
Bytes like "the", "of", "and" are so frequent that a model optimizing mean NLL can pad 128 bytes with padding tokens and get cheap loss.

### Diagnosis 4: **ROOT CAUSE — Dialogue Stage Destroys Structure**
Training mix from final dialogue stage (304K examples):
| Task | Count | % | Target cap | Avg target |
|---|---|---|---|---|
| prose | 78,528 | 25.8% | 160 B | ~1–2 sentences |
| chat | 73,903 | 24.2% | 160 B | ~40 B (short replies) |
| prose_long | 67,498 | 22.1% | 280 B | ~2–3 CoQA sentences |
| qa_excerpt | 60,370 | 19.8% | 160 B | extracted spans (~20 B) |
| qa | 15,581 | 5.1% | 160 B | extracted spans |
| qa_extractive | 8,954 | 2.9% | 160 B | extracted spans |

**Key fact:** `dialogue.py` line ~145:
```python
if len(target.encode()) <= reply_bytes:  # reply_bytes = 160
```
Every supervised target is **capped at 160 bytes (~27 words).** The model is trained to predict the next 160 bytes (or fewer) given a 1536-byte context — this teaches **template completion within a fixed budget**, not **long-form narrative composition**.

Result: Held-out prose BPB dropped 38% (validation: 2.698 → 1.675), but that's because the model **memorized short-span templates**, not because prose became good. Proof: QA held-out exact match is only 18.75% (F1 0.28), so the BPB gain is shape-matching, not semantic understanding.

### Diagnosis 5: **Context Too Short for Clause Boundaries**
- Pretraining context: 512 B = ~85 words = one clause + one governor (barely)
- Dialogue context: 1536 B but all **supervised targets are 160 B**
- Multi-sentence clausal structures need 800+ B of context for realistic dependencies
- **Result:** Model's dependency context is 512 B by training history; fine-tuning at 1536 B doesn't retrain the deep layers' dependency binding

### Diagnosis 6: Byte-Level Tokenizer Leaks Markup
- 25% of prose continuations contain `@-@` / `@,@` (WikiText placeholders)
- Byte-level modeling: every byte is a token, model must learn tokenization from scratch
- **Result:** Model memorizes raw markup instead of normalizing it

---

## Ranked Interventions by Expected Benefit & Reversibility

| Rank | Intervention | Expected gain | Reversibility | Why | Phase |
|---|---|---|---|---|---|
| **1** | Add **sentence-boundary-aware objective** term | +25–40% | High | Direct fix for the loss that has no notion of "completion" | 1 |
| **2** | Raise target length cap (160 B → 500–2000 B) | +15–25% | High | Train the model to emit full thoughts, not padding | 1–2 |
| **3** | Add long-form prose corpus (OpenWebText paragraphs) | +10–20% | High | Give the objective real multi-sentence data with syntactic variety | 2 |
| **4** | Train at 1024–2048 B context | +10–20% | High | Allow clause + governor dependencies to bind | 1–2 |
| **5** | Greedy/low-temp decoding for prose | +5–10% | Trivial (inference-only) | Test immediately — if this alone fixes it, 1–4 are unnecessary | 0 |
| 6 | Byte → BPE tokenizer | +10–20% | Low | Requires full retrain; defer to after 1–4 | Later |
| 7 | Constituency parsing head | +10–15% | Low | Orthogonal; only add after 1–4 | Later |

---

## Plan: Phases 0–3

### **Phase 0: Falsify the hypothesis (inference-only, ~2 hours)**
Test two decoders on the best checkpoint:
1. **Greedy** (argmax at each step)
2. **Sampled** (temp=0.75, top_p=0.9, current)

Measure:
- Complete-sentence rate (% of sentences ending in `.!?`)
- Word count
- Dangling-function-word rate
- Coherence (parse dependency completeness with spaCy)

**Expected result:**
- If greedy >> sampled: problem is sampling strategy → Phase 0.5 (just switch to greedy)
- If greedy ≈ sampled: problem is model weights / objective → Proceed to Phase 1

**Status:** Phase 0 cell added to notebook (cell index 0, greedy vs. sampled comparison).

### **Phase 1: Fix the prose objective (training, ~4–8 hours on A100)**
Current loss (paragraph-level, no boundary notion):
```python
loss = .25*matched.mean() + .50*ending.mean() + .25*ending[:, :32].mean() + .25*F.softplus(-margin).mean()
```

New loss (add **sentence-level contrastive term**):
```python
# Keep paragraph-level term for stability
para_loss = .25*matched.mean() + .25*ending.mean()

# Add sentence-level discrimination: given a clause, pick the right next sentence
# from K=8 real candidates vs. foils from other paragraphs
sent_loss = multi_candidate_ce(logits, sentence_anchors, sentence_foils)  # e.g. K=8

# Combine
loss = 0.6*para_loss + 0.4*sent_loss
```

This directly supervises "which of these candidate sentences completes this clause" at boundary-aware granularity.

**Data:** Use existing CoQA stories + add a small set of WikiText/Project Gutenberg paragraphs with sentence boundaries marked.

**Selection metric:** Best checkpoint by `matched_bpb` (paragraph discrimination) + new **`sentence_completion_accuracy`** (% of multi-candidate discriminations correct).

**Expected outcome:** Prose BPB remains ~1.47, but held-out prose generations gain syntactic structure (fewer dangling ends, more complete sentences).

### **Phase 2: Raise target length & add long-form corpus (training, ~6–12 hours)**
1. Remove/raise the 160-byte cap for prose tasks (keep QA/chat capped — extractive answers should be short)
2. Build `prose_long_deep` with 800–2000 B targets from:
   - OpenWebText paragraphs (via `datasets` library, real uncurated web prose)
   - Project Gutenberg narrative samples (clean, syntactically rich)
3. Add to training schedule at ~20% frequency (replace some existing `prose_long` examples)

**Expected outcome:** Model learns to compose beyond 160 bytes without padding; generations reach 200–300 bytes while maintaining coherence.

### **Phase 3: Context & metrics (training, ~4–8 hours)**
1. Raise training context from 512 B to 1024 B (affects pretraining; re-profile performance)
2. Add held-out **sentence-level coherence score**: parse each generation with spaCy; count ROOT arcs, complete dependencies, dangling arcs
3. Use coherence score + BPB for checkpoint selection

**Expected outcome:** Model's dependency binding improves; 1024 B context allows full clause structures.

---

## Decision Points

### **Should we start with Phase 0?**
✅ **Yes.** If greedy decoding alone fixes 70%+ of the problem, we can ship a quick inference-only fix while designing Phase 1. If it doesn't, Phase 0 validates that the objective needs changing.

### **Which long-form corpus: WikiText, OpenWebText, Gutenberg, or mix?**
✅ **OpenWebText** (chosen).
- Real uncurated prose (more varied than WikiText)
- Already integrated in `datasets` library
- ~200K documents, easy dedup
- Drawback: some noise, but that's reality

Could add Project Gutenberg later if needed (syntactic diversity).

### **Should we also fix tokenization?**
❌ **Not yet.** Byte tokenizer is real but secondary. Fix the objective first (saves a full retrain). Can revisit if post-Phase-3 prose is still hitting byte-level artefacts (unlikely if loss is better).

### **What about the 10M parameter target?**
✅ **Keep it.** Model is **not parameter-limited** (Diagnosis 1). Increasing size won't fix an objective that doesn't recognize sentence boundaries.

---

## Success Metrics

By end of Phase 3:
- **Test prose BPB:** ~1.45–1.50 (should not increase; might decrease slightly as model uses capacity differently)
- **Held-out prose exact-match:** +50–70% (from current ~0% to 50–70% of generations being full sentences)
- **Held-out prose spaCy parse completeness:** >80% of tokens in a complete parse (vs. current ~20–30%)
- **Dangling-function-word rate:** <5% (vs. current ~10–15%)
- **Greedy vs. sampled:** greedy now better (Phase 0 decision point)

---

## Timeline & Effort
- Phase 0 (falsify): ~2 hours (inference only, runs in notebook)
- Phase 1 (objective): ~6 hours (code change + 4–6 hr training)
- Phase 2 (long-form data): ~8 hours (corpus prep + 6–8 hr training)
- Phase 3 (context & metrics): ~6 hours (config change + 4–6 hr training)
- **Total:** ~22 hours (mostly training; actual coding ~4–5 hours)

Assuming A100 GPU: start Phase 0 this session, Phase 1–3 over next 1–2 days.

---

## Code Changes Required

### Phase 0 (notebook cell)
- Compare greedy vs. sampled decode on held-out prompts ✅ **Done** (cell 0)

### Phase 1 (prose_backward + training data)
- Modify `run.ipynb` cell 10 (`prose_backward`): add `sent_loss` term
- Modify `run.ipynb` cell 8: prepare sentence-level targets from corpus anchors
- Modify selection loss: `selection_loss = 0.7*continuation['matched_bpb'] + 0.3*(1 - sentence_completion_acc)`

### Phase 2
- Download OpenWebText via `datasets` → `corpus['owt']`
- Split into paragraphs (using simple heuristic: `\n\n`-delimited)
- Add to training schedule: `TRAIN_STAGES = [..., 'owt_long', ...]`

### Phase 3
- Change `CONTEXT = 1024` (from 512)
- Add `coherence_score()` function using spaCy
- Modify evaluation loop to compute coherence

All changes are **localized and reversible** — no model-architecture changes, no tokenizer changes, only objective-function and data-prep changes.

