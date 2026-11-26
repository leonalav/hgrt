# Analysis Deliverables: Expected Return Files

## Already Delivered

### 1. **DIAGNOSIS_AND_PLAN.md** ✅
**File:** `A:/hgrt10m-full/hgrt10m/DIAGNOSIS_AND_PLAN.md`

**Contents:**
- **Diagnosis 1–6:** Complete forensic analysis of why prose is incoherent
  - Root causes: training objective + dialogue target-length cap
  - Evidence: test BPB, tree ablation, fact-switch accuracy, QA exact-match rates
  - Why NOT capacity-limited, NOT decoding-limited

- **Ranked interventions:** Table of 7 interventions by expected gain (25–40% down to 5–10%)
- **Phases 0–3 detailed:** Specific code changes, data prep, training schedule, success metrics
- **Timeline & effort:** Total 22–30 hours on A100 (mostly training)
- **Decision points:** Which long-form corpus, whether to fix tokenizer first, etc.

**How to use:**
1. Read for full context on root causes
2. Reference for Phase 1–3 implementation (exact code locations to modify)
3. Copy success metrics for validation after each phase completes

---

### 2. **PHASE_STATUS.md** ✅
**File:** `A:/hgrt10m-full/hgrt10m/PHASE_STATUS.md`

**Contents:**
- **Status table:** What's done vs. planned for Phases 0–3
- **Hour breakdown:** Clarifies that all hours are A100 training time (not calendar time)
- **Parallelization notes:** How to overlap phases to reduce total elapsed time
- **Immediate next step:** Run Phase 0 (cell 0 in notebook)

**How to use:**
1. Quick reference for "where are we in the plan?"
2. Time planning: use this to estimate when each phase completes
3. Blocker identification: see which phase depends on which

---

### 3. **run.ipynb** (modified) ✅
**File:** `A:/hgrt10m-full/hgrt10m/run.ipynb`

**Changes:**
- **Cell 0 added:** Phase 0 hypothesis-validation code
  - Greedy vs. sampled decode on best checkpoint
  - Outputs: complete-sentence rates, word counts, dangling-end rates, coherence scores
  - Decision output: "RESULT: Greedy is SUBSTANTIALLY better" vs. "RESULT: Greedy and sampled are SIMILAR"

**Expected outputs when executed:**
```
Checkpoint: kind=C1, slots=8, validation_bpb=X.XXXX

Prompt 0: "The expedition reached the river at dawn. "

  greedy  : XXw X/Xs dangling=FALSE artifact=FALSE | "text..."
  sampled : XXw X/Xs dangling=TRUE  artifact=TRUE  | "text..."

GREEDY aggregate over 4 samples:
  mean words/gen:        XXX.X
  mean complete-sent:    X.XX / X.XX
  dangling-end rate:     XX%
  artifact rate:         XX%

SAMPLED aggregate over 4 samples:
  mean words/gen:        XX.X
  mean complete-sent:    X.XX / X.XX
  dangling-end rate:     XX%
  artifact rate:         XX%

HYPOTHESIS TEST
  Greedy complete-sentence rate:  XX%
  Sampled complete-sentence rate: XX%
  RESULT: [Greedy is SUBSTANTIALLY better / Greedy and sampled are SIMILAR / ...]
```

---

## To Be Generated When You Execute Phase 0

### 4. **Phase 0 Output JSON** (new file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/phase0_results.json` or printed to notebook cell output

**Contents (JSON structure):**
```json
{
  "checkpoint": {
    "path": "runs/best_lm.pt",
    "kind": "C1",
    "validation_bpb": 1.675,
    "steps": 5908
  },
  "test_prompts": 4,
  "greedy": {
    "mean_words": 157.5,
    "mean_sentences": 2.3,
    "mean_complete": 1.8,
    "complete_rate": 0.78,
    "dangling_end_rate": 0.08,
    "artifact_rate": 0.02,
    "sample_1": {
      "prompt": "The expedition...",
      "text": "...",
      "words": 160,
      "complete": 2,
      "dangling": false
    }
  },
  "sampled": {
    "mean_words": 142.0,
    "mean_sentences": 3.1,
    "mean_complete": 1.2,
    "complete_rate": 0.39,
    "dangling_end_rate": 0.22,
    "artifact_rate": 0.08,
    "sample_1": { ... }
  },
  "hypothesis_decision": "GREEDY >> SAMPLED" | "GREEDY ≈ SAMPLED" | "SAMPLED > GREEDY",
  "recommendation": "Switch to greedy decoding (inference fix)" | "Proceed to Phase 1 (retrain with new objective)",
  "timestamp": "2026-11-26T11:22:00Z"
}
```

**How to use:**
- Decision gate for Phases 1–3
- Baseline metrics for future comparisons
- Save for final report

---

## To Be Generated After Phase 1 (If Executed)

### 5. **Phase 1 Training Log** (new file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/runs/phase1_training.jsonl` (or similar)

**Contents (line-delimited JSON, one per checkpoint):**
```json
{"step": 100, "loss": 0.845, "matched_bpb": 1.71, "sent_completion_acc": 0.72, "validation_time_sec": 42}
{"step": 200, "loss": 0.812, "matched_bpb": 1.68, "sent_completion_acc": 0.76, "validation_time_sec": 41}
...
```

**How to use:**
- Plot loss curves to verify objective is working
- Identify best checkpoint by `sent_completion_acc`
- Compare to Phase 0 baseline

---

### 6. **Phase 1 Checkpoint** (new model file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/runs/phase1_best_lm.pt`

**Contents:**
- Model state_dict with new sentence-aware weights
- Optimizer state
- Config, step, bytes_seen, metadata

**How to use:**
- Baseline for Phase 2 fine-tuning (or standalone inference)
- For Phase 0 comparison: generate prose with Phase 1 model vs. original

---

### 7. **Phase 1 Held-Out Prose Samples** (new file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/phase1_prose_samples.json`

**Contents:**
```json
{
  "samples": [
    {
      "prompt": "The expedition reached the river at dawn. ",
      "original_model": "The properties of plants will also attempt to lead to the invertebrate , the results of all planets .",
      "phase1_model": "They prepared their supplies for the journey ahead. The camp was set up quickly before the sun rose higher.",
      "metrics": {
        "complete_sentences": 2,
        "dangling_end": false,
        "dependency_coverage": 0.87
      }
    }
  ],
  "aggregate_metrics": {
    "original_complete_rate": 0.39,
    "phase1_complete_rate": 0.78,
    "improvement": "+100%"
  }
}
```

**How to use:**
- Visual comparison: before/after prose
- Validation that Phase 1 objective is working
- For final report: show improvement

---

## To Be Generated After Phase 2 (If Executed)

### 8. **Phase 2 Corpus Stats** (new file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/phase2_corpus_stats.json`

**Contents:**
```json
{
  "openwebtext": {
    "total_documents": 189651,
    "total_paragraphs": 2847309,
    "avg_paragraph_bytes": 847,
    "dedup_collision_rate": 0.08,
    "byte_range": [100, 3200],
    "example_target": "Long paragraph with multi-sentence structure..."
  },
  "added_to_training": {
    "task_name": "owt_long",
    "examples": 150000,
    "allocation_percent": 20,
    "avg_target_bytes": 920,
    "train_steps_per_epoch": 2341
  }
}
```

**How to use:**
- Confirm corpus prep was successful
- Understand data distribution for Phase 2 training

---

### 9. **Phase 2 Training Log & Checkpoint** (new files)
**Expected files:**
- `A:/hgrt10m-full/hgrt10m/runs/phase2_training.jsonl`
- `A:/hgrt10m-full/hgrt10m/runs/phase2_best_lm.pt`

**Contents:** Similar to Phase 1 but with longer-form targets.

---

## To Be Generated After Phase 3 (If Executed)

### 10. **Phase 3 Coherence Metrics** (new file)
**Expected file:** `A:/hgrt10m-full/hgrt10m/phase3_coherence_scores.json`

**Contents:**
```json
{
  "original_model": {
    "mean_root_arc_coverage": 0.34,
    "mean_complete_dependency_rate": 0.28,
    "mean_dangling_arc_rate": 0.12,
    "complete_sentence_rate": 0.39
  },
  "phase1_model": {
    "mean_root_arc_coverage": 0.68,
    "mean_complete_dependency_rate": 0.65,
    "mean_dangling_arc_rate": 0.05,
    "complete_sentence_rate": 0.78
  },
  "phase2_model": {
    "mean_root_arc_coverage": 0.81,
    "mean_complete_dependency_rate": 0.79,
    "mean_dangling_arc_rate": 0.02,
    "complete_sentence_rate": 0.92
  },
  "phase3_model": {
    "mean_root_arc_coverage": 0.87,
    "mean_complete_dependency_rate": 0.85,
    "mean_dangling_arc_rate": 0.01,
    "complete_sentence_rate": 0.96
  }
}
```

**How to use:**
- Final metric for prose quality improvement
- For report: show progression across phases

---

### 11. **Phase 3 Training Log & Checkpoint** (new files)
**Expected files:**
- `A:/hgrt10m-full/hgrt10m/runs/phase3_training.jsonl`
- `A:/hgrt10m-full/hgrt10m/runs/phase3_best_lm.pt`

---

## Final Deliverable: Analysis Report (After All Phases)

### 12. **FINAL_ANALYSIS_REPORT.md** (to be written)
**File:** `A:/hgrt10m-full/hgrt10m/FINAL_ANALYSIS_REPORT.md`

**Contents (sketch):**
```markdown
# HGRT-10M: Final Analysis & Outcomes

## Executive Summary
- Root cause: short-form training objective + 160-byte target cap
- Phase 0 result: [greedy >> sampled / greedy ≈ sampled / ...]
- Final prose quality: XX% improvement in complete-sentence rate

## Phase 0 Results
- Greedy complete rate: XX%
- Sampled complete rate: XX%
- Decision: [inference fix sufficient / proceed to full retrain]

## Phase 1–3 Results (if executed)
- Table: original vs. phase1 vs. phase2 vs. phase3 metrics
- Prose samples: before/after for each phase
- Learning curves: loss, BPB, coherence scores

## Success Metrics Achieved
- Test prose BPB: X.XX (target: 1.45–1.50)
- Complete-sentence rate: XX% (target: 50–70%)
- Parse completeness: XX% (target: >80%)
- Dangling-end rate: X% (target: <5%)

## Recommendations for Future Work
- Tokenizer upgrade (defer until after Phases 1–3)
- Parsing head (can add orthogonally)
- Larger model size (not capacity-bottlenecked currently)
```

---

## Summary: Expected Deliverables by File

| # | File | When | Type | Purpose |
|---|---|---|---|---|
| 1 | `DIAGNOSIS_AND_PLAN.md` | ✅ Done | Markdown | Full analysis + implementation plan |
| 2 | `PHASE_STATUS.md` | ✅ Done | Markdown | Status tracking + hours breakdown |
| 3 | `run.ipynb` (cell 0) | ✅ Done | Notebook | Phase 0 code (ready to run) |
| 4 | `phase0_results.json` | After Phase 0 | JSON | Decision gate output |
| 5 | `runs/phase1_training.jsonl` | After Phase 1 | JSONL | Training curves |
| 6 | `runs/phase1_best_lm.pt` | After Phase 1 | PyTorch | Model checkpoint |
| 7 | `phase1_prose_samples.json` | After Phase 1 | JSON | Before/after comparison |
| 8 | `phase2_corpus_stats.json` | After Phase 2 | JSON | OpenWebText corpus info |
| 9 | `runs/phase2_training.jsonl`, `.pt` | After Phase 2 | JSONL, PyTorch | Training curves + checkpoint |
| 10 | `phase3_coherence_scores.json` | After Phase 3 | JSON | spaCy parse metrics |
| 11 | `runs/phase3_training.jsonl`, `.pt` | After Phase 3 | JSONL, PyTorch | Final training + checkpoint |
| 12 | `FINAL_ANALYSIS_REPORT.md` | After Phase 3 | Markdown | Executive summary + outcomes |

---

## How To Use These Deliverables

**Immediately (Phase 0):**
1. Read `DIAGNOSIS_AND_PLAN.md` for full context
2. Skim `PHASE_STATUS.md` for status check
3. Execute `run.ipynb` cell 0
4. Check `phase0_results.json` for decision: greedy fix or Phase 1 retrain?

**After Phase 1 (if executed):**
1. Review `phase1_training.jsonl` for loss curves
2. Compare `phase1_prose_samples.json` to original
3. Use `phase1_best_lm.pt` for Phase 2 or standalone inference

**After Phase 3:**
1. Read `FINAL_ANALYSIS_REPORT.md` for executive summary
2. Use coherence metrics in `phase3_coherence_scores.json` for final report
3. Use `phase3_best_lm.pt` as production model

