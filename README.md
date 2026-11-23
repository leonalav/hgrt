# HGRT-10M learnability pilot

A completed CPU check trained C1-S8 for 458 updates (234,496 bytes). On 16,896 held-out bytes its loss fell from 8.044 to **1.979 bits per byte**, compared with 4.654 for the training unigram. Disabling the tree gave 1.97867 BPB and shuffling it gave 1.97860, compared with 1.97863 intact. The architecture learned local text statistics. Tree-content attribution remains unresolved on this endpoint. The A100 40 GB runtime and CUDA kernels still require an on-device run.

The default experiment gives a scratch eight-slot HGRT model **20 minutes on one A100 40 GB** to learn next-byte prediction on real text. It measures held-out bits per byte (BPB) against initialization and a smoothed unigram fit on training bytes. The separate recall command implements the revised, single-seed architecture screen.

A separate, time-reduced CPU A0 recall control reached 4.85% on the fixed 2,000 easy examples after 147 updates and 1,176 supervised query targets. That run stopped before the hard and architectural comparisons. Associative-recall learnability remains unestablished.

The [4,096-token batch follow-up](runs/cpu-controls-4096/report.json) scored 118/2,000 (5.90%) after 112 updates and 7,168 supervised queries. A0 chose the value of a different presented key 351 times and a value absent from the example 1,531 times. Among the 469 predictions that matched any presented value, 25.2% matched the queried key, close to the 25% reference for choosing uniformly among four values. Gate 0A remains unmet at this CPU training budget.

The [nominal easy warm-up](runs/cpu-controls-nominal/report.json) completed 488 updates, 1,998,848 tokens and 31,232 query targets with the same seed and 4,096-token batch. A0 reached **21.05%** (421/2,000) on the same examples. It predicted a value present in the example 87.1% of the time. Among those 1,742 predictions it matched the queried key 24.2% of the time. This single-seed run demonstrates learning which values appeared, with key-specific retrieval still near the four-value guessing reference. It stopped at Gate 0A after 12 minutes 29 seconds on CPU.

## Run on the A100

For the next milestone, paste the complete [portable Python cell](colab_controls.py) into an A100 Colab notebook. It includes the experiment source, verifies a BF16 CUDA matrix multiply, prepares the dependencies in a temporary environment, and tries the fused gated-delta kernel. It writes reports and checkpoints to `My Drive/HGRT-10M/runs` when the Drive mount succeeds. Setup time is additional to the 25-minute experiment budget.

From a project checkout, run the controls with the following command.

```bash
python run.py controls --device cuda --minutes 25 --out runs/a100-controls
```

The first training benchmark measures C1-S8 at 2,048 tokens and records the gradients of the tree parameters under query-only loss. The runner then spends its training quota on A0, up to the registered 2M-token easy warm-up. It proceeds to the long curriculum after 95% easy accuracy. Both endpoints use the fixed screening examples. The report gives counts of exact answers, answers belonging to another key in the example, and answers absent from the example. The recall head projects only the supervised positions to vocabulary logits. Text modeling still predicts every position.

Use a CUDA-enabled PyTorch environment with BF16 support. For the accelerated gated-delta kernel, install the CUDA extra documented by the FLA authors. The runner also has a pure PyTorch backend. [FLA installation](https://github.com/fla-org/flash-linear-attention#installation)

```bash
cd hgrt10m
python -m pip install -r requirements.txt 'flash-linear-attention[cuda]'
python run.py text --backend fla --minutes 20 --out runs/text-42
```

To use an existing CUDA PyTorch installation with NumPy and skip FLA installation, run `python run.py text --backend torch --minutes 20 --out runs/text-torch`. Use a fresh output directory per invocation. The clock covers profiling, kernel warm-up, validation and training. Setup and process startup are additional. A single in-flight CUDA call, JIT compilation or checkpoint write can cross the deadline. Throughput and peak allocated GPU bytes appear in `report.json`. For profiling only, use `python run.py profile --out runs/profile`.

Every invocation first times **C1-S8 at 2,048 tokens, including backward and AdamW**. It verifies a gradient through the learned merge. Three synchronized updates after warm-up supply the median rate. Text training then uses 256-token contexts, even on a fast GPU. Below 5,000 tokens/s at 2K, the report flags the throughput risk. The quota projection in recall mode uses the measured rates for each architecture family and curriculum length. To reduce peak GPU allocation, use `--batch-tokens 4096` and start a fresh run. The batch is a whole number of sequences, with at least one sequence per step.

## Text endpoint

By default the corpus is Python's installed help-topic prose. The loader deduplicates identical topics, partitions whole topics by content hash into train/validation/reserved test groups (approximately 80/10/10 by topic count), and records SHA-256 hashes. The corpus depends on the Python version. Fixed validation windows cover disjoint target bytes. To use your own corpus, add `--text-file /path/to/corpus.txt`. This option splits the file into contiguous byte ranges before drawing windows. The last 10% stays reserved. Use a representative, deduplicated corpus for a more informative language-model result.

A0 gets up to a quarter of the time remaining after profiling and engineering checks. Once its validation BPB beats both initialization and the training unigram, C1-S8 trains from scratch using the remaining time. The two arms draw from the same seeded stream of training windows. They receive different numbers of updates, so their final scores serve as learnability checks. Read C1's `initial_bpb`, `final_bpb`, `off_bpb`, `shuffle_bpb`, update count and training bytes together. Lower BPB is better. This pilot's text-learning criterion was chosen for fast feedback. General language competence requires a broader corpus and evaluation.

The run saves trained weights, a JSON report and a JSONL learning curve. The sampler accepts a UTF-8 prompt.

```bash
python sample.py runs/text-42/C1.pt 'The result of ' --device cuda --bytes 160
```

## Recall screen

```bash
python run.py recall --minutes 120 --backend fla --examples 2000 --out runs/recall-42
```

The generator freezes the matched screening sets and records their hashes before the first optimizer step. There are 32 possible keys and 32 possible answers. Every example assigns a uniformly randomized set of distinct values to distinct keys. Facts occupy a jittered cluster in random key order. Distractors come from a disjoint alphabet. Later queries independently permute the keys. Each key is queried exactly once. Answer targets are external labels at query-key positions and never appear as teacher-forced query responses. Training scores every query using cross entropy. Evaluation scores exactly one prespecified first query per sequence over the entire vocabulary. A random guess from the value vocabulary has expected accuracy 1/32. Uniformly choosing among the 16 presented values would give 1/16 at load 16.

| Endpoint | Length | Pairs | Scored query index (zero-based) | Completed chunks | Popcount |
|---|---:|---:|---:|---:|---:|
| Easy | 256 | 4 | 224 | 3 | 2 |
| Short-load control | 256 | 16 | 224 | 3 | 2 |
| Carry pair | 1,024 / 1,088 | 16 | 992 / 1,056 | 15 / 16 | 4 / 1 |
| Primary / carry | 2,048 / 2,112 | 16 | 2,016 / 2,080 | 31 / 32 | 5 / 1 |

Both sides of a carry pair share the fact identities, values, query order and exact value-to-query distances. The generator translates the facts and queries by 64 tokens. The scored fact lies 1,024–1,984 tokens back on the hard pair, a feasible subset of the 1K–2K stratum. The 15/16 pair uses 704–928. Filler tokens vary. The held-out screen is reused across models and seeds, with a separate seed reserved for the eventual independent test.

### Curriculum and scheduling

The nominal trajectory uses 2M tokens at 256/load 4, then 0.5M at 256/load 4, 0.5M at 512/load 8, 1.5M at 1,024/load 8 and 5.5M at 2,048/load 16. This moves 1.5M continuation tokens into the long stage. Three paid warm-ups plus six continuations total **54M training tokens** before pilots and rounding down to whole updates. The default effective batch is 8,192 tokens. The 54M quota would itself require 45,000 tokens/s to fit in 20 minutes before any overhead.

The runner pilots AdamW learning rates 0.0003, 0.001 and 0.002 independently for A0, B1 and C1 using a separate validation stream. Each candidate uses the same initialization and pilot data within its family. C1 pilots use the B1 warm weights. Warm C1-S4/S8/S16 branches all load every B1 backbone parameter. B1 and the warm branches reset AdamW moments at the fork. Warm-branch counters include inherited training. Scratch C1-S8 follows the whole curriculum independently. The S4 arm is retained.

The scheduler reserves time for evaluation and pilots. If necessary it first trims stage 1 to a floor of 128K tokens, then scales the continuation quotas together across all matched arms. If even a minimal matched continuation cannot fit, it prioritizes a reduced A0-only learnability check. Timed-out and incomplete runs remain inconclusive. The full screen attempts a second seed for B1 and C1-S8 only when measured rates leave room for both complete trajectories and evaluation. This repeat is descriptive. The first seed determines the registered screening gates.

### Screening criteria

| Check | Criterion |
|---|---|
| 0A | A0 or B1 achieves 95% easy accuracy after warm-up |
| 0B | Continued A0 achieves 80% on primary31 |
| B1 short-load check | Continued B1 achieves 80% at short load 16 |
| 1 | Warm S8 beats B1 by at least 5 percentage points at **both** 31 and 32 completed chunks |
| 2 | Each adjacent slot increase loses at most 2 points, and S16 beats S4 by at least 3 points at primary31 |
| 3 | Both tree removal and cross-example tree shuffle cost warm S8 at least 3 points at primary31 |
| 4 | Scratch S8 achieves 95% easy after warm-up and passes the same two 3-point memory checks |
| 5 | Future-token invariance, streaming equivalence, writer/merge gradients and measured state-byte accounting pass |

The B1 short-load threshold and the 3-point shuffled-tree threshold operationalize the qualitative requirements in the proposal. The runner records intact, disabled and shuffled accuracy at both 31 and 32 for both S8 models. Shuffling rotates the entire causal history among independent batch examples with no fixed points. Compare the absolute shuffled accuracy with the chance references as well as the paired drop. A drop after zeroing alone is ambiguous. If accuracy stays high with foreign memory, attribution to tree content remains unresolved.

The report includes paired bootstrap intervals over independent examples and per-example correctness arrays for reanalysis. These intervals describe evaluation sampling noise under the trained seed. Seed variance requires independent training runs. A screening pass motivates the preregistered three-seed comparison with published Log-Linear Attention (C0). A single failed seed with falling loss or a capacity trend remains inconclusive. Carry-specific drops indicate a merge-compression hypothesis to investigate. Warm-only success is labeled initialization-dependent.

## Architecture and verification

All models use width 256, eight blocks, tied input/output embeddings, four 64-dimensional heads and SwiGLU FFNs. A0 uses full causal attention throughout. B1 and C1 use six gated delta-rule mixers with short causal convolutions and two exact 64-token sliding-attention layers, at layers 4 and 8. C1 writes layer 4 outputs into 128-dimensional learned slots and reads completed prefix-tree nodes immediately before layer 8. Binary carries merge aligned slots with a learned projection of their mean. The shared content projection is tied to its readout transpose. A slot sweep changes only the number of learned slot queries. No history is detached. During training the implementation constructs completed nodes by level and selects the prefix forest for each chunk in parallel. Streaming commits one aligned chunk at a time.

| Model | Trainable parameters |
|---|---:|
| A0 | 10,006,784 |
| B1 | 9,963,800 |
| C1-S4 | 10,013,465 |
| C1-S8 | 10,013,977 |
| C1-S16 | 10,015,001 |

Actual storage measurements include the live recurrent matrices, causal-convolution histories, attention K/V and occupied tree nodes. Peak training allocation is reported separately. The test suite checks the optimized recurrence against the single-token equation including gradients, checks the full model at power-of-two carries, and verifies that the disabled warm branch exactly reproduces B1 in FP32.

```bash
OMP_NUM_THREADS=2 python -m unittest -v test_hgrt
```

## Literature check

At 32 pairs, Table 2 of *Adaptive Memory Decay for Log-Linear Attention* lists means of 2.9% and 57.9% for baseline and MLP-softplus. Section 4.2 gives 60.9% ± 46.7% and 99.6% ± 0.1%. Footnote 3 assigns the prose values to a separate run with a different seed configuration. Appendix A.3 reports 5–15 minutes per MQAR seed on an A100 SXM4 80 GB. The pilot measures throughput directly on the run host. [Table 2 and §4.2](https://arxiv.org/html/2605.06946v1#S4.SS2)

The delta update follows the gated delta rule. The CUDA path uses FLA's chunk operator. The reference path solves the same recurrence blockwise in FP32. [Gated Delta Networks](https://arxiv.org/abs/2412.06464) · [FLA chunk implementation](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py)
