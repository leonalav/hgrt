"""Sync embedded module cells (model.py, dialogue.py, fact_switch.py) and upgraded pipeline cells into run.ipynb."""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from phase0_cell import PHASE0_CODE
from resume_block import RESUME_BLOCK
from phase3_probe_cell import PHASE3_PROBE_CODE
import phase1_notebook
import phase2_notebook


def to_lines(text: str) -> list[str]:
    text = text.lstrip('\n')
    lines = text.splitlines(keepends=True)
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
    return lines


def make_cell(cell_type: str, cell_id: str, source: str) -> dict:
    c = dict(cell_type=cell_type, id=cell_id, metadata={}, source=to_lines(source))
    if cell_type == 'code':
        c['execution_count'] = None
        c['outputs'] = []
    return c


INTRO_MD = """
# Train HGRT-10M for fact-to-answer recall, sustained prose and conversation

In Colab, open this notebook directly from GitHub, choose **Runtime → Change runtime type → A100 GPU**, and click **Runtime → Run all**. Every required module (`model.py`, `dialogue.py`, `fact_switch.py`) is embedded verbatim inside this notebook so it runs standalone with no extra files on disk. By default the notebook resumes the newest matching WikiText run or `best_lm.pt` it can find in Drive or the local `runs/` folder; set `AUTO_RESUME = False` for a fresh start, `RESUME_FROM` for a specific `latest.pt`, or `INITIAL_WEIGHTS` to start from saved `C1-S8` weights.

**Research pipeline (in priority order):**
1. **WikiText-103 prose pretraining / resume (`TRAIN_MINUTES`)** — pinned raw Parquet shards ([Dataset card](https://huggingface.co/datasets/Salesforce/wikitext) · [Merity et al.](https://arxiv.org/abs/1609.07843)), with matched vs. mismatched paragraph-continuation scoring.
2. **Counterfactual fact-to-answer learning (`FACT_SWITCH_MINUTES`)** — two-stage training directly targeting fact-to-answer switching on strictly disjoint OOV entities and novel nonce words: Stage 1 (`plain_nll`) locks in multi-byte OOV copying; Stage 2 (`contrastive` on 2×2 value-swap + question-swap quads) cancels both static word priors and Sentence-1 positional shortcuts. Checkpoints are selected on a **disjoint** selection suite and reported on a **fresh-seed** report suite, using the first-byte decision accuracy plus counterfactual pair accuracy and greedy exact match rather than average BPB, and the causal Tree ablations (`on` / `off` / `permute` / `counterfactual_tree_swap`) are reported alongside them. Set `RUN_ARCHITECTURE_SHOWDOWN = True` on an A100 to run a matched 4-arm head-to-head (`C1 L7-only`, `C1 L3+L7`, native `C1 rw=64` Tree-only, and same-size 10M Transformer `A0`).
3. **Rebalanced grounded QA, sustained prose & service chat (`DIALOGUE_MINUTES`)** — separates grounded QA (`Passage:/Question:/Answer:`) from service chat (`User:/Assistant:`) to stop domain leakage, auto-tunes CUDA batch size even when `TRAIN_MINUTES = 0`, rebalances QA + counterfactual quads + multi-sentence `prose_long` continuations, and preserves counterfactual fact-switching during checkpoint selection.
"""

SETTINGS_CODE = """
#@title Run settings
from pathlib import Path

# -- Version guard ------------------------------------------------------------
# Colab autosaves the notebook to Drive and can restore a stale copy over your
# edits. Detect that here so you always know which version is running.
def _notebook_sha():
    import hashlib, json as _json, os as _os
    try:
        nb = globals().get('__nb__')
        if nb is not None:
            body = _json.dumps(
                [{'cell_type': c.cell_type, 'id': c.get('id'), 'source': c.source}
                 for c in nb.cells], sort_keys=True).encode('utf8')
            return hashlib.sha256(body).hexdigest()[:12]
        path = _os.path.join(_os.getcwd(), 'run.ipynb')
        if _os.path.exists(path):
            return hashlib.sha256(open(path, 'rb').read()).hexdigest()[:12]
    except Exception:
        pass
    return None

_NOTEBOOK_SHA = _notebook_sha()
print(f'Running notebook content id: {_NOTEBOOK_SHA or "(unknown)"}')
print('If this differs from the version you pushed, re-open the notebook from GitHub')
print('(File -> Open notebook -> GitHub -> leonalav/hgrt -> run.ipynb), NOT from Drive.')

TRAIN_MINUTES = 20             # Additional WikiText prose training minutes per invocation (0 skips)
CONTEXT = 512                  # UTF-8 bytes per prose sequence
BATCH_BYTES = 8192             # Starting batch budget, in input bytes
AUTO_BATCH = True              # Benchmark larger CUDA batches separately for each training phase
MAX_BATCH_BYTES = 131072       # Upper budget for the automatic search
GPU_MEMORY_FRACTION = .85      # Fraction of available GPU memory the search may use
LR = 0.0005
SEED = 42
DEVICE = 'cuda'
TRY_FUSED = True               # Attempt the optional FLA gated-delta kernel
SAVE_TO_DRIVE = True
AUTO_RESUME = True             # Continue the newest matching prose run when available
INITIAL_WEIGHTS = ''           # Optional best_lm.pt or other C1-S8 weights for a new optimizer
RESUME_FROM = ''               # Optional latest.pt from this notebook
LOCAL_PARQUET_DIR = ''         # Optional offline directory with the four official shard names
OUTPUT_ROOT = ''               # Optional explicit output directory parent
MAX_UPDATES = None             # Optional additional update limit for a short diagnostic
VALID_WINDOWS = 128
TEST_WINDOWS = 256
PROBE_PAIRS = 64               # Per task, each with two matched variants
RUN_REASONING = False          # Optional original probes, after prose evaluation
TASK_MINUTES = 0               # Optional additional practice when RUN_REASONING is True

# Priority #1: Counterfactual fact-to-answer learning & optional A100 architecture showdown
FACT_SWITCH_MINUTES = 6        # Two-stage 2x2 quad fact-switching stage (0 evaluates zero-shot only)
# The fact-switch stage starts from the CONVERGED prose checkpoint with a FRESH
# optimizer, so it must use a prose-safe rate. Measured on the run-2 checkpoint
# (40 steps of the contrastive objective, wikitext validation BPB):
#
#     LR=5e-4   BPB 1.4786 -> 2.9383  (+1.4598)  <-- the old value: destroys the model
#     LR=1e-4   BPB 1.4786 -> 1.5240  (+0.0455)
#     LR=5e-5   BPB 1.4786 -> 1.4932  (+0.0146)
#     LR=2e-5   BPB 1.4786 -> 1.4810  (+0.0024)  <-- prose preserved
#     LR=5e-6   BPB 1.4786 -> 1.4781  (-0.0004)
#
# And the task is still LEARNED at 2e-5: the contrastive loss fell 8.18 -> 7.11
# (-1.07) over the same 40 steps, with the wikitext BPB essentially unchanged.
# At 5e-4 the loss also falls, but only because the model is being destroyed.
FACT_SWITCH_LR = 0.00002
FACT_SWITCH_EVAL_PAIRS = 16    # Held-out OOV pairs per diagnostic split
RUN_ARCHITECTURE_SHOWDOWN = False  # Optional A100 showdown: C1(L7), C1(L3+L7), C1(rw=64 Tree), A0(10M)
SHOWDOWN_STEPS = 1200          # Optimizer steps per showdown arm (batch=4 quads/step on CUDA, 1 on CPU)

# Priority #2: Rebalanced grounded QA, sustained prose and conversation
DIALOGUE_MINUTES = 12          # Additional time for QA, sustained prose, chat and WikiText replay
DIALOGUE_CONTEXT = 1536        # Complete passage + turn history + target, in UTF-8 bytes
DIALOGUE_LR = .00002
DIALOGUE_EVAL_ROWS = 32        # Fixed rows per task and split
QA_PROMPT_STYLE = 'qa'         # 'qa' separates Passage/Question/Answer from User/Assistant chat; 'role' is legacy
DIALOGUE_WIDEN_L3 = False      # Optional: widen Layer 3 as well as Layer 7 to DIALOGUE_CONTEXT (0 new params)
LOCAL_DIALOGUE_DIR = ''        # Optional offline directory with both CoQA files and self-dialogs.json
SAMPLE_BYTES = 360             # Room for several sentences from the fixed river prompt
PROSE_COMPARISON_PARAGRAPHS = 3  # Validation paragraphs for paired free generation (0 skips)
SAMPLE_TEMPERATURE = .75
SAMPLE_TOP_P = .90             # 1.0 samples the full distribution

assert DEVICE in ('cpu', 'cuda') and TRAIN_MINUTES >= 0 and LR > 0
assert CONTEXT >= 128 and CONTEXT % 64 == 0 and BATCH_BYTES >= max(2048, 2*CONTEXT)
assert not AUTO_BATCH or MAX_BATCH_BYTES >= BATCH_BYTES
assert 0 < GPU_MEMORY_FRACTION <= 1
assert VALID_WINDOWS >= 2 and TEST_WINDOWS >= 2 and PROBE_PAIRS > 0
assert TASK_MINUTES >= 0 and SAMPLE_BYTES >= 0 and SAMPLE_TEMPERATURE > 0 and 0 < SAMPLE_TOP_P <= 1
assert PROSE_COMPARISON_PARAGRAPHS >= 0
assert FACT_SWITCH_MINUTES >= 0 and FACT_SWITCH_LR > 0 and FACT_SWITCH_EVAL_PAIRS >= 2
assert SHOWDOWN_STEPS >= 10
assert DIALOGUE_MINUTES >= 0 and DIALOGUE_LR > 0 and DIALOGUE_EVAL_ROWS >= 1
assert DIALOGUE_CONTEXT >= 512 and DIALOGUE_CONTEXT % 64 == 0
assert QA_PROMPT_STYLE in ('qa', 'role')
assert MAX_UPDATES is None or MAX_UPDATES >= 1
assert not (INITIAL_WEIGHTS and RESUME_FROM)

# --- Phase 1: sentence-boundary-aware prose objective -------------------------
# The paragraph-level objective (`continuation_nll`) has no notion of where a
# sentence ends, so a model can minimise it by padding a 128-byte span with
# locally frequent function words. These settings drive the sentence-level
# discrimination term that supervises sentence completion directly.
SENTENCE_CANDIDATES = 8      # 1 true continuation + 7 foils per group
SENTENCE_LEFT_BYTES = 192     # context bytes before the sentence boundary
SENTENCE_RIGHT_BYTES = 96     # bytes of the candidate sentence
SENTENCE_LOSS_WEIGHT = 0.4    # weight of the sentence term
PARAGRAPH_LOSS_WEIGHT = 0.6   # weight of the paragraph term
SELECTION_SENTENCE_WEIGHT = 0.30   # weight of sentence accuracy in selection
SENTENCE_EVAL_GROUPS = 256    # groups used to measure sentence accuracy
WARMUP_STEPS = 1000           # LR ramp length when training from INITIAL_WEIGHTS

# --- Phase 2: long-form prose data and target budgets ------------------------
# Diagnosis 4: every supervised target was capped at 160 bytes, so the model
# learned short clipped spans padded with high-frequency function words.
# PROSE targets now get a much larger budget; QA and chat stay short because
# extractive answers and chat replies are short by nature.
PROSE_TARGET_BYTES = 512        # budget for prose targets
CHAT_TARGET_BYTES = 200         # budget for chat replies
QA_TARGET_BYTES = 160           # budget for extractive QA answers

# OpenWebText supplies real, uncurated prose with far more syntactic variety
# than WikiText, which is largely encyclopaedic. Set OWT_DOCUMENTS = 0 to skip.
OWT_DOCUMENTS = 20000
OWT_REPO = 'Skylion007/openwebtext'
OWT_SUBSET = 'plain_text'
"""

COLAB_CLEANUP_CODE = '''
"""One-time Colab cleanup - run this ONCE, before anything else.

Colab autosaves your notebook into `My Drive/Colab Notebooks/` and can restore that
copy over your edits when you reopen it. Archiving the stale copy removes the
rollback permanently. Originals are kept as `*.colab-backup`.
"""

try:
    from google.colab import drive
    drive.mount('/content/drive')
    import shutil, os, glob
    stale = glob.glob('/content/drive/MyDrive/Colab Notebooks/*.ipynb')
    if not stale:
        print('No Drive notebook copies found - nothing to clean up. (Safe to continue.)')
    for p in stale:
        backup = p + '.colab-backup'
        if os.path.exists(backup):
            os.remove(p)
            print('removed stale copy (backup already existed):', p)
        else:
            shutil.copy2(p, backup)
            os.remove(p)
            print('archived stale copy:', p, '->', backup)
    print()
    print('Done. From now on: File -> Open notebook -> GitHub -> leonalav/hgrt -> run.ipynb')
except ImportError:
    print('Not running in Colab - skipping cleanup (no-op).')
except Exception as exc:
    print('Cleanup skipped:', exc)
    print('You can continue; just always open the notebook from GitHub, not from Drive.')
'''

COLAB_INTRO_MD = """
# HGRT-10M - Colab notebook

> **Before your first run on Colab:** run the *"One-time Colab cleanup"* code cell
> just below. It archives the stale Drive copy that Colab keeps restoring over your
> edits, so your GitHub version is never rolled back again.
>
> **After that:** always re-open this notebook from GitHub
> (*File -> Open notebook -> GitHub -> `leonalav/hgrt` -> `run.ipynb`*),
> **never** from `My Drive/Colab Notebooks/`. Colab's Drive copy is autosave scratch
> space and will contain stale cells.
>
> **Never edit cells in the Colab UI.** Edit the `.py` files locally, run
> `python sync_notebook.py`, commit, push. The notebook embeds `model.py`,
> `dialogue.py` and `fact_switch.py` verbatim, so it runs standalone with no
> external files on disk.
>
> Full details: see `COLAB_ROLLBACK_FIX.md` in the repository.
"""

DIALOG_INTRO_MD = """
## Fact-to-answer learning, sustained prose and conversation

This section runs in two cleanly separated stages:

1. **Counterfactual fact-to-answer learning (`FACT_SWITCH_MINUTES`)**: Uses strictly disjoint OOV evaluation vocabularies and 2×2 counterfactual quads (value-swap + question-swap) with anti-memorization nonce words. Stage 1 (`plain_nll`) builds the multi-byte copy circuit; Stage 2 (`contrastive`) drives key-matched disambiguation without letting Sentence-1 positional shortcuts pass. Checkpoints are selected on a **disjoint** selection suite and reported on a **fresh-seed** report suite (never the same pairs), using the first-byte decision accuracy plus counterfactual pair accuracy and greedy exact match (never average BPB). Causal Tree ablations are reported as `on` / `off` / `permute` (a seeded, content-level permutation with a per-pair donor audit) alongside a real counterfactual activation patch with an unrelated-donor control. The `far_*` splits put the fact ≥ 2048 bytes before the question so that only the tree memory can carry it. Optionally set `RUN_ARCHITECTURE_SHOWDOWN = True` on an A100 to compare `C1(L7)`, `C1(L3+L7)`, native `C1(rw=64 Tree)`, and a 10M full Transformer (`A0`) under identical step and quad budgets.
2. **Rebalanced grounded QA, sustained prose and service chat (`DIALOGUE_MINUTES`)**: Separates grounded QA (`Passage:/Question:/Answer:` when `QA_PROMPT_STYLE = 'qa'`) from Taskmaster service chat (`User:/Assistant:`) to eliminate cross-domain style leakage, auto-tunes CUDA batch size even when `TRAIN_MINUTES = 0`, rebalances grounded QA / extractive QA / counterfactual quads / multi-sentence `prose_long` continuations / WikiText anchors, and selects checkpoints using both validation BPB and live counterfactual fact-switching. [CoQA](https://stanfordnlp.github.io/coqa/) · [Taskmaster-1](https://github.com/google-research-datasets/Taskmaster/tree/master/TM-1-2019)
"""

FACT_SWITCH_TRAIN_CODE = """
#@title Priority #1: Counterfactual fact-to-answer stage and optional A100 showdown
# The SELECTION suite and the REPORT suite are DISJOINT (fresh seed), so the reported
# numbers are never the maximum over the selection evaluations.
fact_select_suite = build_fact_switch_suite(
    qa_style=QA_PROMPT_STYLE, pairs_per_split=FACT_SWITCH_EVAL_PAIRS, require_memory=True)
fact_report_suite = build_fact_switch_suite(
    qa_style=QA_PROMPT_STYLE, pairs_per_split=FACT_SWITCH_EVAL_PAIRS, seed_offset=100_000)
fact_suite = fact_report_suite
fact_base_eval = evaluate_fact_switch_suite(
    prose_model, fact_report_suite, device, do_generate=True, run_tree_ablations=True,
    permutation_audit=True)
print(format_fact_switch_table('Zero-shot prose checkpoint', fact_base_eval), flush=True)

fact_switch_result = dict(
    qa_style=QA_PROMPT_STYLE,
    minutes_requested=FACT_SWITCH_MINUTES,
    widen_l3=DIALOGUE_WIDEN_L3,
    zero_shot=fact_base_eval,
    selection_is_disjoint=suites_are_disjoint(fact_select_suite, fact_report_suite),
    checkpoint=None,
)
fact_model = prose_model

def auto_quads_per_step(net):
    if DEVICE != 'cuda':
        return 2
    if not AUTO_BATCH:
        return max(1, BATCH_BYTES // (4 * 256))
    probe_rng = random.Random(SEED + 909)
    limit = allocation_limit()
    def measure(n_quads):
        trial = deepcopy(net).train()
        opt = torch.optim.AdamW(trial.parameters(), lr=FACT_SWITCH_LR, weight_decay=.01, fused=True)
        def step_once():
            batch = stream_quad_batch(probe_rng, n_quads, TRAIN_VOCAB, filler_choices=(1,), qa_style=QA_PROMPT_STYLE)
            x, y, bpos, dinfo = pair_batch_tensors(batch, device)
            opt.zero_grad(set_to_none=True)
            with precision(device):
                loss, _ = fact_switch_loss(trial(x), y, bpos, dinfo, mode='contrastive')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trial.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        try:
            seconds, peak = timed_updates(step_once)
        finally:
            del trial, opt
        return dict(batch=n_quads, quads_per_second=n_quads/seconds,
                    median_train_step_seconds=seconds, peak_allocated_bytes=peak)
    max_quads = max(2, MAX_BATCH_BYTES // (4 * 256))
    best, _ = choose_batch(measure, 4, 1, max_quads, limit, 'quads_per_second')
    return best['batch']

if FACT_SWITCH_MINUTES > 0:
    fs_net = deepcopy(prose_model)
    fs_net.blocks[-1].mix.window = DIALOGUE_CONTEXT
    if DIALOGUE_WIDEN_L3:
        fs_net.blocks[3].mix.window = DIALOGUE_CONTEXT
    fs_quads = auto_quads_per_step(fs_net)
    print(f'Fact-switch stage: {fs_quads} quads/step ({4*fs_quads} sequences), L3={fs_net.blocks[3].mix.window}, L7={fs_net.blocks[-1].mix.window}', flush=True)
    fs_wiki_rng = np.random.default_rng(SEED + 409)
    wiki_replay_n = max(2, min(batch_size if TRAIN_MINUTES else 8, BATCH_BYTES // CONTEXT))
    fs_net, fs_run = run_two_stage_fact_switch(
        fs_net, fact_select_suite, device,
        minutes=FACT_SWITCH_MINUTES,
        max_steps=MAX_UPDATES,
        quads_per_step=fs_quads,
        lr=FACT_SWITCH_LR,
        qa_style=QA_PROMPT_STYLE,
        seed=SEED,
        replay_fn=lambda: train_batch(fs_wiki_rng, CONTEXT, wiki_replay_n),
        replay_every=6,
        eval_interval_seconds=60.0,
        event_fn=event,
        select_suite=fact_select_suite,
        report_suite=fact_report_suite,
    )
    fs_path = run_dir/'best_fact_switch.pt'
    tmp = fs_path.with_suffix('.partial')
    torch.save(dict(
        kind='C1', slots=8,
        read_window=fs_net.blocks[-1].mix.window,
        mid_window=fs_net.blocks[3].mix.window,
        qa_style=QA_PROMPT_STYLE,
        state_dict=fs_net.state_dict(),
        base_checkpoint=str(run_dir/'best_lm.pt'),
        step=best_step, bytes_seen=selected_prose_bytes,
        fact_switch_step=fs_run['selected_step'],
    ), tmp)
    tmp.replace(fs_path)
    fact_model = fs_net.eval()
    fact_switch_result.update(fs_run, checkpoint=str(fs_path))
    print(format_fact_switch_table(f"Selected fact-switch checkpoint (step {fs_run['selected_step']}/{fs_run['steps']})", fs_run['selected']), flush=True)
    release_memory()

save_json('fact_switch_results.json', fact_switch_result)

showdown_result = None
if RUN_ARCHITECTURE_SHOWDOWN:
    sd_quads = 8 if DEVICE == 'cuda' else 1
    sd_arms = [
        ('c1_l7',   dict(kind='C1', slots=8, from_base=True,  rw=DIALOGUE_CONTEXT, mw=64,               lr=FACT_SWITCH_LR)),
        ('c1_l3l7', dict(kind='C1', slots=8, from_base=True,  rw=DIALOGUE_CONTEXT, mw=DIALOGUE_CONTEXT, lr=FACT_SWITCH_LR)),
        ('c1_rw64', dict(kind='C1', slots=8, from_base=True,  rw=64,               mw=64,               lr=FACT_SWITCH_LR)),
        ('a0_10m',  dict(kind='A0', slots=0, from_base=False, rw=None,             mw=None,             lr=0.001)),
    ]
    showdown_result = dict(steps=SHOWDOWN_STEPS, quads_per_step=sd_quads, qa_style=QA_PROMPT_STYLE, arms={})
    base_sd = {k: v.detach().cpu().clone() for k, v in prose_model.state_dict().items()}
    for arm_name, spec in sd_arms:
        torch.manual_seed(SEED)
        net = LM(spec['kind'], spec['slots'], backend=backend,
                 read_window=spec['rw'] if spec['kind'] == 'C1' else None,
                 mid_window=spec['mw'] if spec['kind'] == 'C1' else None).to(device)
        if spec['from_base']:
            net.load_state_dict(base_sd)
        net, run_out = run_two_stage_fact_switch(
            net, fact_select_suite, device,
            minutes=0.0, max_steps=SHOWDOWN_STEPS, quads_per_step=sd_quads,
            lr=spec['lr'], qa_style=QA_PROMPT_STYLE, seed=SEED,
            eval_interval_seconds=90.0, event_fn=event,
            select_suite=fact_select_suite, report_suite=fact_report_suite,
        )
        print(format_fact_switch_table(f'Showdown {arm_name}', run_out['selected']), flush=True)
        showdown_result['arms'][arm_name] = run_out
        del net
        release_memory()
    save_json('fact_switch_showdown.json', showdown_result)
"""

DIALOG_TRAIN_CODE = """
#@title Priority #2: Rebalanced QA, sustained prose & conversation (A100 auto-batched)
def fixed_dialogue_rows(pools, count, seed):
    rng = np.random.default_rng(seed)
    result = {}
    for task, rows in pools.items():
        if not rows:
            raise ValueError(f'No eligible {task} examples. Check the data and DIALOGUE_CONTEXT.')
        result[task] = [rows[i] for i in rng.choice(len(rows), min(count, len(rows)), replace=False)]
    return result

@torch.inference_mode()
def score_dialogues(net, examples):
    net.eval()
    scores = {}
    for task, rows in examples.items():
        total, count = 0., 0
        for offset in range(0, len(rows), 2):
            x, y = target_batch(rows[offset:offset+2], device)
            with precision(device):
                loss = F.cross_entropy(net(x).float().flatten(0,1), y.flatten(), reduction='sum')
            total += float(loss)
            count += int((y != -100).sum())
        scores[task] = total / (count * math.log(2))
    return scores

def dialogue_selection_score(bpb_metrics, cf_summary):
    mean_bpb = sum(bpb_metrics.values()) / len(bpb_metrics)
    qswap_acc = cf_summary['oov_qswap'].get('first_byte_acc', cf_summary['oov_qswap']['pair_acc'])
    clean_acc = cf_summary['oov_clean'].get('first_byte_acc', cf_summary['oov_clean']['pair_acc'])
    # Lower is better: preserve counterfactual fact-switching alongside held-out BPB.
    # `first_byte_acc` is the PRIMARY decision metric; `pair_acc` is the fallback.
    return mean_bpb - 0.35 * qswap_acc - 0.15 * clean_acc

def tune_dialogue_batch(net, sample_rows):
    # Fix: never clamp CUDA dialogue batch size to prose_tuning['selected_batch']=2 when TRAIN_MINUTES==0.
    base_size = max(1, BATCH_BYTES // DIALOGUE_CONTEXT)
    if DEVICE != 'cuda' or not AUTO_BATCH:
        return max(1, min(base_size, 2 if DEVICE == 'cpu' else base_size))
    limit = allocation_limit()
    longest = sorted(sample_rows, key=lambda r: len((r[0] + r[1]).encode()), reverse=True)
    def measure(n):
        trial = deepcopy(net).train()
        opt = torch.optim.AdamW(trial.parameters(), lr=DIALOGUE_LR, weight_decay=.01, fused=True)
        rows = [longest[i % len(longest)] for i in range(n)]
        def one_step():
            x, y = target_batch(rows, device, history_targets=True)
            opt.zero_grad(set_to_none=True)
            with precision(device):
                loss = F.cross_entropy(trial(x).float().flatten(0,1), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trial.parameters(), 1., error_if_nonfinite=True)
            opt.step()
        try:
            seconds, peak = timed_updates(one_step)
        finally:
            del trial, opt
        return dict(batch=n, seqs_per_second=n/seconds,
                    median_train_step_seconds=seconds, peak_allocated_bytes=peak)
    maximum = max(2, MAX_BATCH_BYTES // DIALOGUE_CONTEXT)
    best, _ = choose_batch(measure, max(2, base_size), 1, maximum, limit, 'seqs_per_second')
    return best['batch']

def train_dialogues(base):
    source_dir = Path(LOCAL_DIALOGUE_DIR) if LOCAL_DIALOGUE_DIR else cache/'conversation_data'
    records, hashes = read_dialogue_data(source_dir, offline=bool(LOCAL_DIALOGUE_DIR))
    pools = prepare_dialogues(records, DIALOGUE_CONTEXT, qa_style=QA_PROMPT_STYLE)
    del records
    counts = {split: {task: len(rows) for task, rows in group.items()} for split, group in pools.items()}
    print('Eligible examples:', counts, flush=True)
    valid = fixed_dialogue_rows(pools['validation'], DIALOGUE_EVAL_ROWS, SEED+71)
    heldout = fixed_dialogue_rows(pools['heldout'], DIALOGUE_EVAL_ROWS, SEED+72)
    if not all(counts['train'][task] for task in valid):
        raise ValueError('Need training examples for all three tasks.')

    val_cf_suite = dict(
        oov_clean=fact_suite['oov_clean'][:8],
        oov_qswap=fact_suite['oov_qswap'][:8],
        user_qswap=fact_suite['user_qswap'],
    )
    baseline = score_dialogues(base, valid)
    baseline['wiki'] = evaluate(base, 'validation')
    base_cf = evaluate_fact_switch_suite(base, val_cf_suite, device, do_generate=False)
    best_score, best_update = dialogue_selection_score(baseline, base_cf), 0
    best_metrics = dict(baseline)
    best_cf = {k: dict(pair=v['pair_acc'], fb_sens=v['fb_mean_sens']) for k, v in base_cf.items()}
    path = run_dir/'best_dialogue.pt'
    def save_candidate(net):
        tmp = path.with_suffix('.partial')
        torch.save(dict(
            kind='C1', slots=8,
            read_window=net.blocks[-1].mix.window,
            mid_window=net.blocks[3].mix.window,
            qa_style=QA_PROMPT_STYLE,
            state_dict=net.state_dict(),
            base_checkpoint=fact_switch_result.get('checkpoint') or str(run_dir/'best_lm.pt'),
            step=best_step, bytes_seen=selected_prose_bytes,
            dialogue_update=best_update, validation_bpb=best_metrics, validation_fact_switch=best_cf,
            context=DIALOGUE_CONTEXT, prompt_format=f'role-lines-{QA_PROMPT_STYLE}-v2',
        ), tmp)
        tmp.replace(path)
    save_candidate(base)

    net = deepcopy(base)
    net.blocks[-1].mix.window = DIALOGUE_CONTEXT
    if DIALOGUE_WIDEN_L3:
        net.blocks[3].mix.window = DIALOGUE_CONTEXT
    opt = torch.optim.AdamW(net.parameters(), lr=DIALOGUE_LR, weight_decay=.01, fused=(DEVICE=='cuda'))
    rng = np.random.default_rng(SEED+73)
    py_rng = random.Random(SEED+74)

    size = tune_dialogue_batch(net, pools['train']['qa'][:16] or pools['train']['chat'][:16])
    short_qa = 'qa_excerpt' if pools['train']['qa_excerpt'] else 'qa'
    ext_qa = 'qa_extractive' if pools['train']['qa_extractive'] else 'qa'
    long_prose = 'prose_long' if pools['train']['prose_long'] else 'prose'
    # Rebalanced 10-step cycle: grounded QA + 2x2 counterfactual quads get 40% of updates
    # (instead of ~1.6% of target bytes), plus multi-sentence sustained prose and WikiText anchors.
    schedule = (short_qa, 'fact_quad', ext_qa, long_prose, 'chat', 'fact_quad', 'qa', 'prose', 'chat', 'wiki')
    updates, supervised = 0, 0
    print(f'Dialogue stage: base batch={size} (L3={net.blocks[3].mix.window}, L7={net.blocks[-1].mix.window}, qa_style={QA_PROMPT_STYLE})', flush=True)

    def backward(n):
        task = schedule[updates % len(schedule)]
        net.train()
        opt.zero_grad(set_to_none=True)
        if task == 'fact_quad':
            quad_rng = random.Random(int(rng.integers(2**32)))
            n_quads = max(1, n // 2 if DEVICE == 'cuda' else min(2, n))
            batch = stream_quad_batch(
                quad_rng, n_quads, TRAIN_VOCAB,
                clean_prob=0.2, filler_choices=(0, 0, 1),
                nonce_val_prob=0.5, nonce_key_prob=0.4, qa_style=QA_PROMPT_STYLE,
            )
            x, y, bpos, dinfo = pair_batch_tensors(batch, device)
            with precision(device):
                loss, _ = fact_switch_loss(
                    net(x), y, bpos, dinfo,
                    mode='contrastive', first_byte_weight=2.5, margin_target=3.0, contrastive_weight=0.6,
                )
        elif task == 'wiki':
            wiki_n = max(1, min(n * 2, MAX_BATCH_BYTES // CONTEXT if dynamic else max(2, BATCH_BYTES // CONTEXT)))
            x, y = train_batch(rng, CONTEXT, wiki_n)
            with precision(device):
                loss = F.cross_entropy(net(x).float().flatten(0,1), y.reshape(-1))
        else:
            rows = pools['train'][task]
            eff_n = min(len(rows), n * 2) if (dynamic and task in ('qa_excerpt', 'prose')) else n
            selected_rows = [rows[i] for i in rng.integers(len(rows), size=eff_n)]
            x, y = target_batch(selected_rows, device, history_targets=task in ('qa', 'qa_extractive', 'chat'))
            with precision(device):
                loss = F.cross_entropy(net(x).float().flatten(0,1), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        return loss.detach(), int((y != -100).sum())

    def validate_candidate():
        nonlocal best_score, best_update, best_metrics, best_cf
        metrics = score_dialogues(net, valid)
        metrics['wiki'] = evaluate(net, 'validation')
        cf = evaluate_fact_switch_suite(net, val_cf_suite, device, do_generate=False)
        value = dialogue_selection_score(metrics, cf)
        if not math.isfinite(value):
            raise FloatingPointError('Nonfinite dialogue validation loss')
        cf_compact = {k: dict(pair=v['pair_acc'], fb_sens=v['fb_mean_sens']) for k, v in cf.items()}
        if value < best_score:
            best_score, best_update, best_metrics, best_cf = value, updates, metrics, cf_compact
            save_candidate(net)
        event(stage='dialogue_validation', update=updates, bpb=metrics, fact_switch=cf_compact,
              composite=round(value, 4), selected_update=best_update)

    event(stage='dialogue_initial', bpb=baseline, fact_switch=best_cf)
    start = last_check = time.perf_counter()
    checked = 0
    while time.perf_counter()-start < 60*DIALOGUE_MINUTES and (MAX_UPDATES is None or updates < MAX_UPDATES):
        for group in opt.param_groups:
            group['lr'] = DIALOGUE_LR * min(1., (updates+1)/20)
        (loss, ntarget), size = backward_with_backoff(backward, opt, rng, size, 1, dynamic)
        opt.step()
        updates += 1
        supervised += ntarget
        if updates == 1 or updates % 100 == 0:
            event(stage='dialogue_train', update=updates, loss=float(loss), supervised_bytes=supervised, batch=size)
        if time.perf_counter()-last_check >= 90:
            validate_candidate()
            last_check, checked = time.perf_counter(), updates
    if updates != checked:
        validate_candidate()
    if updates == 0:
        raise RuntimeError('The dialogue stage was enabled but completed zero optimizer updates.')
    del opt, net
    release_memory()
    result = dict(
        checkpoint=str(path),
        base_checkpoint=fact_switch_result.get('checkpoint') or str(run_dir/'best_lm.pt'),
        counts=counts, sources=SOURCES, sha256=hashes,
        updates=updates, selected_update=best_update,
        validation_before=baseline, validation_selected=best_metrics,
        validation_fact_switch_selected=best_cf,
        supervised_bytes=supervised, context_bytes=DIALOGUE_CONTEXT,
        qa_style=QA_PROMPT_STYLE, widen_l3=DIALOGUE_WIDEN_L3,
        minutes_requested=DIALOGUE_MINUTES, eval_rows_per_task=DIALOGUE_EVAL_ROWS,
        train_seed=SEED+73, validation_seed=SEED+71, heldout_seed=SEED+72,
        lr=DIALOGUE_LR, train_seconds=time.perf_counter()-start, final_batch=size,
        schedule=list(schedule),
        selection='mean validation BPB - 0.35*oov_qswap_pair - 0.15*oov_clean_pair',
    )
    return result, heldout

dialogue_result, dialogue_heldout = None, None
if DIALOGUE_MINUTES > 0:
    dialogue_result, dialogue_heldout = train_dialogues(fact_model)
    save_json('dialogue_results.json', dialogue_result)
    print('Selected dialogue update', dialogue_result['selected_update'], 'of', dialogue_result['updates'])
"""

CHECKPOINT_DEMO_CODE = """
#@title Generate from the saved checkpoint and print the counterfactual scorecard
PROSE_PROMPT = 'The expedition reached the river at dawn. '
CHAT_USER = 'I would like to book a table for dinner.'
CHAT_FOLLOWUP = 'There will be four of us. What else do you need?'
QA_PASSAGE = 'Mira left her green notebook at the library. She put the red folder in her bag.'
QA_USER = 'Where did Mira leave her notebook?'
QA_FOLLOWUP = 'What color was it?'

selected_path = (
    Path(dialogue_result['checkpoint']) if dialogue_result
    else Path(fact_switch_result['checkpoint']) if fact_switch_result.get('checkpoint')
    else run_dir/'best_lm.pt'
)
checkpoint = torch.load(selected_path, map_location='cpu', weights_only=True)
active_qa_style = checkpoint.get('qa_style', QA_PROMPT_STYLE)
preview = LM(
    checkpoint['kind'], checkpoint['slots'], backend=backend,
    read_window=checkpoint.get('read_window'),
    mid_window=checkpoint.get('mid_window'),
).to(device)
preview.load_state_dict(checkpoint['state_dict'])
preview.eval()
print('Loaded', selected_path, '| prose step', checkpoint['step'],
      '| fact_switch step', checkpoint.get('fact_switch_step', 0),
      '| dialogue update', checkpoint.get('dialogue_update', 0),
      f"| L3={preview.blocks[3].mix.window} L7={preview.blocks[-1].mix.window} qa_style={active_qa_style}")
del checkpoint

def preview_text(net, qa_style=active_qa_style):
    with precision(device):
        full = sample(net, PROSE_PROMPT, SAMPLE_BYTES, SAMPLE_TEMPERATURE, device, SAMPLE_TOP_P,
                      seed=123, finish_after=max(1, SAMPLE_BYTES//2))
        turns = []
        for passage, style, questions in [
            ('', 'role', (CHAT_USER, CHAT_FOLLOWUP)),
            (QA_PASSAGE, qa_style, (QA_USER, QA_FOLLOWUP)),
        ]:
            history = []
            for question in questions:
                lead = chat_prompt(question, history, passage, DIALOGUE_CONTEXT-160, qa_style=style)
                if lead is None:
                    raise ValueError('Shorten the passage or question to fit DIALOGUE_CONTEXT.')
                raw = sample(net, lead, 160, SAMPLE_TEMPERATURE, device, SAMPLE_TOP_P,
                             seed=123, stop_at_newline=True)[len(lead):]
                turns.append(dict(passage=passage, qa_style=style, prompt=lead, user=question, generated=raw))
                history.append((question, raw.strip()))
    return dict(prose_prompt=PROSE_PROMPT, prose_completion=full[len(PROSE_PROMPT):], turns=turns)

def show_preview(title, result):
    print('\\n' + title + '\\n' + display_wiki_sample(result['prose_prompt'] + result['prose_completion']))
    for i, turn in enumerate(result['turns']):
        if i == 2:
            print('Passage:', turn['passage'])
        u_tag, a_tag = role_labels(turn['passage'], turn.get('qa_style', 'role'))
        print(f"{u_tag}: {turn['user']}\\n{a_tag}: {turn['generated'].rstrip(chr(10))}")

def heldout_prose_samples(models, examples):
    results = []
    for lead, reference in examples[:2]:
        row = dict(prompt=lead, reference=reference)
        print('\\nHeld-out prose prompt:', lead)
        for label, net in models.items():
            with precision(device):
                full = sample(net, lead, min(SAMPLE_BYTES, 180), SAMPLE_TEMPERATURE, device,
                              SAMPLE_TOP_P, seed=917, finish_after=90)
            row[label] = full[len(lead):]
            print(label, row[label])
        results.append(row)
    return results

preview_result = dict(
    checkpoint=str(selected_path), qa_style=active_qa_style, seed=123,
    temperature=SAMPLE_TEMPERATURE, top_p=SAMPLE_TOP_P,
    prose_max_bytes=SAMPLE_BYTES, reply_max_bytes=160,
)
final_fact_eval = evaluate_fact_switch_suite(preview, fact_suite, device, do_generate=True, run_tree_ablations=True)
print('\\n' + format_fact_switch_table('Final selected checkpoint — Counterfactual Fact-Switch', final_fact_eval))
preview_result['fact_switch_final'] = final_fact_eval

if dialogue_result:
    preview_result['before_dialogue'] = preview_text(prose_model, qa_style=active_qa_style)
    show_preview('Before conversation training', preview_result['before_dialogue'])
    dialogue_result['heldout_selected_bpb'] = score_dialogues(preview, dialogue_heldout)
    dialogue_result['heldout_base_bpb'] = score_dialogues(prose_model, dialogue_heldout)
    preview_result['heldout_prose'] = heldout_prose_samples(
        {'before': prose_model, 'selected': preview}, dialogue_heldout['prose'])
    qa_checks = []
    with precision(device):
        for question, reference in dialogue_heldout['qa'][:8]:
            row = dict(prompt=question, reference=reference.strip())
            for label, net in [('before', prose_model), ('selected', preview)]:
                answer = sample(net, question, 80, SAMPLE_TEMPERATURE, device, SAMPLE_TOP_P,
                                seed=123, stop_at_newline=True)[len(question):]
                row[label] = answer
                row[label+'_exact'] = normalize_answer(answer) == normalize_answer(reference)
            qa_checks.append(row)
    preview_result['heldout_qa'] = qa_checks
    print('Held-out free answers, single-reference exact matches:',
          {k: sum(r[k+'_exact'] for r in qa_checks) for k in ('before', 'selected')}, 'of', len(qa_checks))
    extractive_dev = [(p, t) for p, t in dialogue_heldout['qa'] if is_extractive_qa(p, t, max_answer_bytes=32)][:16]
    if extractive_dev:
        coqa_before = evaluate_coqa_extractive(prose_model, extractive_dev, device)
        coqa_selected = evaluate_coqa_extractive(preview, extractive_dev, device)
        preview_result['heldout_coqa_extractive'] = dict(before=coqa_before, selected=coqa_selected)
        print(f"Held-out CoQA extractive ({len(extractive_dev)} items): "
              f"before EM={coqa_before['exact_match']:.2f} F1={coqa_before['token_f1']:.2f} | "
              f"selected EM={coqa_selected['exact_match']:.2f} F1={coqa_selected['token_f1']:.2f}")
    save_json('dialogue_results.json', dialogue_result)

preview_result['selected'] = preview_text(preview, qa_style=active_qa_style)
show_preview('Selected checkpoint', preview_result['selected'])
save_json('final_generations.json', preview_result)
lm_results.update(
    fact_switch_steps=fact_switch_result.get('steps', 0),
    dialogue_updates=dialogue_result['updates'] if dialogue_result else 0,
    selected_dialogue_update=dialogue_result['selected_update'] if dialogue_result else None,
    final_checkpoint=str(selected_path),
    generation_file='final_generations.json',
)
save_json('lm_results.json', lm_results)
report.update(
    fact_switch=fact_switch_result,
    fact_switch_showdown=showdown_result,
    fact_switch_final=final_fact_eval,
    dialogue=dialogue_result,
    final_checkpoint=str(selected_path),
    generation_file='final_generations.json',
)
save_json('report.json', report)
print('Raw generations:', run_dir/'final_generations.json')
"""


def sync(root: Path | None = None):
    root = root or Path(__file__).parent
    nb_path = root / 'run.ipynb'
    nb = json.loads(nb_path.read_text(encoding='utf8'))
    cells = nb['cells']

    def by_id(cid: str) -> dict:
        for c in cells:
            if c.get('id') == cid:
                return c
        raise KeyError(cid)

    model_src = (root / 'model.py').read_text(encoding='utf8').rstrip()
    dialogue_src = (root / 'dialogue.py').read_text(encoding='utf8').rstrip()
    fact_src = (root / 'fact_switch.py').read_text(encoding='utf8').rstrip()

    by_id('bdcc7c19')['source'] = to_lines(INTRO_MD)

    c1 = by_id('3adb75ac')
    c1['source'] = to_lines(SETTINGS_CODE)
    c1['outputs'], c1['execution_count'] = [], None

    c_model = by_id('775ed5c3')
    c_model['source'] = to_lines('#@title HGRT-10M model definition\n' + model_src)
    c_model['outputs'], c_model['execution_count'] = [], None

    c_corp = by_id('5dda70a4')
    corp_src = ''.join(c_corp['source'])
    old_close = (
        'def close_corpus(maps):\n'
        '    for raw in maps.values():\n'
        '        raw._mmap.close()\n'
        'atexit.register(close_corpus, corpus)'
    )
    new_close = (
        'def close_corpus(maps, tmp_dir=None):\n'
        '    for raw in maps.values():\n'
        '        try:\n'
        '            raw._mmap.close()\n'
        '        except Exception:\n'
        '            pass\n'
        '    if tmp_dir is not None:\n'
        '        try:\n'
        '            tmp_dir.cleanup()\n'
        '        except Exception:\n'
        '            pass\n'
        'atexit.register(close_corpus, corpus, data_cache)'
    )
    if old_close in corp_src:
        c_corp['source'] = to_lines(corp_src.replace(old_close, new_close, 1))
        c_corp['outputs'], c_corp['execution_count'] = [], None

    c_lm = by_id('d1eb119d')
    lm_src = ''.join(c_lm['source'])
    if 'mid_window' not in lm_src:
        lm_src = lm_src.replace(
            "    model.blocks[-1].mix.window = ck.get('read_window', model.s.chunk)",
            "    model.blocks[-1].mix.window = ck.get('read_window') or model.s.chunk\n"
            "    model.blocks[3].mix.window = ck.get('mid_window') or model.s.chunk",
            1,
        )
        lm_src = lm_src.replace(
            "    torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, config=config, state_dict=model.state_dict(),",
            "    torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, mid_window=model.blocks[3].mix.window, config=config, state_dict=model.state_dict(),",
            1,
        )
        lm_src = lm_src.replace(
            "        torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, config=config, state_dict=model.state_dict(),",
            "        torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, mid_window=model.blocks[3].mix.window, config=config, state_dict=model.state_dict(),",
            1,
        )
        c_lm['source'] = to_lines(lm_src)
        c_lm['outputs'], c_lm['execution_count'] = [], None

    by_id('dialog-intro')['source'] = to_lines(DIALOG_INTRO_MD)

    # ---- Colab rollback guards (owned here so they survive every re-sync) ----
    c_intro = by_id('82ee9c05')
    c_intro['cell_type'] = 'markdown'
    c_intro['source'] = to_lines(COLAB_INTRO_MD)
    c_intro.pop('outputs', None)
    c_intro.pop('execution_count', None)

    c_cleanup = make_cell('code', '50b46b5a', COLAB_CLEANUP_CODE)

    # Bootstrap: re-run the settings cell when it has not been executed yet.
    c_setup = by_id('4761bdeb')
    setup_src = ''.join(c_setup['source'])
    if 'bootstrapping it now' not in setup_src:
        setup_src = setup_src.replace(
            "import contextlib, gc, hashlib, importlib, importlib.metadata, importlib.util, json\n"
            "import os, shutil, statistics, subprocess, sys, tempfile, time\n",
            "import contextlib, gc, hashlib, importlib, importlib.metadata, importlib.util, json\n"
            "import os, shutil, statistics, subprocess, sys, tempfile, time\n"
            "\n"
            "# Bootstrap: if this cell is run before \"Run settings\", re-execute it so the\n"
            "# notebook works even when individual cells are executed out of order.\n"
            "if 'DEVICE' not in globals() or 'CONTEXT' not in globals():\n"
            "    print('Run settings cell was not executed; bootstrapping it now.')\n"
            "    _nb = globals().get('__nb__')\n"
            "    if _nb is not None:\n"
            "        try:\n"
            "            exec(''.join(_nb.cells[3].source), globals())\n"
            "        except Exception as exc:\n"
            "            raise RuntimeError(\n"
            "                'Could not auto-run the \"Run settings\" cell. '\n"
            "                'Restart the runtime and run all cells from the top.'\n"
            "            ) from exc\n"
            "    else:\n"
            "        raise RuntimeError(\n"
            "            'The \"Run settings\" cell has not been executed. '\n"
            "            'Restart the runtime and run all cells from the top.'\n"
            "        )\n",
            1,
        )
        c_setup['source'] = to_lines(setup_src)
        c_setup['outputs'], c_setup['execution_count'] = [], None

    # Corpus cell must not fail on a missing CONTEXT.
    c_corp = by_id('5dda70a4')
    corp_src = ''.join(c_corp['source'])
    if "globals().get('CONTEXT'" not in corp_src:
        corp_src = corp_src.replace(
            "prefix_bytes, ending_bytes = min(192, CONTEXT//2), min(128, CONTEXT//2)",
            "# Fall back to the configured default if this cell is run before \"Run settings\".\n"
            "CONTEXT = globals().get('CONTEXT', 512)\n"
            "\n"
            "prefix_bytes, ending_bytes = min(192, CONTEXT//2), min(128, CONTEXT//2)",
            1,
        )
        c_corp['source'] = to_lines(corp_src)
        c_corp['outputs'], c_corp['execution_count'] = [], None

    # Phase 0 hypothesis test lives in the last cell.
    c_p0 = by_id('e7425b56')
    c_p0['source'] = to_lines(PHASE0_CODE)
    c_p0['outputs'], c_p0['execution_count'] = [], None

    # Resume-selection block: survey every run directory and prefer the best
    # prose weights, instead of only globbing wiki-*/latest.pt.
    c_lm = by_id('d1eb119d')
    lm_lines = ''.join(c_lm['source']).split('\n')
    if any(l.startswith('def matching_checkpoint') for l in lm_lines):
        start = next(i for i, l in enumerate(lm_lines) if l.startswith('def matching_checkpoint'))
        end = next(i for i, l in enumerate(lm_lines) if l.strip() == 'del ck')
        new_lines = RESUME_BLOCK.strip('\n').split('\n')
        out_lines = lm_lines[:start] + new_lines + lm_lines[end + 1:]
        c_lm['source'] = [l + '\n' for l in out_lines[:-1]] + [out_lines[-1]]
        c_lm['outputs'], c_lm['execution_count'] = [], None
    elif 'def survey_checkpoints' not in ''.join(c_lm['source']):
        raise RuntimeError('Could not locate the resume-selection block in cell d1eb119d.')

    # Phase 3: the long-range context probe. It re-derives, for whatever
    # checkpoint was selected, the evidence that made a CONTEXT=1024 retrain
    # unnecessary (see PHASE3_DECISION.md).
    # The cell is ALWAYS refreshed (not only inserted when missing), so a fix
    # to `phase3_probe_cell.py` -- e.g. holding the probe content fixed across
    # context lengths (W9) -- reaches the notebook.
    c_probe = next((c for c in cells if c.get('id') == 'context-probe'), None)
    if c_probe is None:
        cells.insert(next(i for i, c in enumerate(cells) if c.get('id') == '2b82a875') + 1,
                     make_cell('code', 'context-probe', PHASE3_PROBE_CODE))
    else:
        c_probe['source'] = to_lines(PHASE3_PROBE_CODE)
        c_probe['outputs'], c_probe['execution_count'] = [], None

    c_dd = by_id('dialog-data')
    c_dd['source'] = to_lines(dialogue_src)
    c_dd['outputs'], c_dd['execution_count'] = [], None

    c_fs_lib = make_cell(
        'code', 'fact-switch-lib',
        '#@title Counterfactual fact-to-answer benchmark and trainer\n' + fact_src,
    )
    c_fs_train = make_cell('code', 'fact-switch-train', FACT_SWITCH_TRAIN_CODE)

    c_dt = by_id('dialog-train')
    c_dt['source'] = to_lines(DIALOG_TRAIN_CODE)
    c_dt['outputs'], c_dt['execution_count'] = [], None

    c_demo = by_id('checkpoint-demo')
    c_demo['source'] = to_lines(CHECKPOINT_DEMO_CODE)
    c_demo['outputs'], c_demo['execution_count'] = [], None

    cells[:] = [c for c in cells if c.get('id') not in ('fact-switch-lib', 'fact-switch-train', '50b46b5a')]
    dd_idx = next(i for i, c in enumerate(cells) if c.get('id') == 'dialog-data')
    cells[dd_idx + 1:dd_idx + 1] = [c_fs_lib, c_fs_train]

    # The cleanup cell must be the very first thing that runs.
    if not cells or cells[0].get('id') != '50b46b5a':
        cells.insert(0, c_cleanup)

    # Phase 1: sentence-boundary-aware prose objective. Applied last so it sees
    # the final state of every cell it touches.
    objective_src = (root / 'phase1_objective.py').read_text(encoding='utf8').rstrip()
    applied = phase1_notebook.apply_phase1(nb, objective_src)
    if applied:
        print(f'Applied Phase 1 objective changes: {applied}')

    # Phase 2: long-form corpus and larger prose target budgets.
    applied2 = phase2_notebook.apply_phase2(nb)
    if applied2:
        print(f'Applied Phase 2 data changes: {applied2}')

    payload = json.dumps(nb, indent=1, ensure_ascii=False) + '\n'
    # Windows can transiently lock the file (indexer, antivirus, an open
    # editor tab). Retry briefly instead of failing the whole sync.
    for attempt in range(6):
        try:
            nb_path.write_text(payload, encoding='utf8')
            break
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.4)
    print(f'Synced {nb_path} ({len(cells)} cells).')
    for i, c in enumerate(cells):
        print(f"  [{i:02d}] {c['cell_type']:8s} id={c.get('id', '')}")


if __name__ == '__main__':
    sync()
