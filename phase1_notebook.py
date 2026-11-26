"""Notebook edits for Phase 1 (sentence-boundary-aware prose objective).

These functions transform a parsed notebook (a dict with 'cells') in place.
They are idempotent: applying them twice yields the same notebook.

The notebook must remain standalone on Colab, so `phase1_objective.py` is
EMBEDDED into a dedicated cell rather than imported from disk.
"""

import re

# ---------------------------------------------------------------------------
# Cell text templates
# ---------------------------------------------------------------------------

SETTINGS_ADDITIONS = '''
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
'''

CORPUS_SENTENCE_ANCHORS_OLD = """                    for paragraph in row.splitlines(keepends=True):
                        cut = sentence_boundary(paragraph.rstrip(b'\\r\\n'), prefix_bytes, ending_bytes)
                        if cut is not None:
                            anchors.append(offset + cut)
                        offset += len(paragraph)
"""

CORPUS_SENTENCE_ANCHORS_NEW = """                    for paragraph in row.splitlines(keepends=True):
                        stripped = paragraph.rstrip(b'\\r\\n')
                        cut = sentence_boundary(stripped, prefix_bytes, ending_bytes)
                        if cut is not None:
                            anchors.append(offset + cut)
                        # Sentence-level anchors: EVERY boundary that can host a
                        # candidate sentence. The paragraph-level anchor keeps
                        # only the first usable cut per paragraph, which is too
                        # coarse for the sentence discrimination task.
                        for scut in sentence_cuts(stripped):
                            if SENTENCE_LEFT_BYTES <= scut <= len(stripped) - SENTENCE_RIGHT_BYTES:
                                sentence_anchors.append(offset + scut)
                        offset += len(paragraph)
"""

CORPUS_RETURN_OLD = """    return (np.memmap(path, dtype=np.uint8, mode='c'),
            dict(bytes=path.stat().st_size, rows=count, sha256=digest.hexdigest(), files=FILES[split],
                 continuation_paragraphs=len(anchors)), np.asarray(anchors, dtype=np.int64))

corpus, split_info, anchors = {}, {}, {}
for split in FILES:
    corpus[split], split_info[split], anchors[split] = make_split(split)
    print(split, split_info[split]['rows'], 'rows,', len(corpus[split]), 'bytes')
"""

CORPUS_RETURN_NEW = """    return (np.memmap(path, dtype=np.uint8, mode='c'),
            dict(bytes=path.stat().st_size, rows=count, sha256=digest.hexdigest(), files=FILES[split],
                 continuation_paragraphs=len(anchors),
                 sentence_anchors=len(sentence_anchors)), np.asarray(anchors, dtype=np.int64),
            np.asarray(sentence_anchors, dtype=np.int64))

corpus, split_info, anchors, sentence_anchors = {}, {}, {}, {}
for split in FILES:
    (corpus[split], split_info[split], anchors[split],
     sentence_anchors[split]) = make_split(split)
    print(split, split_info[split]['rows'], 'rows,', len(corpus[split]), 'bytes,',
          len(sentence_anchors[split]), 'sentence anchors')
"""

CORPUS_DECLARE_OLD = """    count, offset = 0, 0
    anchors = array('q')
"""

CORPUS_DECLARE_NEW = """    count, offset = 0, 0
    anchors = array('q')
    sentence_anchors = array('q')
"""

CORPUS_USE_OLD = """use_continuations = len(anchors['train']) >= 2 and len(anchors['validation']) >= 2
if not use_continuations:
    print('Too few eligible paragraphs for continuation practice. Using ordinary next-byte training.')
"""

CORPUS_USE_NEW = """use_continuations = len(anchors['train']) >= 2 and len(anchors['validation']) >= 2
if not use_continuations:
    print('Too few eligible paragraphs for continuation practice. Using ordinary next-byte training.')

use_sentences = (len(sentence_anchors['train']) >= SENTENCE_CANDIDATES * 2 and
                 len(sentence_anchors['validation']) >= SENTENCE_CANDIDATES * 2)
if not use_sentences:
    print('Too few sentence anchors for the sentence-level objective. Using the paragraph objective only.')
else:
    print('Sentence-level objective active:', len(sentence_anchors['train']),
          'train anchors,', len(sentence_anchors['validation']), 'validation anchors.')
"""

PROSE_BACKWARD_OLD = """def prose_backward(model, optimizer, x, y, check_tree=False, continuation=False):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision(device):
        logits = model(x)
        if continuation:
            matched, mismatched = continuation_nll(logits, y, prefix_bytes)
            margin = (mismatched - matched[:, prefix_bytes-1:]).sum(1) / math.sqrt(ending_bytes)
            ending = matched[:, prefix_bytes-1:]
            # Give the start of the next sentence its own prediction signal.
            loss = .25*matched.mean() + .50*ending.mean() + .25*ending[:, :32].mean()
            loss = loss + .25*F.softplus(-margin).mean()
        else:
            loss = F.cross_entropy(logits.reshape(-1,256).float(), y.reshape(-1))
    loss.backward()
"""

PROSE_BACKWARD_NEW = '''def prose_backward(model, optimizer, x, y, check_tree=False, continuation=False, sentence=None):
    """Prose training step.

    `sentence` is an optional (ids, targets) pair of multi-candidate sentence
    groups. When present, the sentence discrimination term is added to the loss
    with weight SENTENCE_LOSS_WEIGHT and the paragraph term is scaled by
    PARAGRAPH_LOSS_WEIGHT, so the two objectives stay balanced regardless of
    whether the ordinary next-byte term is also active.
    """
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with precision(device):
        logits = model(x)
        if continuation:
            matched, mismatched = continuation_nll(logits, y, prefix_bytes)
            margin = (mismatched - matched[:, prefix_bytes-1:]).sum(1) / math.sqrt(ending_bytes)
            ending = matched[:, prefix_bytes-1:]
            # Give the start of the next sentence its own prediction signal.
            para_loss = .25*matched.mean() + .50*ending.mean() + .25*ending[:, :32].mean()
            para_loss = para_loss + .25*F.softplus(-margin).mean()
        else:
            para_loss = F.cross_entropy(logits.reshape(-1,256).float(), y.reshape(-1))
        if sentence is not None:
            s_ids, s_targets = sentence
            s_logits = model(s_ids.reshape(-1, s_ids.size(-1)))
            sent_loss, _ = sentence_choice_loss(
                s_logits.reshape(s_ids.size(0), s_ids.size(1), s_ids.size(2), -1),
                s_targets, SENTENCE_LEFT_BYTES)
            loss = PARAGRAPH_LOSS_WEIGHT * para_loss + SENTENCE_LOSS_WEIGHT * sent_loss
        else:
            loss = para_loss
    loss.backward()
'''

TRAIN_BATCH_OLD = """def train_prose_batch(size):
    contrast = use_continuations and (step+1) % 4 == 0
    if contrast:
        real, other = choose_paragraphs(train_rng, len(anchors['train']), max(1, size//2))
        x, y = continuation_batch(train_storage, anchors['train'], real, other, prefix_bytes, ending_bytes)
    else:
        x, y = train_batch(train_rng, CONTEXT, size)
    loss, _ = prose_backward(model, optimizer, x, y, continuation=contrast)
    return loss, y.numel()
"""

TRAIN_BATCH_NEW = """def train_prose_batch(size):
    contrast = use_continuations and (step+1) % 4 == 0
    # The sentence term runs on a different step from the paragraph contrast so
    # the two objectives never share a batch.
    sentence_step = use_sentences and (step+1) % 4 == 2
    sentence = None
    if sentence_step:
        # Phase 2: prefer the OpenWebText pool when it is available, because
        # uncurated web prose has far more varied sentence structure than
        # WikiText. Fall back to WikiText otherwise.
        pool, storage = sentence_anchors['train'], train_storage
        if use_owt_sentences:
            pool, storage = owt_sentence_pool, owt_train_storage
        groups = max(1, size // SENTENCE_CANDIDATES)
        real, foils = choose_sentence_foils(train_rng, len(pool), groups, SENTENCE_CANDIDATES)
        sentence = sentence_group_batch(storage, pool, real, foils,
                                        SENTENCE_LEFT_BYTES, SENTENCE_RIGHT_BYTES, device)
    if contrast:
        real, other = choose_paragraphs(train_rng, len(anchors['train']), max(1, size//2))
        x, y = continuation_batch(train_storage, anchors['train'], real, other, prefix_bytes, ending_bytes)
    else:
        x, y = train_batch(train_rng, CONTEXT, size)
    loss, _ = prose_backward(model, optimizer, x, y, continuation=contrast, sentence=sentence)
    return loss, y.numel()
"""

WARMUP_OLD = """    warmup_step = step+1 if resume_path else step-initial_step+1
    for group in optimizer.param_groups:
        group['lr'] = LR*min(1., warmup_step/50)
"""

WARMUP_NEW = """    # The warm-up ramp exists to stabilise a FRESH model. Re-running it on an
    # already-trained model knocks the weights out of their basin: in run
    # wiki-20261126-055649-yrlg5b4l the ramp restarted at LR~0 and validation
    # BPB rose from 1.417 to 1.518 over 19k steps without recovering.
    # `resume_path` restores the optimizer state, so no ramp is needed at all.
    # `INITIAL_WEIGHTS` starts a fresh optimizer, so a short ramp lets AdamW's
    # moments settle without perturbing the weights.
    warmup_step = step+1 if resume_path else step-initial_step+1
    for group in optimizer.param_groups:
        group['lr'] = LR*min(1., warmup_step/WARMUP_STEPS)
"""

EVALUATE_SENTENCES = '''

@torch.inference_mode()
def evaluate_sentences(model, split):
    """Measure sentence-level completion accuracy on a held-out split."""
    if not use_sentences:
        return None
    pool = sentence_anchors[split]
    if len(pool) < SENTENCE_CANDIDATES * 2:
        return None
    size = min(SENTENCE_EVAL_GROUPS, len(pool))
    rng = np.random.default_rng(3703 if split == 'validation' else 3704)
    real, foils = choose_sentence_foils(rng, len(pool), size, SENTENCE_CANDIDATES)
    ids, targets = sentence_group_batch(corpus[split], pool, real, foils,
                                        SENTENCE_LEFT_BYTES, SENTENCE_RIGHT_BYTES, device)
    correct = 0.0
    seen = 0
    for i in range(0, size, 4):
        block = slice(i, min(i + 4, size))
        rows = block.stop - block.start
        with precision(device):
            logits = model(ids[block].reshape(-1, ids.size(-1)))
        _, acc = sentence_choice_loss(
            logits.reshape(rows, SENTENCE_CANDIDATES, ids.size(-1), -1),
            targets[block], SENTENCE_LEFT_BYTES)
        correct += float(acc) * rows
        seen += rows
    if not seen:
        return None
    return dict(groups=seen, candidates=SENTENCE_CANDIDATES, accuracy=correct/seen)
'''

KEEP_BEST_OLD = """best_bpb, best_step, best_selection_loss = float('inf'), None, float('inf')

def keep_best_lm(bpb):
    global best_bpb, best_step, best_selection_loss
    continuation = evaluate_continuations(model, 'validation') if use_continuations else None
    selection_loss = continuation['matched_bpb'] if continuation else bpb
    if not math.isfinite(bpb) or not math.isfinite(selection_loss):
        raise FloatingPointError(f'Nonfinite validation BPB at step {step}')
    if selection_loss < best_selection_loss:
        tmp = run_dir/'best_lm.partial'
        torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, mid_window=model.blocks[3].mix.window, config=config, state_dict=model.state_dict(),
                        step=step, bytes_seen=bytes_seen, validation_bpb=bpb, continuation=continuation), tmp)
        tmp.replace(run_dir/'best_lm.pt')
        best_bpb, best_step, best_selection_loss = bpb, step, selection_loss
    return continuation
"""

KEEP_BEST_NEW = """best_bpb, best_step, best_selection_loss = float('inf'), None, float('inf')
best_sentence_accuracy = None

def keep_best_lm(bpb):
    global best_bpb, best_step, best_selection_loss, best_sentence_accuracy
    continuation = evaluate_continuations(model, 'validation') if use_continuations else None
    sentences = evaluate_sentences(model, 'validation')
    # Selection now combines the paragraph-level BPB with sentence-level
    # discrimination accuracy, so a checkpoint that completes sentences better
    # is preferred even at a marginally worse BPB.
    selection_loss = selection_score(continuation['matched_bpb'] if continuation else bpb,
                                     sentences['accuracy'] if sentences else None)
    if not math.isfinite(bpb) or not math.isfinite(selection_loss):
        raise FloatingPointError(f'Nonfinite validation BPB at step {step}')
    if selection_loss < best_selection_loss:
        tmp = run_dir/'best_lm.partial'
        torch.save(dict(kind='C1', slots=8, read_window=model.blocks[-1].mix.window, mid_window=model.blocks[3].mix.window, config=config, state_dict=model.state_dict(),
                        step=step, bytes_seen=bytes_seen, validation_bpb=bpb, continuation=continuation,
                        sentences=sentences, selection_loss=selection_loss), tmp)
        tmp.replace(run_dir/'best_lm.pt')
        best_bpb, best_step, best_selection_loss = bpb, step, selection_loss
        best_sentence_accuracy = sentences['accuracy'] if sentences else None
    return continuation, sentences
"""

INITIAL_CALL_OLD = """initial_continuation = keep_best_lm(before_bpb)
event(stage='initial', step=step, bytes_seen=bytes_seen, validation_bpb=before_bpb, continuation=initial_continuation, **valid_base)
"""

INITIAL_CALL_NEW = """initial_continuation, initial_sentences = keep_best_lm(before_bpb)
event(stage='initial', step=step, bytes_seen=bytes_seen, validation_bpb=before_bpb,
      continuation=initial_continuation, sentences=initial_sentences, **valid_base)
"""

TRAIN_EVENT_OLD = """        cont = keep_best_lm(bpb)
        event(stage='train', step=step, bytes_seen=bytes_seen, loss=float(loss), validation_bpb=bpb, continuation=cont,
"""

TRAIN_EVENT_NEW = """        cont, sents = keep_best_lm(bpb)
        event(stage='train', step=step, bytes_seen=bytes_seen, loss=float(loss), validation_bpb=bpb, continuation=cont,
              sentences=sents,
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _replace_once(text, old, new, label, required=True):
    """Replace `old` with `new`, raising if the anchor is missing."""
    if old not in text:
        if required:
            raise RuntimeError(f'Could not find the {label} anchor in the notebook.')
        return text
    if new in text and old not in text:
        return text
    return text.replace(old, new, 1)


def _cell_by_id(nb, cell_id):
    for c in nb['cells']:
        if c.get('id') == cell_id:
            return c
    raise KeyError(cell_id)


def _set_source(cell, text):
    """Write text into a cell's source list.

    The trailing newline is preserved as it was: a cell whose source ends with
    a newline keeps that newline, so a revert/apply round-trip is lossless.
    """
    ends_with_newline = text.endswith('\n')
    body = text[:-1] if ends_with_newline else text
    lines = body.split('\n')
    cell['source'] = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None


def revert_phase1(nb):
    """Undo the Phase 1 changes. Used to test the round-trip.

    Returns the list of reverted items.
    """
    reverted = []

    # ---- settings cell ----------------------------------------------------
    # The settings cell is owned by sync_notebook.py, which writes the Phase 1
    # and Phase 2 constants directly. Nothing to revert here.

    # ---- corpus cell ------------------------------------------------------
    c = _cell_by_id(nb, '5dda70a4')
    text = ''.join(c['source'])
    # Match the same precise marker as apply_phase1. A bare 'sentence_anchors'
    # substring also matches Phase 2's 'owt_sentence_anchors'.
    if 'sentence_anchors = array(' in text:
        text = text.replace(CORPUS_DECLARE_NEW, CORPUS_DECLARE_OLD, 1)
        text = text.replace(CORPUS_SENTENCE_ANCHORS_NEW, CORPUS_SENTENCE_ANCHORS_OLD, 1)
        text = text.replace(CORPUS_RETURN_NEW, CORPUS_RETURN_OLD, 1)
        text = text.replace(CORPUS_USE_NEW, CORPUS_USE_OLD, 1)
        _set_source(c, text)
        reverted.append('corpus')

    # ---- objective cell ---------------------------------------------------
    cells = nb['cells']
    keep = [x for x in cells if x.get('id') != 'sentence-objective']
    if len(keep) != len(cells):
        nb['cells'] = keep
        reverted.append('objective-cell')

    # ---- prose_backward ---------------------------------------------------
    c = _cell_by_id(nb, 'f997eb16')
    text = ''.join(c['source'])
    if 'sentence=None' in text:
        text = text.replace(PROSE_BACKWARD_NEW, PROSE_BACKWARD_OLD, 1)
        _set_source(c, text)
        reverted.append('prose_backward')

    # ---- train loop -------------------------------------------------------
    c = _cell_by_id(nb, '69ac46a1')
    text = ''.join(c['source'])
    if 'sentence_step' in text:
        text = text.replace(TRAIN_BATCH_NEW, TRAIN_BATCH_OLD, 1)
        reverted.append('train_prose_batch')
    if 'WARMUP_STEPS' in text:
        text = text.replace(WARMUP_NEW, WARMUP_OLD, 1)
        reverted.append('warmup')
    if 'sentences=sents' in text:
        text = text.replace(TRAIN_EVENT_NEW, TRAIN_EVENT_OLD, 1)
        reverted.append('train-event')
    _set_source(c, text)

    # ---- evaluation cell --------------------------------------------------
    c = _cell_by_id(nb, 'd1eb119d')
    text = ''.join(c['source'])
    if 'def evaluate_sentences' in text:
        marker = "\n\n@torch.inference_mode()\ndef evaluate_sentences(model, split):"
        start = text.find(marker)
        if start != -1:
            end = text.find("config = dict(kind='C1', slots=8, seed=SEED, context=CONTEXT, batch=batch_size, lr=LR,", start)
            if end != -1:
                text = text[:start] + '\n\n' + text[end:]
                reverted.append('evaluate_sentences')
    if 'selection_score(' in text:
        text = text.replace(KEEP_BEST_NEW, KEEP_BEST_OLD, 1)
        reverted.append('keep_best_lm')
    if 'initial_sentences' in text:
        text = text.replace(INITIAL_CALL_NEW, INITIAL_CALL_OLD, 1)
        reverted.append('initial-call')
    _set_source(c, text)

    return reverted


def apply_phase1(nb, objective_source):
    """Apply every Phase 1 change to the notebook. Idempotent."""
    changed = []

    # ---- 1. settings cell -------------------------------------------------
    # The settings cell is owned by sync_notebook.py, which writes the Phase 1
    # and Phase 2 constants directly. Nothing to add here.

    # ---- 2. corpus cell: collect sentence anchors --------------------------
    c = _cell_by_id(nb, '5dda70a4')
    text = ''.join(c['source'])
    # The guard must match a marker that only Phase 1 introduces. A bare
    # 'sentence_anchors' substring is NOT safe: Phase 2 adds
    # 'owt_sentence_anchors' to the same cell, which would make the guard
    # always false and silently skip every corpus edit. Use the declaration
    # line instead, which is unique to Phase 1.
    if 'sentence_anchors = array(' not in text:
        text = _replace_once(text, CORPUS_DECLARE_OLD, CORPUS_DECLARE_NEW, 'corpus declare')
        text = _replace_once(text, CORPUS_SENTENCE_ANCHORS_OLD, CORPUS_SENTENCE_ANCHORS_NEW, 'corpus anchors')
        text = _replace_once(text, CORPUS_RETURN_OLD, CORPUS_RETURN_NEW, 'corpus return')
        text = _replace_once(text, CORPUS_USE_OLD, CORPUS_USE_NEW, 'corpus use')
        _set_source(c, text)
        changed.append('corpus')

    # ---- 3. objective cell: embed phase1_objective.py ----------------------
    if not any(cell.get('id') == 'sentence-objective' for cell in nb['cells']):
        # Strip the module-level constants: they live in the settings cell so a
        # user can change them there. Re-defining them here would silently
        # override the settings cell values.
        objective_lines = objective_source.split('\n')
        kept = []
        for line in objective_lines:
            if re.match(r'^(SENTENCE_CANDIDATES|SENTENCE_LEFT_BYTES|SENTENCE_RIGHT_BYTES|'
                        r'SENTENCE_LOSS_WEIGHT|PARAGRAPH_LOSS_WEIGHT|SELECTION_SENTENCE_WEIGHT)\s*=',
                        line):
                continue
            kept.append(line)
        objective_text = '\n'.join(kept)
        cell = dict(cell_type='code', id='sentence-objective', metadata={},
                    source=[l + '\n' for l in objective_text.split('\n')[:-1]] + [objective_text.split('\n')[-1]],
                    execution_count=None, outputs=[])
        cells = nb['cells']
        idx = next(i for i, x in enumerate(cells) if x.get('id') == '775ed5c3')
        cells.insert(idx + 1, cell)
        changed.append('objective-cell')

    # ---- 4. training helper cell: prose_backward + train_prose_batch ------
    c = _cell_by_id(nb, 'f997eb16')
    text = ''.join(c['source'])
    if 'sentence=None' not in text:
        text = _replace_once(text, PROSE_BACKWARD_OLD, PROSE_BACKWARD_NEW, 'prose_backward')
        _set_source(c, text)
        changed.append('prose_backward')

    # ---- 5. train loop cell: schedule + warm-up ----------------------------
    c = _cell_by_id(nb, '69ac46a1')
    text = ''.join(c['source'])
    if 'sentence_step' not in text:
        text = _replace_once(text, TRAIN_BATCH_OLD, TRAIN_BATCH_NEW, 'train_prose_batch')
        changed.append('train_prose_batch')
    if 'WARMUP_STEPS' not in text:
        text = _replace_once(text, WARMUP_OLD, WARMUP_NEW, 'warmup')
        changed.append('warmup')
    _set_source(c, text)

    # ---- 6. evaluation cell: sentence accuracy + selection ------------------
    c = _cell_by_id(nb, 'd1eb119d')
    text = ''.join(c['source'])
    if "corpus_hashes={k:v['sha256'] for k,v in split_info.items()}" in text:
        # Phase 2 adds an 'openwebtext' entry to split_info. Including it in
        # corpus_hashes would change the config of every existing checkpoint
        # and make it look incompatible, so the hashes stay restricted to the
        # three pretraining splits.
        text = text.replace(
            "corpus_hashes={k:v['sha256'] for k,v in split_info.items()}",
            "corpus_hashes={k:split_info[k]['sha256'] for k in ('train','validation','test')}",
            1)
        _set_source(c, text)
        changed.append('corpus-hashes')
    if 'def evaluate_sentences' not in text:
        anchor = "config = dict(kind='C1', slots=8, seed=SEED, context=CONTEXT, batch=batch_size, lr=LR,"
        if anchor not in text:
            raise RuntimeError('evaluation cell anchor not found')
        text = text.replace(anchor, EVALUATE_SENTENCES.strip('\n') + '\n\n' + anchor, 1)
        changed.append('evaluate_sentences')
    if 'selection_score(' not in text:
        text = _replace_once(text, KEEP_BEST_OLD, KEEP_BEST_NEW, 'keep_best_lm')
        changed.append('keep_best_lm')
    if 'initial_sentences' not in text:
        text = _replace_once(text, INITIAL_CALL_OLD, INITIAL_CALL_NEW, 'initial call')
        changed.append('initial-call')
    _set_source(c, text)

    # ---- 7. train loop cell: the training event also reports sentences -----
    c = _cell_by_id(nb, '69ac46a1')
    text = ''.join(c['source'])
    if 'sentences=sents' not in text:
        text = _replace_once(text, TRAIN_EVENT_OLD, TRAIN_EVENT_NEW, 'train event')
        changed.append('train-event')
    _set_source(c, text)

    return changed
