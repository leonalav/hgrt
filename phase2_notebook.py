"""Notebook edits for Phase 2 (long-form corpus and larger target budgets).

Applied on top of Phase 1. Idempotent, and reversible for testing.
"""

# The Phase 1 version of the shared corpus block, with the Phase 2 sentinels
# removed. Phase 2's revert restores this, not the pre-Phase-1 original.
PHASE1_CORPUS_BLOCK = """corpus, split_info, anchors, sentence_anchors = {}, {}, {}, {}
for split in FILES:
    (corpus[split], split_info[split], anchors[split],
     sentence_anchors[split]) = make_split(split)
    print(split, split_info[split]['rows'], 'rows,', len(corpus[split]), 'bytes,',
          len(sentence_anchors[split]), 'sentence anchors')
"""

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

SETTINGS_ADDITIONS = '''
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
'''

# ---------------------------------------------------------------------------
# Corpus cell: load OpenWebText and expose it as an extra training source
# ---------------------------------------------------------------------------

CORPUS_OWT_OLD = """corpus, split_info, anchors, sentence_anchors = {}, {}, {}, {}
for split in FILES:
    (corpus[split], split_info[split], anchors[split],
     sentence_anchors[split]) = make_split(split)
    print(split, split_info[split]['rows'], 'rows,', len(corpus[split]), 'bytes,',
          len(sentence_anchors[split]), 'sentence anchors')
"""

CORPUS_OWT_NEW = """
# --- Phase 2: OpenWebText long-form prose ------------------------------------
# WikiText-103 is largely encyclopaedic. OpenWebText is uncurated web prose,
# which gives the sentence-level objective a far wider variety of sentence
# structures to discriminate between.
owt_corpus = None
owt_anchors = None
owt_sentence_anchors = None
if OWT_DOCUMENTS > 0:
    try:
        owt_path = cache/'openwebtext.bin'
        owt_digest = hashlib.sha256()
        owt_offset = 0
        owt_paras = array('q')
        owt_sentence_offsets = array('q')
        owt_rows = 0
        with owt_path.open('wb') as dst:
            shard = hf_hub_download(OWT_REPO, f'{OWT_SUBSET}/openwebtext.tar.xz',
                                    repo_type='dataset')
            import tarfile
            with tarfile.open(shard) as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if owt_rows >= OWT_DOCUMENTS:
                        break
                    handle = archive.extractfile(member)
                    if handle is None:
                        continue
                    text = handle.read().decode('utf8', errors='replace')
                    for paragraph in text.split('\\n\\n'):
                        paragraph = paragraph.strip()
                        if not paragraph:
                            continue
                        row = (paragraph + '\\n').encode('utf8')
                        for cut in sentence_cuts(row.rstrip(b'\\n')):
                            if SENTENCE_LEFT_BYTES <= cut <= len(row) - SENTENCE_RIGHT_BYTES - 1:
                                owt_sentence_offsets.append(owt_offset + cut)
                        owt_paras.append(owt_offset)
                        dst.write(row)
                        owt_digest.update(row)
                        owt_offset += len(row)
                        owt_rows += 1
        owt_corpus = np.memmap(owt_path, dtype=np.uint8, mode='c')
        owt_anchors = np.asarray(owt_paras, dtype=np.int64)
        owt_sentence_anchors = np.asarray(owt_sentence_offsets, dtype=np.int64)
        split_info['openwebtext'] = dict(bytes=owt_path.stat().st_size, rows=owt_rows,
                                         sha256=owt_digest.hexdigest(), source=OWT_REPO)
        print('openwebtext', owt_rows, 'documents,', len(owt_corpus), 'bytes,',
              len(owt_sentence_anchors), 'sentence anchors')
    except Exception as exc:
        print('OpenWebText unavailable, continuing with WikiText only:', exc)
        owt_corpus = None
"""

CORPUS_TRAIN_STORAGE_OLD = """train_storage = corpus['train']
if DEVICE == 'cuda' and train_storage.nbytes <= .10 * torch.cuda.mem_get_info(device)[0] * GPU_MEMORY_FRACTION:
    try:
        train_storage = torch.from_numpy(train_storage).to(device)
    except torch.OutOfMemoryError:
        pass  # The memory-mapped training split is already available.
print('Training windows:', 'GPU byte cache' if isinstance(train_storage, torch.Tensor) else 'local memory map')
"""

CORPUS_TRAIN_STORAGE_NEW = """train_storage = corpus['train']
if DEVICE == 'cuda' and train_storage.nbytes <= .10 * torch.cuda.mem_get_info(device)[0] * GPU_MEMORY_FRACTION:
    try:
        train_storage = torch.from_numpy(train_storage).to(device)
    except torch.OutOfMemoryError:
        pass  # The memory-mapped training split is already available.
print('Training windows:', 'GPU byte cache' if isinstance(train_storage, torch.Tensor) else 'local memory map')

# Phase 2: mix OpenWebText into the sentence-level objective. The paragraph
# objective still uses WikiText alone, because the OpenWebText paragraph
# offsets are document starts rather than sentence boundaries.
owt_train_storage = owt_corpus
owt_sentence_pool = owt_sentence_anchors if owt_sentence_anchors is not None else None
use_owt_sentences = owt_sentence_pool is not None and len(owt_sentence_pool) >= SENTENCE_CANDIDATES * 2
if use_owt_sentences:
    print('OpenWebText sentence pool active:', len(owt_sentence_pool), 'anchors')
"""

# ---------------------------------------------------------------------------
# Dialogue training: the larger prose budget
# ---------------------------------------------------------------------------

DIALOG_CALL_OLD = "    pools = prepare_dialogues(records, DIALOGUE_CONTEXT, qa_style=QA_PROMPT_STYLE)\n"
DIALOG_CALL_NEW = ("    pools = prepare_dialogues(records, DIALOGUE_CONTEXT, qa_style=QA_PROMPT_STYLE,\n"
                   "                      prose_bytes=PROSE_TARGET_BYTES, chat_bytes=CHAT_TARGET_BYTES)\n")


def _replace_once(text, old, new, label):
    if old not in text:
        raise RuntimeError(f'Could not find the {label} anchor in the notebook.')
    return text.replace(old, new, 1)


def _replace_block(text, marker, new_block, label):
    """Replace the text between the >>> and <<< sentinels with new_block."""
    start_marker = f'# >>> {marker} >>>'
    end_marker = f'# <<< {marker} <<<'
    start = text.find(start_marker)
    end = text.find(end_marker)
    if start == -1 or end == -1:
        raise RuntimeError(f'Could not find the {label} block in the notebook.')
    end += len(end_marker)
    return text[:start] + new_block + text[end:]


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


def apply_phase2(nb):
    """Apply the Phase 2 changes. Idempotent."""
    changed = []

    # ---- settings ---------------------------------------------------------
    # The settings cell is owned by sync_notebook.py, which writes the Phase 1
    # and Phase 2 constants directly. Nothing to add here.

    # ---- corpus: OpenWebText ----------------------------------------------
    c = _cell_by_id(nb, '5dda70a4')
    text = ''.join(c['source'])
    if 'owt_corpus' not in text:
        anchor = "# On Windows the byte mappings must be closed before the temporary directory is removed."
        if anchor not in text:
            raise RuntimeError('corpus owt anchor not found')
        text = text.replace(anchor, CORPUS_OWT_NEW.strip('\n') + '\n\n' + anchor, 1)
        text = _replace_once(text, CORPUS_TRAIN_STORAGE_OLD, CORPUS_TRAIN_STORAGE_NEW, 'train storage')
        _set_source(c, text)
        changed.append('corpus-owt')

    # ---- dialogue training: larger prose budget ---------------------------
    c = _cell_by_id(nb, 'dialog-train')
    text = ''.join(c['source'])
    if 'prose_bytes=PROSE_TARGET_BYTES' not in text:
        text = _replace_once(text, DIALOG_CALL_OLD, DIALOG_CALL_NEW, 'dialogue call')
        _set_source(c, text)
        changed.append('dialog-budget')

    return changed


def revert_phase2(nb):
    """Undo the Phase 2 changes. Used to test the round-trip."""
    reverted = []

    # The settings cell is owned by sync_notebook.py, so nothing to revert.

    c = _cell_by_id(nb, '5dda70a4')
    text = ''.join(c['source'])
    if 'owt_corpus' in text:
        start = text.find('\n# --- Phase 2: OpenWebText long-form prose')
        end = text.find('# On Windows the byte mappings must be closed')
        if start != -1 and end != -1:
            text = text[:start] + '\n' + text[end:]
        text = text.replace(CORPUS_TRAIN_STORAGE_NEW, CORPUS_TRAIN_STORAGE_OLD, 1)
        _set_source(c, text)
        reverted.append('corpus-owt')

    c = _cell_by_id(nb, 'dialog-train')
    text = ''.join(c['source'])
    if 'prose_bytes=PROSE_TARGET_BYTES' in text:
        text = text.replace(DIALOG_CALL_NEW, DIALOG_CALL_OLD, 1)
        _set_source(c, text)
        reverted.append('dialog-budget')

    return reverted