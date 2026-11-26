"""Phase 2: long-form prose training data and a larger target budget.

Two changes:

1. **Target length.** Diagnosis 4 in DIAGNOSIS_AND_PLAN.md: every supervised
   dialogue target was capped at 160 bytes (~27 words), so the model learned to
   emit short clipped spans and pad with high-frequency function words. The cap
   is now raised for PROSE tasks only. QA answers stay short because they are
   extractive, and chat replies stay short because that is what a chat turn
   is.

2. **Long-form corpus.** OpenWebText supplies real, uncurated prose with
   multi-sentence structure and far more syntactic variety than WikiText
   (which is largely encyclopaedic). The paragraphs are split into sentence
   spans so they can be used for the same sentence-level objective.
"""

import re

# Target budgets, in bytes, per task family.
#   QA and chat stay short: an extractive answer and a chat reply are short by
#   nature, and lengthening them would dilute the supervision.
#   Prose tasks get a long budget so the model learns to compose across
#   sentence boundaries instead of padding a short span.
QA_TARGET_BYTES = 160
CHAT_TARGET_BYTES = 200
PROSE_TARGET_BYTES = 512

# Sentence span limits for the long-form corpus.
LONG_MIN_SENTENCES = 3
LONG_MAX_SENTENCES = 8

# How many OpenWebText documents to ingest (0 disables the corpus).
OWT_DOCUMENTS = 20000

OWT_REPO = 'Skylion007/openwebtext'
OWT_SUBSET = 'plain_text'


def split_sentences(text):
    """Split text into sentences.

    Uses the same rule as the notebook's `sentence_cuts`, adapted for str:
    a terminator followed by whitespace and an opening capital or quote.
    Abbreviations such as "Mr." and initials are not treated as boundaries.
    """
    out = []
    start = 0
    for match in re.finditer(r'''[.!?]["']?[ \t]+(?=[A-Z"'])''', text):
        end = match.start()
        if text[end] == '.':
            word = text[max(0, end - 12):end].rstrip()
            if re.search(r'\b(?:Mr|Mrs|Ms|Dr|Prof|St|Ste|Mt|Gen|Sr|Jr|[A-Z])$', word):
                continue
        sentence = text[start:match.end()].strip()
        if sentence:
            out.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def long_form_paragraphs(text, min_sentences=LONG_MIN_SENTENCES,
                         max_sentences=LONG_MAX_SENTENCES,
                         max_bytes=PROSE_TARGET_BYTES):
    """Yield (prompt, target) pairs of sustained prose from a document.

    The prompt is the text before the target sentence group; the target is a
    run of consecutive sentences. Both come from the same document, so the
    continuation is genuinely coherent.
    """
    sentences = split_sentences(text)
    if len(sentences) < min_sentences:
        return
    for i in range(min_sentences - 1, len(sentences)):
        window = min(max_sentences, len(sentences) - i)
        target = ' '.join(sentences[i:i + window])
        while len(target.encode()) > max_bytes and window > 1:
            window -= 1
            target = ' '.join(sentences[i:i + window])
        if len(target.encode()) > max_bytes:
            continue
        prompt = ' '.join(sentences[:i])
        if not prompt or not target:
            continue
        yield prompt, target


def target_budget_for(task):
    """Return the target byte budget for a task name."""
    if task in ('qa', 'qa_excerpt', 'qa_extractive'):
        return QA_TARGET_BYTES
    if task == 'chat':
        return CHAT_TARGET_BYTES
    return PROSE_TARGET_BYTES


def split_wiki_markup(text):
    """Remove the WikiText markup tokens that leak into generations.

    WikiText-103-raw-v1 stores "@-@" for a hyphen and "@,@" for a comma.
    Byte-level modelling has to spend capacity learning these, and they appear
    verbatim in generated text (25% of continuations in the diagnosis).
    """
    return text.replace('@-@', '-').replace('@,@', ',').replace('@.@', '.')
