"""Prepare disjoint prose continuations, grounded QA turns and assistant replies."""
import hashlib
import json
import re
from pathlib import Path
from urllib.request import urlopen

import torch


SOURCES = {
    'coqa-train-v1.0.json': 'https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-train-v1.0.json',
    'coqa-dev-v1.0.json': 'https://downloads.cs.stanford.edu/nlp/data/coqa/coqa-dev-v1.0.json',
    'self-dialogs.json': (
        'https://raw.githubusercontent.com/google-research-datasets/Taskmaster/'
        'd92cb6af3005f1dc09c39e75e7daf4a04905e00b/TM-1-2019/self-dialogs.json'
    ),
}


def read_dialogue_data(directory, offline):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data, hashes = {}, {}
    for name, url in SOURCES.items():
        path = directory / name
        if not path.is_file() and not offline:
            with urlopen(url, timeout=120) as response:
                raw = response.read()
            json.loads(raw)  # Validate before publishing a cached download.
            tmp = path.with_suffix('.partial')
            tmp.write_bytes(raw)
            tmp.replace(path)
        raw = path.read_bytes()
        data[name] = json.loads(raw)
        hashes[name] = hashlib.sha256(raw).hexdigest()
    return data, hashes


def single_line(text):
    return ' '.join(text.split())


def role_labels(passage, qa_style='role'):
    """Use distinct Question/Answer cues for grounded QA when qa_style='qa' to avoid chat leakage."""
    if passage and qa_style == 'qa':
        return 'Question', 'Answer'
    if qa_style not in ('role', 'qa'):
        raise ValueError(f'Unknown qa_style: {qa_style}')
    return 'User', 'Assistant'


def chat_prompt(user, history, passage, budget, qa_style='role'):
    """Keep the passage, current question and as many complete recent turns as fit."""
    u_tag, a_tag = role_labels(passage, qa_style)
    lead = 'Passage: ' + single_line(passage) + '\n' if passage else ''
    tail = f'{u_tag}: {single_line(user)}\n{a_tag}: '
    room = budget - len((lead + tail).encode())
    if room < 0:
        return None
    for question, answer in reversed(history):
        turn = f'{u_tag}: {single_line(question)}\n{a_tag}: {single_line(answer)}\n'
        size = len(turn.encode())
        if size > room:
            break
        tail, room = turn + tail, room - size
    return lead + tail


def story_continuations(story, context):
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', story)
    for i in range(1, len(sentences)):
        prompt = ' '.join(sentences[max(0, i-2):i]) + ' '
        target = ' '.join(sentences[i:i+2])
        if len(target.encode()) > 224:
            target = sentences[i]
        if len(target.encode()) <= 224 and len((prompt + target).encode()) <= context:
            yield prompt, target


def long_story_continuations(story, context, min_prefix_bytes=220, target_bytes=280):
    """Multi-sentence prefixes (up to context - target_bytes) for practicing sustained prose."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"“])', story)
    prefix_budget = context - target_bytes
    for i in range(3, len(sentences)):
        target = ' '.join(sentences[i:i+3])
        if len(target.encode()) > target_bytes:
            target = ' '.join(sentences[i:i+2])
        if len(target.encode()) > target_bytes:
            target = sentences[i]
        if not target or len(target.encode()) > target_bytes:
            continue
        # Grow backwards from sentence i-1 so the prompt keeps the fullest recent narrative arc.
        kept, used = [], 0
        for sent in reversed(sentences[:i]):
            add = len(sent.encode()) + 1
            if used + add > prefix_budget:
                break
            kept.append(sent)
            used += add
        if used < min_prefix_bytes or len(kept) < 3:
            continue
        prompt = ' '.join(reversed(kept)) + ' '
        if len((prompt + target).encode()) <= context:
            yield prompt, target


def training_excerpt(story, answer):
    """Use an annotated evidence span only for short, extractive training questions."""
    start, end = answer.get('span_start', -1), answer.get('span_end', -1)
    reply = single_line(answer['input_text']).casefold()
    if not (0 <= start < end <= len(story)) or reply in ('', 'yes', 'no', 'unknown'):
        return ''
    left = max(0, start-100)
    right = min(len(story), end+100)
    while left and not story[left-1].isspace():
        left -= 1
    while right < len(story) and not story[right].isspace():
        right += 1
    excerpt = single_line(story[left:right])
    return excerpt if reply in excerpt.casefold() else ''


def is_extractive_qa(prompt, target, max_answer_bytes=48):
    """True for non-yes/no/unknown answers whose normalized text appears inside the prompt passage."""
    ans = single_line(target).casefold()
    if not ans or ans in ('yes', 'no', 'unknown') or len(ans.encode()) > max_answer_bytes:
        return False
    passage = prompt.split('\n', 1)[0].casefold()
    return ans in passage


def prepare_dialogues(data, context, reply_bytes=160, qa_style='role'):
    splits = {split: {task: [] for task in ('prose', 'qa', 'chat')}
              for split in ('train', 'validation', 'heldout')}
    prompt_budget = context - reply_bytes
    splits['train']['qa_excerpt'] = []
    splits['train']['qa_extractive'] = []
    splits['train']['prose_long'] = []

    def group(text):
        return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'big') % 100

    def add_reply(rows, user, reply, history, passage, style='role'):
        prompt = chat_prompt(user, history, passage, prompt_budget, qa_style=style)
        target = reply + '\n'
        if prompt is not None and reply and len(target.encode()) <= reply_bytes:
            rows.append((prompt, target))
            return prompt, target
        return None

    # Read the official development set first so duplicated passages stay reserved.
    seen_stories = set()
    for name in ('coqa-dev-v1.0.json', 'coqa-train-v1.0.json'):
        for item in data[name]['data']:
            story = single_line(item['story'])
            if not story or story in seen_stories:
                continue
            seen_stories.add(story)
            split = 'heldout' if name.startswith('coqa-dev') else (
                'validation' if group(story) < 5 else 'train')
            pools = splits[split]
            pools['prose'].extend(story_continuations(story, context))
            if split == 'train':
                pools['prose_long'].extend(long_story_continuations(story, context))
            history = []
            for q, a in zip(item['questions'], item['answers'], strict=True):
                if q['turn_id'] != a['turn_id']:
                    raise ValueError('Misaligned CoQA turn')
                question, answer = single_line(q['input_text']), single_line(a['input_text'])
                added = add_reply(pools['qa'], question, answer, history, story, style=qa_style)
                if split == 'train':
                    if added is not None and is_extractive_qa(*added):
                        pools['qa_extractive'].append(added)
                    excerpt = training_excerpt(item['story'], a)
                    if excerpt:
                        add_reply(pools['qa_excerpt'], question, answer, history[-1:], excerpt, style=qa_style)
                history.append((question, answer))

    for item in data['self-dialogs.json']:
        turns = []
        for utterance in item['utterances']:
            speaker, text = utterance['speaker'], single_line(utterance['text'])
            if not text:
                continue
            if turns and speaker == turns[-1][0]:
                turns[-1] = (speaker, turns[-1][1] + ' ' + text)
            else:
                turns.append((speaker, text))
        bucket = group(json.dumps(turns, ensure_ascii=False))
        split = 'heldout' if bucket < 5 else 'validation' if bucket < 10 else 'train'
        history = []
        for (speaker, question), (next_speaker, answer) in zip(turns, turns[1:]):
            if (speaker, next_speaker) == ('USER', 'ASSISTANT'):
                add_reply(splits[split]['chat'], question, answer, history, '', style='role')
                history.append((question, answer))
    return splits


def target_batch(rows, device, history_targets=False):
    """Mask prompts and padding. Training can also supervise preceding assistant/answer turns."""
    encoded = [(prompt.encode(), target.encode()) for prompt, target in rows]
    length = max(len(prompt) + len(target) - 1 for prompt, target in encoded)
    x = torch.zeros((len(rows), length), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, (prompt, target) in enumerate(encoded):
        tokens = torch.tensor(list(prompt + target))
        end = len(tokens) - 1
        start = len(prompt) - 1
        x[i, :end] = tokens[:-1]
        if history_targets:
            for reply in re.finditer(rb'(?m)^(?:Assistant|Answer): ([^\n]*\n)', prompt):
                first, last = reply.span(1)
                y[i, first-1:last-1] = tokens[first:last]
        y[i, start:end] = tokens[start+1:]
    return x.to(device), y.to(device)
