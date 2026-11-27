"""Counterfactual fact-to-answer benchmark, 2x2 quad trainer, and Tree causal ablation suite.

Designed to run both as a standalone Python module/CLI and embedded verbatim inside run.ipynb
on Google Colab (where LM and dialogue helpers are already defined in notebook globals).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import re
import time
from typing import NamedTuple

import numpy as np
import torch
from torch.nn import functional as F

if 'chat_prompt' not in globals():
    from dialogue import chat_prompt, prepare_dialogues, read_dialogue_data, single_line, target_batch
if 'LM' not in globals():
    from model import LM


# Broad training entity vocabulary — strictly disjoint from OOV_VOCAB and user_audit_pairs().
TRAIN_VOCAB = {
    'people': [
        'Liam', 'Noah', 'Nora', 'Maya', 'Omar', 'Sara', 'Paul', 'Ruth',
        'Carl', 'Leah', 'Owen', 'Iris', 'Hugo', 'Vera', 'Adam', 'Cora',
        'Alan', 'Beth', 'Cole', 'Dana', 'Eric', 'Faye', 'Gabe', 'Hana',
        'Ivan', 'Jade', 'Karl', 'Lola', 'Marc', 'Nadia', 'Otto', 'Rosa',
        'Seth', 'Tara', 'Umar', 'Veda', 'Wade', 'Yara', 'Zane', 'Alec',
        'Bria', 'Cleo', 'Dean', 'Elia', 'Finn', 'Gina', 'Hank', 'Inez',
        'Joel', 'Kira', 'Leon', 'Mona', 'Neil', 'Olga', 'Pete', 'Rhea',
        'Saul', 'Thea', 'Vince', 'Wren', 'Bram', 'Dora', 'Emil', 'Freya',
    ],
    'colors': [
        'red', 'yellow', 'black', 'white', 'brown', 'silver', 'gold', 'purple',
        'orange', 'amber', 'ivory', 'scarlet', 'teal', 'coral', 'maroon', 'olive',
        'beige', 'bronze', 'copper', 'indigo', 'jade', 'khaki', 'lilac', 'ochre',
        'peach', 'plum', 'ruby', 'rust', 'sage', 'tan', 'taupe', 'azure',
        'cyan', 'ebony', 'hazel', 'lemon', 'mauve', 'mint', 'navy', 'pearl',
    ],
    'objects': [
        'jacket', 'bottle', 'camera', 'ticket', 'letter', 'wallet', 'helmet', 'mirror',
        'candle', 'hammer', 'guitar', 'parcel', 'lantern', 'compass', 'blanket', 'kettle',
        'ribbon', 'scarf', 'shovel', 'teapot', 'violin', 'whistle', 'wrench', 'apron',
        'badge', 'basket', 'beaker', 'binder', 'buckle', 'button', 'carpet', 'chisel',
        'cloak', 'collar', 'crayon', 'dagger', 'flute', 'funnel', 'gloves', 'goblet',
        'handle', 'hatchet', 'locket', 'magnet', 'mallet', 'mitten', 'muffler', 'napkin',
        'paddle', 'pebble', 'pencil', 'pepper', 'pillow', 'pitcher', 'plaque', 'pocket',
        'poster', 'pouch', 'pulley', 'quiver', 'racket', 'rattle', 'saddle', 'sandal',
    ],
    'places': [
        'kitchen', 'garden', 'station', 'museum', 'harbor', 'bakery', 'office', 'clinic',
        'market', 'theater', 'cellar', 'studio', 'archive', 'attic', 'balcony', 'barn',
        'bazaar', 'boathouse', 'brewery', 'canteen', 'castle', 'cavern', 'closet', 'courtyard',
        'depot', 'dormitory', 'factory', 'foundry', 'gallery', 'greenhouse', 'hangar', 'hostel',
        'kennel', 'laundry', 'lobby', 'lounge', 'mill', 'nursery', 'orchard', 'parlor',
        'pavilion', 'pharmacy', 'pier', 'plaza', 'porch', 'quarry', 'shed', 'stable',
        'tavern', 'terrace', 'tower', 'tunnel', 'vault', 'veranda', 'warehouse', 'workshop',
    ],
    'cities': [
        'Paris', 'Rome', 'Cairo', 'Tokyo', 'Seoul', 'Prague', 'Vienna', 'Dublin',
        'Lisbon', 'Athens', 'Madrid', 'Zurich', 'Accra', 'Baku', 'Basel', 'Bilbao',
        'Bonn', 'Brest', 'Bruges', 'Cadiz', 'Cannes', 'Cebu', 'Cusco', 'Dakar',
        'Doha', 'Essen', 'Faro', 'Genoa', 'Ghent', 'Graz', 'Hanoi', 'Hull',
        'Izmir', 'Jena', 'Kobe', 'Krakow', 'Kyoto', 'Lagos', 'Leeds', 'Lille',
        'Linz', 'Lucca', 'Malmo', 'Milan', 'Minsk', 'Modena', 'Munich', 'Nantes',
        'Naples', 'Nice', 'Osaka', 'Padua', 'Pisa', 'Porto', 'Quito', 'Rabat',
        'Reno', 'Riga', 'Rouen', 'Siena', 'Sofia', 'Split', 'Suva', 'Tallinn',
        'Tampa', 'Trier', 'Tunis', 'Turin', 'Ulm', 'Vigo', 'Warsaw', 'York',
    ],
    'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Saturday'],
    'events': [
        'concert', 'lecture', 'briefing', 'workshop', 'hearing', 'summit',
        'auction', 'ballet', 'contest', 'debate', 'exhibit', 'festival',
        'gala', 'match', 'opera', 'parade', 'rally', 'recital',
        'rehearsal', 'review', 'show', 'speech', 'symposium', 'trial',
    ],
}

OOV_VOCAB = {
    'people': ['Mira', 'Anna', 'Emma', 'Luca', 'Nina', 'Ezra', 'Tessa', 'Jonas'],
    'colors': ['blue', 'pink', 'green', 'grey', 'violet', 'crimson'],
    'objects': ['notebook', 'folder', 'atlas', 'map', 'journal', 'satchel'],
    'places': ['library', 'cabin', 'chapel', 'garage', 'temple', 'pantry'],
    'cities': ['Lima', 'Oslo', 'Bern', 'Lyon', 'Cork', 'Perth'],
    'days': ['Friday', 'Sunday'],
    'events': ['meeting', 'dinner', 'seminar', 'forum', 'session', 'banquet'],
}

_OOV_FORBIDDEN = {w.casefold() for vals in OOV_VOCAB.values() for w in vals} | {'desk', 'bag'}

FILLER_SENTENCES = [
    'The weather stayed mild throughout the afternoon while trains ran on schedule.',
    'Several clerks checked the roster near the front entrance before closing time.',
    'A quiet breeze moved across the courtyard as streetlights flickered along the lane.',
    'Two delivery vans parked outside the west gate and unloaded wooden crates.',
    'The corridor remained calm except for footsteps echoing from the upper stairs.',
    'Rain clouds gathered above the ridge, though no storm arrived before evening.',
]

# Long-form filler used by the `far_*` splits (E1).  Every sentence is ~85-95 bytes, so a
# 26-sentence tail pushes the fact more than MIN_MEMORY_DISTANCE_BYTES away from the
# question.  The fact therefore lives in a chunk that the widened layer-7 local attention
# window can no longer reach, and the only route to it is the tree memory.
FAR_FILLER_SENTENCES = [
    'The reading room stayed empty while dust motes drifted past the tall shelves.',
    'A delivery bicycle leaned against the railing, its basket full of parcels.',
    'Footsteps crossed the courtyard and faded towards the eastern staircase.',
    'The kettle whistled twice before the shutters were finally closed.',
    'Several bicycles stood in a row near the entrance, chained to a metal rail.',
    'Clouds moved slowly across the ridge, shading the fields below the hill.',
    'A wooden cart rattled past the gate, leaving a faint smell of fresh hay.',
    'The caretaker swept the entrance twice before locking the gate for the night.',
    'Paperwork sat stacked on the counter, curling at the edges from the damp.',
    'Two neighbours argued gently about a hedge that had grown far too tall.',
    'The power hummed steadily overhead while the lamps flickered into life.',
    'An old bicycle leaned against the wall, its chain loosely around the frame.',
    'The smell of fresh bread drifted from the bakery down towards the river.',
    'Sparrows hopped along the fence, pecking at the crumbs beneath the bench.',
    'The storeroom shelves were restocked twice before the midday shift ended.',
    'A clerk stamped a stack of forms and set them neatly into a wire tray.',
    'Rainwater gathered in the gutter and ran slowly towards the open drain.',
    'The gardener trimmed the hedge on one side and left the tools in the shed.',
    'A brass clock above the doorway ticked steadily through the empty hallway.',
    'Pigeons wheeled above the rooftops and settled on the church tower ledge.',
    'The porter wheeled a trolley of crates across the yard towards the store.',
    'A window on the upper floor was left open and the curtain stirred.',
    'Freshly painted railings gleamed along the path towards the rear gate.',
    'The washing on the line outside rippled gently in the light afternoon air.',
    'Someone had left a wheelbarrow half filled with gravel beside the shed.',
    'The market stalls were packed away and the swept ground smelled of pine.',
    'The caretaker counted the keys twice before locking the store for the night.',
    'A stack of empty crates waited beside the loading dock in the fading light.',
    'The lane behind the bakery was quiet except for a dripping downpipe.',
    'Dust filmed the windowpanes of the empty offices on the upper floor.',
    'The gardener left the wheelbarrow outside the potting shed again.',
    'A stack of empty baskets leaned against the side of the dairy barn.',
    'Wind stirred the dried weeds along the embankment below the railway.',
    'The porter wheeled a trolley of sacks across the yard towards the store.',
    'Lamps flickered on along the covered walkway as the light began to fade.',
    'The damp flagstones of the courtyard shone faintly in the moonlight.',
    'A rope was coiled neatly on the wall beside the heavy oak door.',
]

# Default Tree chunk size in bytes (the model is byte-level, so tokens == bytes).
MEMORY_CHUNK_BYTES = 64
# A fact must be at least this many bytes before the question for the pair to be a
# genuine *memory* test rather than a 1-hop induction test.
MIN_MEMORY_DISTANCE_BYTES = 2048
# Splits that measure binding *generalisation* (unseen nonce keys/values) rather than
# memory retrieval.  They are reported separately and prominently.
GENERALIZATION_SPLITS = ('nonce_qswap',)

_CONSONANTS = 'bcdfghjklmnprstvwz'
_VOWELS = 'aeiou'

ALL_FAMILIES = ('city', 'place', 'object', 'color_fwd', 'color', 'person')
FAMILY_DIRECTION = {
    'city': 'forward',
    'place': 'forward',
    'object': 'forward',
    'color_fwd': 'forward',
    'color': 'reverse',
    'person': 'reverse',
}


def random_nonce(rng: random.Random, capitalize: bool = False) -> str:
    """Generate a pronounceable novel 4-7 byte word guaranteed outside OOV_VOCAB."""
    while True:
        length = rng.choice((4, 5, 6, 7))
        chars = []
        start_vowel = rng.random() < 0.25
        for i in range(length):
            use_vowel = ((i % 2 == 0) == start_vowel)
            chars.append(rng.choice(_VOWELS if use_vowel else _CONSONANTS))
        word = ''.join(chars)
        if word.casefold() not in _OOV_FORBIDDEN:
            return word.capitalize() if capitalize else word


def user_audit_pairs(qa_style: str = 'role', id_prefix: str = '') -> list[dict]:
    """The six diagnostic pairs (clean + distractor) matching the reported failure cases."""
    specs = [
        (
            'color_clean', 'color', False,
            'Mira left her {a} notebook at the library.',
            "What color was Mira's notebook?",
            'blue', 'pink',
        ),
        (
            'color_dist', 'color', True,
            'Mira left her {a} notebook at the library. She put the {b} folder in her bag.',
            "What color was Mira's notebook?",
            'blue', 'pink',
        ),
        (
            'city_clean', 'city', False,
            'The meeting will be held in {a} on Friday.',
            'Where will the meeting be held?',
            'Lima', 'Oslo',
        ),
        (
            'city_dist', 'city', True,
            'The meeting will be held in {a} on Friday. The dinner will be in {b}.',
            'Where will the meeting be held?',
            'Lima', 'Oslo',
        ),
        (
            'person_clean', 'person', False,
            '{a} borrowed the green atlas from the desk.',
            'Who borrowed the atlas?',
            'Anna', 'Emma',
        ),
        (
            'person_dist', 'person', True,
            '{a} borrowed the green atlas from the desk. {b} took the grey map.',
            'Who borrowed the atlas?',
            'Anna', 'Emma',
        ),
    ]
    pairs = []
    for pid, kind, has_dist, tmpl, q, va, vb in specs:
        pa = chat_prompt(q, [], tmpl.format(a=va, b=vb), 1400, qa_style=qa_style)
        pb = chat_prompt(q, [], tmpl.format(a=vb, b=va), 1400, qa_style=qa_style)
        pairs.append(dict(
            id=f'{id_prefix}{pid}', family=kind, direction=FAMILY_DIRECTION[kind], distractor=has_dist,
            prompt_a=pa, prompt_b=pb, answer_a=va, answer_b=vb,
        ))
    return pairs


def user_qswap_pairs(qa_style: str = 'role', id_prefix: str = '') -> list[dict]:
    """Question-swap counterparts of the user's distractor passages (cancels S1/S2 shortcuts)."""
    specs = [
        (
            'color_qswap', 'color', 'reverse',
            'Mira left her blue notebook at the library. Mira left her pink folder in her bag.',
            "What color was Mira's notebook?", "What color was Mira's folder?",
            'blue', 'pink',
        ),
        (
            'city_qswap', 'city', 'forward',
            'The meeting will be held in Lima on Friday. The dinner will be held in Oslo on Sunday.',
            'Where will the meeting be held?', 'Where will the dinner be held?',
            'Lima', 'Oslo',
        ),
        (
            'person_qswap', 'person', 'reverse',
            'Anna borrowed the green atlas from the desk. Emma borrowed the grey map from the desk.',
            'Who borrowed the atlas?', 'Who borrowed the map?',
            'Anna', 'Emma',
        ),
    ]
    return [
        dict(
            id=f'{id_prefix}{pid}', family=fam, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], passage, 1400, qa_style=qa_style),
            prompt_b=chat_prompt(qb, [], passage, 1400, qa_style=qa_style),
            answer_a=va, answer_b=vb,
        )
        for pid, fam, direction, passage, qa, qb, va, vb in specs
    ]


def _draw_distinct_pair(rng: random.Random, pool: list[str], nonce_prob: float, capitalize: bool) -> tuple[str, str]:
    if nonce_prob > 0 and rng.random() < nonce_prob:
        a = random_nonce(rng, capitalize=capitalize)
        b = random_nonce(rng, capitalize=capitalize)
        while b.casefold() == a.casefold() or b[0] == a[0]:
            b = random_nonce(rng, capitalize=capitalize)
        return a, b
    a, b = rng.sample(pool, 2)
    tries = 0
    while b[0] == a[0] and tries < 12:
        a, b = rng.sample(pool, 2)
        tries += 1
    return a, b


def _sample_fillers(rng: random.Random, pool: list[str], count: int) -> list[str]:
    """Sample `count` filler sentences, cycling through the pool when needed.

    The pool is small (6 short sentences for the normal splits, 26 medium ones for
    the `far_*` splits) but the `far_*` splits need ~30 sentences to push the fact
    more than 2048 bytes away from the question.  Sampling WITH replacement (and
    shuffling) keeps the filler text varied while allowing the required count.
    """
    if not count or not pool:
        return []
    out = list(rng.sample(pool, len(pool)))
    while len(out) < count:
        out.extend(rng.sample(pool, len(pool)))
    return out[:count]


def _build_pair(
    rng: random.Random,
    vocab: dict[str, list[str]],
    family: str,
    distractor: bool,
    unseen_template: bool,
    filler_count: int,
    pair_id: str,
    *,
    nonce_value_prob: float = 0.0,
    nonce_key_prob: float = 0.0,
    qa_style: str = 'role',
    filler_pool: list[str] | None = None,
    prompt_budget: int = 1400,
    pre_filler_count: int = 0,
) -> dict:
    """Build a strictly symmetric counterfactual value-swap pair."""
    p1, p2 = _draw_distinct_pair(rng, vocab['people'], nonce_value_prob if family == 'person' else nonce_key_prob, True)
    c1, c2 = _draw_distinct_pair(rng, vocab['colors'], nonce_value_prob if family in ('color', 'color_fwd') else 0.0, False)
    o1, o2 = _draw_distinct_pair(rng, vocab['objects'], nonce_value_prob if family == 'object' else nonce_key_prob, False)
    l1, l2 = _draw_distinct_pair(rng, vocab['places'], nonce_value_prob if family == 'place' else 0.0, False)
    t1, t2 = _draw_distinct_pair(rng, vocab['cities'], nonce_value_prob if family == 'city' else 0.0, True)
    e1, e2 = _draw_distinct_pair(rng, vocab['events'], nonce_key_prob if family == 'city' else 0.0, False)
    day1, day2 = rng.sample(vocab['days'], 2)

    if family == 'color':
        va, vb = c1, c2
        if not unseen_template:
            tpl = rng.randrange(3)
            if tpl == 0:
                s1 = lambda v: f'{p1} left the {v} {o1} at the {l1}.'
                s2 = lambda v: f'{p1} left the {v} {o2} at the {l2}.'
                q = f"What color was {p1}'s {o1}?"
            elif tpl == 1:
                s1 = lambda v: f'{p1} bought a {v} {o1} on {day1}.'
                s2 = lambda v: f'{p1} bought a {v} {o2} on {day2}.'
                q = f'What color is the {o1}?'
            else:
                s1 = lambda v: f'The {v} {o1} sat at the {l1}.'
                s2 = lambda v: f'The {v} {o2} sat at the {l2}.'
                q = f'What color was the {o1}?'
        else:
            s1 = lambda v: f'Inside the {l1}, {p1} noticed a {v} {o1}.'
            s2 = lambda v: f'Inside the {l2}, {p1} noticed a {v} {o2}.'
            q = f'Which color described the {o1}?'
    elif family == 'color_fwd':
        va, vb = c1, c2
        if not unseen_template:
            tpl = rng.randrange(2)
            if tpl == 0:
                s1 = lambda v: f'The {o1} on the table was {v}.'
                s2 = lambda v: f'The {o2} on the table was {v}.'
                q = f'What color was the {o1}?'
            else:
                s1 = lambda v: f'{p1} chose a {v} {o1}.'
                s2 = lambda v: f'{p2} chose a {v} {o1}.'
                q = f"What color was {p1}'s {o1}?"
        else:
            s1 = lambda v: f'The paint on the {o1} looked {v}.'
            s2 = lambda v: f'The paint on the {o2} looked {v}.'
            q = f'Which color was the {o1}?'
    elif family == 'place':
        va, vb = l1, l2
        if not unseen_template:
            tpl = rng.randrange(3)
            if tpl == 0:
                s1 = lambda v: f'{p1} left the {o1} at the {v}.'
                s2 = lambda v: f'{p2} left the {o2} at the {v}.'
                q = f'Where did {p1} leave the {o1}?'
            elif tpl == 1:
                s1 = lambda v: f'The {c1} {o1} is stored in the {v}.'
                s2 = lambda v: f'The {c2} {o2} is stored in the {v}.'
                q = f'Where is the {o1} stored?'
            else:
                s1 = lambda v: f'{p1} found the {o1} inside the {v}.'
                s2 = lambda v: f'{p2} found the {o2} inside the {v}.'
                q = f'Where did {p1} find the {o1}?'
        else:
            s1 = lambda v: f'According to the note, the {o1} remained at the {v}.'
            s2 = lambda v: f'According to the note, the {o2} remained at the {v}.'
            q = f'At which location did the {o1} remain?'
    elif family == 'city':
        va, vb = t1, t2
        if not unseen_template:
            tpl = rng.randrange(3)
            if tpl == 0:
                s1 = lambda v: f'The {e1} will be held in {v} on {day1}.'
                s2 = lambda v: f'The {e2} will be held in {v} on {day2}.'
                q = f'Where will the {e1} be held?'
            elif tpl == 1:
                s1 = lambda v: f'{p1} traveled to {v} for the {e1}.'
                s2 = lambda v: f'{p2} traveled to {v} for the {e2}.'
                q = f'Which city did {p1} travel to?'
            else:
                s1 = lambda v: f'The {e1} opens in {v} on {day1}.'
                s2 = lambda v: f'The {e2} opens in {v} on {day2}.'
                q = f'Where does the {e1} open?'
        else:
            s1 = lambda v: f'Organizers scheduled the {e1} in {v}.'
            s2 = lambda v: f'Organizers scheduled the {e2} in {v}.'
            q = f'Which city is hosting the {e1}?'
    elif family == 'person':
        va, vb = p1, p2
        if not unseen_template:
            tpl = rng.randrange(3)
            if tpl == 0:
                s1 = lambda v: f'{v} borrowed the {c1} {o1} from the {l1}.'
                s2 = lambda v: f'{v} borrowed the {c2} {o2} from the {l2}.'
                q = f'Who borrowed the {o1}?'
            elif tpl == 1:
                s1 = lambda v: f'{v} brought the {o1} to the {e1}.'
                s2 = lambda v: f'{v} brought the {o2} to the {e2}.'
                q = f'Who brought the {o1}?'
            else:
                s1 = lambda v: f'{v} repaired the {c1} {o1} on {day1}.'
                s2 = lambda v: f'{v} repaired the {c2} {o2} on {day2}.'
                q = f'Who repaired the {o1}?'
        else:
            s1 = lambda v: f'The {c1} {o1} was returned to the {l1} by {v}.'
            s2 = lambda v: f'The {c2} {o2} was returned to the {l2} by {v}.'
            q = f'Who returned the {o1}?'
    elif family == 'object':
        va, vb = o1, o2
        if not unseen_template:
            tpl = rng.randrange(2)
            if tpl == 0:
                s1 = lambda v: f'{p1} carried a {v} into the {l1}.'
                s2 = lambda v: f'{p2} carried a {v} into the {l2}.'
                q = f'What did {p1} carry?'
            else:
                s1 = lambda v: f'{p1} packed a {v} before leaving for {t1}.'
                s2 = lambda v: f'{p2} packed a {v} before leaving for {t2}.'
                q = f'What did {p1} pack?'
        else:
            s1 = lambda v: f'Before noon, {p1} placed a {v} inside the {l1}.'
            s2 = lambda v: f'Before noon, {p2} placed a {v} inside the {l2}.'
            q = f'Which item did {p1} place inside the {l1}?'
    else:
        raise ValueError(f'Unknown family: {family}')

    flip_order = distractor and (rng.randrange(2) == 1)
    pool = filler_pool if filler_pool is not None else FILLER_SENTENCES
    fillers = _sample_fillers(rng, pool, filler_count)
    # `pre_filler_count` sentences are inserted BEFORE the fact sentence.  They push
    # the fact further away from the question in the 'role' style, where the fact
    # sentence comes first in the passage.
    pre_fillers = _sample_fillers(rng, pool, pre_filler_count)

    def compose(main_val: str, dist_val: str) -> str:
        sents = (
            [s2(dist_val), s1(main_val)] if flip_order
            else ([s1(main_val), s2(dist_val)] if distractor else [s1(main_val)])
        )
        sents.extend(fillers)
        return ' '.join(pre_fillers + sents)

    pass_a = compose(va, vb)
    pass_b = compose(vb, va)
    return dict(
        id=pair_id,
        family=family,
        direction=FAMILY_DIRECTION[family],
        distractor=distractor,
        filler_count=filler_count,
        unseen_template=unseen_template,
        prompt_a=chat_prompt(q, [], pass_a, prompt_budget, qa_style=qa_style),
        prompt_b=chat_prompt(q, [], pass_b, prompt_budget, qa_style=qa_style),
        answer_a=va,
        answer_b=vb,
    )


def make_quad_as_two_pairs(
    rng: random.Random,
    vocab: dict[str, list[str]],
    family: str = 'city',
    *,
    nonce_val_prob: float = 0.5,
    nonce_key_prob: float = 0.35,
    filler_count: int = 0,
    qa_style: str = 'role',
    filler_pool: list[str] | None = None,
    prompt_budget: int = 1400,
) -> list[dict]:
    """Build a 2x2 counterfactual quad as two question-swap pairs with swapped values.

    Each pair holds the passage fixed while swapping which entity is queried (cancels S1/S2
    positional shortcuts). Across the two pairs, values swap (cancels static word priors).
    """
    p1, p2 = _draw_distinct_pair(rng, vocab['people'], nonce_val_prob if family == 'person' else nonce_key_prob, True)
    c1, c2 = _draw_distinct_pair(rng, vocab['colors'], nonce_val_prob if family in ('color', 'color_fwd') else 0.0, False)
    o1, o2 = _draw_distinct_pair(rng, vocab['objects'], nonce_val_prob if family == 'object' else nonce_key_prob, False)
    l1, l2 = _draw_distinct_pair(rng, vocab['places'], nonce_val_prob if family == 'place' else 0.0, False)
    t1, t2 = _draw_distinct_pair(rng, vocab['cities'], nonce_val_prob if family == 'city' else 0.0, True)
    e1, e2 = _draw_distinct_pair(rng, vocab['events'], nonce_key_prob if family == 'city' else 0.0, False)
    d1, d2 = rng.sample(vocab['days'], 2)

    if family == 'city':
        va, vb = t1, t2
        s = lambda a, b: f'The {e1} will be held in {a} on {d1}. The {e2} will be held in {b} on {d2}.'
        qa, qb = f'Where will the {e1} be held?', f'Where will the {e2} be held?'
    elif family == 'color':
        va, vb = c1, c2
        s = lambda a, b: f'{p1} left her {a} {o1} at the {l1}. {p1} left her {b} {o2} at the {l2}.'
        qa, qb = f"What color was {p1}'s {o1}?", f"What color was {p1}'s {o2}?"
    elif family == 'place':
        va, vb = l1, l2
        s = lambda a, b: f'{p1} left the {o1} at the {a}. {p1} left the {o2} at the {b}.'
        qa, qb = f'Where did {p1} leave the {o1}?', f'Where did {p1} leave the {o2}?'
    elif family == 'person':
        va, vb = p1, p2
        s = lambda a, b: f'{a} borrowed the {c1} {o1} from the {l1}. {b} borrowed the {c2} {o2} from the {l2}.'
        qa, qb = f'Who borrowed the {o1}?', f'Who borrowed the {o2}?'
    elif family == 'object':
        va, vb = o1, o2
        s = lambda a, b: f'{p1} carried a {a} into the {l1}. {p2} carried a {b} into the {l2}.'
        qa, qb = f'What did {p1} carry?', f'What did {p2} carry?'
    elif family == 'color_fwd':
        va, vb = c1, c2
        s = lambda a, b: f'The {o1} on the table was {a}. The {o2} on the table was {b}.'
        qa, qb = f'What color was the {o1}?', f'What color was the {o2}?'
    else:
        raise ValueError(f'Unknown family: {family}')

    pool = filler_pool if filler_pool is not None else FILLER_SENTENCES
    tail = (' ' + ' '.join(_sample_fillers(rng, pool, filler_count))) if filler_count else ''
    p_ab = s(va, vb) + tail
    p_ba = s(vb, va) + tail
    direction = FAMILY_DIRECTION[family]
    return [
        dict(
            id='q1', family=family, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ab, prompt_budget, qa_style=qa_style),
            prompt_b=chat_prompt(qb, [], p_ab, prompt_budget, qa_style=qa_style),
            answer_a=va, answer_b=vb,
        ),
        dict(
            id='q2', family=family, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ba, prompt_budget, qa_style=qa_style),
            prompt_b=chat_prompt(qb, [], p_ba, prompt_budget, qa_style=qa_style),
            answer_a=vb, answer_b=va,
        ),
    ]


def generate_pairs(
    n_pairs: int,
    seed: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ALL_FAMILIES,
    distractor_prob: float = 0.5,
    unseen_template: bool = False,
    filler_choices: tuple[int, ...] = (0,),
    nonce_value_prob: float = 0.0,
    nonce_key_prob: float = 0.0,
    qa_style: str = 'role',
    excluded_keys: set[tuple] | None = None,
    filler_pool: list[str] | None = None,
    prompt_budget: int = 1400,
    id_prefix: str = '',
) -> list[dict]:
    """Generate unique counterfactual value-swap pairs with optional exclusion set."""
    rng = random.Random(seed)
    seen = set(excluded_keys or ())
    pairs = []
    while len(pairs) < n_pairs:
        fam = families[len(pairs) % len(families)]
        dist = rng.random() < distractor_prob
        fillers = rng.choice(filler_choices)
        pair = _build_pair(
            rng, vocab, fam, dist, unseen_template, fillers, f'{id_prefix}p{len(pairs):04d}',
            nonce_value_prob=nonce_value_prob, nonce_key_prob=nonce_key_prob, qa_style=qa_style,
            filler_pool=filler_pool, prompt_budget=prompt_budget,
        )
        key = (pair['prompt_a'], pair['prompt_b'])
        if key in seen:
            continue
        seen.add(key)
        pairs.append(pair)
    return pairs


def generate_qswap_pairs(
    n_pairs: int,
    seed: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ('city', 'color', 'place', 'person'),
    filler_choices: tuple[int, ...] = (0,),
    nonce_val_prob: float = 0.0,
    nonce_key_prob: float = 0.0,
    qa_style: str = 'role',
    filler_pool: list[str] | None = None,
    prompt_budget: int = 1400,
    id_prefix: str = '',
) -> list[dict]:
    """Generate fixed held-out question-swap pairs (identical passage, swapped query)."""
    rng = random.Random(seed)
    pairs = []
    for i in range(n_pairs):
        fam = families[i % len(families)]
        fc = rng.choice(filler_choices)
        pair = make_quad_as_two_pairs(
            rng, vocab, fam, nonce_val_prob=nonce_val_prob, nonce_key_prob=nonce_key_prob,
            filler_count=fc, qa_style=qa_style, filler_pool=filler_pool, prompt_budget=prompt_budget,
        )[0]
        pair['id'] = f'{id_prefix}qs{i:03d}'
        pairs.append(pair)
    return pairs


# ---------------------------------------------------------------------------
# Far (memory-required) splits (E1)
#
# The `far_*` splits put the fact at least MIN_MEMORY_DISTANCE_BYTES before the
# question, so the fact lands in a chunk that the widened layer-7 local attention
# window can no longer reach.  Only the Tree memory can carry the fact forward.
# ---------------------------------------------------------------------------

FAR_FILLER_COUNT = 34
# The `chat_prompt` budget must be large enough to hold the long passage.
FAR_PROMPT_BUDGET = 8192


def generate_far_pairs(
    n_pairs: int,
    seed: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ('city', 'color', 'place', 'person'),
    distractor_prob: float = 0.0,
    filler_count: int = FAR_FILLER_COUNT,
    pre_filler_count: int = 0,
    qa_style: str = 'role',
    min_distance: int = MIN_MEMORY_DISTANCE_BYTES,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
    excluded_keys: set[tuple] | None = None,
    max_attempts: int = 400,
    id_prefix: str = '',
) -> list[dict]:
    """Generate value-swap pairs whose fact is far enough away to require memory.

    Pairs that fail the memory precondition are rejected and regenerated, so the
    returned list always satisfies `check_memory_precondition` (assuming
    `min_distance` is achievable).
    """
    rng = random.Random(seed)
    seen = set(excluded_keys or ())
    pairs: list[dict] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        fam = families[len(pairs) % len(families)]
        dist = rng.random() < distractor_prob
        pair = _build_pair(
            rng, vocab, fam, dist, False, filler_count, f'{id_prefix}far{len(pairs):04d}',
            nonce_value_prob=0.0, nonce_key_prob=0.0, qa_style=qa_style,
            filler_pool=FAR_FILLER_SENTENCES, prompt_budget=FAR_PROMPT_BUDGET,
            pre_filler_count=pre_filler_count,
        )
        key = (pair['prompt_a'], pair['prompt_b'])
        if key in seen:
            continue
        try:
            check_memory_precondition(
                [pair], min_distance=min_distance, chunk_bytes=chunk_bytes,
                split_name='far',
            )
        except ValueError:
            continue
        seen.add(key)
        pairs.append(pair)
    if len(pairs) < n_pairs:
        raise ValueError(
            f'Could only build {len(pairs)}/{n_pairs} memory-valid far pairs after '
            f'{attempts} attempts (min_distance={min_distance}).'
        )
    return pairs


def generate_far_qswap_pairs(
    n_pairs: int,
    seed: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ('city', 'color', 'place', 'person'),
    filler_count: int = FAR_FILLER_COUNT,
    qa_style: str = 'role',
    min_distance: int = MIN_MEMORY_DISTANCE_BYTES,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
    max_attempts: int = 400,
    id_prefix: str = '',
) -> list[dict]:
    """Generate question-swap pairs whose fact is far enough away to require memory."""
    rng = random.Random(seed)
    pairs: list[dict] = []
    attempts = 0
    while len(pairs) < n_pairs and attempts < max_attempts:
        attempts += 1
        fam = families[len(pairs) % len(families)]
        pair = make_quad_as_two_pairs(
            rng, vocab, fam, nonce_val_prob=0.0, nonce_key_prob=0.0,
            filler_count=filler_count, qa_style=qa_style,
            filler_pool=FAR_FILLER_SENTENCES, prompt_budget=FAR_PROMPT_BUDGET,
        )[0]
        pair['id'] = f'{id_prefix}farqs{len(pairs):03d}'
        try:
            check_memory_precondition(
                [pair], min_distance=min_distance, chunk_bytes=chunk_bytes,
                split_name='far_qswap',
            )
        except ValueError:
            continue
        pairs.append(pair)
    if len(pairs) < n_pairs:
        raise ValueError(
            f'Could only build {len(pairs)}/{n_pairs} memory-valid far qswap pairs after '
            f'{attempts} attempts (min_distance={min_distance}).'
        )
    return pairs


def stream_batch(
    rng: random.Random,
    n_pairs: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ALL_FAMILIES,
    distractor_prob: float = 0.75,
    filler_count: int = 0,
    nonce_value_prob: float = 0.5,
    nonce_key_prob: float = 0.25,
    qa_style: str = 'role',
    excluded_keys: set[tuple] | None = None,
) -> list[dict]:
    """Draw a fresh online batch of value-swap pairs."""
    excluded = excluded_keys or set()
    batch = []
    while len(batch) < n_pairs:
        fam = rng.choice(families)
        dist = rng.random() < distractor_prob
        pair = _build_pair(
            rng, vocab, fam, dist, False, filler_count, f's{len(batch)}',
            nonce_value_prob=nonce_value_prob, nonce_key_prob=nonce_key_prob, qa_style=qa_style,
        )
        if (pair['prompt_a'], pair['prompt_b']) in excluded:
            continue
        batch.append(pair)
    return batch


def stream_quad_batch(
    rng: random.Random,
    n_quads: int,
    vocab: dict[str, list[str]],
    *,
    families: tuple[str, ...] = ('city', 'color', 'place', 'person'),
    clean_prob: float = 0.0,
    filler_choices: tuple[int, ...] = (0,),
    nonce_val_prob: float = 0.5,
    nonce_key_prob: float = 0.35,
    qa_style: str = 'role',
) -> list[dict]:
    """Draw n_quads (each = 2 pairs = 4 sequences): either a 2x2 question-swap quad or 2 clean pairs."""
    batch = []
    for _ in range(n_quads):
        fam = rng.choice(families)
        fc = rng.choice(filler_choices)
        if clean_prob > 0 and rng.random() < clean_prob:
            for k in range(2):
                batch.append(_build_pair(
                    rng, vocab, fam, False, False, fc, f'c{k}',
                    nonce_value_prob=nonce_val_prob, nonce_key_prob=0.0, qa_style=qa_style,
                ))
        else:
            batch.extend(make_quad_as_two_pairs(
                rng, vocab, fam, nonce_val_prob=nonce_val_prob, nonce_key_prob=nonce_key_prob,
                filler_count=fc, qa_style=qa_style,
            ))
    return batch


def first_diff_index(a: bytes, b: bytes) -> int:
    """First byte index where distinct byte strings a and b diverge."""
    for i, (ba, bb) in enumerate(zip(a, b)):
        if ba != bb:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    raise ValueError('Answers must be distinct')


# ---------------------------------------------------------------------------
# Memory geometry (E1)
#
# The Tree memory is only load-bearing if the fact lives OUTSIDE the local
# attention window that layer 7 (and layer 3) can see.  These helpers make that
# precondition measurable instead of assumed.
# ---------------------------------------------------------------------------

QUESTION_MARKERS = ('\nQuestion: ', '\nUser: ')
_WORD_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_')


def question_byte_offset(prompt: str) -> int:
    """Byte offset of the question text inside a `chat_prompt` prompt.

    `chat_prompt` renders `...\\n{U_tag}: {question}\\n{A_tag}: ` at the tail, so the
    question is the text after the LAST question marker.
    """
    best = None
    for marker in QUESTION_MARKERS:
        i = prompt.rfind(marker)
        if i >= 0 and (best is None or i > best):
            best = i + len(marker)
    if best is None:
        # Fall back to the final newline (no marker found).
        return len(prompt.encode('utf8'))
    return best


def question_text(prompt: str) -> str:
    """The question string carried by a `chat_prompt` prompt (empty if not found)."""
    start = question_byte_offset(prompt)
    end = prompt.find('\n', start)
    text = prompt[start:end] if end != -1 else prompt[start:]
    return text.strip()


def prompt_qa_style(prompt: str) -> str:
    """Infer the `chat_prompt` style ('role' or 'qa') used to render a prompt."""
    return 'qa' if '\nQuestion: ' in prompt else 'role'


def passage_text(prompt: str) -> str:
    """The `Passage: ` body of a `chat_prompt` prompt (empty string if absent)."""
    if not prompt.startswith('Passage: '):
        return ''
    end = prompt.find('\n', len('Passage: '))
    return prompt[len('Passage: '):end] if end != -1 else prompt[len('Passage: '):]


def find_answer_span(prompt: str, answer: str, *, before: int | None = None) -> tuple[int, int]:
    """Locate `answer` in `prompt` as a whole-word occurrence.

    Returns the (start, end) byte offsets of the LAST whole-word occurrence that
    starts before `before` (default: the end of the prompt).  Taking the last
    occurrence makes every distance measurement CONSERVATIVE: the reported gap is
    the smallest possible gap between the fact and the question, so a pair can
    never pass the memory precondition by accident.

    Raises `ValueError` when the answer does not occur.
    """
    pb = prompt.encode('utf8')
    ab = answer.encode('utf8')
    if not ab:
        raise ValueError('Empty answer')
    limit = len(pb) if before is None else max(0, min(before, len(pb)))
    span = None
    start = pb.find(ab)
    while start != -1:
        end = start + len(ab)
        if end <= limit:
            left_ok = start == 0 or pb[start - 1] not in _WORD_CHARS
            right_ok = end >= len(pb) or pb[end] not in _WORD_CHARS
            if left_ok and right_ok:
                span = (start, end)
        start = pb.find(ab, start + 1)
    if span is None:
        raise ValueError(f'Answer {answer!r} does not occur in the prompt')
    return span


def fact_chunk_offset(
    prompt: str,
    answer: str,
    *,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
) -> tuple[int, int]:
    """Return (fact_chunk_index, question_chunk_index) for `answer` inside `prompt`.

    The chunk index is the byte offset divided by `chunk_bytes` -- the model is
    byte-level, so one byte is one token and the Tree chunk is `chunk_bytes` wide.

    Raises `ValueError` when the answer does not occur in the prompt (the pair is
    then not a fact-binding pair at all).
    """
    qpos = question_byte_offset(prompt)
    start, _ = find_answer_span(prompt, answer, before=qpos)
    return start // chunk_bytes, qpos // chunk_bytes


def fact_question_distance(prompt: str, answer: str) -> tuple[int, int]:
    """Return (gap_bytes, start_to_start_bytes) between the fact and the question.

    `gap_bytes` is the number of bytes between the END of the answer occurrence and
    the START of the question (the conservative distance).  `start_to_start_bytes`
    is the distance between the START of the answer and the START of the question.
    """
    qpos = question_byte_offset(prompt)
    start, end = find_answer_span(prompt, answer, before=qpos)
    return max(0, qpos - end), max(0, qpos - start)



def check_memory_precondition(
    pairs: list[dict],
    *,
    min_distance: int = MIN_MEMORY_DISTANCE_BYTES,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
    split_name: str = 'split',
) -> dict:
    """Assert that every pair genuinely requires the Tree memory.

    A pair requires memory when, for each of its two (prompt, answer) bindings:
      * the fact occurrence sits in a STRICTLY EARLIER chunk than the question
        (so the local attention window cannot reach it), and
      * the fact is at least `min_distance` bytes before the question.

    Raises `ValueError` on the first violation and returns a small stats dict on
    success.
    """
    if not pairs:
        raise ValueError(f'Memory precondition for {split_name!r}: no pairs to check')
    worst_gap = None
    checked = 0
    for p in pairs:
        for prompt_key, answer_key in (('prompt_a', 'answer_a'), ('prompt_b', 'answer_b')):
            prompt = p[prompt_key]
            answer = p[answer_key]
            fact_chunk, q_chunk = fact_chunk_offset(prompt, answer, chunk_bytes=chunk_bytes)
            gap, start_to_start = fact_question_distance(prompt, answer)
            if fact_chunk >= q_chunk:
                raise ValueError(
                    f'Memory precondition violated for {split_name!r}/{p.get("id")}: '
                    f'the fact is in chunk {fact_chunk} and the question is in chunk {q_chunk} '
                    f'-- the fact is inside the local attention window, so the tree '
                    f'cannot be what produces the answer.'
                )
            if gap < min_distance:
                raise ValueError(
                    f'Memory precondition violated for {split_name!r}/{p.get("id")}: '
                    f'the fact is only {gap} bytes before the question '
                    f'(minimum {min_distance}).'
                )
            worst_gap = gap if worst_gap is None else min(worst_gap, gap)
            checked += 1
    return dict(
        split=split_name,
        pairs=len(pairs),
        bindings_checked=checked,
        min_gap_bytes=int(worst_gap),
        min_distance_bytes=int(min_distance),
        chunk_bytes=int(chunk_bytes),
    )


def pair_batch_tensors(pairs: list[dict], device: torch.device | str):
    """Pack N counterfactual pairs into 2N sequences [A0, B0, A1, B1, ...] for single-pass scoring/training."""
    seqs = []
    prompt_lens = []
    target_lens = []
    diff_info = []
    for pair in pairs:
        ta = (pair['answer_a'] + '\n').encode('utf8')
        tb = (pair['answer_b'] + '\n').encode('utf8')
        pa = pair['prompt_a'].encode('utf8')
        pb = pair['prompt_b'].encode('utf8')
        k = first_diff_index(ta, tb)
        row_a = len(seqs)
        seqs.append(pa + ta)
        prompt_lens.append(len(pa))
        target_lens.append(len(ta))
        diff_info.append((row_a, len(pa) - 1 + k, ta[k], tb[k]))
        row_b = len(seqs)
        seqs.append(pb + tb)
        prompt_lens.append(len(pb))
        target_lens.append(len(tb))
        diff_info.append((row_b, len(pb) - 1 + k, tb[k], ta[k]))

    max_len = max(len(s) - 1 for s in seqs)
    x = torch.zeros((len(seqs), max_len), dtype=torch.long)
    y = torch.full_like(x, -100)
    w = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    for i, (seq, plen, tlen) in enumerate(zip(seqs, prompt_lens, target_lens)):
        toks = torch.tensor(list(seq), dtype=torch.long)
        end = len(seq) - 1
        start = plen - 1
        x[i, :end] = toks[:-1]
        y[i, start:end] = toks[start + 1:]
        w[i, start:end] = torch.arange(1, tlen + 1, dtype=torch.float32)
    diff_tensor = torch.tensor(diff_info, dtype=torch.long)
    return x.to(device), y.to(device), w.to(device), diff_tensor.to(device)


def fact_switch_loss(
    logits: torch.Tensor,
    y: torch.Tensor,
    byte_pos: torch.Tensor,
    diff_info: torch.Tensor,
    *,
    mode: str = 'contrastive',
    first_byte_weight: float = 2.0,
    margin_target: float = 3.0,
    contrastive_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute plain NLL (Phase 1 copy lock-in) or counterfactual contrastive loss (Phase 2 routing)."""
    nll_tok = F.cross_entropy(logits.float().transpose(1, 2), y, ignore_index=-100, reduction='none')
    valid = y != -100
    plain_nll = nll_tok[valid].mean()

    rows, pos, true_b, foil_b = diff_info.unbind(1)
    div_logits = logits.float()[rows, pos]
    true_z = div_logits.gather(1, true_b[:, None]).squeeze(1)
    foil_z = div_logits.gather(1, foil_b[:, None]).squeeze(1)
    div_margin = true_z - foil_z
    pair_sens = div_margin.reshape(-1, 2).sum(1)

    if mode == 'plain_nll':
        loss = plain_nll
    elif mode == 'weighted_nll':
        weights = torch.where(byte_pos == 1, first_byte_weight, 1.0) * valid.float()
        loss = (nll_tok * weights).sum() / weights.sum().clamp_min(1.0)
    elif mode == 'contrastive':
        weights = torch.where(byte_pos == 1, first_byte_weight, 1.0) * valid.float()
        weighted_nll = (nll_tok * weights).sum() / weights.sum().clamp_min(1.0)
        indiv_hinge = F.softplus(margin_target - div_margin).mean()
        swap_hinge = F.softplus(2.0 * margin_target - pair_sens).mean()
        loss = weighted_nll + contrastive_weight * (0.5 * indiv_hinge + 0.5 * swap_hinge)
    else:
        raise ValueError(f'Unknown loss mode: {mode}')

    stats = dict(
        loss=float(loss.detach()),
        plain_nll=float(plain_nll.detach()),
        mean_div_margin=float(div_margin.detach().mean()),
        mean_pair_sens=float(pair_sens.detach().mean()),
        div_pair_acc=float((div_margin.detach().reshape(-1, 2) > 0).all(1).float().mean()),
    )
    return loss, stats


# ---------------------------------------------------------------------------
# Memory ablation modes (E2)
#
# 'permute' is a genuine, seeded, content-level permutation of the Tree source
# chunks.  'shuffle' is kept only as a deprecated alias.  Because `model.py` is
# being upgraded concurrently, every use goes through `resolve_tree_mode()` which
# feature-detects the supported modes and records a fallback note.
# ---------------------------------------------------------------------------

PERMUTE_MODE = 'permute'
LEGACY_PERMUTE_MODE = 'shuffle'
TREE_MODES = ('on', 'off', PERMUTE_MODE, LEGACY_PERMUTE_MODE)


def supported_tree_modes(model: LM | None = None) -> set[str]:
    """The Tree modes the given model actually supports."""
    modes = {'on', 'off'}
    if model is None:
        return modes
    tree = getattr(model, 'tree', None)
    if tree is None:
        return modes
    try:
        src = inspect.getsource(type(tree).forward)
    except (OSError, TypeError):
        src = ''
    if f"'{PERMUTE_MODE}'" in src or f'"{PERMUTE_MODE}"' in src:
        modes.add(PERMUTE_MODE)
    if f"'{LEGACY_PERMUTE_MODE}'" in src or f'"{LEGACY_PERMUTE_MODE}"' in src:
        modes.add(LEGACY_PERMUTE_MODE)
    return modes


def resolve_tree_mode(mode: str, model: LM | None = None) -> tuple[str, str | None]:
    """Map a requested Tree mode onto a mode the model supports.

    Returns (resolved_mode, note).  `note` is None when the requested mode was used
    as-is, and otherwise explains the fallback (e.g. the model does not yet
    implement 'permute' so the legacy 'shuffle' alias was used).
    """
    if mode not in TREE_MODES:
        raise ValueError(f'Unknown Tree mode: {mode!r}')
    if mode in ('on', 'off'):
        return mode, None
    supported = supported_tree_modes(model)
    if mode == PERMUTE_MODE:
        if PERMUTE_MODE in supported:
            return PERMUTE_MODE, None
        if LEGACY_PERMUTE_MODE in supported:
            return LEGACY_PERMUTE_MODE, (
                "mode='permute' is not implemented by this model; fell back to the "
                "deprecated mode='shuffle' alias"
            )
        raise ValueError(
            "This model supports neither mode='permute' nor mode='shuffle'; "
            "the memory ablation cannot be run."
        )
    # Legacy 'shuffle' request.
    if LEGACY_PERMUTE_MODE in supported:
        return LEGACY_PERMUTE_MODE, "mode='shuffle' is deprecated; use mode='permute'"
    if PERMUTE_MODE in supported:
        return PERMUTE_MODE, "mode='shuffle' is deprecated; mapped to mode='permute'"
    raise ValueError("This model supports neither mode='permute' nor mode='shuffle'")


def permutation_seed_for(pair_id: str, seed: int) -> int:
    """A stable per-pair permutation seed derived from the base seed and the pair id.

    Deterministic across runs and machines (unlike `hash()`, which is salted).
    """
    digest = hashlib.sha256(f'{seed}:{pair_id}'.encode('utf8')).digest()
    return int.from_bytes(digest[:4], 'big')


def donor_signature(source: torch.Tensor, row: int = 0) -> str:
    """A short, human-auditable signature of a Tree source activation row.

    The signature identifies the DONOR CONTENT that was fed to the tree, so the
    permutation ablation can be inspected instead of merely trusted.

    The WHOLE row is hashed, not just its head: the tree memory is a stack of
    per-query-chunk frontiers, and the first frontier (`C = 1`) is a single
    leaf which is IDENTICAL under every permutation.  Hashing only the head
    would therefore report "no difference" for every pair.
    """
    flat = source[row].detach().float().reshape(-1)
    if flat.numel() == 0:
        return 'empty'
    return 'sha1:' + hashlib.sha1(flat.cpu().numpy().tobytes()).hexdigest()[:16]


def _tree_memory_signature(model: LM, ids: torch.Tensor, tree, mode: str, device) -> str:
    """A content signature of the memory the Tree BUILDS from `ids` under `mode`.

    The signature has to be taken at a point where the ablation is VISIBLE.
    `Tree.write` is the wrong place: it produces the level-0 leaves BEFORE
    the permutation is applied (the permutation happens inside `Tree.build`,
    on the already-written leaves), so wrapping `write` yields the SAME
    signature for 'on' and 'permute' and the audit would always report
    `donor_differs = False`.

    `Tree.read` receives the BUILT memory -- i.e. after the permutation has
    been applied -- so wrapping `read` captures exactly the content the
    model actually read.  `read` is a plain method (not an `nn.Module`), so
    it is wrapped directly rather than hooked.
    """
    captured: list[torch.Tensor] = []
    original = tree.read

    def wrapper(x, nodes, mask=None):
        captured.append(nodes.detach().clone())
        return original(x, nodes, mask)

    tree.read = wrapper
    try:
        with torch.no_grad():
            model(ids, mode=mode)
    finally:
        tree.read = original
    if not captured:
        return 'none'
    return donor_signature(captured[0], 0)


@torch.inference_mode()
def _score_prompt_endings(
    model: LM,
    prompts: list[str],
    endings: list[str],
    device: torch.device | str,
    mode: str = 'on',
    batch_size: int = 16,
) -> tuple[list[float], list[float], list[list[float]]]:
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    totals, per_bytes, details = [], [], []
    resolved, _ = resolve_tree_mode(mode, model)
    permuted = resolved in (PERMUTE_MODE, LEGACY_PERMUTE_MODE)
    eff_bs = max(2, batch_size) if permuted else batch_size
    for start in range(0, len(prompts), eff_bs):
        ps = prompts[start:start + eff_bs]
        es = endings[start:start + eff_bs]
        if permuted and len(ps) == 1:
            ps = ps + [prompts[0]]
            es = es + [endings[0]]
            drop_last = True
        else:
            drop_last = False
        encoded = [(p.encode('utf8'), e.encode('utf8')) for p, e in zip(ps, es)]
        max_len = max(len(p) + len(e) - 1 for p, e in encoded)
        ids = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, (pb, eb) in enumerate(encoded):
            seq = pb + eb
            ids[i, :len(seq) - 1] = torch.tensor(list(seq[:-1]), dtype=torch.long)
        ids = ids.to(device)
        with amp:
            logits = model(ids, mode=resolved).float()
        logp = logits.log_softmax(-1)
        limit = len(encoded) - (1 if drop_last else 0)
        for i in range(limit):
            pb, eb = encoded[i]
            p0 = len(pb) - 1
            toks = torch.tensor(list(eb), device=device, dtype=torch.long)
            vals = logp[i, p0:p0 + len(eb)].gather(1, toks[:, None]).squeeze(1).tolist()
            totals.append(float(sum(vals)))
            per_bytes.append(float(sum(vals) / len(vals)))
            details.append(vals)
    return totals, per_bytes, details


@torch.inference_mode()
def greedy_generate(
    model: LM,
    prompts: list[str],
    device: torch.device | str,
    *,
    max_bytes: int = 20,
    mode: str = 'on',
    batch_size: int = 16,
) -> list[str]:
    """Batched greedy decode from each prompt up to newline or max_bytes."""
    if not prompts:
        return []
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    resolved, _ = resolve_tree_mode(mode, model)
    permuted = resolved in (PERMUTE_MODE, LEGACY_PERMUTE_MODE)
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        if permuted and len(chunk_prompts) < 2:
            chunk_prompts = chunk_prompts + [prompts[0]]
            drop_last = True
        else:
            drop_last = False
        bufs = [list(p.encode('utf8')) for p in chunk_prompts]
        prompt_lens = [len(b) for b in bufs]
        done = [False] * len(bufs)
        for _ in range(max_bytes):
            if all(done):
                break
            max_len = max(len(b) for b in bufs)
            ids = torch.zeros((len(bufs), max_len), dtype=torch.long)
            last_pos = torch.tensor([len(b) - 1 for b in bufs], dtype=torch.long)
            for i, b in enumerate(bufs):
                ids[i, :len(b)] = torch.tensor(b, dtype=torch.long)
            ids = ids.to(device)
            with amp:
                logits = model(ids, mode=resolved, positions=last_pos.to(device)[:, None]).squeeze(1).float()
            nxt = logits.argmax(-1).tolist()
            for i, byte in enumerate(nxt):
                if not done[i]:
                    bufs[i].append(int(byte))
                    if byte == 10:
                        done[i] = True
        limit = len(bufs) - (1 if drop_last else 0)
        for i in range(limit):
            gen = bytes(bufs[i][prompt_lens[i]:]).decode('utf8', errors='replace')
            outputs.append(gen.split('\n', 1)[0].strip())
    return outputs


def normalize_answer(text: str) -> str:
    return ' '.join(re.findall(r'\w+', text.casefold()))


def build_unrelated_control_prompts(pairs: list[dict], *, seed: int = 0) -> list[str]:
    """Build a THIRD prompt per pair that asks the SAME question about an UNRELATED passage.

    This is the `unrelated_key_acc` control (E5): if the model scores the gold answer
    highly even when the passage says nothing about the question, then it is not
    reading the fact at all -- it is preferring a plausible-sounding answer.
    """
    rng = random.Random(seed + 7717)
    prompts = []
    for i, p in enumerate(pairs):
        question = question_text(p['prompt_a'])
        style = prompt_qa_style(p['prompt_a'])
        # An unrelated passage: no answer words, no shared entities.
        filler = rng.choice(FAR_FILLER_SENTENCES)
        distractor_passage = (
            f'{filler} Nothing of note was recorded during the '
            f'{rng.choice(ALL_FAMILIES)} and no further details were noted.'
        )
        prompt = chat_prompt(question, [], distractor_passage, 1400, qa_style=style)
        prompts.append(prompt if prompt is not None else p['prompt_a'])
    return prompts


@torch.inference_mode()
def _first_byte_decisions(
    model: LM,
    prompts: list[str],
    endings: list[str],
    device: torch.device | str,
    mode: str = 'on',
    batch_size: int = 16,
    offsets: list[int] | None = None,
) -> list[dict]:
    """Read the model's decision at the answer divergence position.

    For each (prompt, ending) we take the FULL logit row at the position that predicts
    the answer's divergence byte and record the argmax over the whole 256-byte
    vocabulary.  `offsets[i]` is the byte offset of the divergence byte INSIDE the
    ending (default 0, i.e. the very first answer byte).

    Returns one dict per prompt with:
      argmax_byte : int    -- the model's argmax byte at the divergence position
      logp_gold    : float  -- log-prob of the gold byte there
      logp_foil    : float  -- log-prob of the foil byte there
    """
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    resolved, _ = resolve_tree_mode(mode, model)
    permuted = resolved in (PERMUTE_MODE, LEGACY_PERMUTE_MODE)
    if offsets is None:
        offsets = [0] * len(prompts)
    out: list[dict] = []
    for start in range(0, len(prompts), batch_size):
        ps = prompts[start:start + batch_size]
        es = endings[start:start + batch_size]
        os_ = offsets[start:start + batch_size]
        # A permutation needs at least two independent donors, so a single-item batch
        # is padded with a duplicate and the duplicate's result is discarded.
        drop_last = False
        if permuted and len(ps) == 1:
            ps = ps + [prompts[0]]
            es = es + [endings[0]]
            os_ = os_ + [os_[0]]
            drop_last = True
        encoded = [(p.encode('utf8'), e.encode('utf8')) for p, e in zip(ps, es)]
        max_len = max(len(p) + len(e) - 1 for p, e in encoded)
        ids = torch.zeros((len(encoded), max_len), dtype=torch.long)
        for i, (pb, eb) in enumerate(encoded):
            seq = pb + eb
            ids[i, :len(seq) - 1] = torch.tensor(list(seq[:-1]), dtype=torch.long)
        ids = ids.to(device)
        with amp:
            logits = model(ids, mode=resolved).float()
        logp = logits.log_softmax(-1)
        limit = len(encoded) - (1 if drop_last else 0)
        for i in range(limit):
            pb, eb = encoded[i]
            k = min(os_[i], len(eb) - 1)
            # ids index j predicts seq byte j+1; we want to predict eb[k], which lives
            # at absolute index len(pb)+k, so the ids index is len(pb)+k-1.
            pos = len(pb) + k - 1
            row = logp[i, pos]
            out.append(dict(
                argmax_byte=int(row.argmax().item()),
                logp_at_pos=row,
                position=pos,
                offset=k,
            ))
    return out


@torch.inference_mode()
def evaluate_pairs(
    model: LM,
    pairs: list[dict],
    device: torch.device | str,
    *,
    mode: str = 'on',
    do_generate: bool = True,
    gen_limit: int | None = None,
    batch_size: int = 16,
    unrelated_control: bool = True,
    permutation_seed: int = 0,
) -> dict:
    """Evaluate forced-choice pair flipping, first-byte sensitivity, direction breakdown, and greedy generation.

    PRIMARY METRIC (E5): `first_byte_acc` -- at the answer divergence position, is the
    model's ARGMAX the correct first byte?  This is a decision metric: it asks "does
    the model commit to the right answer's first byte", not "how much probability mass
    did the gold prefix collect".  It is therefore immune to the teacher-forcing
    artefact that most of a gold answer's log-prob comes from trivially predictable
    bytes.

    `sum_sens_diagnostic` (and the legacy `mean_sens`) are SUM-over-bytes diagnostics,
    NOT memory metrics; they mix units when the two answers differ in length.
    `fb_mean_sens` is the per-byte-mean version of the same diagnostic and is kept for
    backwards compatibility with existing callers.
    """
    prompts_a = [p['prompt_a'] for p in pairs]
    prompts_b = [p['prompt_b'] for p in pairs]
    ans_a = [p['answer_a'] + '\n' for p in pairs]
    ans_b = [p['answer_b'] + '\n' for p in pairs]

    s_aa, pb_aa, det_aa = _score_prompt_endings(model, prompts_a, ans_a, device, mode=mode, batch_size=batch_size)
    s_ab, pb_ab, det_ab = _score_prompt_endings(model, prompts_a, ans_b, device, mode=mode, batch_size=batch_size)
    s_ba, pb_ba, det_ba = _score_prompt_endings(model, prompts_b, ans_a, device, mode=mode, batch_size=batch_size)
    s_bb, pb_bb, det_bb = _score_prompt_endings(model, prompts_b, ans_b, device, mode=mode, batch_size=batch_size)

    # --- PRIMARY: the model's first-byte DECISION at the divergence position ---
    # For prompt_a the divergence position is the first byte where answer_a and
    # answer_b differ; the same position for prompt_b.  We read the full logit row
    # at that position and take the argmax over the whole vocabulary.
    div_positions: list[int] = []
    gold_bytes: list[int] = []
    foil_bytes: list[int] = []
    for i, p in enumerate(pairs):
        ta = ans_a[i].encode('utf8')
        tb = ans_b[i].encode('utf8')
        k = first_diff_index(ta, tb)
        div_positions.append(k)
        gold_bytes.append(ta[k])
        foil_bytes.append(tb[k])

    fb_dec_a = _first_byte_decisions(
        model, prompts_a, ans_a, device, mode=mode, batch_size=batch_size, offsets=div_positions,
    )
    fb_dec_b = _first_byte_decisions(
        model, prompts_b, ans_b, device, mode=mode, batch_size=batch_size, offsets=div_positions,
    )

    # --- unrelated-passage control (E5) -------------------------------------
    unrelated_first_byte_acc = None
    unrelated_rows: list[dict] = []
    if unrelated_control and pairs:
        unrelated_prompts = build_unrelated_control_prompts(pairs, seed=permutation_seed)
        # Score the GOLD ending on the unrelated passage and check whether the model
        # still commits to the gold byte.  A model that reads the fact should NOT.
        u_dec = _first_byte_decisions(
            model, unrelated_prompts, ans_a, device, mode=mode, batch_size=batch_size,
            offsets=div_positions,
        )
        for i, p in enumerate(pairs):
            k = div_positions[i]
            gold_byte = gold_bytes[i]
            foil_byte = foil_bytes[i]
            row = u_dec[i]
            argmax = row['argmax_byte']
            unrelated_rows.append(dict(
                id=p['id'],
                argmax_byte=int(argmax),
                gold_first_byte=int(gold_byte),
                foil_first_byte=int(foil_byte),
                logp_gold=float(row['logp_at_pos'][gold_byte].item()),
                logp_foil=float(row['logp_at_pos'][foil_byte].item()),
                # True when the model still prefers the gold byte even though the
                # passage says nothing about the question.
                prefers_gold=bool(argmax == gold_byte),
            ))
        unrelated_first_byte_acc = sum(r['prefers_gold'] for r in unrelated_rows) / len(unrelated_rows)

    gen_n = len(pairs) if gen_limit is None else min(len(pairs), gen_limit)
    gen_a = greedy_generate(model, prompts_a[:gen_n], device, mode=mode, batch_size=batch_size) if do_generate else None
    gen_b = greedy_generate(model, prompts_b[:gen_n], device, mode=mode, batch_size=batch_size) if do_generate else None

    rows = []
    for i, p in enumerate(pairs):
        ta = ans_a[i].encode('utf8')
        tb = ans_b[i].encode('utf8')
        k = first_diff_index(ta, tb)
        same_len = len(ta) == len(tb)
        dA = (s_aa[i] - s_ab[i]) if same_len else (pb_aa[i] - pb_ab[i])
        dB = (s_bb[i] - s_ba[i]) if same_len else (pb_bb[i] - pb_ba[i])
        fb_dA = det_aa[i][k] - det_ab[i][k]
        fb_dB = det_bb[i][k] - det_ba[i][k]
        # --- PRIMARY: first-byte decision (argmax over the whole vocabulary) ---
        argmax_a = fb_dec_a[i]['argmax_byte']
        argmax_b = fb_dec_b[i]['argmax_byte']
        # prompt_a asks about answer_a, prompt_b asks about answer_b.
        fb_a_correct = bool(argmax_a == gold_bytes[i])
        fb_b_correct = bool(argmax_b == foil_bytes[i])
        row = dict(
            id=p['id'],
            family=p['family'],
            direction=p.get('direction', FAMILY_DIRECTION.get(p['family'], 'unknown')),
            distractor=p['distractor'],
            answer_a=p['answer_a'],
            answer_b=p['answer_b'],
            dA=dA,
            dB=dB,
            sens=dA + dB,
            abs_bias=abs(dA - dB) / 2.0,
            both_correct=(dA > 0) and (dB > 0),
            indiv_correct=int(dA > 0) + int(dB > 0),
            same_choice=((dA > 0) != (dB > 0)),
            fb_dA=fb_dA,
            fb_dB=fb_dB,
            fb_sens=fb_dA + fb_dB,
            fb_abs_bias=abs(fb_dA - fb_dB) / 2.0,
            fb_both_correct=(fb_dA > 0) and (fb_dB > 0),
            # PRIMARY metric, per-binding.
            first_byte_a_correct=fb_a_correct,
            first_byte_b_correct=fb_b_correct,
            first_byte_pair_correct=bool(fb_a_correct and fb_b_correct),
            first_byte_argmax_a=int(argmax_a),
            first_byte_argmax_b=int(argmax_b),
        )
        if do_generate and i < gen_n:
            ga, gb = gen_a[i], gen_b[i]
            ea = normalize_answer(ga) == normalize_answer(p['answer_a'])
            eb = normalize_answer(gb) == normalize_answer(p['answer_b'])
            row.update(gen_a=ga, gen_b=gb, gen_a_exact=ea, gen_b_exact=eb, gen_both_exact=ea and eb)
        rows.append(row)

    n = len(rows)
    by_dir = {}
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['direction']].append(r)
    for dname, rs in grouped.items():
        entry = dict(
            pairs=len(rs),
            pair_acc=sum(r['both_correct'] for r in rs) / len(rs),
            fb_mean_sens=float(np.mean([r['fb_sens'] for r in rs])),
            # PRIMARY metric, per direction.
            first_byte_acc=sum(r['first_byte_pair_correct'] for r in rs) / len(rs),
            first_byte_indiv_acc=sum(int(r['first_byte_a_correct']) + int(r['first_byte_b_correct']) for r in rs) / (2 * len(rs)),
        )
        grs = [r for r in rs if 'gen_both_exact' in r]
        if grs:
            entry['gen_pair_exact'] = sum(r['gen_both_exact'] for r in grs) / len(grs)
            entry['gen_indiv_exact'] = sum(int(r['gen_a_exact']) + int(r['gen_b_exact']) for r in grs) / (2 * len(grs))
        by_dir[dname] = entry

    summary = dict(
        pairs=n,
        mode=mode,
        pair_acc=sum(r['both_correct'] for r in rows) / n,
        indiv_acc=sum(r['indiv_correct'] for r in rows) / (2 * n),
        same_choice_rate=sum(r['same_choice'] for r in rows) / n,
        # Legacy keys kept for backwards compatibility.  `mean_sens` is a SUM over
        # the answer bytes -- it mixes units when the two answers differ in length
        # and is dominated by teacher-forced bytes.  It is a DIAGNOSTIC, not the
        # headline metric.
        mean_sens=float(np.mean([r['sens'] for r in rows])),
        mean_abs_bias=float(np.mean([r['abs_bias'] for r in rows])),
        fb_pair_acc=sum(r['fb_both_correct'] for r in rows) / n,
        fb_mean_sens=float(np.mean([r['fb_sens'] for r in rows])),
        fb_mean_abs_bias=float(np.mean([r['fb_abs_bias'] for r in rows])),
        # ---- NEW: clearly-labelled primary metrics (E5) --------------------
        first_byte_acc=sum(r['first_byte_pair_correct'] for r in rows) / n,
        first_byte_indiv_acc=sum(int(r['first_byte_a_correct']) + int(r['first_byte_b_correct']) for r in rows) / (2 * n),
        first_byte_a_acc=sum(r['first_byte_a_correct'] for r in rows) / n,
        first_byte_b_acc=sum(r['first_byte_b_correct'] for r in rows) / n,
        sum_sens_diagnostic=float(np.mean([r['sens'] for r in rows])),
        fb_sum_sens_diagnostic=float(np.mean([r['fb_sens'] for r in rows])),
        by_direction=by_dir,
        rows=rows,
    )
    if unrelated_control and unrelated_rows:
        summary['unrelated_key_acc'] = unrelated_first_byte_acc
        summary['unrelated_control_rows'] = unrelated_rows
    if do_generate and gen_n > 0:
        gen_rows = rows[:gen_n]
        summary['gen_pairs_evaluated'] = gen_n
        summary['gen_indiv_exact'] = sum(int(r['gen_a_exact']) + int(r['gen_b_exact']) for r in gen_rows) / (2 * gen_n)
        summary['gen_pair_exact'] = sum(r['gen_both_exact'] for r in gen_rows) / gen_n
    return summary


class SourceCapture:
    """Captures (and optionally overrides) the `source` activation the Tree receives.

    `LM.forward` computes `source = x` after layer 3 and hands it to `Tree.forward`.
    We intercept the Tree's `forward` with a pre-hook so we can (a) record the
    donor activations and (b) patch them with activations captured from a different
    run -- a genuine counterfactual activation patch rather than a relabelled
    ablation.
    """

    def __init__(self, tree):
        self.tree = tree
        self.captured: torch.Tensor | None = None
        self.override: torch.Tensor | None = None
        self._handle = None

    def _pre_hook(self, module, args):
        if len(args) < 2:
            return None
        source = args[1]
        if self.captured is None:
            self.captured = source
        if self.override is not None:
            return (args[0], self.override) + tuple(args[2:])
        return None

    def __enter__(self):
        self._handle = self.tree.register_forward_pre_hook(self._pre_hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


@contextmanager
def _captured_source(tree):
    """Context manager yielding a `SourceCapture` bound to `tree`."""
    cap = SourceCapture(tree)
    with cap:
        yield cap


@torch.inference_mode()
def _run_with_source(
    model: LM,
    ids: torch.Tensor,
    tree,
    *,
    override: torch.Tensor | None = None,
    capture: bool = True,
    mode: str = 'on',
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the model, optionally capturing / overriding the Tree source activations.

    Returns (logits, captured_source).
    """
    with _captured_source(tree) as cap:
        cap.override = override
        logits = model(ids, mode=mode)
        captured = cap.captured if capture else None
    return logits, captured


def _fit_source_length(source: torch.Tensor, length: int) -> torch.Tensor:
    """Trim or zero-pad a captured source tensor to exactly `length` positions.

    The Tree requires `source` and `x` to have the same sequence length, so a donor
    captured from a differently-sized sequence has to be fitted before it can be
    patched in.  Trimming keeps the donor's earliest content (the fact usually lives
    near the start of the passage), and zero-padding keeps the shape honest.
    """
    if source.size(1) == length:
        return source
    if source.size(1) > length:
        return source[:, :length].contiguous()
    pad = length - source.size(1)
    return F.pad(source, (0, 0, 0, pad)).contiguous()


@torch.inference_mode()
def evaluate_tree_counterfactual_swap(
    model: LM,
    pairs: list[dict],
    device: torch.device | str,
    *,
    unrelated_pairs: list[dict] | None = None,
) -> dict:
    """REAL counterfactual activation patch on the Tree memory (E3).

    For each pair we run the prompt_a sequence and the prompt_b sequence, capture the
    Tree `source` activations (the layer-3 output) of each run, then re-run with the
    two source activations SWAPPED between the two runs.  We report how the model's
    first-byte preference at the answer divergence position changes.

    We also run a CONTROL: the same patch, but with the source activations taken from
    an UNRELATED pair (a different family).  The control isolates the FACT-SPECIFIC
    effect from generic damage caused by feeding the tree a mismatched memory.

    The old implementation of this function did NOT swap anything: it compared
    `mode='on'` against `mode='shuffle'` and relabelled the difference as a causal
    swap.  That is exactly the permutation ablation, so the "swap" number was not
    independent evidence.  This version performs a genuine activation patch.
    """
    model.eval()
    tree = getattr(model, 'tree', None)
    if tree is None:
        return dict(
            pairs=0,
            available=False,
            note='Model has no Tree; the counterfactual activation patch cannot run.',
        )
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    if unrelated_pairs is None:
        unrelated_pairs = _pick_unrelated_pairs(pairs)

    rows = []
    for idx, p in enumerate(pairs):
        ta = (p['answer_a'] + '\n').encode('utf8')
        tb = (p['answer_b'] + '\n').encode('utf8')
        k = first_diff_index(ta, tb)
        pa = p['prompt_a'].encode('utf8')
        pb = p['prompt_b'].encode('utf8')
        seqs = [pa + ta, pb + tb]
        max_len = max(len(s) - 1 for s in seqs)
        ids = torch.zeros((2, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s) - 1] = torch.tensor(list(s[:-1]), dtype=torch.long)
        ids = ids.to(device)
        pos_a, pos_b = len(pa) - 1 + k, len(pb) - 1 + k

        # --- baseline run: capture both source activations ---------------------
        with amp:
            base_logits, captured = _run_with_source(model, ids, tree, mode='on')
        if captured is None:
            return dict(
                pairs=0, available=False,
                note='Could not capture the Tree source activation; the model layout changed.',
            )
        src_a, src_b = captured[0:1], captured[1:2]

        base_argmax = [
            int(base_logits[0, pos_a].argmax().item()),
            int(base_logits[1, pos_b].argmax().item()),
        ]
        base_prefers_gold = [
            bool(base_argmax[0] == ta[k]),
            bool(base_argmax[1] == tb[k]),
        ]

        # --- the counterfactual patch: swap the two sources -------------------
        seq_len = ids.size(1)
        swapped = torch.cat([
            _fit_source_length(src_b, seq_len),
            _fit_source_length(src_a, seq_len),
        ], dim=0)
        with amp:
            swap_logits, _ = _run_with_source(model, ids, tree, override=swapped, mode='on')
        swap_argmax = [
            int(swap_logits[0, pos_a].argmax().item()),
            int(swap_logits[1, pos_b].argmax().item()),
        ]
        swap_prefers_gold = [
            bool(swap_argmax[0] == ta[k]),
            bool(swap_argmax[1] == tb[k]),
        ]

        # --- CONTROL: patch in an UNRELATED pair's sources -------------------
        control = None
        if unrelated_pairs:
            up = unrelated_pairs[idx % len(unrelated_pairs)]
            ua = (up['answer_a'] + '\n').encode('utf8')
            ub = (up['answer_b'] + '\n').encode('utf8')
            upa = up['prompt_a'].encode('utf8')
            upb = up['prompt_b'].encode('utf8')
            useqs = [upa + ua, upb + ub]
            umax = max(len(s) - 1 for s in useqs)
            uids = torch.zeros((2, umax), dtype=torch.long)
            for i, s in enumerate(useqs):
                uids[i, :len(s) - 1] = torch.tensor(list(s[:-1]), dtype=torch.long)
            uids = uids.to(device)
            with amp:
                _, ucaptured = _run_with_source(model, uids, tree, mode='on')
            if ucaptured is not None:
                unrel = torch.cat([
                    _fit_source_length(ucaptured[0:1], seq_len),
                    _fit_source_length(ucaptured[1:2], seq_len),
                ], dim=0)
                with amp:
                    ctrl_logits, _ = _run_with_source(model, ids, tree, override=unrel, mode='on')
                ctrl_argmax = [
                    int(ctrl_logits[0, pos_a].argmax().item()),
                    int(ctrl_logits[1, pos_b].argmax().item()),
                ]
                ctrl_prefers_gold = [
                    bool(ctrl_argmax[0] == ta[k]),
                    bool(ctrl_argmax[1] == tb[k]),
                ]
                control = dict(
                    argmax=ctrl_argmax,
                    prefers_gold=ctrl_prefers_gold,
                    unrelated_pair_id=up.get('id'),
                    unrelated_family=up.get('family'),
                    fact_specific_drop=float(
                        sum(base_prefers_gold) - sum(swap_prefers_gold)
                    ) - float(
                        sum(base_prefers_gold) - sum(ctrl_prefers_gold)
                    ),
                )

        rows.append(dict(
            id=p.get('id'),
            family=p.get('family'),
            base_argmax=base_argmax,
            swapped_argmax=swap_argmax,
            base_prefers_gold=base_prefers_gold,
            swapped_prefers_gold=swap_prefers_gold,
            swap_drop=float(sum(base_prefers_gold) - sum(swap_prefers_gold)),
            control=control,
        ))

    n = len(rows)
    base_acc = sum(sum(r['base_prefers_gold']) for r in rows) / (2 * n)
    swap_acc = sum(sum(r['swapped_prefers_gold']) for r in rows) / (2 * n)
    ctrl_acc = None
    if any(r['control'] is not None for r in rows):
        ctrl_acc = sum(
            sum(r['control']['prefers_gold']) for r in rows if r['control'] is not None
        ) / (2 * sum(1 for r in rows if r['control'] is not None))
    return dict(
        pairs=n,
        available=True,
        # Baseline first-byte accuracy at the divergence position.
        base_first_byte_acc=base_acc,
        # First-byte accuracy after swapping the two runs' tree sources.
        swapped_first_byte_acc=swap_acc,
        # First-byte accuracy after patching in an UNRELATED pair's sources.
        unrelated_control_first_byte_acc=ctrl_acc,
        # The fact-specific effect and the generic-damage effect.
        fact_swap_drop=base_acc - swap_acc,
        unrelated_control_drop=(base_acc - ctrl_acc) if ctrl_acc is not None else None,
        fact_specific_excess=(
            (base_acc - swap_acc) - (base_acc - ctrl_acc) if ctrl_acc is not None else None
        ),
        # Legacy keys, kept so existing report readers do not break.
        fb_sens_on=float(np.mean([sum(r['base_prefers_gold']) for r in rows])),
        fb_sens_tree_swapped=float(np.mean([sum(r['swapped_prefers_gold']) for r in rows])),
        tree_swap_drop=float(base_acc - swap_acc),
        rows=rows,
    )


def _pick_unrelated_pairs(pairs: list[dict]) -> list[dict]:
    """Pick a DIFFERENT-family pair for each pair, for the unrelated-donor control."""
    unrelated = []
    for i, p in enumerate(pairs):
        candidates = [
            q for q in pairs
            if q.get('family') != p.get('family') and q.get('id') != p.get('id')
        ]
        if not candidates:
            candidates = [q for q in pairs if q.get('id') != p.get('id')]
        unrelated.append(candidates[i % len(candidates)] if candidates else p)
    return unrelated


@torch.inference_mode()
def evaluate_coqa_extractive(
    model: LM,
    coqa_rows: list[tuple[str, str]],
    device: torch.device | str,
    mode: str = 'on',
) -> dict:
    """Evaluate greedy exact match & token F1 on real held-out CoQA extractive questions."""
    prompts = [p for p, _ in coqa_rows]
    refs = [t.strip() for _, t in coqa_rows]
    preds = greedy_generate(model, prompts, device, max_bytes=28, mode=mode, batch_size=8)
    exacts, f1s, samples = [], [], []
    for ref, pred in zip(refs, preds):
        nr, np_ = normalize_answer(ref), normalize_answer(pred)
        exact = nr == np_
        rt, pt = nr.split(), np_.split()
        if not rt or not pt:
            f1 = float(rt == pt)
        else:
            common = 0
            rem = list(rt)
            for tok in pt:
                if tok in rem:
                    common += 1
                    rem.remove(tok)
            f1 = (2 * common / (len(rt) + len(pt))) if common else 0.0
        exacts.append(exact)
        f1s.append(f1)
        samples.append(dict(reference=ref, predicted=pred, exact=exact, f1=f1))
    return dict(
        examples=len(coqa_rows),
        exact_match=float(np.mean(exacts)),
        token_f1=float(np.mean(f1s)),
        samples=samples[:8],
    )


# The seeds of every split.  The `report_*` seeds are deliberately DISJOINT from the
# `select_*` seeds (E4): the checkpoint that is reported must never have been selected
# on the pairs it is reported on.
SUITE_SEEDS = {
    'user_6pairs': None,
    'user_qswap': None,
    'oov_clean': 2001,
    'oov_qswap': 2002,
    'nonce_qswap': 2005,
    'long_clean': 2004,
    'far_clean': 2006,
    'far_qswap': 2007,
}

# Splits whose fact is at least MIN_MEMORY_DISTANCE_BYTES before the question.
MEMORY_SPLITS = ('far_clean', 'far_qswap')
# Splits that measure BINDING GENERALISATION (unseen nonce keys and values) rather than
# memory retrieval.  They are expected to fail and are reported separately (E7).
GENERALIZATION_SPLITS = ('nonce_qswap',)
# Splits whose content is FIXED (the user's reported diagnostic pairs).  They are the
# same in every suite and are excluded from the disjointness content comparison.
FIXED_DIAGNOSTIC_SPLITS = ('user_6pairs', 'user_qswap')


def require_memory_on(
    suite: dict[str, list[dict]],
    *,
    min_distance: int = MIN_MEMORY_DISTANCE_BYTES,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
    splits: tuple[str, ...] = MEMORY_SPLITS,
) -> dict:
    """Apply the memory precondition to a suite's memory-required splits.

    Raises `ValueError` when any of `splits` contains a pair whose fact is in the SAME
    chunk as the question (the tree could not possibly contribute) or is closer than
    `min_distance` bytes to it.

    This is the guard behind `build_fact_switch_suite(..., require_memory=True)`; it is
    exposed separately so an EXTERNALLY supplied suite can be validated too.
    """
    stats = {}
    for name in splits:
        if name not in suite:
            continue
        stats[name] = check_memory_precondition(
            suite[name], min_distance=min_distance, chunk_bytes=chunk_bytes, split_name=name,
        )
    if not stats:
        raise ValueError(
            f'No memory-required split ({", ".join(splits)}) present in the suite; '
            f'cannot verify the memory precondition.'
        )
    return stats


def build_fact_switch_suite(
    qa_style: str = 'role',
    pairs_per_split: int = 16,
    *,
    require_memory: bool = False,
    min_distance: int = MIN_MEMORY_DISTANCE_BYTES,
    chunk_bytes: int = MEMORY_CHUNK_BYTES,
    seed_offset: int = 0,
    include_far: bool = True,
) -> dict[str, list[dict]]:
    """Standard OOV fact-switch diagnostic suite separating Phase-1 copying from Phase-2 key binding.

    NEW splits (E1): `far_clean` and `far_qswap` place the fact at least
    `min_distance` bytes before the question, so the fact lands in a chunk the widened
    layer-7 local attention window cannot reach.  Only the Tree memory can carry it.

    `require_memory=True` raises `ValueError` when any generated pair fails the memory
    precondition -- i.e. when the fact sits in the SAME chunk as the question (the tree
    could not possibly contribute) or is closer than `min_distance` bytes.

    `seed_offset` shifts every split seed, which is how a DISJOINT report suite is
    built from the same generator (E4).
    """
    def seed_for(name: str) -> int:
        base = SUITE_SEEDS[name]
        return base + seed_offset if base is not None else base

    # The id prefix makes the pair ids of two suites built with different seed offsets
    # distinct, so a disjointness check on ids is meaningful.
    id_prefix = '' if not seed_offset else f's{seed_offset}_'

    suite: dict[str, list[dict]] = {
        'user_6pairs': user_audit_pairs(qa_style=qa_style, id_prefix=id_prefix),
        'user_qswap': user_qswap_pairs(qa_style=qa_style, id_prefix=id_prefix),
        'oov_clean': generate_pairs(
            pairs_per_split, seed=seed_for('oov_clean'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), distractor_prob=0.0, qa_style=qa_style,
            id_prefix=id_prefix,
        ),
        'oov_qswap': generate_qswap_pairs(
            pairs_per_split, seed=seed_for('oov_qswap'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), qa_style=qa_style,
            id_prefix=id_prefix,
        ),
        'nonce_qswap': generate_qswap_pairs(
            pairs_per_split, seed=seed_for('nonce_qswap'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), nonce_val_prob=1.0, nonce_key_prob=1.0, qa_style=qa_style,
            id_prefix=id_prefix,
        ),
        'long_clean': generate_pairs(
            pairs_per_split, seed=seed_for('long_clean'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), distractor_prob=0.0,
            filler_choices=(2,), qa_style=qa_style, id_prefix=id_prefix,
        ),
    }
    if include_far:
        suite['far_clean'] = generate_far_pairs(
            pairs_per_split, seed=seed_for('far_clean'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), distractor_prob=0.0,
            qa_style=qa_style, min_distance=min_distance, chunk_bytes=chunk_bytes,
            id_prefix=id_prefix,
        )
        suite['far_qswap'] = generate_far_qswap_pairs(
            pairs_per_split, seed=seed_for('far_qswap'), vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'),
            qa_style=qa_style, min_distance=min_distance, chunk_bytes=chunk_bytes,
            id_prefix=id_prefix,
        )

    if require_memory:
        # Only the `far_*` splits are required to satisfy the memory precondition; the
        # short splits are deliberately 1-hop induction tests and are reported as such.
        require_memory_on(
            suite, min_distance=min_distance, chunk_bytes=chunk_bytes, splits=MEMORY_SPLITS,
        )
    return suite


def suite_pair_ids(suite: dict[str, list[dict]]) -> dict[str, set[str]]:
    """Map split name -> the set of pair ids it contains (for disjointness checks)."""
    return {name: {p.get('id') for p in items} for name, items in suite.items()}


def suite_pair_keys(suite: dict[str, list[dict]]) -> set[tuple]:
    """The CONTENT keys of every pair in a suite: (prompt_a, prompt_b).

    Disjointness must be checked on content, not only on ids: two suites built from
    different seeds can still reuse the same id strings while carrying different
    content, and two suites built from the same seed carry identical content under
    identical ids.

    The `user_*` splits are FIXED diagnostics (the six pairs the user reported), so
    their content is intentionally identical in every suite and they are excluded from
    the content comparison.
    """
    keys: set[tuple] = set()
    for name, items in suite.items():
        if name in FIXED_DIAGNOSTIC_SPLITS:
            continue
        for p in items:
            keys.add((p.get('prompt_a'), p.get('prompt_b')))
    return keys


def suites_are_disjoint(a: dict[str, list[dict]], b: dict[str, list[dict]]) -> bool:
    """True when the two suites share neither a pair id nor a pair's content."""
    ids_a = set().union(*suite_pair_ids(a).values()) if a else set()
    ids_b = set().union(*suite_pair_ids(b).values()) if b else set()
    if ids_a & ids_b:
        return False
    return not (suite_pair_keys(a) & suite_pair_keys(b))


def audit_permutation_donors(
    model: LM,
    pairs: list[dict],
    device: torch.device | str,
    *,
    permutation_seed: int = 0,
    mode: str = PERMUTE_MODE,
) -> dict:
    """Record WHICH donor content the permutation ablation used for each pair (E2).

    The old `mode='shuffle'` was a BATCH-axis `roll(1, 0)`: the "donor" was simply
    the adjacent item in a fixed list, so the ablation said nothing about content.
    A genuine ablation must show WHICH content replaced the memory.

    We hook `Tree.write` -- the function that turns the (possibly permuted) source
    chunks into memory nodes -- because that is the first place the permuted content
    is actually visible.  Hooking the Tree's `forward` argument would capture the
    source BEFORE the permutation is applied and would therefore be blind to it.

    For each pair we record a short content signature for the memory built from the
    pair's OWN source and for the memory built from the PERMUTED source, so the report
    shows that the donor really was different content.
    """
    tree = getattr(model, 'tree', None)
    if tree is None:
        return dict(available=False, note='Model has no Tree; permutation audit unavailable.', rows=[])
    resolved, note = resolve_tree_mode(mode, model)
    if resolved not in (PERMUTE_MODE, LEGACY_PERMUTE_MODE):
        return dict(available=False, note='Model supports no permutation mode.', rows=[])

    rows = []
    original_seed = getattr(tree, 'permute_seed', None)
    for p in pairs:
        pa = p['prompt_a'].encode('utf8')
        pb = p['prompt_b'].encode('utf8')
        ids = torch.zeros((2, max(len(pa), len(pb))), dtype=torch.long)
        ids[0, :len(pa)] = torch.tensor(list(pa), dtype=torch.long)
        ids[1, :len(pb)] = torch.tensor(list(pb), dtype=torch.long)
        ids = ids.to(device)
        # The seed that is ACTUALLY used for this pair.  It is installed on the
        # tree so the ablation honours it -- recording a seed that the
        # ablation ignores would make the audit report fictional.
        pair_seed = permutation_seed_for(str(p.get('id')), permutation_seed)
        if original_seed is not None:
            tree.permute_seed = pair_seed
        own_sig = _tree_memory_signature(model, ids, tree, 'on', device)
        perm_sig = _tree_memory_signature(model, ids, tree, resolved, device)
        rows.append(dict(
            id=p.get('id'),
            family=p.get('family'),
            permutation_seed=pair_seed,
            own_source_signature=own_sig,
            permuted_source_signature=perm_sig,
            donor_differs=bool(own_sig != perm_sig),
        ))
    if original_seed is not None:
        tree.permute_seed = original_seed
    return dict(
        available=True,
        mode=resolved,
        note=note,
        permutation_seed=permutation_seed,
        pairs=len(rows),
        donor_differs_fraction=(
            sum(r['donor_differs'] for r in rows) / len(rows) if rows else 0.0
        ),
        rows=rows,
    )


def evaluate_fact_switch_suite(
    model: LM,
    suite: dict[str, list[dict]],
    device: torch.device | str,
    *,
    do_generate: bool = True,
    gen_limit: int = 8,
    run_tree_ablations: bool = False,
    permutation_seed: int = 0,
    permutation_audit: bool = False,
) -> dict:
    """Evaluate all fact-switch splits plus optional causal Tree ablations.

    The returned dict contains, per split:
      * `first_byte_acc`      -- the PRIMARY decision metric (E5)
      * `pair_acc`            -- the legacy teacher-forced forced-choice metric
      * `sum_sens_diagnostic` -- the legacy SUM-over-bytes diagnostic, relabelled
      * `unrelated_key_acc`   -- the unrelated-passage control (E5)

    Plus:
      * `generalization_splits` -- the binding-generalisation numbers, reported
        separately and prominently (E7)
      * `memory_splits`        -- the memory-required (`far_*`) splits (E1)
    """
    results = {}
    for name, items in suite.items():
        limit = len(items) if name in ('user_6pairs', 'user_qswap') else gen_limit
        r = evaluate_pairs(
            model, items, device, mode='on', do_generate=do_generate, gen_limit=limit,
            permutation_seed=permutation_seed,
        )
        if name in ('user_6pairs', 'user_qswap'):
            results[name] = r
        else:
            compact = {k: v for k, v in r.items() if k != 'rows'}
            if do_generate:
                compact['sample_generations'] = [
                    {k: row[k] for k in ('id', 'family', 'answer_a', 'answer_b', 'gen_a', 'gen_b', 'gen_both_exact') if k in row}
                    for row in r['rows'][:4]
                ]
            results[name] = compact

    # --- E7: report the binding-GENERALISATION splits separately -------------
    generalization = {}
    for sp in GENERALIZATION_SPLITS:
        if sp not in results:
            continue
        v = results[sp]
        generalization[sp] = dict(
            first_byte_acc=v.get('first_byte_acc'),
            pair_acc=v.get('pair_acc'),
            fb_pair_acc=v.get('fb_pair_acc'),
            by_direction={
                d: dict(
                    first_byte_acc=e.get('first_byte_acc'),
                    pair_acc=e.get('pair_acc'),
                )
                for d, e in (v.get('by_direction') or {}).items()
            },
        )
    if generalization:
        results['generalization_splits'] = generalization

    # --- E1: report the MEMORY-required splits separately --------------------
    memory_section = {}
    for sp in MEMORY_SPLITS:
        if sp not in results or sp not in suite:
            continue
        try:
            stats = check_memory_precondition(
                suite[sp], split_name=sp,
            )
        except ValueError as exc:
            stats = dict(split=sp, valid=False, error=str(exc))
        memory_section[sp] = dict(
            first_byte_acc=results[sp].get('first_byte_acc'),
            pair_acc=results[sp].get('pair_acc'),
            fb_pair_acc=results[sp].get('fb_pair_acc'),
            memory_precondition=stats,
        )
    if memory_section:
        results['memory_splits'] = memory_section

    if run_tree_ablations and getattr(model, 'tree', None) is not None:
        ablations = {}
        abl_splits = [sp for sp in ('far_clean', 'far_qswap', 'long_clean', 'oov_qswap') if sp in suite]
        for sp in abl_splits:
            items = suite[sp]
            abl = {}
            for m in ('on', 'off', PERMUTE_MODE):
                try:
                    resolved, note = resolve_tree_mode(m, model)
                except ValueError as exc:
                    abl[m] = dict(available=False, note=str(exc))
                    continue
                r = evaluate_pairs(
                    model, items, device, mode=m, do_generate=False,
                    permutation_seed=permutation_seed,
                )
                entry = {k: v for k, v in r.items() if k != 'rows'}
                entry['resolved_mode'] = resolved
                if note:
                    entry['mode_note'] = note
                abl[m] = entry
            # Legacy key so old report readers keep working.
            if PERMUTE_MODE not in abl and LEGACY_PERMUTE_MODE in abl:
                abl[PERMUTE_MODE] = abl[LEGACY_PERMUTE_MODE]
            abl['counterfactual_tree_swap'] = evaluate_tree_counterfactual_swap(model, items, device)
            if permutation_audit:
                abl['permutation_audit'] = audit_permutation_donors(
                    model, items, device, permutation_seed=permutation_seed,
                )
            ablations[sp] = abl
        results['tree_ablations'] = ablations
    return results


def fact_switch_selection_key(summary: dict) -> tuple[float, float, float, float, float]:
    """Rank checkpoints by Phase-2 key-matched qswap first, then clean OOV & greedy generation.

    E6: the key now also uses the NEW first-byte quantities (which are the primary
    metric, E5).  The first component is the first-byte accuracy on the
    key-matched (question-swap) splits, because that is the quantity that actually
    measures binding rather than teacher-forced copying.

    IMPORTANT (E4/E6): this function must only ever be called on the SELECTION
    suite.  Calling it on the suite that is later reported produces an upward-biased
    maximum over the number of evaluations.
    """
    qswap = summary.get('oov_qswap', {})
    clean = summary.get('oov_clean', {})
    user_q = summary.get('user_qswap', {})
    far = summary.get('far_qswap', summary.get('far_clean', {}))
    qswap_pair = 0.5 * qswap.get('pair_acc', 0.0) + 0.5 * user_q.get('pair_acc', 0.0)
    clean_pair = clean.get('pair_acc', 0.0)
    gen_exact = 0.5 * qswap.get('gen_indiv_exact', 0.0) + 0.5 * clean.get('gen_indiv_exact', 0.0)
    sens = qswap.get('fb_mean_sens', 0.0) + 0.25 * clean.get('fb_mean_sens', 0.0)
    # NEW: the primary first-byte decision accuracy.
    fb_key = 0.5 * qswap.get('first_byte_acc', 0.0) + 0.5 * user_q.get('first_byte_acc', 0.0)
    fb_far = far.get('first_byte_acc', 0.0)
    return (
        round(fb_key, 4),
        round(qswap_pair, 4),
        round(gen_exact, 4),
        round(clean_pair, 4),
        round(fb_far + 0.25 * sens, 4),
    )


def format_fact_switch_table(label: str, summary: dict) -> str:
    lines = [f'=== {label} ===']
    for sp in ('user_6pairs', 'user_qswap', 'oov_clean', 'oov_qswap', 'long_clean', 'far_clean', 'far_qswap'):
        if sp not in summary or not isinstance(summary[sp], dict):
            continue
        v = summary[sp]
        gen_s = f" | gen_pair={v['gen_pair_exact']:.2f} gen_ind={v['gen_indiv_exact']:.2f}" if 'gen_pair_exact' in v else ''
        fb = v.get('first_byte_acc')
        unrel = v.get('unrelated_key_acc')
        unrel_s = f" unrel={unrel:.2f}" if isinstance(unrel, float) else ''
        lines.append(
            f"  {sp:12s}: fb_acc={fb if isinstance(fb, float) else float('nan'):.2f} "
            f"pair={v['pair_acc']:.2f} fb_pair={v['fb_pair_acc']:.2f} "
            f"sum_sens={v.get('sum_sens_diagnostic', v.get('mean_sens', 0.0)):+5.2f} "
            f"fb_sens={v['fb_mean_sens']:+5.2f}{unrel_s}{gen_s}"
        )

    # --- E7: the binding-GENERALISATION numbers, reported prominently --------
    gen_section = summary.get('generalization_splits')
    if gen_section:
        lines.append('  BINDING GENERALISATION (unseen nonce keys+values) -- EXPECTED TO FAIL:')
        lines.append('    ^ this split tests whether the model generalises the BINDING RULE to')
        lines.append('      never-seen words. It is NOT a memory-retrieval test and is not')
        lines.append('      evidence about the tree. Reported separately so it is not buried.')
        for sp, v in gen_section.items():
            lines.append(
                f"    {sp:14s}: fb_acc={v.get('first_byte_acc', float('nan')):.2f} "
                f"pair={v.get('pair_acc', float('nan')):.2f} "
                f"fb_pair={v.get('fb_pair_acc', float('nan')):.2f}"
            )
            for d, e in (v.get('by_direction') or {}).items():
                lines.append(
                    f"      {d:8s}     : fb_acc={e.get('first_byte_acc', float('nan')):.2f} "
                    f"pair={e.get('pair_acc', float('nan')):.2f}"
                )

    mem_section = summary.get('memory_splits')
    if mem_section:
        lines.append('  MEMORY-REQUIRED splits (fact >= 2048 bytes before the question):')
        for sp, v in mem_section.items():
            pre = v.get('memory_precondition') or {}
            valid = pre.get('valid', True) if isinstance(pre, dict) else True
            lines.append(
                f"    {sp:12s}: fb_acc={v.get('first_byte_acc', float('nan')):.2f} "
                f"pair={v.get('pair_acc', float('nan')):.2f} "
                f"precondition_ok={valid}"
            )

    if 'tree_ablations' in summary:
        lines.append('  Causal Tree ablations (fb_mean_sens diagnostic):')
        for sp, abl in summary['tree_ablations'].items():
            perm = abl.get(PERMUTE_MODE) or abl.get(LEGACY_PERMUTE_MODE) or {}
            lines.append(
                f"    {sp:10s}: on={abl['on']['fb_mean_sens']:+5.2f} "
                f"off={abl['off']['fb_mean_sens']:+5.2f} "
                f"perm={perm.get('fb_mean_sens', float('nan')):+5.2f}"
            )
            cf = abl.get('counterfactual_tree_swap') or {}
            if cf.get('available'):
                lines.append(
                    f"      counterfactual swap: base_fb={cf.get('base_first_byte_acc', float('nan')):.2f} "
                    f"swapped_fb={cf.get('swapped_first_byte_acc', float('nan')):.2f} "
                    f"unrelated_ctrl={cf.get('unrelated_control_first_byte_acc', float('nan')):.2f} "
                    f"(fact_specific_excess={cf.get('fact_specific_excess', float('nan')):+.2f})"
                )
            else:
                lines.append(f"      counterfactual swap: unavailable ({cf.get('note', 'n/a')})")
    return '\n'.join(lines)


def load_coqa_pools(dialogue_dir: str | Path, context: int = 1024, eval_limit: int = 16, seed: int = 42, qa_style: str = 'role'):
    """Load training qa_excerpt rows and held-out extractive CoQA dev questions."""
    data, _ = read_dialogue_data(dialogue_dir, offline=True)
    pools = prepare_dialogues(data, context, qa_style=qa_style)
    train_excerpts = [(p, t) for p, t in pools['train']['qa_excerpt'] if len((p + t).encode()) <= 420]
    eligible = []
    for prompt, target in pools['heldout']['qa']:
        ans = single_line(target).casefold()
        if ans and ans not in ('yes', 'no', 'unknown') and len(ans) <= 24 and ans in prompt.casefold():
            eligible.append((prompt, target))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(eligible), min(eval_limit, len(eligible)), replace=False)
    heldout = [eligible[int(i)] for i in idx]
    return train_excerpts, heldout


def run_two_stage_fact_switch(
    model: LM,
    suite: dict[str, list[dict]],
    device: torch.device | str,
    *,
    minutes: float = 6.0,
    max_steps: int | None = None,
    quads_per_step: int = 2,
    lr: float = 5e-4,
    qa_style: str = 'qa',
    seed: int = 42,
    replay_fn=None,
    replay_every: int = 0,
    eval_interval_seconds: float = 60.0,
    event_fn=None,
    select_suite: dict[str, list[dict]] | None = None,
    report_suite: dict[str, list[dict]] | None = None,
    permutation_seed: int = 0,
) -> tuple[LM, dict]:
    """Two-stage fact-to-answer adaptation:
      Stage 1 (first 35% budget): plain_nll on mixed clean + 2x2 quads + 60% nonce values (copy-head lock-in).
      Stage 2 (remaining 65%):    contrastive loss on 100% 2x2 counterfactual quads + nonce keys (Phase-2 routing).
    Selects the best weights by counterfactual pair accuracy + greedy exact match (never BPB alone).

    SELECTION vs REPORTING (E4): the checkpoint is chosen using ONLY `select_suite`;
    the returned `selected` summary is computed on `report_suite`.  When only one suite
    is supplied it is used for both, and the returned dict carries
    `selection_is_disjoint: False` so the upward bias is explicit rather than
    silent.
    """
    dev = torch.device(device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(dev.type == 'cuda'))
    py_rng = random.Random(seed + 401)

    # --- E4: separate the SELECTION suite from the REPORT suite -------------
    if select_suite is None and report_suite is None:
        select_suite = suite
        report_suite = suite
        selection_is_disjoint = False
        selection_note = (
            'Only one suite was supplied, so the checkpoint was selected and reported '
            'on the SAME pairs.  The reported number is an upward-biased maximum over '
            'the number of evaluations; pass disjoint select_suite/report_suite to '
            'remove this bias.'
        )
    elif select_suite is None:
        select_suite = suite
        selection_is_disjoint = suites_are_disjoint(suite, report_suite)
        selection_note = (
            'No select_suite was supplied; the legacy `suite` argument was used for '
            'selection and report_suite for reporting.'
        )
    elif report_suite is None:
        report_suite = suite
        selection_is_disjoint = suites_are_disjoint(select_suite, suite)
        selection_note = (
            'No report_suite was supplied; the legacy `suite` argument was used for '
            'reporting and select_suite for selection.'
        )
    else:
        selection_is_disjoint = suites_are_disjoint(select_suite, report_suite)
        selection_note = 'Selection and reporting used disjoint suites.'

    t0 = time.perf_counter()
    deadline = t0 + 60.0 * minutes if minutes > 0 else float('inf')
    stage1_deadline = t0 + 60.0 * minutes * 0.35 if minutes > 0 else float('inf')

    before = evaluate_fact_switch_suite(
        model, report_suite, dev, do_generate=True, run_tree_ablations=True,
        permutation_seed=permutation_seed,
    )
    # E6: the selection key is computed on the SELECTION suite ONLY.
    sel_before = evaluate_fact_switch_suite(
        model, select_suite, dev, do_generate=True, run_tree_ablations=False,
        permutation_seed=permutation_seed,
    )
    best_key = fact_switch_selection_key(sel_before)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step, best_summary = 0, sel_before
    if event_fn:
        event_fn(stage='fact_switch_initial', key=list(best_key),
                 oov_clean=sel_before['oov_clean']['pair_acc'], oov_qswap=sel_before['oov_qswap']['pair_acc'])

    step, last_check = 0, time.perf_counter()
    last_stats = {}
    while (max_steps is None or step < max_steps) and time.perf_counter() < deadline:
        in_stage1 = (
            (step < int(0.35 * max_steps)) if max_steps is not None
            else (time.perf_counter() < stage1_deadline)
        )
        warmup = 20
        cur_lr = lr * min(1.0, (step + 1) / warmup) * (1.0 if in_stage1 else 0.7)
        for grp in opt.param_groups:
            grp['lr'] = cur_lr

        model.train()
        opt.zero_grad(set_to_none=True)
        if replay_fn is not None and replay_every > 0 and (step + 1) % replay_every == 0:
            x, y = replay_fn()
            with amp:
                loss = F.cross_entropy(model(x).float().flatten(0, 1), y.reshape(-1))
        else:
            batch = stream_quad_batch(
                py_rng, quads_per_step, TRAIN_VOCAB,
                clean_prob=0.35 if in_stage1 else 0.0,
                filler_choices=(0,) if in_stage1 else (0, 0, 1),
                nonce_val_prob=0.6 if in_stage1 else 0.5,
                nonce_key_prob=0.25 if in_stage1 else 0.5,
                qa_style=qa_style,
            )
            x, y, bpos, dinfo = pair_batch_tensors(batch, dev)
            with amp:
                logits = model(x)
                loss, last_stats = fact_switch_loss(
                    logits, y, bpos, dinfo,
                    mode='plain_nll' if in_stage1 else 'contrastive',
                    first_byte_weight=2.5, margin_target=3.0, contrastive_weight=0.75,
                )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        step += 1

        now = time.perf_counter()
        if now - last_check >= eval_interval_seconds or (max_steps is not None and step == max_steps) or now >= deadline:
            # E6: the selection key is computed on the SELECTION suite ONLY.
            val_sum = evaluate_fact_switch_suite(
                model, select_suite, dev, do_generate=True, gen_limit=6,
                run_tree_ablations=False, permutation_seed=permutation_seed,
            )
            key = fact_switch_selection_key(val_sum)
            if key > best_key:
                best_key, best_step, best_summary = key, step, val_sum
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if event_fn:
                event_fn(
                    stage='fact_switch_val', step=step, phase=1 if in_stage1 else 2,
                    oov_clean=val_sum['oov_clean']['pair_acc'], oov_qswap=val_sum['oov_qswap']['pair_acc'],
                    user6=val_sum['user_6pairs']['pair_acc'], user_qswap=val_sum['user_qswap']['pair_acc'],
                    gen_oov_clean=val_sum['oov_clean'].get('gen_indiv_exact', 0.0),
                    selected_step=best_step,
                )
            last_check = time.perf_counter()

    model.load_state_dict(best_state)
    model.eval()
    # E4: the REPORTED summary comes from the REPORT suite only.
    selected = evaluate_fact_switch_suite(
        model, report_suite, dev, do_generate=True, run_tree_ablations=True,
        permutation_seed=permutation_seed,
    )
    return model, dict(
        steps=step,
        selected_step=best_step,
        quads_per_step=quads_per_step,
        lr=lr,
        qa_style=qa_style,
        seconds=time.perf_counter() - t0,
        last_train_stats=last_stats,
        before=before,
        selected=selected,
        # --- E4 bookkeeping ---------------------------------------------------
        selection_is_disjoint=selection_is_disjoint,
        selection_note=selection_note,
        selection_suite_splits=sorted(select_suite.keys()),
        report_suite_splits=sorted(report_suite.keys()),
        selection_summary=best_summary,
        permutation_seed=permutation_seed,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/best_lm.pt')
    p.add_argument('--out', default='')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--backend', default='torch', choices=('torch', 'fla'))
    p.add_argument('--minutes', type=float, default=5.0)
    p.add_argument('--max-steps', type=int, default=None)
    p.add_argument('--quads-per-step', type=int, default=2)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--qa-style', default='qa', choices=('role', 'qa'))
    p.add_argument('--read-window', type=int, default=1536)
    p.add_argument('--widen-l3', action='store_true')
    a = p.parse_args()

    device = torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    out_dir = Path(a.out) if a.out else Path('runs') / time.strftime('fact-switch-%Y%m%d-%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    model = LM(
        ck['kind'], ck['slots'], backend=a.backend,
        read_window=a.read_window if ck['kind'] == 'C1' else None,
        mid_window=a.read_window if (ck['kind'] == 'C1' and a.widen_l3) else None,
    ).to(device)
    model.load_state_dict(ck['state_dict'])
    # E4: build a DISJOINT selection suite and report suite.  The report suite uses a
    # fresh seed offset, so the reported numbers are never the maximum over the
    # selection evaluations.
    select_suite = build_fact_switch_suite(qa_style=a.qa_style, pairs_per_split=16, require_memory=True)
    report_suite = build_fact_switch_suite(qa_style=a.qa_style, pairs_per_split=16, seed_offset=100_000)
    model, res = run_two_stage_fact_switch(
        model, select_suite, device,
        minutes=a.minutes, max_steps=a.max_steps, quads_per_step=a.quads_per_step,
        lr=a.lr, qa_style=a.qa_style, event_fn=lambda **row: print(row, flush=True),
        select_suite=select_suite, report_suite=report_suite,
    )
    print(f"selection_is_disjoint={res['selection_is_disjoint']}: {res['selection_note']}")
    print(format_fact_switch_table('Before', res['before']))
    print(format_fact_switch_table('Selected', res['selected']))
    torch.save(dict(
        kind=ck['kind'], slots=ck['slots'],
        read_window=model.blocks[-1].mix.window if model.kind == 'C1' else None,
        mid_window=model.blocks[3].mix.window if model.kind == 'C1' else None,
        state_dict=model.state_dict(), qa_style=a.qa_style,
    ), out_dir / 'best_fact_switch.pt')
    (out_dir / 'report.json').write_text(json.dumps(res, indent=2), encoding='utf8')
    print(f'Saved results to {out_dir / "report.json"}')


if __name__ == '__main__' and '__file__' in globals():
    main()
