"""Counterfactual fact-to-answer benchmark, 2x2 quad trainer, and Tree causal ablation suite.

Designed to run both as a standalone Python module/CLI and embedded verbatim inside run.ipynb
on Google Colab (where LM and dialogue helpers are already defined in notebook globals).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import re
import time

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


def user_audit_pairs(qa_style: str = 'role') -> list[dict]:
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
            id=pid, family=kind, direction=FAMILY_DIRECTION[kind], distractor=has_dist,
            prompt_a=pa, prompt_b=pb, answer_a=va, answer_b=vb,
        ))
    return pairs


def user_qswap_pairs(qa_style: str = 'role') -> list[dict]:
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
            id=pid, family=fam, direction=direction, distractor=True,
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
    fillers = rng.sample(FILLER_SENTENCES, filler_count) if filler_count else []

    def compose(main_val: str, dist_val: str) -> str:
        sents = (
            [s2(dist_val), s1(main_val)] if flip_order
            else ([s1(main_val), s2(dist_val)] if distractor else [s1(main_val)])
        )
        sents.extend(fillers)
        return ' '.join(sents)

    pass_a = compose(va, vb)
    pass_b = compose(vb, va)
    return dict(
        id=pair_id,
        family=family,
        direction=FAMILY_DIRECTION[family],
        distractor=distractor,
        filler_count=filler_count,
        unseen_template=unseen_template,
        prompt_a=chat_prompt(q, [], pass_a, 1400, qa_style=qa_style),
        prompt_b=chat_prompt(q, [], pass_b, 1400, qa_style=qa_style),
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

    tail = (' ' + ' '.join(rng.sample(FILLER_SENTENCES, filler_count))) if filler_count else ''
    p_ab = s(va, vb) + tail
    p_ba = s(vb, va) + tail
    direction = FAMILY_DIRECTION[family]
    return [
        dict(
            id='q1', family=family, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ab, 1400, qa_style=qa_style),
            prompt_b=chat_prompt(qb, [], p_ab, 1400, qa_style=qa_style),
            answer_a=va, answer_b=vb,
        ),
        dict(
            id='q2', family=family, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ba, 1400, qa_style=qa_style),
            prompt_b=chat_prompt(qb, [], p_ba, 1400, qa_style=qa_style),
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
            rng, vocab, fam, dist, unseen_template, fillers, f'p{len(pairs):04d}',
            nonce_value_prob=nonce_value_prob, nonce_key_prob=nonce_key_prob, qa_style=qa_style,
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
) -> list[dict]:
    """Generate fixed held-out question-swap pairs (identical passage, swapped query)."""
    rng = random.Random(seed)
    pairs = []
    for i in range(n_pairs):
        fam = families[i % len(families)]
        fc = rng.choice(filler_choices)
        pair = make_quad_as_two_pairs(
            rng, vocab, fam, nonce_val_prob=nonce_val_prob, nonce_key_prob=nonce_key_prob,
            filler_count=fc, qa_style=qa_style,
        )[0]
        pair['id'] = f'qs{i:03d}'
        pairs.append(pair)
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
    eff_bs = max(2, batch_size) if mode == 'shuffle' else batch_size
    for start in range(0, len(prompts), eff_bs):
        ps = prompts[start:start + eff_bs]
        es = endings[start:start + eff_bs]
        if mode == 'shuffle' and len(ps) == 1:
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
            logits = model(ids, mode=mode).float()
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
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk_prompts = prompts[start:start + batch_size]
        if mode == 'shuffle' and len(chunk_prompts) < 2:
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
                logits = model(ids, mode=mode, positions=last_pos.to(device)[:, None]).squeeze(1).float()
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
) -> dict:
    """Evaluate forced-choice pair flipping, first-byte sensitivity, direction breakdown, and greedy generation."""
    prompts_a = [p['prompt_a'] for p in pairs]
    prompts_b = [p['prompt_b'] for p in pairs]
    ans_a = [p['answer_a'] + '\n' for p in pairs]
    ans_b = [p['answer_b'] + '\n' for p in pairs]

    s_aa, pb_aa, det_aa = _score_prompt_endings(model, prompts_a, ans_a, device, mode=mode, batch_size=batch_size)
    s_ab, pb_ab, det_ab = _score_prompt_endings(model, prompts_a, ans_b, device, mode=mode, batch_size=batch_size)
    s_ba, pb_ba, det_ba = _score_prompt_endings(model, prompts_b, ans_a, device, mode=mode, batch_size=batch_size)
    s_bb, pb_bb, det_bb = _score_prompt_endings(model, prompts_b, ans_b, device, mode=mode, batch_size=batch_size)

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
        mean_sens=float(np.mean([r['sens'] for r in rows])),
        mean_abs_bias=float(np.mean([r['abs_bias'] for r in rows])),
        fb_pair_acc=sum(r['fb_both_correct'] for r in rows) / n,
        fb_mean_sens=float(np.mean([r['fb_sens'] for r in rows])),
        fb_mean_abs_bias=float(np.mean([r['fb_abs_bias'] for r in rows])),
        by_direction=by_dir,
        rows=rows,
    )
    if do_generate and gen_n > 0:
        gen_rows = rows[:gen_n]
        summary['gen_pairs_evaluated'] = gen_n
        summary['gen_indiv_exact'] = sum(int(r['gen_a_exact']) + int(r['gen_b_exact']) for r in gen_rows) / (2 * gen_n)
        summary['gen_pair_exact'] = sum(r['gen_both_exact'] for r in gen_rows) / gen_n
    return summary


@torch.inference_mode()
def evaluate_tree_counterfactual_swap(model: LM, pairs: list[dict], device: torch.device | str) -> dict:
    """Directly swap Tree source activations between prompt_a and prompt_b while keeping Delta & Attn intact."""
    model.eval()
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else nullcontext()
    deltas_on, deltas_swap = [], []
    for p in pairs:
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
        with amp:
            lon = model(ids, mode='on').float()
            lsw = model(ids, mode='shuffle').float()
        sens_on = float((lon[0, pos_a, ta[k]] - lon[0, pos_a, tb[k]]) + (lon[1, pos_b, tb[k]] - lon[1, pos_b, ta[k]]))
        sens_sw = float((lsw[0, pos_a, ta[k]] - lsw[0, pos_a, tb[k]]) + (lsw[1, pos_b, tb[k]] - lsw[1, pos_b, ta[k]]))
        deltas_on.append(sens_on)
        deltas_swap.append(sens_sw)
    return dict(
        pairs=len(pairs),
        fb_sens_on=float(np.mean(deltas_on)),
        fb_sens_tree_swapped=float(np.mean(deltas_swap)),
        tree_swap_drop=float(np.mean(deltas_on) - np.mean(deltas_swap)),
    )


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


def build_fact_switch_suite(qa_style: str = 'role', pairs_per_split: int = 16) -> dict[str, list[dict]]:
    """Standard OOV fact-switch diagnostic suite separating Phase-1 copying from Phase-2 key binding."""
    return {
        'user_6pairs': user_audit_pairs(qa_style=qa_style),
        'user_qswap': user_qswap_pairs(qa_style=qa_style),
        'oov_clean': generate_pairs(
            pairs_per_split, seed=2001, vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), distractor_prob=0.0, qa_style=qa_style,
        ),
        'oov_qswap': generate_qswap_pairs(
            pairs_per_split, seed=2002, vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), qa_style=qa_style,
        ),
        'nonce_qswap': generate_qswap_pairs(
            pairs_per_split, seed=2005, vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), nonce_val_prob=1.0, nonce_key_prob=1.0, qa_style=qa_style,
        ),
        'long_clean': generate_pairs(
            pairs_per_split, seed=2004, vocab=OOV_VOCAB,
            families=('city', 'color', 'place', 'person'), distractor_prob=0.0,
            filler_choices=(2,), qa_style=qa_style,
        ),
    }


def evaluate_fact_switch_suite(
    model: LM,
    suite: dict[str, list[dict]],
    device: torch.device | str,
    *,
    do_generate: bool = True,
    gen_limit: int = 8,
    run_tree_ablations: bool = False,
) -> dict:
    """Evaluate all fact-switch splits plus optional causal Tree ablations."""
    results = {}
    for name, items in suite.items():
        limit = len(items) if name in ('user_6pairs', 'user_qswap') else gen_limit
        r = evaluate_pairs(model, items, device, mode='on', do_generate=do_generate, gen_limit=limit)
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
    if run_tree_ablations and model.tree is not None:
        ablations = {}
        for sp in ('long_clean', 'oov_qswap'):
            if sp not in suite:
                continue
            items = suite[sp]
            abl = {
                m: {k: v for k, v in evaluate_pairs(model, items, device, mode=m, do_generate=False).items() if k != 'rows'}
                for m in ('on', 'off', 'shuffle')
            }
            abl['counterfactual_tree_swap'] = evaluate_tree_counterfactual_swap(model, items, device)
            ablations[sp] = abl
        results['tree_ablations'] = ablations
    return results


def fact_switch_selection_key(summary: dict) -> tuple[float, float, float, float]:
    """Rank checkpoints by Phase-2 key-matched qswap first, then clean OOV & greedy generation."""
    qswap = summary.get('oov_qswap', {})
    clean = summary.get('oov_clean', {})
    user_q = summary.get('user_qswap', {})
    qswap_pair = 0.5 * qswap.get('pair_acc', 0.0) + 0.5 * user_q.get('pair_acc', 0.0)
    clean_pair = clean.get('pair_acc', 0.0)
    gen_exact = 0.5 * qswap.get('gen_indiv_exact', 0.0) + 0.5 * clean.get('gen_indiv_exact', 0.0)
    sens = qswap.get('fb_mean_sens', 0.0) + 0.25 * clean.get('fb_mean_sens', 0.0)
    return (round(qswap_pair, 4), round(gen_exact, 4), round(clean_pair, 4), round(sens, 4))


def format_fact_switch_table(label: str, summary: dict) -> str:
    lines = [f'=== {label} ===']
    for sp in ('user_6pairs', 'user_qswap', 'oov_clean', 'oov_qswap', 'nonce_qswap', 'long_clean'):
        if sp not in summary:
            continue
        v = summary[sp]
        gen_s = f" | gen_pair={v['gen_pair_exact']:.2f} gen_ind={v['gen_indiv_exact']:.2f}" if 'gen_pair_exact' in v else ''
        lines.append(
            f"  {sp:12s}: pair={v['pair_acc']:.2f} fb_pair={v['fb_pair_acc']:.2f} "
            f"sens={v['mean_sens']:+5.2f} fb_sens={v['fb_mean_sens']:+5.2f}{gen_s}"
        )
    if 'tree_ablations' in summary:
        lines.append('  Causal Tree ablations (fb_sens):')
        for sp, abl in summary['tree_ablations'].items():
            lines.append(
                f"    {sp:10s}: on={abl['on']['fb_mean_sens']:+5.2f} "
                f"off={abl['off']['fb_mean_sens']:+5.2f} "
                f"shuf={abl['shuffle']['fb_mean_sens']:+5.2f} "
                f"cf_swap={abl['counterfactual_tree_swap']['fb_sens_tree_swapped']:+5.2f} "
                f"(drop={abl['counterfactual_tree_swap']['tree_swap_drop']:+5.2f})"
            )
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
) -> tuple[LM, dict]:
    """Two-stage fact-to-answer adaptation:
      Stage 1 (first 35% budget): plain_nll on mixed clean + 2x2 quads + 60% nonce values (copy-head lock-in).
      Stage 2 (remaining 65%):    contrastive loss on 100% 2x2 counterfactual quads + nonce keys (Phase-2 routing).
    Selects the best weights by counterfactual pair accuracy + greedy exact match (never BPB alone).
    """
    dev = torch.device(device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if dev.type == 'cuda' else nullcontext()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(dev.type == 'cuda'))
    py_rng = random.Random(seed + 401)

    t0 = time.perf_counter()
    deadline = t0 + 60.0 * minutes if minutes > 0 else float('inf')
    stage1_deadline = t0 + 60.0 * minutes * 0.35 if minutes > 0 else float('inf')

    before = evaluate_fact_switch_suite(model, suite, dev, do_generate=True, run_tree_ablations=True)
    best_key = fact_switch_selection_key(before)
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_step, best_summary = 0, before
    if event_fn:
        event_fn(stage='fact_switch_initial', key=list(best_key),
                 oov_clean=before['oov_clean']['pair_acc'], oov_qswap=before['oov_qswap']['pair_acc'])

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
            val_sum = evaluate_fact_switch_suite(model, suite, dev, do_generate=True, gen_limit=6, run_tree_ablations=False)
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
    selected = evaluate_fact_switch_suite(model, suite, dev, do_generate=True, run_tree_ablations=True)
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
    suite = build_fact_switch_suite(qa_style=a.qa_style, pairs_per_split=16)
    model, res = run_two_stage_fact_switch(
        model, suite, device,
        minutes=a.minutes, max_steps=a.max_steps, quads_per_step=a.quads_per_step,
        lr=a.lr, qa_style=a.qa_style, event_fn=lambda **row: print(row, flush=True),
    )
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
