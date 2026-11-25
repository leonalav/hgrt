"""Focused decisive experiment: counterfactual fact-switching & exact OOV byte copying in HGRT-10M.

Isolates the core question on two strictly symmetric, shortcut-free templates (1 forward KEY->VALUE
city template + 1 reverse VALUE->KEY color template) with anti-memorization broad+nonce training:
  - Can HGRT-10M (C1-S8) reliably switch its answer AND greedily generate exact unseen OOV words?
  - How do L7-only (L3=64, L7=1536), L3+L7 (both=1536), native HGRT (L3=64, L7=64, Delta+Tree),
    plain NLL vs. contrastive loss, and same-size 10M Transformer (A0) compare?
  - How much does the Hierarchical Tree causally contribute when wide attention is disabled (rw=64)?
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from dialogue import chat_prompt
from fact_switch import (
    OOV_VOCAB,
    TRAIN_VOCAB,
    evaluate_pairs,
    evaluate_tree_counterfactual_swap,
    fact_switch_loss,
    pair_batch_tensors,
    random_nonce,
    user_audit_pairs,
)
from model import LM


def _draw_pair(rng: random.Random, pool: list[str], nonce_prob: float, cap: bool) -> tuple[str, str]:
    if nonce_prob > 0 and rng.random() < nonce_prob:
        a = random_nonce(rng, capitalize=cap)
        b = random_nonce(rng, capitalize=cap)
        while b.casefold() == a.casefold() or b[0] == a[0]:
            b = random_nonce(rng, capitalize=cap)
        return a, b
    a, b = rng.sample(pool, 2)
    return a, b


def make_focused_pair(
    rng: random.Random,
    vocab: dict[str, list[str]],
    direction: str,
    distractor: bool,
    *,
    nonce_val_prob: float = 0.0,
    nonce_key_prob: float = 0.0,
    pair_id: str = 'p',
) -> dict:
    """Strictly symmetric focused passage (>64 bytes from sentence 1 to Assistant: )."""
    if direction == 'forward':
        # Forward KEY -> VALUE: event precedes city. Both sentences identical in structure!
        e1, e2 = _draw_pair(rng, vocab['events'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['cities'], nonce_val_prob, True)
        d1, d2 = rng.sample(vocab['days'], 2)
        s1 = lambda v: f'The {e1} will be held in {v} on {d1}.'
        s2 = lambda v: f'The {e2} will be held in {v} on {d2}.'
        q = f'Where will the {e1} be held?'
        fam = 'city'
        va, vb = c1, c2
    elif direction == 'reverse':
        # Reverse VALUE -> KEY: color precedes object. Same person -> ONLY object distinguishes them!
        p1 = rng.choice(vocab['people'])
        o1, o2 = _draw_pair(rng, vocab['objects'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['colors'], nonce_val_prob, False)
        l1, l2 = rng.sample(vocab['places'], 2)
        s1 = lambda v: f'{p1} left her {v} {o1} at the {l1}.'
        s2 = lambda v: f'{p1} left her {v} {o2} at the {l2}.'
        q = f"What color was {p1}'s {o1}?"
        fam = 'color'
        va, vb = c1, c2
    else:
        raise ValueError(direction)

    flip = distractor and (rng.randrange(2) == 1)

    def compose(v_main: str, v_dist: str) -> str:
        if not distractor:
            return s1(v_main)
        return f'{s2(v_dist)} {s1(v_main)}' if flip else f'{s1(v_main)} {s2(v_dist)}'

    return dict(
        id=pair_id,
        family=fam,
        direction=direction,
        distractor=distractor,
        prompt_a=chat_prompt(q, [], compose(va, vb), 512),
        prompt_b=chat_prompt(q, [], compose(vb, va), 512),
        answer_a=va,
        answer_b=vb,
    )


def build_focused_eval_suite() -> dict[str, list[dict]]:
    def gen(n: int, seed: int, direction: str, dist: bool, nv: float = 0.0, nk: float = 0.0):
        rng = random.Random(seed)
        return [
            make_focused_pair(rng, OOV_VOCAB, direction, dist, nonce_val_prob=nv, nonce_key_prob=nk, pair_id=f'e{i:02d}')
            for i in range(n)
        ]

    u6 = user_audit_pairs()
    return {
        'user_4': [p for p in u6 if p['family'] in ('city', 'color')],
        'oov_fwd_clean': gen(16, 3001, 'forward', False),
        'oov_fwd_dist': gen(16, 3002, 'forward', True),
        'oov_rev_clean': gen(16, 3003, 'reverse', False),
        'oov_rev_dist': gen(16, 3004, 'reverse', True),
        'nonce_fwd_dist': gen(16, 3005, 'forward', True, nv=1.0, nk=0.5),
        'nonce_rev_dist': gen(16, 3006, 'reverse', True, nv=1.0, nk=0.5),
    }


def eval_focused(model: LM, suite: dict[str, list[dict]], device: torch.device, do_gen: bool = True) -> dict:
    out = {}
    for name, items in suite.items():
        r = evaluate_pairs(model, items, device, mode='on', do_generate=do_gen, gen_limit=8 if name != 'user_4' else 4)
        compact = {k: v for k, v in r.items() if k != 'rows'}
        if name == 'user_4':
            compact['rows'] = r['rows']
        elif do_gen:
            compact['sample_generations'] = [
                {k: row[k] for k in ('answer_a', 'answer_b', 'gen_a', 'gen_b', 'gen_both_exact') if k in row}
                for row in r['rows'][:4]
            ]
        out[name] = compact
    return out


def train_and_eval_arm(
    arm_name: str,
    *,
    kind: str,
    slots: int,
    base_state: dict | None,
    read_window: int | None,
    widen_l3: bool,
    loss_mode: str,
    steps: int,
    pairs_per_step: int,
    lr: float,
    seed: int,
    device: torch.device,
    suite: dict[str, list[dict]],
    eval_every: int,
    backend: str = 'torch',
    nonce_val_prob: float = 0.6,
    nonce_key_prob: float = 0.25,
    distractor_only: bool = False,
) -> tuple[LM, dict]:
    torch.manual_seed(seed)
    model = LM(kind, slots, backend=backend, read_window=read_window if kind == 'C1' else None).to(device)
    if base_state is not None:
        model.load_state_dict(base_state)
    if kind == 'C1' and widen_l3 and read_window is not None:
        model.blocks[3].mix.window = read_window
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, fused=(device.type == 'cuda'))
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()

    py_rng = random.Random(seed + 777)
    curve = []
    t0 = time.perf_counter()

    def snap(step: int):
        do_gen = (step == steps)
        res = eval_focused(model, suite, device, do_gen=do_gen)
        row = dict(step=step, seconds=round(time.perf_counter() - t0, 1))
        for k, v in res.items():
            entry = dict(pair=round(v['pair_acc'], 3), fb_pair=round(v['fb_pair_acc'], 3), fb_sens=round(v['fb_mean_sens'], 2))
            if 'gen_pair_exact' in v:
                entry['gen_pair'] = round(v['gen_pair_exact'], 3)
                entry['gen_indiv'] = round(v['gen_indiv_exact'], 3)
            row[k] = entry
        curve.append(row)
        g = ''
        if do_gen:
            g = (
                f' | GEN pair: u4={row["user_4"]["gen_pair"]:.2f} '
                f'fc={row["oov_fwd_clean"]["gen_pair"]:.2f} fd={row["oov_fwd_dist"]["gen_pair"]:.2f} '
                f'rc={row["oov_rev_clean"]["gen_pair"]:.2f} rd={row["oov_rev_dist"]["gen_pair"]:.2f} '
                f'nfd={row["nonce_fwd_dist"]["gen_pair"]:.2f} nrd={row["nonce_rev_dist"]["gen_pair"]:.2f}'
            )
        print(
            f'[{arm_name:22s} step={step:3d} {row["seconds"]:5.1f}s] '
            f'u4={row["user_4"]["pair"]:.2f}({row["user_4"]["fb_sens"]:+.1f}) '
            f'fc={row["oov_fwd_clean"]["pair"]:.2f} fd={row["oov_fwd_dist"]["pair"]:.2f} '
            f'rc={row["oov_rev_clean"]["pair"]:.2f} rd={row["oov_rev_dist"]["pair"]:.2f} '
            f'nfd={row["nonce_fwd_dist"]["pair"]:.2f} nrd={row["nonce_rev_dist"]["pair"]:.2f}'
            + g,
            flush=True,
        )
        return res

    init_res = snap(0)
    warmup = max(15, steps // 12)
    last_stats = {}
    for step in range(1, steps + 1):
        if step <= warmup:
            cur_lr = lr * (step / warmup)
        else:
            prog = (step - warmup) / max(1, steps - warmup)
            cur_lr = lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for grp in opt.param_groups:
            grp['lr'] = cur_lr

        # Progressive distractor curriculum unless distractor_only is forced.
        dist_p = 1.0 if distractor_only else (0.35 if step <= steps // 4 else 0.85)
        batch = []
        for j in range(pairs_per_step):
            d = 'forward' if (j % 2 == 0) else 'reverse'
            batch.append(make_focused_pair(
                py_rng, TRAIN_VOCAB, d, py_rng.random() < dist_p,
                nonce_val_prob=nonce_val_prob, nonce_key_prob=nonce_key_prob, pair_id=f't{j}',
            ))
        x, y, bpos, dinfo = pair_batch_tensors(batch, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        with amp:
            logits = model(x)
            loss, last_stats = fact_switch_loss(logits, y, bpos, dinfo, mode=loss_mode)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % eval_every == 0 or step == steps:
            final_res = snap(step)

    ablations = {}
    if model.tree is not None:
        for sp in ('oov_fwd_dist', 'oov_rev_dist'):
            abl = {}
            for m in ('on', 'off', 'shuffle'):
                r = evaluate_pairs(model, suite[sp], device, mode=m, do_generate=False)
                abl[m] = {k: v for k, v in r.items() if k != 'rows'}
            abl['counterfactual_tree_swap'] = evaluate_tree_counterfactual_swap(model, suite[sp], device)
            ablations[sp] = abl

    return model, dict(
        arm=arm_name,
        kind=kind,
        pretrained=base_state is not None,
        read_window=read_window,
        widen_l3=widen_l3,
        loss_mode=loss_mode,
        steps=steps,
        pairs_per_step=pairs_per_step,
        lr=lr,
        seconds=time.perf_counter() - t0,
        last_train_stats=last_stats,
        initial=init_res,
        final=final_res,
        curve=curve,
        tree_ablations=ablations,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/best_lm.pt')
    p.add_argument('--out', default='runs/decisive-probe')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--backend', default='torch', choices=('torch', 'fla'))
    p.add_argument('--steps', type=int, default=260)
    p.add_argument('--pairs-per-step', type=int, default=6)
    p.add_argument('--lr', type=float, default=6e-4)
    p.add_argument('--eval-every', type=int, default=65)
    p.add_argument('--distractor-only', action='store_true')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--arms', nargs='+', default=['c1_l7_contrastive'])
    a = p.parse_args()

    device = torch.device(a.device)
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf8')) if report_path.is_file() else dict(
        checkpoint=a.checkpoint, device=a.device, steps=a.steps, pairs_per_step=a.pairs_per_step, arms={},
    )
    suite = build_focused_eval_suite()

    specs = {
        'c1_l7_plain':         dict(kind='C1', slots=8, base=ck['state_dict'], rw=1536, l3=False, loss='plain_nll',   lr=a.lr),
        'c1_l3l7_plain':       dict(kind='C1', slots=8, base=ck['state_dict'], rw=1536, l3=True,  loss='plain_nll',   lr=a.lr),
        'c1_l7_contrastive':   dict(kind='C1', slots=8, base=ck['state_dict'], rw=1536, l3=False, loss='contrastive', lr=a.lr),
        'c1_l3l7_contrastive': dict(kind='C1', slots=8, base=ck['state_dict'], rw=1536, l3=True,  loss='contrastive', lr=a.lr),
        'c1_rw64_contrastive': dict(kind='C1', slots=8, base=ck['state_dict'], rw=64,   l3=False, loss='contrastive', lr=a.lr),
        'c1_rw64_plain':       dict(kind='C1', slots=8, base=ck['state_dict'], rw=64,   l3=False, loss='plain_nll',   lr=a.lr),
        'a0_scratch_contr':    dict(kind='A0', slots=0, base=None,             rw=None, l3=False, loss='contrastive', lr=1e-3),
        'c1_scratch_l3l7':     dict(kind='C1', slots=8, base=None,             rw=1536, l3=True,  loss='contrastive', lr=1e-3),
    }

    for name in a.arms:
        s = specs[name]
        model, res = train_and_eval_arm(
            name,
            kind=s['kind'], slots=s['slots'], base_state=s['base'],
            read_window=s['rw'], widen_l3=s['l3'], loss_mode=s['loss'],
            steps=a.steps, pairs_per_step=a.pairs_per_step, lr=s['lr'],
            seed=a.seed, device=device, suite=suite, eval_every=a.eval_every,
            backend=a.backend, distractor_only=a.distractor_only,
        )
        report['arms'][name] = res
        torch.save(dict(
            kind=s['kind'], slots=s['slots'], read_window=s['rw'], widen_l3=s['l3'],
            state_dict=model.state_dict(), arm=name, steps=a.steps,
        ), out_dir / f'{name}.pt')
        report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(f'Updated {report_path}')


if __name__ == '__main__':
    main()
