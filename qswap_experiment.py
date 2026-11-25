"""Test Question-Swap + Value-Swap (2x2 Counterfactual Quad) training on HGRT-10M.

Value-swap pairs cancel static word priors, but a positional shortcut ("always copy Sentence 1")
still gets 100% margin whenever the queried fact is in Sentence 1.
Question-swap pairs ((P, Q1) -> v1 paired with (P, Q2) -> v2 on the IDENTICAL passage P)
cancel positional shortcuts identically (S1-only gets margin = 0, pair_acc = 0% on every pair).
Together in a 2x2 quad, the ONLY zero-loss solution is genuine key-matched entity binding.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time

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
    while b[0] == a[0]:
        a, b = rng.sample(pool, 2)
    return a, b


def make_qswap_pair(
    rng: random.Random,
    vocab: dict[str, list[str]],
    direction: str,
    *,
    nonce_val_prob: float = 0.0,
    nonce_key_prob: float = 0.0,
    pair_id: str = 'q',
) -> dict:
    """Question-swap counterfactual pair on the EXACT SAME two-sentence passage.

    prompt_a queries Sentence 1 -> answer_a.
    prompt_b queries Sentence 2 -> answer_b (identical passage bytes!).
    A positional shortcut (always S1 or always S2) gets EXACTLY 0% pair accuracy and 0.0 sensitivity.
    """
    if direction == 'forward':
        e1, e2 = _draw_pair(rng, vocab['events'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['cities'], nonce_val_prob, True)
        d1, d2 = rng.sample(vocab['days'], 2)
        passage = f'The {e1} will be held in {c1} on {d1}. The {e2} will be held in {c2} on {d2}.'
        qa = f'Where will the {e1} be held?'
        qb = f'Where will the {e2} be held?'
        va, vb, fam = c1, c2, 'city'
    elif direction == 'reverse':
        p1 = rng.choice(vocab['people'])
        o1, o2 = _draw_pair(rng, vocab['objects'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['colors'], nonce_val_prob, False)
        l1, l2 = rng.sample(vocab['places'], 2)
        passage = f'{p1} left her {c1} {o1} at the {l1}. {p1} left her {c2} {o2} at the {l2}.'
        qa = f"What color was {p1}'s {o1}?"
        qb = f"What color was {p1}'s {o2}?"
        va, vb, fam = c1, c2, 'color'
    else:
        raise ValueError(direction)

    return dict(
        id=pair_id,
        family=fam,
        direction=direction,
        distractor=True,
        prompt_a=chat_prompt(qa, [], passage, 512),
        prompt_b=chat_prompt(qb, [], passage, 512),
        answer_a=va,
        answer_b=vb,
    )


def make_quad_as_two_pairs(
    rng: random.Random,
    vocab: dict[str, list[str]],
    direction: str,
    *,
    nonce_val_prob: float = 0.5,
    nonce_key_prob: float = 0.5,
) -> list[dict]:
    """Return two question-swap pairs with swapped values (P_ab and P_ba) -> full 2x2 counterfactual quad!

    Batching both pairs together cancels BOTH positional shortcuts (within each pair)
    AND static word priors (across the two pairs)!
    """
    if direction == 'forward':
        e1, e2 = _draw_pair(rng, vocab['events'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['cities'], nonce_val_prob, True)
        d1, d2 = rng.sample(vocab['days'], 2)
        p_ab = f'The {e1} will be held in {c1} on {d1}. The {e2} will be held in {c2} on {d2}.'
        p_ba = f'The {e1} will be held in {c2} on {d1}. The {e2} will be held in {c1} on {d2}.'
        qa = f'Where will the {e1} be held?'
        qb = f'Where will the {e2} be held?'
        va, vb, fam = c1, c2, 'city'
    else:
        p1 = rng.choice(vocab['people'])
        o1, o2 = _draw_pair(rng, vocab['objects'], nonce_key_prob, False)
        c1, c2 = _draw_pair(rng, vocab['colors'], nonce_val_prob, False)
        l1, l2 = rng.sample(vocab['places'], 2)
        p_ab = f'{p1} left her {c1} {o1} at the {l1}. {p1} left her {c2} {o2} at the {l2}.'
        p_ba = f'{p1} left her {c2} {o1} at the {l1}. {p1} left her {c1} {o2} at the {l2}.'
        qa = f"What color was {p1}'s {o1}?"
        qb = f"What color was {p1}'s {o2}?"
        va, vb, fam = c1, c2, 'color'

    return [
        dict(
            id='q1', family=fam, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ab, 512), prompt_b=chat_prompt(qb, [], p_ab, 512),
            answer_a=va, answer_b=vb,
        ),
        dict(
            id='q2', family=fam, direction=direction, distractor=True,
            prompt_a=chat_prompt(qa, [], p_ba, 512), prompt_b=chat_prompt(qb, [], p_ba, 512),
            answer_a=vb, answer_b=va,
        ),
    ]


def build_qswap_eval_suite() -> dict[str, list[dict]]:
    def gen(n: int, seed: int, vocab, direction: str, nv: float = 0.0, nk: float = 0.0):
        rng = random.Random(seed)
        return [
            make_qswap_pair(rng, vocab, direction, nonce_val_prob=nv, nonce_key_prob=nk, pair_id=f'qs{i:02d}')
            for i in range(n)
        ]

    # Also build question-swap versions of the user's exact distractor passages!
    user_qswap = [
        dict(
            id='user_color_qswap', family='color', direction='reverse', distractor=True,
            prompt_a=chat_prompt("What color was Mira's notebook?", [], 'Mira left her blue notebook at the library. Mira left her pink folder in her bag.', 512),
            prompt_b=chat_prompt("What color was Mira's folder?", [], 'Mira left her blue notebook at the library. Mira left her pink folder in her bag.', 512),
            answer_a='blue', answer_b='pink',
        ),
        dict(
            id='user_city_qswap', family='city', direction='forward', distractor=True,
            prompt_a=chat_prompt('Where will the meeting be held?', [], 'The meeting will be held in Lima on Friday. The dinner will be held in Oslo on Sunday.', 512),
            prompt_b=chat_prompt('Where will the dinner be held?', [], 'The meeting will be held in Lima on Friday. The dinner will be held in Oslo on Sunday.', 512),
            answer_a='Lima', answer_b='Oslo',
        ),
    ]
    return {
        'user_qswap': user_qswap,
        'user_4_vswap': [p for p in user_audit_pairs() if p['family'] in ('city', 'color')],
        'train_qswap_fwd': gen(16, 4000, TRAIN_VOCAB, 'forward'),
        'train_qswap_rev': gen(16, 4001, TRAIN_VOCAB, 'reverse'),
        'oov_qswap_fwd': gen(16, 4002, OOV_VOCAB, 'forward'),
        'oov_qswap_rev': gen(16, 4003, OOV_VOCAB, 'reverse'),
        'nonce_qswap_fwd': gen(16, 4004, OOV_VOCAB, 'forward', nv=1.0, nk=1.0),
        'nonce_qswap_rev': gen(16, 4005, OOV_VOCAB, 'reverse', nv=1.0, nk=1.0),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='runs/decisive-probe-stage2/c1_l3l7_contrastive.pt',
                   help='Checkpoint path, or "scratch" to initialize fresh weights')
    p.add_argument('--kind', default='C1', choices=('A0', 'B1', 'C1'))
    p.add_argument('--slots', type=int, default=8)
    p.add_argument('--backend', default='torch', choices=('torch', 'fla'))
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--out', default='runs/qswap-quad')
    p.add_argument('--arm-name', default='c1_l3l7_quad')
    p.add_argument('--read-window', type=int, default=1536)
    p.add_argument('--widen-l3', action='store_true')
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--quads-per-direction', type=int, default=1,
                   help='2x2 quads per direction per step (each quad = 4 sequences; 1 -> batch 8, 4 -> batch 32)')
    p.add_argument('--lr', type=float, default=4e-4)
    p.add_argument('--eval-every', type=int, default=80)
    p.add_argument('--loss-mode', default='contrastive', choices=('plain_nll', 'weighted_nll', 'contrastive'))
    p.add_argument('--contrastive-weight', type=float, default=1.0)
    p.add_argument('--nonce-val-prob', type=float, default=0.5)
    p.add_argument('--nonce-key-prob', type=float, default=0.5)
    p.add_argument('--seed', type=int, default=300)
    a = p.parse_args()

    device = torch.device(a.device)
    amp = torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    torch.manual_seed(a.seed)
    if a.checkpoint != 'scratch':
        ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        kind, slots = ck['kind'], ck['slots']
        model = LM(kind, slots, backend=a.backend, read_window=a.read_window if kind == 'C1' else None).to(device)
        model.load_state_dict(ck['state_dict'])
    else:
        kind, slots = a.kind, (a.slots if a.kind == 'C1' else 0)
        model = LM(kind, slots, backend=a.backend, read_window=a.read_window if kind == 'C1' else None).to(device)
    if kind == 'C1' and a.widen_l3:
        model.blocks[3].mix.window = a.read_window

    suite = build_qswap_eval_suite()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, fused=(device.type == 'cuda'))
    py_rng = random.Random(a.seed + 17)
    t0 = time.perf_counter()
    curve = []

    def snap(step: int):
        do_gen = (step == a.steps)
        res = {}
        row = dict(step=step, seconds=round(time.perf_counter() - t0, 1))
        for k, items in suite.items():
            r = evaluate_pairs(model, items, device, mode='on', do_generate=do_gen, gen_limit=8 if len(items) > 4 else len(items))
            compact = {kk: vv for kk, vv in r.items() if kk != 'rows'}
            if k in ('user_qswap', 'user_4_vswap'):
                compact['rows'] = r['rows']
            elif do_gen:
                compact['sample_generations'] = [
                    {kk: x[kk] for kk in ('answer_a', 'answer_b', 'gen_a', 'gen_b', 'gen_both_exact') if kk in x}
                    for x in r['rows'][:4]
                ]
            res[k] = compact
            row[k] = dict(pair=round(r['pair_acc'], 3), fb_pair=round(r['fb_pair_acc'], 3), fb_sens=round(r['fb_mean_sens'], 2))
            if do_gen:
                row[k]['gen_pair'] = round(r['gen_pair_exact'], 3)
                row[k]['gen_indiv'] = round(r['gen_indiv_exact'], 3)
        curve.append(row)
        g = ''
        if do_gen:
            g = (
                f' | GEN pair: uqs={row["user_qswap"]["gen_pair"]:.2f} '
                f't_f={row["train_qswap_fwd"]["gen_pair"]:.2f} t_r={row["train_qswap_rev"]["gen_pair"]:.2f} '
                f'o_f={row["oov_qswap_fwd"]["gen_pair"]:.2f} o_r={row["oov_qswap_rev"]["gen_pair"]:.2f} '
                f'n_f={row["nonce_qswap_fwd"]["gen_pair"]:.2f} n_r={row["nonce_qswap_rev"]["gen_pair"]:.2f}'
            )
        print(
            f'[{a.arm_name} step={step:3d} {row["seconds"]:5.1f}s] '
            f'uqs={row["user_qswap"]["pair"]:.2f}({row["user_qswap"]["fb_sens"]:+.1f}) '
            f't_f={row["train_qswap_fwd"]["pair"]:.2f} t_r={row["train_qswap_rev"]["pair"]:.2f} '
            f'o_f={row["oov_qswap_fwd"]["pair"]:.2f}({row["oov_qswap_fwd"]["fb_sens"]:+.1f}) '
            f'o_r={row["oov_qswap_rev"]["pair"]:.2f}({row["oov_qswap_rev"]["fb_sens"]:+.1f}) '
            f'n_f={row["nonce_qswap_fwd"]["pair"]:.2f} n_r={row["nonce_qswap_rev"]["pair"]:.2f}'
            + g,
            flush=True,
        )
        return res

    init_res = snap(0)
    warmup = max(15, a.steps // 12)
    last_stats = {}
    for step in range(1, a.steps + 1):
        if step <= warmup:
            cur_lr = a.lr * (step / warmup)
        else:
            prog = (step - warmup) / max(1, a.steps - warmup)
            cur_lr = a.lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * prog)))
        for grp in opt.param_groups:
            grp['lr'] = cur_lr

        batch = []
        for _ in range(a.quads_per_direction):
            batch.extend(make_quad_as_two_pairs(py_rng, TRAIN_VOCAB, 'forward', nonce_val_prob=a.nonce_val_prob, nonce_key_prob=a.nonce_key_prob))
            batch.extend(make_quad_as_two_pairs(py_rng, TRAIN_VOCAB, 'reverse', nonce_val_prob=a.nonce_val_prob, nonce_key_prob=a.nonce_key_prob))
        x, y, bpos, dinfo = pair_batch_tensors(batch, device)
        model.train()
        opt.zero_grad(set_to_none=True)
        with amp:
            logits = model(x)
            loss, last_stats = fact_switch_loss(
                logits, y, bpos, dinfo,
                mode=a.loss_mode, first_byte_weight=3.0, margin_target=3.5, contrastive_weight=a.contrastive_weight,
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            final_res = snap(step)

    ablations = {}
    if model.tree is not None:
        for sp in ('oov_qswap_fwd', 'oov_qswap_rev'):
            items = suite[sp]
            abl = {m: {kk: vv for kk, vv in evaluate_pairs(model, items, device, mode=m, do_generate=False).items() if kk != 'rows'}
                   for m in ('on', 'off', 'shuffle')}
            abl['counterfactual_tree_swap'] = evaluate_tree_counterfactual_swap(model, items, device)
            ablations[sp] = abl

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / 'report.json'
    report = json.loads(report_path.read_text(encoding='utf8')) if report_path.is_file() else dict(arms={})
    report['arms'][a.arm_name] = dict(
        checkpoint=a.checkpoint,
        kind=kind,
        slots=slots,
        device=a.device,
        read_window=a.read_window if kind == 'C1' else None,
        widen_l3=a.widen_l3 if kind == 'C1' else False,
        loss_mode=a.loss_mode,
        steps=a.steps,
        quads_per_direction=a.quads_per_direction,
        lr=a.lr,
        seconds=time.perf_counter() - t0,
        last_train_stats=last_stats,
        initial=init_res,
        final=final_res,
        curve=curve,
        tree_ablations=ablations,
    )
    torch.save(dict(
        kind=kind, slots=slots, read_window=a.read_window if kind == 'C1' else None,
        widen_l3=a.widen_l3 if kind == 'C1' else False,
        state_dict=model.state_dict(), arm=a.arm_name, steps=a.steps,
    ), out_dir / f'{a.arm_name}.pt')
    report_path.write_text(json.dumps(report, indent=2), encoding='utf8')
    print(f'Saved to {report_path}')


if __name__ == '__main__':
    main()
