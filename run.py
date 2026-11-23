"""Time-limited text pilot and learnability-gated recall screening."""
import argparse
import contextlib
import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import FINAL_SEED, STRATA, TRAIN_STAGES, Text, recall, screen_sets
from model import LM


class OutOfTime(Exception):
    pass


def save_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False))
    tmp.replace(path)


class Run:
    def __init__(self, args):
        self.args = args
        self.start = time.monotonic()
        self.end = self.start + 60*args.minutes
        self.dir = Path(args.out)
        self.dir.mkdir(parents=True, exist_ok=True)
        if any(self.dir.iterdir()):
            raise ValueError("Choose an empty output directory to preserve previous runs")
        self.device = torch.device(args.device)
        self.cuda = self.device.type == 'cuda'
        if self.cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use an A100 host or --device cpu for a functional check.")
        if self.cuda and not torch.cuda.is_bf16_supported():
            raise RuntimeError("The GPU must support BF16")
        self.eval_rates = {}
        self.backend = 'torch'
        if args.backend == 'fla' or (args.backend == 'auto' and self.cuda):
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                if not self.cuda:
                    raise ValueError("FLA requires CUDA")
                self.backend = 'fla'
            except ImportError:
                if args.backend == 'fla':
                    raise
        self.report = dict(protocol=args.task, config=vars(args), torch=torch.__version__, numpy=np.__version__,
                           device=torch.cuda.get_device_name() if self.cuda else 'CPU',
                           backend=self.backend, status='in_progress', models={})
        if self.cuda:
            props = torch.cuda.get_device_properties(self.device)
            self.report['cuda'] = dict(runtime=torch.version.cuda, compute_capability=[props.major, props.minor],
                                       device_bytes=props.total_memory, precision='bf16')
        self.checkpoint_report()

    def checkpoint_report(self):
        self.report['elapsed_seconds'] = round(time.monotonic()-self.start, 2)
        save_json(self.dir/'report.json', self.report)

    def event(self, **data):
        data['elapsed'] = round(time.monotonic()-self.start, 2)
        with (self.dir/'metrics.jsonl').open('a') as file:
            print(json.dumps(data, allow_nan=False), file=file)
        print(json.dumps(data), flush=True)

    def guard(self, reserve=0):
        if time.monotonic()+reserve >= self.end:
            raise OutOfTime()

    def model(self, kind, slots=8, seed=None):
        torch.manual_seed(self.args.seed if seed is None else seed)
        m = LM(kind, slots, backend=self.backend).to(self.device)
        assert abs(sum(p.numel() for p in m.parameters())/10_000_000 - 1) <= .005
        return m

    def amp(self):
        return torch.autocast('cuda', torch.bfloat16) if self.cuda else contextlib.nullcontext()

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def optimizer(self, model, lr):
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.01, fused=self.cuda)

    def step(self, m, opt, batch):
        x, *rest = [t.to(self.device, dtype=torch.long) for t in batch]
        opt.zero_grad(set_to_none=True)
        with self.amp():
            if len(rest) == 2:
                positions, targets = rest
                logits = m(x, positions=positions)
            else:
                targets, = rest
                logits = m(x)
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        return loss.item(), x.numel(), targets.numel()

    def batch_size(self, length):
        return max(1, self.args.batch_tokens // length)

    def profile(self, kind, stratum, first=False):
        self.guard()
        m = self.model(kind)
        opt = self.optimizer(m, .001)
        length = STRATA[stratum][0]
        data = recall(np.random.default_rng(700), self.batch_size(length), stratum)
        if self.cuda:
            torch.cuda.reset_peak_memory_stats()
        self.step(m, opt, data)  # includes any first-use kernel compilation
        if first:
            tree_gradients = {name: p.grad.norm().item() if p.grad is not None else 0.
                              for name, p in m.tree.named_parameters()}
            if not all(0 < norm < math.inf for norm in tree_gradients.values()):
                raise RuntimeError(f"2K query loss failed the tree gradient check: {tree_gradients}")
        elapsed = []
        for _ in range(self.args.profile_steps):
            self.guard()
            self.sync()
            start = time.monotonic()
            self.step(m, opt, data)
            self.sync()
            elapsed.append(time.monotonic()-start)
        rate = data[0].numel()/statistics.median(elapsed)
        info = dict(model=kind, length=length, batch=len(data[0]), tokens_per_second=rate,
                    measured_steps=len(elapsed), peak_allocated_bytes=torch.cuda.max_memory_allocated() if self.cuda else None)
        if first:
            info.update(slots=len(m.tree.slots), tree_gradient_norms=tree_gradients,
                        parameters=sum(p.numel() for p in m.parameters()))
        if self.args.task in ('recall', 'controls'):
            # Same batch sizing as screening, including a valid shuffle donor.
            batch = max(2, self.batch_size(length)//2*2)
            ids = recall(np.random.default_rng(701), batch, stratum)[0].to(self.device, dtype=torch.long)
            pos = data[1][:1].to(self.device)
            with torch.no_grad(), self.amp():
                m(ids, positions=pos)
                self.sync()
                start = time.monotonic()
                m(ids, positions=pos)
                self.sync()
                inference_rate = ids.numel()/(time.monotonic()-start)
            self.eval_rates[kind,length] = inference_rate
            info['evaluation_tokens_per_second'] = inference_rate
        self.report.setdefault('profiles', []).append(info)
        self.event(profile=info)
        self.checkpoint_report()
        return rate

    def save_weights(self, model, name):
        dest = self.dir / (name+'.pt')
        torch.save(dict(kind=model.kind, slots=len(model.tree.slots) if model.tree else 0,
                        state_dict=model.state_dict()), dest.with_suffix('.tmp'))
        dest.with_suffix('.tmp').replace(dest)


def measured_state(state, model, batch, chunks):
    caches, forest, offset = state
    s = model.s
    def size(t):
        return t.untyped_storage().nbytes()
    # Every saved tensor owns compact storage. Compare actual bytes with the
    # architecture count (recurrent matrices, conv histories, local K/V, slots).
    actual = sum(size(t) for cache in caches for t in cache) + sum(size(t) for t in forest.values())
    expected = 0
    for i, (x, y) in enumerate(caches):
        if i in (3, 7):
            expected += batch * 2 * min(offset, s.chunk-1)*s.width*x.element_size()
        else:
            expected += batch * (3*3*s.width*x.element_size() +
                                 s.heads*(s.width//s.heads)**2*y.element_size())
    tree_bytes = sum(size(t) for t in forest.values())
    expected += batch * chunks.bit_count() * model.tree.slots.numel() * next(iter(forest.values())).element_size()
    assert len(forest) == chunks.bit_count() and expected == actual, (expected, actual)
    return dict(chunks=chunks, popcount=chunks.bit_count(), tree_bytes=tree_bytes, all_state_bytes=actual)


def engineering(r):
    r.guard()
    model = r.model('C1')
    length = 2049 if r.args.task in ('recall', 'controls') else 257
    x = torch.randint(256, (2, length), device=r.device)
    with torch.no_grad(), r.amp():
        logits = model(x)
        other = x.clone()
        other[:, 159:] = (other[:, 159:]+31) % 256
        future_delta = (logits[:, :159]-model(other)[:, :159]).abs().max().item()
        state, outputs, sizes = None, [], []
        for i, tokens in enumerate(x.split(64, 1)):
            y, state = model.stream_chunk(tokens, state)
            outputs.append(y)
            if i+1 in (1,2,3,4,15,16,31,32) and tokens.size(1) == 64:
                sizes.append(measured_state(state, model, len(x), i+1))
        stream_delta = (logits-torch.cat(outputs, 1)).float().abs().max().item()
    tolerance = .045 if r.cuda else 1e-4
    assert future_delta < 1e-6 and stream_delta < tolerance, (future_delta, stream_delta)
    with r.amp():
        loss = model(x)[:, -1].float().square().mean()
    loss.backward()
    gradients = {k: p.grad.norm().item() for k, p in model.tree.named_parameters()}
    assert all(0 < val < math.inf for val in gradients.values())
    r.report['engineering'] = dict(pass_all=True, future_delta=future_delta, stream_delta=stream_delta,
                                   tolerance=tolerance, gradients=gradients, state_bytes=sizes)
    r.checkpoint_report()


@torch.no_grad()
def text_eval(r, model, valid, mode='on'):
    total = count = 0
    # Pair windows for foreign-memory controls. The very short corpus case uses
    # only an even number of windows, identically across all conditions.
    x, y = valid
    n = len(x) - len(x)%2
    batch = max(2, r.batch_size(x.size(1))//2*2)
    for start in range(0, n, batch):
        r.guard()
        ids = x[start:min(start+batch,n)].to(r.device, dtype=torch.long)
        labels = y[start:min(start+batch,n)].to(r.device, dtype=torch.long)
        with r.amp():
            logits = model(ids, mode)
        total += F.cross_entropy(logits.flatten(0, 1).float(), labels.flatten(), reduction='sum').item()
        count += labels.numel()
    return total / (count*math.log(2))


def text_pilot(r, corpus, rate):
    valid = corpus.validation()
    n = len(valid[0])//2*2
    unigram = corpus.unigram_nll[valid[1][:n].numpy()].mean()/math.log(2)
    r.report['text'] = dict(source=corpus.source, split_sha256=corpus.hashes,
                            validation_bytes=int(valid[1][:n].numel()), train_unigram_bpb=unigram,
                            learning_rule='final BPB below initialization and train-unigram BPB')
    # Spend up to one quarter of remaining time on the text positive control.
    # The C1 arm trains from scratch on the same prefix of training batches.
    for kind in ('A0', 'C1'):
        r.guard()
        m = r.model(kind)
        opt = r.optimizer(m, .001)
        rng = np.random.default_rng(r.args.seed)
        length = 256
        bsz = r.batch_size(length)
        started = time.monotonic()
        initial = text_eval(r, m, valid)
        eval_secs = max(.2, time.monotonic()-started)
        available = r.end-time.monotonic()
        stop = time.monotonic() + (.25*available if kind == 'A0' else available)
        reserve = max((2 if kind == 'A0' else 5)*eval_secs, 2.)
        record = dict(initial_bpb=initial, parameters=sum(p.numel() for p in m.parameters()),
                      steps=0, train_tokens=0)
        r.report['models'][kind] = record
        next_check = 16
        step_time = max(.05, bsz*length/rate)
        while time.monotonic() + reserve + step_time < stop:
            ts = time.monotonic()
            loss, tokens, _ = r.step(m, opt, corpus.sample(rng, bsz, length))
            step_time = .75*step_time + .25*(time.monotonic()-ts)
            record['steps'] += 1
            record['train_tokens'] += tokens
            if record['steps'] == next_check:
                val = text_eval(r, m, valid)
                r.event(model=kind, step=record['steps'], loss=loss, validation_bpb=val)
                next_check *= 2
                if kind == 'A0' and val < min(initial, unigram):
                    break
        final = text_eval(r, m, valid)
        learned = record['steps'] > 0 and final < min(initial, unigram)
        record.update(final_bpb=final, learned=bool(learned), final_perplexity=2**final)
        r.save_weights(m, kind)
        r.checkpoint_report()
        if kind == 'A0' and not learned:
            r.report['status'] = 'inconclusive_text_positive_control'
            return
        if kind == 'C1':
            for mode in ('off', 'shuffle'):
                record[mode+'_bpb'] = text_eval(r, m, valid, mode)
            r.report['status'] = 'text_learning_observed' if learned else 'inconclusive_text_learning'


@torch.no_grad()
def first_predictions(r, model, data, mode='on'):
    x, pos, answers = data
    batch = max(2, r.batch_size(x.size(1))//2*2)
    predictions = []
    position = pos[:1].to(r.device)
    for i in range(0, len(x), batch):
        r.guard()
        with r.amp():
            logits = model(x[i:i+batch].to(r.device, dtype=torch.long), mode, positions=position)
        # Exactly one prespecified first query per independent sequence.
        predictions.append(logits[:, 0].argmax(1).cpu())
    return torch.cat(predictions)


def score(r, model, data, mode='on'):
    return (first_predictions(r, model, data, mode) == data[2][:, 0]).numpy()


def score_control(r, model, data, rec, name):
    predicted = first_predictions(r, model, data)
    answers = data[2]
    correct = (predicted == answers[:, 0]).numpy()
    in_set = (predicted[:, None] == answers).any(dim=1)
    rec.setdefault('first_query_counts', {})[name] = {
        'examples': len(correct), 'correct_key_value': int(correct.sum()),
        'other_presented_value': int(in_set.sum()) - int(correct.sum()),
        'outside_presented_values': int((~in_set).sum()),
    }
    np.save(r.dir / f'A0_{name}.npy', correct)
    return float(correct.mean())


def recall_steps(r, m, opt, stage, steps, seed, record, reserve_seconds=0):
    rng = np.random.default_rng(seed)
    size = r.batch_size(STRATA[stage][0])
    for i in range(steps):
        if reserve_seconds and time.monotonic() + reserve_seconds >= r.end:
            break
        r.guard()
        loss, tokens, targets = r.step(m, opt, recall(rng, size, stage))
        record['steps'] += 1
        record['train_tokens'] += tokens
        record['query_targets'] += targets
        if i % 50 == 0 or i+1 == steps:
            r.event(model=record['name'], stage=stage, step=record['steps'], train_tokens=record['train_tokens'],
                    query_targets=record['query_targets'], loss=loss)


def pilot_lr(r, kind, state=None):
    valid = recall(np.random.default_rng(8901), 128, 'easy')
    trials = {}
    for lr in (.0003, .001, .002):
        m = r.model(kind)
        if state is not None:
            inherit(m, state)
        opt = r.optimizer(m, lr)
        rng = np.random.default_rng(8900)
        for _ in range(r.args.pilot_steps):
            r.guard()
            r.step(m, opt, recall(rng, r.batch_size(256), 'easy'))
        with torch.no_grad():
            total = 0
            x, p, y = valid
            size = r.batch_size(256)
            for i in range(0, len(x), size):
                r.guard()
                with r.amp():
                    logits = m(x[i:i+size].to(r.device, dtype=torch.long), positions=p.to(r.device))
                total += F.cross_entropy(logits.float().flatten(0,1), y[i:i+size].flatten().to(r.device), reduction='sum').item()
            trials[lr] = total/y.numel()
    r.report.setdefault('lr_pilots', {})[kind] = trials
    return min(trials, key=trials.get)


def inherit(model, state):
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected and set(missing) == {k for k in model.state_dict() if k.startswith('tree.')}


def paired(new, old):
    delta = new.astype(float)-old.astype(float)
    rng = np.random.default_rng(130)
    boots = [delta[rng.integers(len(delta), size=(100, len(delta)))].mean(1) for _ in range(20)]
    return dict(difference=delta.mean(), paired_interval95=np.quantile(np.concatenate(boots), [.025,.975]).tolist())


def positive_only(r, sets, rates):
    # First reserve enough time to actually measure the easy endpoint.
    available = r.end-time.monotonic()
    easy_eval_cost = sets['easy'][0].numel()/r.eval_rates['A0',256]
    pilot_cost = 3*r.args.pilot_steps*r.args.batch_tokens/rates['A0','easy'] + 3*128*256/r.eval_rates['A0',256]
    if available < 1.1*(easy_eval_cost+pilot_cost):
        r.report['status'] = 'insufficient_time_for_easy_endpoint'
        return
    lr = pilot_lr(r,'A0')
    model = r.model('A0')
    opt = r.optimizer(model,lr)
    rec = dict(name='A0', steps=0, train_tokens=0, query_targets=0,
               parameters=sum(p.numel() for p in model.parameters()), accuracy={})
    r.report['models']['A0'] = rec
    gates = r.report['gates'] = {}
    available = max(0, r.end-time.monotonic()-1.2*easy_eval_cost-3)
    # In controls mode, first fund the easy warm-up up to its registered quota.
    fraction = .85 if r.args.task == 'controls' else .4
    count = int(min(2e6, available*fraction*rates['A0','easy'])//(256*r.batch_size(256)))
    if not count:
        r.report['status'] = 'insufficient_time_for_easy_endpoint'
        return
    rec['planned_easy_updates'] = count
    recall_steps(r,model,opt,'easy',count,r.args.seed,rec,reserve_seconds=1.2*easy_eval_cost+3)
    # Keep the checkpoint even if the deadline interrupts the held-out sweep.
    r.save_weights(model,'A0')
    r.checkpoint_report()
    easy = score_control(r,model,sets['easy'],rec,'easy')
    rec['accuracy']['easy'] = easy
    gates['0A'] = bool(easy>=.95)
    r.checkpoint_report()
    if not gates['0A']:
        r.report['status'] = 'inconclusive_easy_A0_only'
        return
    for stage in dict.fromkeys(TRAIN_STAGES[1:]):
        if ('A0', stage) not in rates:
            rates['A0', stage] = r.profile('A0', stage)
    avail = r.end-time.monotonic()-1.1*sets['primary31'][0].numel()/r.eval_rates['A0',2048]
    quotas = [.5e6,.5e6,1.5e6,5.5e6]
    factor = min(1., avail / sum(t/rates['A0',s] for t,s in zip(quotas,TRAIN_STAGES[1:])))
    for i,(quota,stage) in enumerate(zip(quotas,TRAIN_STAGES[1:]),1):
        length = STRATA[stage][0]
        count = int(quota*factor//(length*r.batch_size(length)))
        if count < 1:
            r.report['status'] = 'insufficient_time_for_hard_endpoint'
            return
        recall_steps(r,model,opt,stage,count,r.args.seed+100*i,rec)
    r.save_weights(model,'A0')
    r.checkpoint_report()
    hard = score_control(r,model,sets['primary31'],rec,'primary31')
    rec['accuracy']['primary31'] = hard
    gates['0B'] = bool(hard>=.8)
    r.report['status'] = 'learnability_controls_only'


def run_recall(r, sets, base_rate):
    rates = {}
    for kind in ('C1', 'A0', 'B1'):
        for key in dict.fromkeys(TRAIN_STAGES):
            rates[kind, key] = base_rate if (kind,key)==('C1','primary31') else r.profile(kind, key)
    # Budget for 3 independent warm-ups and 6 continuations. Preserve all slots.
    tokens = np.array([2e6, .5e6, .5e6, 1.5e6, 5.5e6])
    cost = np.array([sum(1/rates[k,stage] for k in
                         (['A0','B1','C1'] if i==0 else ['A0','B1']+['C1']*4))
                     for i,stage in enumerate(TRAIN_STAGES)])*tokens
    # Boundary conditions use the nearest benchmark length. Reserve measured
    # forward-only cost separately from full optimizer-step throughput.
    evaluation_seconds = 0.
    for kind in ['A0','B1'] + ['C1']*4:
        for data in sets.values():
            length = data[0].size(1)
            nearest = min((256,512,1024,2048), key=lambda t: abs(t-length))
            evaluation_seconds += data[0].numel()/r.eval_rates[kind,nearest]
    evaluation_seconds += sum(4*sets[key][0].numel()/r.eval_rates['C1',2048]
                              for key in ('primary31','carry32'))
    overhead = evaluation_seconds + sum(
        3*r.args.pilot_steps*r.args.batch_tokens/rates[k,'easy'] +
        (3*128*256+2*sets['easy'][0].numel())/r.eval_rates[k,256]
        for k in ('A0','B1','C1')
    )
    remaining = max(0, (r.end-time.monotonic())*.94-overhead)
    # Cut stage one first (floor of 128k tokens), then all continuations in lockstep.
    warm_factor = min(1., max(.064, (remaining-cost[1:].sum())/cost[0]))
    cont_factor = min(1., max(0., (remaining-cost[0]*warm_factor)/cost[1:].sum()))
    tokens[0] *= warm_factor
    tokens[1:] *= cont_factor
    steps = [int(tok//(r.batch_size(STRATA[s][0])*STRATA[s][0])) for tok,s in zip(tokens,TRAIN_STAGES)]
    r.report['schedule'] = dict(steps=steps, warm_factor=warm_factor, continuation_factor=cont_factor,
                                projected_full_training_seconds=cost.sum(), reserved_overhead_seconds=overhead,
                                evaluation_examples=len(sets['easy'][0]))
    r.checkpoint_report()
    if min(steps) < 1:
        # A complete comparison cannot fit. Spend the available time testing
        # learnability with A0, keeping the same frozen endpoints and quotas.
        r.report['reduced_screen'] = 'positive_control_only'
        positive_only(r, sets, rates)
        return
    r.report['reduced_screen'] = bool(len(sets['easy'][0]) != 2000 or min(cont_factor,warm_factor)<1)

    predictions, warm_scores = {}, {}
    def record_for(name, m):
        rec = dict(name=name, steps=0, train_tokens=0, query_targets=0,
                   parameters=sum(p.numel() for p in m.parameters()), accuracy={})
        r.report['models'][name] = rec
        return rec

    def train_stage(m, opt, i, rec, seed=None):
        seed = r.args.seed if seed is None else seed
        recall_steps(r, m, opt, TRAIN_STAGES[i], steps[i], seed+100*i, rec)

    def evaluate(m, rec, names, mode='on'):
        pred = predictions.setdefault(rec['name'], {})
        for name in names:
            key = name if mode=='on' else name+'_'+mode
            pred[key] = score(r, m, sets[name], mode)
            rec['accuracy'][key] = pred[key].mean()
            if name in ('carry15','carry16','primary31','carry32'):
                length, _, _, _, anchor = STRATA[name]
                popcount = (anchor // 64).bit_count()
                by_count = rec.setdefault('boundary_by_popcount_and_length', {}).setdefault(mode, {})
                by_count.setdefault(popcount, {})[length] = rec['accuracy'][key]
            np.save(r.dir / (rec['name']+'_'+key+'.npy'), pred[key])
            r.checkpoint_report()

    controls = {}
    for kind in ('A0','B1'):
        lr = pilot_lr(r, kind)
        m = r.model(kind)
        rec = record_for(kind, m)
        opt = r.optimizer(m, lr)
        train_stage(m, opt, 0, rec)
        evaluate(m, rec, ['easy'])
        warm_scores[kind] = rec['accuracy']['easy']
        rec['warm_easy'] = warm_scores[kind]
        controls[kind] = m, opt, rec
    gates = r.report['gates'] = {'0A': bool(max(warm_scores.values()) >= .95)}
    if not gates['0A']:
        for kind, (m, _, _) in controls.items():
            r.save_weights(m, kind+'-warm')
        r.report['status'] = 'inconclusive_easy_learnability'
        return
    b1 = controls['B1'][0]
    warm = {k: v.detach().cpu().clone() for k,v in b1.state_dict().items()}
    warm_counts = {k: controls['B1'][2][k] for k in ('steps','train_tokens','query_targets')}
    r.save_weights(b1, 'B1-warm')
    for kind in ('A0','B1'):
        m, opt, rec = controls.pop(kind)
        if kind == 'B1':
            opt = r.optimizer(m, opt.param_groups[0]['lr'])  # same moment reset as C1 branches
        for i in range(1,5):
            train_stage(m, opt, i, rec)
        evaluate(m, rec, sets)
        r.save_weights(m, kind)
        if kind == 'A0':
            gates['0B'] = bool(rec['accuracy']['primary31'] >= .8)
            if not gates['0B']:
                r.report['status'] = 'inconclusive_hard_learnability'
                return
    lr = pilot_lr(r, 'C1', warm)
    for slots, scratch in [(4,False),(8,False),(16,False),(8,True)]:
        name = 'S'+str(slots)+('-scratch' if scratch else '')
        m = r.model('C1', slots)
        rec = record_for(name, m)
        if not scratch:
            inherit(m, warm)
            rec.update(warm_counts)
        opt = r.optimizer(m, lr)
        for i in range(0 if scratch else 1,5):
            train_stage(m, opt, i, rec)
            if scratch and i == 0:
                rec['warm_easy'] = float(score(r, m, sets['easy']).mean())
        evaluate(m, rec, sets)
        if slots == 8:
            for mode in ('off','shuffle'):
                evaluate(m, rec, ('primary31','carry32'), mode)
        r.save_weights(m, name)

    def acc(name, key='primary31'):
        return predictions[name][key].mean()

    comparisons = r.report['paired_S8_minus_B1'] = {key: paired(predictions['S8'][key], predictions['B1'][key])
                                                     for key in sets}
    # Integer counts avoid floating point edge cases at exact gate thresholds.
    eps = 1e-9
    gates['1'] = all(comparisons[k]['difference']+eps >= .05 for k in ('primary31','carry32'))
    gates['2'] = bool(acc('S8')+eps >= acc('S4')-.02 and acc('S16')+eps >= acc('S8')-.02
                      and acc('S16')-acc('S4')+eps >= .03)
    def contributes(name):
        return all(acc(name)-acc(name,'primary31_'+mode)+eps >= .03 for mode in ('off','shuffle'))
    gates['3'] = bool(contributes('S8'))
    gates['4'] = bool(r.report['models']['S8-scratch']['warm_easy'] >= .95 and contributes('S8-scratch'))
    r.report['paired_tree_controls'] = {
        name: {key+'_'+mode: paired(predictions[name][key], predictions[name][key+'_'+mode])
               for key in ('primary31','carry32') for mode in ('off','shuffle')}
        for name in ('S8','S8-scratch')
    }
    gates['5'] = r.report['engineering']['pass_all']
    gates['B1_short_load_check'] = bool(acc('B1','short16') >= .80)
    r.report['B1_short16_accuracy'] = acc('B1','short16')
    if all(gates.values()):
        r.report['status'] = 'candidate_for_three_seed_comparison'
    elif all(value for key, value in gates.items() if key != '4') and not gates['4']:
        r.report['status'] = 'warm_start_dependent'
    else:
        r.report['status'] = 'inconclusive_single_seed_screen'
    r.checkpoint_report()
    # A second seed estimates repeatability only if both complete trajectories fit.
    seed = r.args.seed+1
    warm_seconds = steps[0]*r.batch_size(256)*256/rates['B1','easy']
    need = warm_seconds
    for kind in ('B1','C1'):
        for i in range(1,5):
            stage = TRAIN_STAGES[i]
            length = STRATA[stage][0]
            need += steps[i]*r.batch_size(length)*length/rates[kind,stage]
        need += sum(sets[key][0].numel()/rates[kind,'primary31'] for key in ('easy','primary31','carry32'))
    r.report['repeat_estimated_seconds'] = need
    if time.monotonic()+1.2*need < r.end:
        b_lr = min(r.report['lr_pilots']['B1'], key=r.report['lr_pilots']['B1'].get)
        base = r.model('B1', seed=seed)
        rec = record_for(f'B1-seed{seed}', base)
        train_stage(base, r.optimizer(base,b_lr), 0, rec, seed)
        branch = r.model('C1', seed=seed)
        inherit(branch, base.state_dict())
        branch_rec = record_for(f'S8-seed{seed}', branch)
        for key in ('steps','train_tokens','query_targets'):
            branch_rec[key] = rec[key]
        for m, rec, trial_lr in ((base,rec,b_lr),(branch,branch_rec,lr)):
            opt = r.optimizer(m,trial_lr)
            for i in range(1,5):
                train_stage(m,opt,i,rec,seed)
            evaluate(m,rec,('easy','primary31','carry32'))
            r.save_weights(m,rec['name'])
        r.report['repeat_paired_S8_minus_B1'] = {
            key: paired(predictions[f'S8-seed{seed}'][key], predictions[f'B1-seed{seed}'][key])
            for key in ('primary31','carry32')
        }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('task', choices=['text','recall','controls','profile'])
    p.add_argument('--minutes', type=float, default=20)
    p.add_argument('--out', required=True)
    p.add_argument('--text-file', help='UTF-8 plain text; defaults to installed Python help topics')
    p.add_argument('--backend', choices=['auto','fla','torch'], default='auto')
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-tokens', type=int, default=8192)
    p.add_argument('--examples', type=int, default=2000, help='even, fixed screening set size per stratum')
    p.add_argument('--pilot-steps', type=int, default=12)
    p.add_argument('--profile-steps', type=int, default=3)
    a = p.parse_args()
    if a.minutes <= 0 or min(a.batch_tokens,a.pilot_steps,a.profile_steps)<1 or a.examples<2 or a.examples%2:
        p.error('positive budgets and an even evaluation count >=2 are required')
    r = Run(a)
    try:
        if a.task in ('recall', 'controls'):
            sets, manifest = screen_sets(a.examples)
            save_json(r.dir/'screen.json', dict(strata=manifest, reserved_final_seed=FINAL_SEED))
        corpus = Text(a.text_file) if a.task == 'text' else None
        rate = r.profile('C1','primary31',first=True)
        r.report['below_5k_at_2k'] = rate < 5000
        engineering(r)
        if a.task == 'text':
            text_pilot(r, corpus, rate)
        elif a.task == 'recall':
            run_recall(r, sets, rate)
        elif a.task == 'controls':
            r.report['reduced_screen'] = 'positive_control_only'
            positive_only(r, sets, {('A0', 'easy'): r.profile('A0', 'easy')})
        else:
            r.report['status'] = 'profile_complete'
    except OutOfTime:
        if r.report['status'] == 'in_progress':
            r.report['status'] = 'inconclusive_time_limit'
        else:
            r.report['repeat_incomplete'] = True
    except Exception as exc:
        r.report.update(status='failed', error=repr(exc))
        raise
    finally:
        r.checkpoint_report()
        print(r.report['status'], r.dir/'report.json', flush=True)


if __name__ == '__main__':
    main()
