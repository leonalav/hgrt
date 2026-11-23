import tempfile
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from data import STRATA, Text, recall, screen_sets
from model import LM, Shape, delta_reference
from run import inherit, measured_state, paired, positive_only, run_recall, score_control


class RecallData(unittest.TestCase):
    def test_oracle_all_queries(self):
        for name, (length, load, near, far, anchor) in STRATA.items():
            x, positions, answers = recall(np.random.default_rng(42), 64, name)
            for seq, labels in zip(x.numpy(), answers.numpy()):
                facts = np.where(seq == 1)[0]
                values = dict(zip(seq[facts+1], seq[facts+2]))
                self.assertEqual(len(values), load)
                qkeys = seq[positions.numpy()]
                np.testing.assert_array_equal([values[k] for k in qkeys], labels)
                self.assertEqual(len(set(qkeys)), load)
                self.assertTrue(np.all(seq[positions.numpy()-1] == 2))
                self.assertFalse(np.any((seq[anchor-1:]>=64) & (seq[anchor-1:]<96)))
                distance = anchor - int(facts[np.where(seq[facts+1] == qkeys[0])[0][0]]) - 2
                self.assertTrue(near <= distance <= far)
                self.assertEqual(len(seq), length)

    def test_boundary_matching(self):
        sets, manifests = screen_sets(8)
        for a,b in [('carry15','carry16'),('primary31','carry32')]:
            left,p,y = sets[a]
            right,q,z = sets[b]
            torch.testing.assert_close(y,z)
            self.assertEqual(int(q[0]-p[0]),64)
            # All fact and query tokens match after translation.
            mask = left < 128
            torch.testing.assert_close(right[:,64:][mask],left[mask])
        self.assertEqual(manifests['primary31']['chunks'],31)
        self.assertEqual(manifests['carry32']['popcount'],1)
        self.assertEqual(manifests,screen_sets(8)[1])

    def test_text_splits(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'text'
            p.write_bytes(b'a'*16000+b'b'*2000+b'c'*2000)
            corpus = Text(p)
            x,y = corpus.sample(np.random.default_rng(0),8,256)
            self.assertTrue((x==ord('a')).all() and (y==ord('a')).all())
            self.assertTrue((corpus.validation()[1]==ord('b')).all())


class Architecture(unittest.TestCase):
    def test_query_logits_and_gradients_match_full_head(self):
        shape = Shape(width=32, heads=2, chunk=8, slot_dim=16, target=200000)
        tokens = torch.randint(256, (2, 33))
        pos = torch.tensor([17, 24, 32])
        for kind in ('A0', 'B1', 'C1'):
            model = LM(kind, s=shape)
            full = model(tokens)[:, pos]
            sparse = model(tokens, positions=pos)
            torch.testing.assert_close(full, sparse)
            grads = torch.autograd.grad(full.square().mean(), model.parameters())
            selected_grads = torch.autograd.grad(sparse.square().mean(), model.parameters())
            for expected, actual in zip(grads, selected_grads):
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-4)

    def test_recurrence_values_and_gradients(self):
        torch.manual_seed(12)
        q,k,v = [torch.randn(2,67,2,4) for _ in range(3)]
        args = [q,F.normalize(k,dim=-1),v,-torch.rand(2,67,2)*.2,torch.rand(2,67,2),torch.randn(2,2,4,4)]
        q,k,v,g,beta,state = [t.requires_grad_() for t in args]
        y, last = delta_reference(*args)
        stream = []
        for i in range(67):
            state = state*g[:,i].exp()[...,None,None]
            key = k[:,i].unsqueeze(-1)
            err = v[:,i]-(key.transpose(-1,-2)@state).squeeze(-2)
            state = state + key @ (beta[:,i].unsqueeze(-1)*err).unsqueeze(-2)
            stream.append((q[:,i].unsqueeze(-2)@state).squeeze(-2))
        oracle = torch.stack(stream,1)
        torch.testing.assert_close(y,oracle,atol=5e-6,rtol=1e-4)
        torch.testing.assert_close(last,state,atol=3e-6,rtol=1e-4)
        grad = torch.autograd.grad(y.square().mean()+last.square().mean(),args)
        oracle_grad = torch.autograd.grad(oracle.square().mean()+state.square().mean(),args)
        for a,b in zip(grad,oracle_grad):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=1e-4)

    def test_carries_ablation_and_stream(self):
        torch.manual_seed(3)
        s = Shape(width=32,heads=2,chunk=8,slot_dim=16,target=200000)
        m = LM('C1',s=s)
        x = torch.randint(256,(2,257))
        with torch.no_grad():
            y = m(x)
            state = None
            parts = []
            for i,t in enumerate(x.split(8,1)):
                part,state = m.stream_chunk(t,state)
                parts.append(part)
                if i+1 in (15,16,31,32):
                    measured_state(state,m,len(x),i+1)
            torch.testing.assert_close(torch.cat(parts,1),y,rtol=2e-4,atol=2e-5)
            for mode in ('on','off','shuffle'):
                z = x.clone()
                z[:,171:] = 0
                torch.testing.assert_close(m(z,mode)[:,:171],m(x,mode)[:,:171])
            dup = x[:1].repeat(2,1)
            torch.testing.assert_close(m(dup),m(dup,'shuffle'))
            self.assertGreater((y-m(x,'shuffle')).abs().max().item(),1e-5)
            self.assertGreater((y-m(x,'off')).abs().max().item(),1e-5)

    def test_historical_gradient(self):
        s = Shape(width=32,heads=2,slot_dim=16,chunk=8,target=200000)
        tree = LM(s=s).tree
        x = torch.randn(2,33,32)
        source = torch.randn_like(x,requires_grad=True)
        tree(x,source,'on')[:,32:].sum().backward()
        for chunk in source.grad[:,:32].split(8,1):
            self.assertGreater(chunk.norm().item(),0)
        self.assertEqual(source.grad[:,32:].abs().max().item(),0)

    def test_budget_and_warm_transfer(self):
        b = LM('B1')
        x = torch.randint(256,(1,129))
        with torch.no_grad():
            control = b(x)
            for n in (4,8,16):
                m = LM('C1',n)
                self.assertLessEqual(abs(sum(p.numel() for p in m.parameters())/1e7-1),.005)
                inherit(m,b.state_dict())
                torch.testing.assert_close(control,m(x,'off'),rtol=0,atol=0)
        pair = paired(np.ones(10),np.zeros(10))
        self.assertEqual(pair['paired_interval95'],[1.,1.])


class Protocol(unittest.TestCase):
    def test_control_error_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = SimpleNamespace(dir=Path(tmp))
            answers = torch.tensor([[64, 65], [70, 71], [66, 67]])
            record = {}
            with patch('run.first_predictions', return_value=torch.tensor([64, 71, 72])):
                accuracy = score_control(r, None, (None, None, answers), record, 'easy')
            self.assertAlmostEqual(accuracy, 1/3)
            self.assertEqual(record['first_query_counts']['easy'], dict(
                examples=3, correct_key_value=1, other_presented_value=1, outside_presented_values=1))
            np.testing.assert_array_equal(np.load(r.dir/'A0_easy.npy'), [True, False, False])

    def test_control_proceeds_to_long_lengths_only_after_easy_gate(self):
        sets, _ = screen_sets(8)
        for easy, passed in [(.94, False), (.95, True)]:
            r = Mock()
            r.report = {'models': {}}
            r.args = SimpleNamespace(task='controls', pilot_steps=1, batch_tokens=8192, seed=42)
            r.model.return_value = torch.nn.Linear(4, 4)
            r.end = time.monotonic() + 600
            r.eval_rates = {('A0', 256): 1e7, ('A0', 2048): 1e7}
            r.batch_size.side_effect = lambda length: 8192 // length
            r.profile.return_value = 1e7
            with patch('run.pilot_lr', return_value=.001), patch('run.recall_steps'), \
                 patch('run.score_control', side_effect=[easy, .80]):
                positive_only(r, sets, {('A0', 'easy'): 1e7})
            self.assertEqual(r.report['gates']['0A'], passed)
            if passed:
                self.assertTrue(r.report['gates']['0B'])
                self.assertEqual(r.profile.call_count, 3)
            else:
                r.profile.assert_not_called()
                self.assertNotIn('0B', r.report['gates'])

    def test_gating_and_paired_decision(self):
        # Test routing independently of whether one stochastic training run converges.
        sets, _ = screen_sets(100)
        shape = Shape(width=16, heads=2, slot_dim=8, target=50000)
        for a0_hard, expected in [(79, 'inconclusive_hard_learnability'),
                                  (80, 'candidate_for_three_seed_comparison')]:
            with tempfile.TemporaryDirectory() as d:
                r = Mock()
                r.dir = Path(d)
                r.args = SimpleNamespace(batch_tokens=8192, seed=42, pilot_steps=4)
                r.end = time.monotonic()+999
                r.report = {'models':{}, 'engineering':{'pass_all':True}}
                r.eval_rates = {(k,l):1e9 for k in ('A0','B1','C1') for l in (256,512,1024,2048)}
                r.profile.return_value = 1e9
                r.batch_size.side_effect = lambda l: 8192//l
                r.model.side_effect = lambda kind, slots=8, seed=None: LM(kind,slots,s=shape)
                r.optimizer.side_effect = lambda m,lr: SimpleNamespace(param_groups=[{'lr':lr}])

                def pilot(r,kind,state=None):
                    r.report.setdefault('lr_pilots',{})[kind] = {.001:1.5}
                    return .001

                def fixed_score(r,m,data,mode='on'):
                    if data is sets['easy'] or data is sets['short16']:
                        n = 95
                    elif m.kind == 'A0':
                        n = a0_hard
                    elif m.kind == 'B1':
                        n = 50
                    else:
                        n = {4:52,8:55,16:56}[len(m.tree.slots)] - (3 if mode!='on' else 0)
                    return np.arange(100)<n

                with patch('run.pilot_lr',pilot), patch('run.score',fixed_score), patch('run.recall_steps'):
                    run_recall(r,sets,1e9)
                self.assertEqual(r.report['status'],expected)
                if a0_hard < 80:
                    self.assertNotIn('S8',r.report['models'])
                else:
                    self.assertTrue(all(r.report['gates'].values()))
                    self.assertIn('B1-seed43',r.report['models'])
                    self.assertAlmostEqual(r.report['paired_S8_minus_B1']['carry32']['difference'],.05)
                    self.assertEqual(r.report['models']['S8']['boundary_by_popcount_and_length']['on'][1][2112],.55)


if __name__ == '__main__':
    unittest.main()
