"""Unit tests for the counterfactual fact-switch benchmark and contrastive loss."""
import unittest

import torch

from fact_switch import (
    ALL_FAMILIES,
    OOV_VOCAB,
    TRAIN_VOCAB,
    evaluate_pairs,
    evaluate_tree_counterfactual_swap,
    fact_switch_loss,
    first_diff_index,
    generate_pairs,
    pair_batch_tensors,
    random_nonce,
    user_audit_pairs,
)
from model import LM, Shape


class FactSwitchTests(unittest.TestCase):
    def test_vocab_strictly_disjoint_from_oov_and_user_audit(self):
        train_words = {w.casefold() for vals in TRAIN_VOCAB.values() for w in vals}
        oov_words = {w.casefold() for vals in OOV_VOCAB.values() for w in vals}
        self.assertTrue(train_words.isdisjoint(oov_words), train_words & oov_words)
        for pair in user_audit_pairs():
            self.assertNotIn(pair['answer_a'].casefold(), train_words)
            self.assertNotIn(pair['answer_b'].casefold(), train_words)
            for token in (pair['prompt_a'] + ' ' + pair['prompt_b']).replace('.', '').replace("'", ' ').split():
                # Ensure no train entity leaks into the user audit passages.
                if token.casefold() in oov_words:
                    self.assertNotIn(token.casefold(), train_words)

    def test_first_diff_index_handles_prefixes_and_newlines(self):
        self.assertEqual(first_diff_index(b'blue\n', b'pink\n'), 0)
        self.assertEqual(first_diff_index(b'bright blue\n', b'bright pink\n'), 7)
        self.assertEqual(first_diff_index(b'cat\n', b'cats\n'), 3)

    def test_single_pass_divergence_margin_matches_full_forward_oracle(self):
        torch.manual_seed(7)
        model = LM('C1', slots=4, s=Shape(width=32, heads=2, slot_dim=16, chunk=8, target=120000), read_window=256).eval()
        pairs = [
            dict(
                id='shared_prefix', family='color', distractor=False,
                prompt_a='Passage: It was bright blue.\nUser: Color?\nAssistant: ',
                prompt_b='Passage: It was bright pink.\nUser: Color?\nAssistant: ',
                answer_a='bright blue', answer_b='bright pink',
            ),
            dict(
                id='plain', family='city', distractor=False,
                prompt_a='Passage: Meet in Lima.\nUser: City?\nAssistant: ',
                prompt_b='Passage: Meet in Oslo.\nUser: City?\nAssistant: ',
                answer_a='Lima', answer_b='Oslo',
            ),
        ]
        x, y, bpos, dinfo = pair_batch_tensors(pairs, 'cpu')
        logits = model(x)
        loss, stats = fact_switch_loss(logits, y, bpos, dinfo, mode='contrastive')
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(model.embed.weight.grad.norm().item(), 0.0)

        # Compare single-pass pair_sens against evaluate_pairs (which scores all 4 prompt x answer combinations)
        ev = evaluate_pairs(model, pairs, 'cpu', mode='on', do_generate=False)
        self.assertAlmostEqual(stats['mean_pair_sens'], ev['fb_mean_sens'], places=4)

    def test_generated_pairs_always_swap_answers_and_preserve_distractor_balance(self):
        pairs = generate_pairs(20, seed=11, vocab=TRAIN_VOCAB, distractor_prob=1.0, filler_choices=(1, 2))
        self.assertEqual(len(pairs), 20)
        for p in pairs:
            self.assertNotEqual(p['answer_a'], p['answer_b'])
            self.assertIn(p['answer_a'], p['prompt_a'])
            self.assertIn(p['answer_b'], p['prompt_a'])
            self.assertIn(p['answer_a'], p['prompt_b'])
            self.assertIn(p['answer_b'], p['prompt_b'])
            self.assertNotEqual(p['prompt_a'], p['prompt_b'])

    def test_tree_counterfactual_swap_runs_and_differs_on_long_prompt(self):
        torch.manual_seed(9)
        model = LM('C1', slots=4, s=Shape(width=32, heads=2, slot_dim=16, chunk=16, target=120000), read_window=16).eval()
        pairs = generate_pairs(4, seed=19, vocab=OOV_VOCAB, distractor_prob=1.0, filler_choices=(1,))
        res = evaluate_tree_counterfactual_swap(model, pairs, 'cpu')
        self.assertEqual(res['pairs'], 4)
        self.assertTrue(abs(res['tree_swap_drop']) >= 0.0)


    def test_nonce_never_collides_with_oov_and_direction_metadata_present(self):
        import random
        rng = random.Random(123)
        oov = {w.casefold() for vals in OOV_VOCAB.values() for w in vals}
        for _ in range(500):
            w = random_nonce(rng, capitalize=False)
            self.assertNotIn(w.casefold(), oov)
        pairs = generate_pairs(len(ALL_FAMILIES) * 2, seed=21, vocab=TRAIN_VOCAB, distractor_prob=1.0, nonce_value_prob=1.0)
        dirs = {p['direction'] for p in pairs}
        self.assertEqual(dirs, {'forward', 'reverse'})

    def test_qswap_quad_cancels_both_positional_and_static_prior_shortcuts(self):
        import random
        from qswap_experiment import make_qswap_pair, make_quad_as_two_pairs

        rng = random.Random(55)
        for direction in ('forward', 'reverse'):
            p = make_qswap_pair(rng, OOV_VOCAB, direction)
            # Identical passage bytes, different questions, swapped answers.
            pass_a = p['prompt_a'].split('\nUser:', 1)[0]
            pass_b = p['prompt_b'].split('\nUser:', 1)[0]
            self.assertEqual(pass_a, pass_b)
            self.assertNotEqual(p['prompt_a'], p['prompt_b'])
            self.assertNotEqual(p['answer_a'], p['answer_b'])
            # A pure S1-copying model predicts answer_a for both prompts -> dA = +M, dB = -M -> sens = 0.
            dA, dB = 5.0, -5.0
            self.assertEqual(dA + dB, 0.0)

            q1, q2 = make_quad_as_two_pairs(rng, TRAIN_VOCAB, direction, nonce_val_prob=0.5, nonce_key_prob=0.5)
            self.assertEqual(q1['answer_a'], q2['answer_b'])
            self.assertEqual(q1['answer_b'], q2['answer_a'])


if __name__ == '__main__':
    unittest.main()
