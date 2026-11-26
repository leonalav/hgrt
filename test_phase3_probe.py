"""Tests for the Phase 3 context probe and coherence metrics."""
import unittest

import numpy as np
import torch

from phase3_probe import (
    bpb_by_context,
    coherence_penalty,
    coherence_score,
    context_sensitivity,
)


class CoherenceScoreTests(unittest.TestCase):
    def test_clean_text_scores_well(self):
        text = "The committee approved the plan on Tuesday. The budget was increased afterwards."
        score = coherence_score(text)
        self.assertEqual(score['sentences'], 2)
        self.assertEqual(score['complete'], 2)
        self.assertAlmostEqual(score['complete_rate'], 1.0)
        self.assertFalse(score['dangling'])

    def test_dangling_ending_is_detected(self):
        score = coherence_score("The ship was able to return to the")
        self.assertTrue(score['dangling'])

    def test_repetition_is_detected(self):
        text = "the ship was able the ship was able the ship was able the ship was able"
        score = coherence_score(text)
        self.assertGreater(score['repetition'], 0.0)

    def test_distinct_text_has_low_repetition(self):
        text = "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
        score = coherence_score(text)
        self.assertEqual(score['repetition'], 0.0)

    def test_markup_is_normalised(self):
        text = "The plan @-@ the budget @,@ the result ."
        score = coherence_score(text)
        # The markup tokens are removed, leaving six plain words.
        self.assertEqual(score['words'], 6)
        self.assertNotIn('@', ' '.join(
            __import__('re').findall(r"[A-Za-z']+", text.replace('@-@', '-').replace('@,@', ','))))

    def test_empty_text_does_not_crash(self):
        score = coherence_score('')
        self.assertEqual(score['words'], 0)
        self.assertEqual(score['sentences'], 0)
        self.assertEqual(score['complete_rate'], 0.0)


class CoherencePenaltyTests(unittest.TestCase):
    def test_clean_text_has_no_penalty(self):
        clean = coherence_score("The plan was approved. The budget was increased.")
        self.assertAlmostEqual(coherence_penalty(clean), 0.0)

    def test_dangling_text_is_penalised(self):
        dangling = coherence_score("The ship was able to return to the")
        self.assertGreater(coherence_penalty(dangling), 0.0)

    def test_repetitive_text_is_penalised(self):
        rep = coherence_score("the ship was the ship was the ship was the ship was")
        self.assertGreater(coherence_penalty(rep), 0.0)

    def test_penalty_is_bounded(self):
        worst = coherence_score("the the the the the the the the")
        self.assertLess(coherence_penalty(worst), 1.0)


class ContextSensitivityTests(unittest.TestCase):
    def test_runs_on_a_small_model(self):
        from model import LM
        model = LM('C1', 8)
        model.eval()
        rng = np.random.default_rng(0)
        result = context_sensitivity(model, rng, distances=(5, 20), trials=2, context_bytes=16)
        self.assertEqual(set(result), {5, 20})
        for value in result.values():
            self.assertGreaterEqual(value, 0.0)

    def test_bpb_by_context_shapes(self):
        from model import LM
        model = LM('C1', 8)
        model.eval()
        data = np.frombuffer(b'The committee approved the plan . ' * 400, dtype=np.uint8)
        result = bpb_by_context(model, data, lengths=(64, 128), windows=3, offset=100)
        self.assertEqual(set(result), {64, 128})
        for value in result.values():
            self.assertGreater(value, 0.0)


if __name__ == '__main__':
    unittest.main()
