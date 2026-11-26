"""Tests for the Phase 1 sentence-boundary-aware prose objective."""
import unittest

import numpy as np
import torch

from phase1_objective import (
    SENTENCE_CANDIDATES,
    build_sentence_anchors,
    choose_sentence_foils,
    sentence_choice_loss,
    sentence_group_batch,
    selection_score,
    warmup_steps,
)


def fake_cuts(paragraph):
    """Sentence cuts for a bytes paragraph, mirroring the notebook's rule."""
    import re
    for match in re.finditer(rb"""[.!?]["']?[ \t]+(?=[A-Z"'])""", paragraph):
        yield match.end()


SAMPLE = (
    b"The first sentence ends here. "
    b"The second sentence follows. "
    b"Third sentence appears now. "
    b"And a fourth one closes the paragraph. "
)


class SentenceAnchorTests(unittest.TestCase):
    def test_anchors_found_for_each_sentence(self):
        raw = SAMPLE
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=10, right=10, cuts_fn=fake_cuts)
        # Four sentences, but the last one has no following boundary, so it is
        # not a valid continuation point: three anchors.
        self.assertEqual(len(anchors), 3)

    def test_anchors_point_at_sentence_starts(self):
        raw = SAMPLE
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=10, right=10, cuts_fn=fake_cuts)
        for anchor in anchors:
            self.assertTrue(raw[anchor:anchor + 5].startswith(b'The ') or
                            raw[anchor:anchor + 7].startswith(b'Third ') or
                            raw[anchor:anchor + 5].startswith(b'And a'))

    def test_bounds_respected(self):
        raw = SAMPLE
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=40, right=40, cuts_fn=fake_cuts)
        for anchor in anchors:
            self.assertGreaterEqual(anchor, 40)
            self.assertLessEqual(anchor, len(raw) - 40)

    def test_no_anchors_when_span_too_small(self):
        raw = b"Short. "
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=50, right=50, cuts_fn=fake_cuts)
        self.assertEqual(len(anchors), 0)


class FoilSelectionTests(unittest.TestCase):
    def test_foils_exclude_the_true_anchor(self):
        rng = np.random.default_rng(0)
        real, foils = choose_sentence_foils(rng, 100, 32, SENTENCE_CANDIDATES)
        self.assertEqual(foils.shape, (32, SENTENCE_CANDIDATES - 1))
        self.assertEqual(len(real), 32)
        for row in range(32):
            self.assertNotIn(int(real[row]), foils[row].tolist(),
                             'a foil must not be the true anchor')
            self.assertEqual(len(set(foils[row].tolist())), len(foils[row]),
                             'foils must be distinct within a group')

    def test_real_and_foils_are_consistent(self):
        """The returned true anchor and foils must belong to the same group."""
        rng = np.random.default_rng(7)
        real, foils = choose_sentence_foils(rng, 50, 10, 4)
        for row in range(10):
            self.assertLess(int(real[row]), 50)
            for foil in foils[row]:
                self.assertLess(int(foil), 50)


class SentenceGroupBatchTests(unittest.TestCase):
    def test_shape(self):
        raw = SAMPLE * 20
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=20, right=20, cuts_fn=fake_cuts)
        rng = np.random.default_rng(3)
        real, foils = choose_sentence_foils(rng, len(anchors), 5, 4)
        ids, targets = sentence_group_batch(raw, anchors, real, foils, 20, 20, 'cpu')
        self.assertEqual(tuple(ids.shape), (5, 4, 40))
        self.assertEqual(tuple(targets.shape), (5, 4, 40))

    def test_all_candidates_share_the_same_context(self):
        raw = SAMPLE * 20
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=20, right=20, cuts_fn=fake_cuts)
        rng = np.random.default_rng(4)
        real, foils = choose_sentence_foils(rng, len(anchors), 3, 4)
        ids, _ = sentence_group_batch(raw, anchors, real, foils, 20, 20, 'cpu')
        for row in range(ids.shape[0]):
            contexts = ids[row, :, :20]
            for slot in range(1, contexts.shape[0]):
                self.assertTrue(torch.equal(contexts[0], contexts[slot]),
                                'candidates in a group must share the context')

    def test_true_candidate_differs_from_foils(self):
        raw = SAMPLE * 20
        anchors = build_sentence_anchors(raw, [(0, len(raw))], left=20, right=20, cuts_fn=fake_cuts)
        rng = np.random.default_rng(5)
        real, foils = choose_sentence_foils(rng, len(anchors), 3, 4)
        ids, _ = sentence_group_batch(raw, anchors, real, foils, 20, 20, 'cpu')
        for row in range(ids.shape[0]):
            self.assertFalse(torch.equal(ids[row, 0, 20:], ids[row, 1, 20:]))


class SentenceChoiceLossTests(unittest.TestCase):
    def test_perfect_predictions_give_accuracy_one(self):
        g, k, t, v = 4, 3, 5, 8
        targets = torch.randint(0, v, (g, k, t))
        # Make the true candidate (index 0) the only one that can predict its targets.
        logits = torch.full((g, k, t, v), -20.0)
        for row in range(g):
            for step in range(t):
                logits[row, 0, step, targets[row, 0, step]] = 20.0
        loss, accuracy = sentence_choice_loss(logits, targets, left=2)
        self.assertAlmostEqual(float(accuracy), 1.0)
        # The true candidate has ~0 NLL and the foils ~log(8) each, so the
        # loss is log(1 + (K-1)*exp(log(8))) ~ 2.83 with the raw NLL scale.
        # The temperature sharpens the softmax over candidates, so the loss
        # drops towards 0 as the true candidate becomes clearly preferred.
        self.assertLess(float(loss), 3.0)
        hot, _ = sentence_choice_loss(logits, targets, left=2, temperature=0.05)
        self.assertLess(float(hot), 0.01)

    def test_scored_region_excludes_the_context(self):
        """Corrupting the context must not change the sentence ranking."""
        g, k, t, v = 2, 3, 6, 16
        targets = torch.randint(0, v, (g, k, t))
        logits = torch.randn(g, k, t, v)
        loss_a, acc_a = sentence_choice_loss(logits, targets, left=3)
        # Make the context region (target positions 0..left-2) uninformative.
        corrupted = logits.clone()
        corrupted[:, :, :2, :] = 0.0
        loss_b, acc_b = sentence_choice_loss(corrupted, targets, left=3)
        self.assertAlmostEqual(float(acc_a), float(acc_b))

    def test_inverted_predictions_give_accuracy_zero(self):
        g, k, t, v = 4, 3, 5, 8
        targets = torch.randint(0, v, (g, k, t))
        logits = torch.full((g, k, t, v), -20.0)
        for row in range(g):
            for step in range(t):
                # The last candidate predicts perfectly; the true one does not.
                logits[row, k - 1, step, targets[row, k - 1, step]] = 20.0
        loss, accuracy = sentence_choice_loss(logits, targets, left=2)
        self.assertAlmostEqual(float(accuracy), 0.0)

    def test_loss_is_finite(self):
        g, k, t, v = 2, 4, 6, 16
        targets = torch.randint(0, v, (g, k, t))
        logits = torch.randn(g, k, t, v)
        loss, accuracy = sentence_choice_loss(logits, targets, left=3)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)


class SelectionScoreTests(unittest.TestCase):
    def test_better_sentence_accuracy_lowers_the_score(self):
        a = selection_score(1.43, 0.50)
        b = selection_score(1.43, 0.90)
        self.assertLess(b, a)

    def test_better_bpb_lowers_the_score(self):
        a = selection_score(1.50, 0.70)
        b = selection_score(1.40, 0.70)
        self.assertLess(b, a)

    def test_none_accuracy_falls_back_to_bpb(self):
        self.assertAlmostEqual(selection_score(1.43, None), 1.43)

    def test_none_bpb_returns_none(self):
        self.assertIsNone(selection_score(None, 0.5))


class WarmupTests(unittest.TestCase):
    def test_fresh_model_gets_full_ramp(self):
        self.assertEqual(warmup_steps(None, None), 50)

    def test_resume_with_optimizer_gets_no_ramp(self):
        self.assertEqual(warmup_steps('path/latest.pt', None), 1)

    def test_seeded_weights_get_short_ramp(self):
        self.assertEqual(warmup_steps(None, 'path/best_lm.pt'), 5)

    def test_seeded_ramp_is_shorter_than_fresh(self):
        self.assertLess(warmup_steps(None, 'w.pt'), warmup_steps(None, None))

    def test_resume_ramp_is_shortest(self):
        self.assertLessEqual(warmup_steps('r.pt', None), warmup_steps(None, 'w.pt'))


if __name__ == '__main__':
    unittest.main()
