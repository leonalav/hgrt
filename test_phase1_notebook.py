"""Tests for the Phase 1 notebook transformations.

Each anchor in phase1_notebook.py must exist in a notebook that has NOT yet
had Phase 1 applied, so the round-trip (revert -> apply) can be verified.
"""
import json
import unittest
from pathlib import Path

import phase1_notebook as P


NOTEBOOK = Path(__file__).with_name('run.ipynb')


def load():
    return json.loads(NOTEBOOK.read_text(encoding='utf8'))


def cell_text(nb, cell_id):
    for c in nb['cells']:
        if c.get('id') == cell_id:
            return ''.join(c['source'])
    raise AssertionError(f'cell {cell_id} not found')


def objective_source():
    return Path(__file__).with_name('phase1_objective.py').read_text(encoding='utf8').rstrip()


class AnchorTests(unittest.TestCase):
    """The anchors must match the PRE-Phase-1 notebook."""

    def setUp(self):
        self.nb = load()
        P.revert_phase1(self.nb)

    def test_corpus_anchors_exist(self):
        text = cell_text(self.nb, '5dda70a4')
        self.assertIn(P.CORPUS_DECLARE_OLD, text)
        self.assertIn(P.CORPUS_SENTENCE_ANCHORS_OLD, text)
        self.assertIn(P.CORPUS_RETURN_OLD, text)
        self.assertIn(P.CORPUS_USE_OLD, text)

    def test_prose_backward_anchor_exists(self):
        text = cell_text(self.nb, 'f997eb16')
        self.assertIn(P.PROSE_BACKWARD_OLD, text)

    def test_train_batch_anchor_exists(self):
        text = cell_text(self.nb, '69ac46a1')
        self.assertIn(P.TRAIN_BATCH_OLD, text)
        self.assertIn(P.WARMUP_OLD, text)

    def test_evaluation_anchors_exist(self):
        text = cell_text(self.nb, 'd1eb119d')
        self.assertIn(P.KEEP_BEST_OLD, text)
        self.assertIn(P.INITIAL_CALL_OLD, text)
        self.assertIn("config = dict(kind='C1', slots=8, seed=SEED, context=CONTEXT, batch=batch_size, lr=LR,", text)

    def test_train_event_anchor_exists_in_the_train_loop_cell(self):
        text = cell_text(self.nb, '69ac46a1')
        self.assertIn(P.TRAIN_EVENT_OLD, text)

    def test_settings_anchor_exists(self):
        text = cell_text(self.nb, '3adb75ac')
        self.assertIn('assert not (INITIAL_WEIGHTS and RESUME_FROM)', text)


class ApplyTests(unittest.TestCase):
    def test_apply_is_idempotent(self):
        nb = load()
        P.revert_phase1(nb)
        first = P.apply_phase1(nb, objective_source())
        snapshot = json.dumps(nb, sort_keys=True)
        second = P.apply_phase1(nb, objective_source())
        self.assertEqual(snapshot, json.dumps(nb, sort_keys=True),
                         'applying Phase 1 twice must not change the notebook')
        self.assertEqual(second, [], 'the second application must report no changes')
        self.assertTrue(first, 'the first application must report changes')

    def test_revert_then_apply_restores_the_notebook(self):
        """A revert/apply round-trip must be lossless."""
        nb = load()
        P.revert_phase1(nb)
        reverted = json.dumps(nb, sort_keys=True)
        P.apply_phase1(nb, objective_source())
        P.revert_phase1(nb)
        self.assertEqual(json.dumps(nb, sort_keys=True), reverted,
                         'revert -> apply -> revert must be lossless')

    def test_apply_adds_the_expected_content(self):
        nb = load()
        P.revert_phase1(nb)
        P.apply_phase1(nb, objective_source())
        self.assertIn('SENTENCE_CANDIDATES', cell_text(nb, '3adb75ac'))
        self.assertIn('sentence_anchors', cell_text(nb, '5dda70a4'))
        self.assertIn('sentence=None', cell_text(nb, 'f997eb16'))
        self.assertIn('sentence_step', cell_text(nb, '69ac46a1'))
        self.assertIn('WARMUP_STEPS', cell_text(nb, '69ac46a1'))
        self.assertIn('def evaluate_sentences', cell_text(nb, 'd1eb119d'))
        self.assertIn('selection_score(', cell_text(nb, 'd1eb119d'))
        self.assertTrue(any(c.get('id') == 'sentence-objective' for c in nb['cells']))

    def test_the_objective_cell_does_not_redefine_settings(self):
        nb = load()
        P.revert_phase1(nb)
        P.apply_phase1(nb, objective_source())
        obj = cell_text(nb, 'sentence-objective')
        for name in ('SENTENCE_CANDIDATES', 'SENTENCE_LEFT_BYTES', 'SENTENCE_RIGHT_BYTES',
                     'SENTENCE_LOSS_WEIGHT', 'PARAGRAPH_LOSS_WEIGHT', 'SELECTION_SENTENCE_WEIGHT'):
            self.assertNotIn(f'\n{name} =', obj,
                             f'{name} belongs in the settings cell, not the objective cell')

    def test_corpus_edits_survive_phase2_openwebtext(self):
        """Regression: Phase 2's 'owt_sentence_anchors' must not block Phase 1.

        Phase 1 guards its corpus edits on a substring check. Phase 2 adds
        'owt_sentence_anchors' to the same cell, so a bare 'sentence_anchors'
        guard always matched and every corpus edit was silently skipped. The
        result was a NameError at runtime: 'sentence_anchors' is not defined.
        """
        nb = load()
        P.revert_phase1(nb)
        P.apply_phase1(nb, objective_source())
        text = cell_text(nb, '5dda70a4')

        # The sentence-anchor accumulator must exist.
        self.assertIn('sentence_anchors = array(', text)
        # make_split must return the sentence anchors, and the caller must
        # unpack all four values.
        self.assertIn('np.asarray(sentence_anchors, dtype=np.int64))', text)
        self.assertIn('sentence_anchors[split]) = make_split(split)', text)
        # use_sentences must be defined before anything reads it.
        self.assertIn('use_sentences = (', text)
        # And the ordering must hold: anchors are populated before use.
        self.assertLess(text.index('sentence_anchors[split]) = make_split(split)'),
                        text.index('use_sentences = ('))

    def test_apply_reports_corpus_change_even_with_phase2_present(self):
        """The 'corpus' item must appear in the applied list."""
        nb = load()
        P.revert_phase1(nb)
        applied = P.apply_phase1(nb, objective_source())
        self.assertIn('corpus', applied)


if __name__ == '__main__':
    unittest.main()
