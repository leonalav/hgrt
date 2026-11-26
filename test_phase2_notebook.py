"""Tests for the Phase 2 notebook transformations."""
import json
import unittest
from pathlib import Path

import phase1_notebook
import phase2_notebook as P


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


def base_notebook():
    """A notebook with neither Phase 1 nor Phase 2 applied."""
    nb = load()
    phase1_notebook.revert_phase1(nb)
    P.revert_phase2(nb)
    return nb


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.nb = base_notebook()

    def test_corpus_anchor_exists(self):
        text = cell_text(self.nb, '5dda70a4')
        self.assertIn('# On Windows the byte mappings must be closed', text)
        self.assertIn(P.CORPUS_TRAIN_STORAGE_OLD, text)

    def test_dialogue_call_anchor_exists(self):
        text = cell_text(self.nb, 'dialog-train')
        self.assertIn(P.DIALOG_CALL_OLD, text)

    def test_settings_anchor_exists(self):
        text = cell_text(self.nb, '3adb75ac')
        self.assertIn('assert not (INITIAL_WEIGHTS and RESUME_FROM)', text)


class ApplyTests(unittest.TestCase):
    def test_apply_is_idempotent(self):
        nb = base_notebook()
        first = P.apply_phase2(nb)
        snapshot = json.dumps(nb, sort_keys=True)
        second = P.apply_phase2(nb)
        self.assertEqual(snapshot, json.dumps(nb, sort_keys=True))
        self.assertEqual(second, [])
        self.assertTrue(first)

    def test_revert_then_apply_restores_the_notebook(self):
        nb = base_notebook()
        reverted = json.dumps(nb, sort_keys=True)
        P.apply_phase2(nb)
        P.revert_phase2(nb)
        self.assertEqual(json.dumps(nb, sort_keys=True), reverted)

    def test_apply_adds_the_expected_content(self):
        nb = base_notebook()
        P.apply_phase2(nb)
        self.assertIn('OWT_DOCUMENTS', cell_text(nb, '3adb75ac'))
        self.assertIn('PROSE_TARGET_BYTES', cell_text(nb, '3adb75ac'))
        self.assertIn('owt_corpus', cell_text(nb, '5dda70a4'))
        self.assertIn('prose_bytes=PROSE_TARGET_BYTES', cell_text(nb, 'dialog-train'))

    def test_phase1_and_phase2_compose(self):
        """Applying then reverting both phases must be lossless."""
        nb = base_notebook()
        base = json.dumps(nb, sort_keys=True)
        phase1_notebook.apply_phase1(nb, objective_source())
        P.apply_phase2(nb)
        self.assertNotEqual(json.dumps(nb, sort_keys=True), base,
                            'the phases must change the notebook')

        # Revert in reverse order of application.
        P.revert_phase2(nb)
        phase1_notebook.revert_phase1(nb)
        self.assertEqual(json.dumps(nb, sort_keys=True), base,
                         'reverting both phases must restore the base notebook')


if __name__ == '__main__':
    unittest.main()
