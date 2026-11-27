"""Integration test for the Phase 1 sentence objective inside run.ipynb.

This executes the real notebook cells (not a copy) to check that:
  - the sentence anchors are collected from the corpus,
  - the multi-candidate batch has the right shape and shares contexts,
  - the sentence loss and accuracy work on the real model,
  - the warm-up ramp does not restart on an already-trained model.
"""
import ast
import json
import math
import re
import unittest
from pathlib import Path

import numpy as np
import torch

from model import LM, Shape


NOTEBOOK = Path(__file__).with_name('run.ipynb')


def cells_by_id():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf8'))
    return {c.get('id'): c for c in notebook['cells']}


def cell_source(cell_id):
    return ''.join(cells_by_id()[cell_id]['source'])


def load_functions(cell_id, names, scope):
    """Exec the named top-level functions of a notebook cell into `scope`."""
    source = cell_source(cell_id)
    tree = ast.parse(source)
    found = set()
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            found.add(node.name)
            chunks.append(ast.get_source_segment(source, node))
    missing = names - found
    assert not missing, f'missing functions in cell {cell_id}: {missing}'
    exec('\n\n'.join(chunks), scope)
    return scope


class NotebookPhase1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cells = cells_by_id()

    # ---- the notebook must contain the Phase 1 wiring ----------------------
    def test_settings_cell_defines_the_constants(self):
        text = cell_source('3adb75ac')
        for name in ('SENTENCE_CANDIDATES', 'SENTENCE_LEFT_BYTES', 'SENTENCE_RIGHT_BYTES',
                     'SENTENCE_LOSS_WEIGHT', 'PARAGRAPH_LOSS_WEIGHT', 'SELECTION_SENTENCE_WEIGHT'):
            self.assertIn(f'{name} =', text)

    def test_corpus_cell_collects_sentence_anchors(self):
        text = cell_source('5dda70a4')
        self.assertIn('sentence_anchors', text)
        self.assertIn('sentence_cuts', text)

    def test_objective_cell_exists_and_defines_the_loss(self):
        self.assertIn('sentence-objective', self.cells)
        text = cell_source('sentence-objective')
        self.assertIn('def sentence_choice_loss', text)
        self.assertIn('def sentence_group_batch', text)
        self.assertIn('def choose_sentence_foils', text)
        self.assertIn('def selection_score', text)

    def test_prose_backward_accepts_the_sentence_term(self):
        text = cell_source('f997eb16')
        self.assertIn('sentence=None', text)
        self.assertIn('SENTENCE_LOSS_WEIGHT', text)
        self.assertIn('PARAGRAPH_LOSS_WEIGHT', text)

    def test_train_loop_uses_the_sentence_step_and_fixed_warmup(self):
        text = cell_source('69ac46a1')
        self.assertIn('sentence_step', text)
        # The ramp length and the base learning rate are POLICY, derived from
        # where the weights came from -- not hard-coded constants.
        self.assertIn('warmup_steps(resume_path, INITIAL_WEIGHTS)', text)
        self.assertIn('learning_rate(resume_path, INITIAL_WEIGHTS)', text)
        self.assertNotIn('warmup_step/50', text)

    def test_selection_uses_the_combined_score(self):
        text = cell_source('d1eb119d')
        self.assertIn('selection_score(', text)
        self.assertIn('def evaluate_sentences', text)

    # ---- functional tests on the real notebook code -----------------------
    def test_sentence_group_batch_shapes_and_shared_context(self):
        scope = {'torch': torch, 'np': np, 'device': 'cpu',
                 'F': torch.nn.functional}
        load_functions('sentence-objective',
                       {'sentence_group_batch', 'choose_sentence_foils'}, scope)

        raw = (b'The first sentence ends here. The second sentence follows. '
               b'Third sentence appears now. And a fourth one closes it. ') * 20
        anchors = np.arange(0, len(raw) - 200, 37, dtype=np.int64)
        rng = np.random.default_rng(0)
        real, foils = scope['choose_sentence_foils'](rng, len(anchors), 4, 6)
        ids, targets = scope['sentence_group_batch'](raw, anchors, real, foils, 40, 40, 'cpu')

        self.assertEqual(tuple(ids.shape), (4, 6, 80))
        self.assertEqual(tuple(targets.shape), (4, 6, 80))
        for row in range(4):
            ctx = ids[row, :, :40]
            for slot in range(1, 6):
                self.assertTrue(torch.equal(ctx[0], ctx[slot]))

    def test_sentence_choice_loss_runs_on_the_real_model(self):
        scope = {'torch': torch, 'np': np, 'device': 'cpu',
                 'F': torch.nn.functional}
        load_functions('sentence-objective', {'sentence_choice_loss'}, scope)

        torch.manual_seed(0)
        model = LM('C1', 8)
        model.eval()
        g, k, t = 2, 4, 24
        ids = torch.randint(0, 256, (g, k, t))
        targets = ids.roll(-1, dims=-1)
        with torch.inference_mode():
            logits = model(ids.reshape(-1, t)).reshape(g, k, t, 256)
        loss, accuracy = scope['sentence_choice_loss'](logits, targets, left=12)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(0.0 <= float(accuracy) <= 1.0)

    def test_selection_score_combines_bpb_and_accuracy(self):
        scope = {'torch': torch, 'np': np, 'device': 'cpu', 're': re}
        # The weight lives in the settings cell; the notebook has it in scope.
        scope['SELECTION_SENTENCE_WEIGHT'] = 0.30
        load_functions('sentence-objective', {'selection_score'}, scope)
        better_accuracy = scope['selection_score'](1.43, 0.95)
        worse_accuracy = scope['selection_score'](1.43, 0.50)
        self.assertLess(better_accuracy, worse_accuracy)

    def test_warmup_steps_helper_is_consistent_with_the_notebook(self):
        scope = {'torch': torch, 'np': np, 'device': 'cpu'}
        load_functions('sentence-objective', {'warmup_steps'}, scope)
        # The notebook divides by WARMUP_STEPS, so the helper must agree.
        self.assertEqual(scope['warmup_steps'](None, None), 50)
        self.assertEqual(scope['warmup_steps'](None, 'best_lm.pt'), 5)
        self.assertEqual(scope['warmup_steps']('latest.pt', None), 1)

    def test_the_notebook_uses_the_same_warmup_values(self):
        """The ramp length and base LR must come from the shared policy."""
        text = cell_source('69ac46a1')
        self.assertIn('warmup_steps(resume_path, INITIAL_WEIGHTS)', text)
        self.assertIn('learning_rate(resume_path, INITIAL_WEIGHTS)', text)
        self.assertIn('base_lr*min(1., warmup_step/ramp)', text)

    def test_learning_rate_policy_is_smaller_for_seeded_and_resumed_models(self):
        """A converged model must not be trained at the scratch learning rate.

        Measured: at LR=5e-4 the run-2 checkpoint's validation BPB rose from
        1.4717 to 1.7310 in 40 steps; at 2e-5 it fell to 1.4153.
        """
        scope = {'torch': torch, 'np': np, 'device': 'cpu'}
        load_functions('sentence-objective', {'learning_rate'}, scope)
        fresh = scope['learning_rate'](None, None)
        seeded = scope['learning_rate'](None, 'best_lm.pt')
        resumed = scope['learning_rate']('latest.pt', None)
        self.assertEqual(fresh, 5e-4)
        self.assertLess(seeded, fresh)
        self.assertLess(resumed, fresh)
        # The reduced rate must be the measured optimum (~2e-5).
        self.assertAlmostEqual(seeded, 2e-5, places=6)
        self.assertAlmostEqual(resumed, 2e-5, places=6)

    def test_sentence_anchor_collection_from_a_real_paragraph(self):
        scope = {'torch': torch, 'np': np, 'device': 'cpu', 're': re}
        load_functions('sentence-objective',
                       {'build_sentence_anchors', 'sentence_anchor_cuts'}, scope)
        corpus = cell_source('5dda70a4')
        # Use the notebook's own sentence_cuts implementation.
        tree = ast.parse(corpus)
        chunk = None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == 'sentence_cuts':
                chunk = ast.get_source_segment(corpus, node)
        exec(chunk, scope)

        paragraph = (b'The first sentence ends here. The second sentence follows. '
                     b'Third sentence appears now. And a fourth closes the set. ')
        anchors = scope['build_sentence_anchors'](paragraph, [(0, len(paragraph))],
                                                   10, 10, scope['sentence_cuts'])
        self.assertEqual(len(anchors), 3)
        for a in anchors:
            self.assertTrue(paragraph[a:].startswith((b'The ', b'Third ', b'And ')))

    def test_downstream_stage_learning_rates_are_prose_safe(self):
        """The fact-switch and dialogue stages must not use a destructive rate.

        Both stages start from the CONVERGED prose checkpoint with a FRESH
        optimizer, so they are in exactly the situation the Phase-1 learning
        rate policy exists for -- and they were not covered by it.

        Measured on the run-2 checkpoint (40 steps of the contrastive
        objective, held-out wikitext BPB):

            LR=5e-4   1.4786 -> 2.9383  (+1.4598)   destroys the model
            LR=1e-4   1.4786 -> 1.5240  (+0.0455)
            LR=2e-5   1.4786 -> 1.4810  (+0.0024)   prose preserved

        And the task is still learned at 2e-5: the contrastive loss fell
        8.18 -> 7.11 (-1.07) over the same 40 steps.
        """
        text = cell_source('3adb75ac')
        scope = {}
        lines = [l for l in text.splitlines()
                 if l.startswith(('FACT_SWITCH_LR', 'DIALOGUE_LR'))]
        self.assertEqual(len(lines), 2, f'expected both stage rates, got {lines}')
        exec('\n'.join(lines), scope)
        self.assertLessEqual(scope['FACT_SWITCH_LR'], 2e-5)
        self.assertLessEqual(scope['DIALOGUE_LR'], 2e-5)
        # The scratch prose rate is 5e-4; a downstream stage must never use it.
        self.assertLess(scope['FACT_SWITCH_LR'], 5e-4)
        self.assertLess(scope['DIALOGUE_LR'], 5e-4)


if __name__ == '__main__':
    unittest.main()
