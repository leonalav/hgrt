"""Static scope analysis of run.ipynb.

Colab executes cells top to bottom in one namespace. A name that is read at
module level before any cell defines it is a NameError waiting to happen, and
that is exactly the failure mode that has bitten this project repeatedly
(`use_sentences`, `sentence_anchors`, `WARMUP_STEPS`).

This test asserts two things:

1. Every module-level name read is defined either by an earlier cell or by
   the same cell. A violation is a guaranteed runtime NameError.

2. Every module-level name read resolves to *somewhere* in the notebook. A
   name that is never defined anywhere is a guaranteed NameError regardless
   of cell order.

The check is a static analysis, not an execution: it does not need a GPU,
a network, or a Drive mount.
"""
import ast
import json
import unittest
from pathlib import Path

from notebook_scope import analyse

NOTEBOOK = Path(__file__).with_name('run.ipynb')


def load():
    return json.loads(NOTEBOOK.read_text(encoding='utf8'))


class ModuleScopeTests(unittest.TestCase):
    def setUp(self):
        self.nb = load()
        self.result = analyse(self.nb)

    def test_no_name_is_read_before_it_is_defined(self):
        """A module-level read with no prior definition is a NameError."""
        problems = self.result['undefined_reads']
        if problems:
            detail = '\n'.join(f'  cell {cid} line {line}: {name}'
                                for cid, name, line in problems)
            self.fail(f'Names read before they are defined:\n{detail}')

    def test_every_read_name_exists_somewhere(self):
        """A name that no cell defines is a NameError no matter the order."""
        problems = self.result['unknown_globals']
        if problems:
            detail = '\n'.join(f'  cell {cid} line {line}: {name}'
                                for cid, name, line in problems)
            self.fail(f'Names that are never defined:\n{detail}')

    def test_every_code_cell_parses(self):
        for i, cell in enumerate(self.nb['cells']):
            if cell.get('cell_type') != 'code':
                continue
            src = ''.join(cell.get('source', []))
            try:
                ast.parse(src)
            except SyntaxError as exc:
                self.fail(f'cell {cell.get("id", i)} has a syntax error: {exc}')

    def test_phase1_corpus_cell_defines_its_own_globals(self):
        """Regression: the corpus cell must define everything it reads.

        Phase 1's corpus edits were skipped because the guard matched a
        substring that Phase 2 also introduced. The symptom was a NameError
        for `sentence_anchors` at runtime.
        """
        cell = next(c for c in self.nb['cells'] if c.get('id') == '5dda70a4')
        src = ''.join(cell['source'])
        self.assertIn('sentence_anchors = array(', src)
        self.assertIn('sentence_anchors[split]) = make_split(split)', src)
        self.assertIn('use_sentences = (', src)
        # The anchors must be populated before use_sentences reads them.
        self.assertLess(src.index('sentence_anchors[split]) = make_split(split)'),
                        src.index('use_sentences = ('))

    def test_settings_cell_defines_every_phase_constant(self):
        """The training cells read these; the settings cell must define them."""
        cell = next(c for c in self.nb['cells'] if c.get('id') == '3adb75ac')
        src = ''.join(cell['source'])
        for name in ('WARMUP_STEPS', 'SENTENCE_CANDIDATES', 'SENTENCE_LEFT_BYTES',
                     'SENTENCE_RIGHT_BYTES', 'SENTENCE_LOSS_WEIGHT',
                     'PARAGRAPH_LOSS_WEIGHT', 'SELECTION_SENTENCE_WEIGHT',
                     'PROSE_TARGET_BYTES', 'CHAT_TARGET_BYTES', 'QA_TARGET_BYTES',
                     'OWT_DOCUMENTS', 'OWT_REPO', 'OWT_SUBSET'):
            self.assertRegex(src, rf'(?m)^{name}\s*=\s*\S',
                             f'{name} must be defined in the settings cell')


if __name__ == '__main__':
    unittest.main()
