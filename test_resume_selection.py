"""Tests for the resume-selection logic in run.ipynb cell d1eb119d.

These run offline on CPU: they build fake run directories on disk and check that
the selector picks the right checkpoint and explains every rejection.
"""
import ast
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import torch


NOTEBOOK = Path(__file__).with_name('run.ipynb')


def notebook_source():
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf8'))
    for cell in notebook['cells']:
        if cell.get('id') == 'd1eb119d':
            return ''.join(cell['source'])
    raise AssertionError('cell d1eb119d not found')


def load_resume_module():
    """Execute just the resume-selection part of the notebook cell."""
    source = notebook_source()
    tree = ast.parse(source)
    wanted = {'compatible_config', '_selection_loss', '_config_diff',
              'survey_checkpoints', 'print_resume_report'}
    chunks = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            chunks.append(ast.get_source_segment(source, node))
    assert len(chunks) == len(wanted), f'expected {len(wanted)} functions, found {len(chunks)}'

    namespace = {'torch': torch, 'json': json, 'math': math, 'Path': Path}
    # `run_dir` is a notebook global that the selector closes over to skip the
    # directory it is currently writing into.
    namespace['run_dir'] = Path('/nonexistent-current-run')
    exec('\n\n'.join(chunks), namespace)
    return namespace


CORPUS_HASHES = {
    'train': 'trainhash',
    'validation': 'validhash',
    'test': 'testhash',
}


def base_config(**overrides):
    config = {'kind': 'C1', 'slots': 8, 'seed': 42, 'context': 512, 'batch': 2,
              'lr': 0.0005, 'tokenizer': 'utf8-bytes', 'corpus_hashes': CORPUS_HASHES}
    config.update(overrides)
    return config


def make_state_dict():
    return {'embed.weight': torch.zeros(1)}


def write_best_lm(run_dir, config, validation_bpb=None, matched_bpb=None,
                  kind='C1', slots=8, corpus_hashes=None):
    payload = dict(kind=kind, slots=slots, read_window=64, config=config,
                   state_dict=make_state_dict(), step=1, bytes_seen=1)
    if validation_bpb is not None:
        payload['validation_bpb'] = validation_bpb
    if matched_bpb is not None:
        payload['continuation'] = {'matched_bpb': matched_bpb}
    if corpus_hashes is not None:
        payload['config'] = dict(config, corpus_hashes=corpus_hashes)
    torch.save(payload, run_dir / 'best_lm.pt')


def write_latest(run_dir, config, optimizer=True, **extra):
    payload = dict(kind='C1', slots=8, read_window=64, config=config,
                   state_dict=make_state_dict(), step=1, bytes_seen=1,
                   batch_at_save=2, train_rng=json.dumps({'bit_generator': 'state'}))
    if optimizer:
        payload['optimizer'] = {}
    payload.update(extra)
    torch.save(payload, run_dir / 'latest.pt')


class ResumeSelectionTests(unittest.TestCase):
    def setUp(self):
        self.ns = load_resume_module()
        self.tmp = Path(tempfile.mkdtemp())
        self.runs = self.tmp / 'runs'
        self.runs.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_run(self, name):
        run = self.runs / name
        run.mkdir(parents=True)
        return run

    def survey(self, config=None, adaptive=True):
        config = config or base_config()
        return self.ns['survey_checkpoints']({self.runs}, config, adaptive)

    # ---- selection loss -------------------------------------------------
    def test_selection_loss_prefers_continuation_bpb(self):
        weights = {'validation_bpb': 1.5, 'continuation': {'matched_bpb': 1.4}}
        self.assertAlmostEqual(self.ns['_selection_loss'](weights), 1.4)

    def test_selection_loss_falls_back_to_validation_bpb(self):
        self.assertAlmostEqual(self.ns['_selection_loss']({'validation_bpb': 1.5}), 1.5)

    def test_selection_loss_infinite_when_absent(self):
        self.assertEqual(self.ns['_selection_loss']({}), float('inf'))

    def test_selection_loss_ignores_nonfinite(self):
        weights = {'validation_bpb': float('nan'), 'continuation': {'matched_bpb': 1.2}}
        self.assertAlmostEqual(self.ns['_selection_loss'](weights), 1.2)

    # ---- config diff ----------------------------------------------------
    def test_config_diff_explains_mismatch(self):
        saved = base_config(lr=0.00015)
        current = base_config(lr=0.0005)
        detail = self.ns['_config_diff'](saved, current)
        self.assertIn('lr', detail)
        self.assertIn('0.00015', detail)

    def test_config_diff_handles_missing(self):
        self.assertIn('no readable saved config', self.ns['_config_diff'](None, base_config()))

    # ---- best_lm only directory is visible ------------------------------
    def test_best_lm_only_run_is_discovered(self):
        """The regression: a run with only best_lm.pt must not be invisible."""
        run = self.make_run('wiki-20261125-172147-6z1pap8b')
        write_best_lm(run, base_config(), validation_bpb=1.4170)

        resume, weights, report = self.survey()
        self.assertIsNone(resume, 'no latest.pt exists, so nothing can resume')
        self.assertEqual(weights, run / 'best_lm.pt')
        self.assertEqual(len(report), 1)
        self.assertEqual(report[0]['best_lm_status'], 'usable')

    def test_best_lm_with_lower_loss_wins(self):
        older = self.make_run('wiki-20261101-000000-old')
        newer = self.make_run('wiki-20261125-000000-new')
        write_best_lm(older, base_config(), validation_bpb=1.4600)
        write_best_lm(newer, base_config(), validation_bpb=1.4170)

        resume, weights, _ = self.survey()
        self.assertEqual(weights, newer / 'best_lm.pt')

    def test_older_directory_with_better_loss_still_wins(self):
        """Selection is by quality, not recency."""
        older = self.make_run('wiki-20260101-000000-a')
        newer = self.make_run('wiki-20260102-000000-b')
        write_best_lm(older, base_config(), validation_bpb=1.4000)
        write_best_lm(newer, base_config(), validation_bpb=1.5000)

        resume, weights, _ = self.survey()
        self.assertEqual(weights, older / 'best_lm.pt')

    def test_best_lm_ranks_by_continuation_bpb(self):
        a = self.make_run('wiki-20260101-000000-a')
        b = self.make_run('wiki-20260102-000000-b')
        write_best_lm(a, base_config(), validation_bpb=1.4000, matched_bpb=1.4500)
        write_best_lm(b, base_config(), validation_bpb=1.6000, matched_bpb=1.4300)

        resume, weights, _ = self.survey()
        self.assertEqual(weights, b / 'best_lm.pt')

    # ---- rejection reasons ----------------------------------------------
    def test_different_corpus_is_rejected(self):
        run = self.make_run('wiki-20261125-000000-x')
        other = dict(CORPUS_HASHES, train='different')
        write_best_lm(run, base_config(), validation_bpb=1.41, corpus_hashes=other)

        resume, weights, report = self.survey()
        self.assertIsNone(weights)
        self.assertEqual(report[0]['best_lm_status'], 'different_corpus')

    def test_wrong_architecture_is_rejected(self):
        run = self.make_run('wiki-20261125-000000-x')
        write_best_lm(run, base_config(), validation_bpb=1.41, slots=4)

        resume, weights, report = self.survey()
        self.assertIsNone(weights)
        self.assertEqual(report[0]['best_lm_status'], 'wrong_architecture')

    def test_incompatible_latest_is_rejected_with_detail(self):
        run = self.make_run('wiki-20261123-000000-y')
        (run / 'config.json').write_text(
            json.dumps({'config': base_config(lr=0.00015)}), encoding='utf8')
        write_latest(run, base_config(lr=0.00015))

        resume, weights, report = self.survey()
        self.assertIsNone(resume)
        self.assertEqual(report[0]['latest_status'], 'config_mismatch')
        self.assertIn('lr', report[0]['latest_detail'])

    def test_compatible_latest_is_selected(self):
        run = self.make_run('wiki-20261123-000000-y')
        (run / 'config.json').write_text(json.dumps({'config': base_config()}), encoding='utf8')
        write_latest(run, base_config())

        resume, weights, report = self.survey()
        self.assertEqual(resume, run / 'latest.pt')
        self.assertEqual(report[0]['latest_status'], 'compatible')

    def test_newest_compatible_latest_wins(self):
        a = self.make_run('wiki-20261101-000000-a')
        b = self.make_run('wiki-20261125-000000-b')
        for run in (a, b):
            (run / 'config.json').write_text(json.dumps({'config': base_config()}), encoding='utf8')
            write_latest(run, base_config())

        resume, weights, _ = self.survey()
        self.assertEqual(resume, b / 'latest.pt')

    def test_batch_mismatch_allowed_when_adaptive(self):
        run = self.make_run('wiki-20261123-000000-y')
        (run / 'config.json').write_text(
            json.dumps({'config': base_config(batch=256)}), encoding='utf8')
        write_latest(run, base_config(batch=256))

        resume, _, report = self.survey(adaptive=True)
        self.assertEqual(resume, run / 'latest.pt')
        self.assertEqual(report[0]['latest_status'], 'compatible')

    def test_batch_mismatch_rejected_when_not_adaptive(self):
        run = self.make_run('wiki-20261123-000000-y')
        (run / 'config.json').write_text(
            json.dumps({'config': base_config(batch=256)}), encoding='utf8')
        write_latest(run, base_config(batch=256))

        resume, _, report = self.survey(adaptive=False)
        self.assertIsNone(resume)
        self.assertEqual(report[0]['latest_status'], 'config_mismatch')

    def test_run_dir_is_excluded(self):
        run = self.make_run('wiki-20261125-000000-self')
        write_best_lm(run, base_config(), validation_bpb=1.41)
        self.ns['run_dir'] = run

        resume, weights, report = self.survey()
        self.assertIsNone(weights)
        self.assertEqual(report, [])

    # ---- reporting ------------------------------------------------------
    def test_report_lists_every_directory(self):
        a = self.make_run('wiki-20261101-000000-a')
        b = self.make_run('wiki-20261125-000000-b')
        write_best_lm(a, base_config(), validation_bpb=1.46)
        write_latest(b, base_config())
        (b / 'config.json').write_text(json.dumps({'config': base_config()}), encoding='utf8')

        resume, weights, report = self.survey()
        self.assertEqual(len(report), 2)
        names = {Path(e['run']).name for e in report}
        self.assertEqual(names, {'wiki-20261101-000000-a', 'wiki-20261125-000000-b'})

    def test_print_resume_report_runs(self):
        run = self.make_run('wiki-20261125-000000-x')
        write_best_lm(run, base_config(), validation_bpb=1.4170)
        resume, weights, report = self.survey()
        self.ns['print_resume_report'](report, resume, weights)


if __name__ == '__main__':
    unittest.main()
