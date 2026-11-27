"""Tests for the memory-validity remediation of the fact-switch benchmark.

These cover the seven methodology defects that were fixed in `fact_switch.py`:

  E1  the suite must REQUIRE memory -- the `far_*` splits put the fact >= 2048 bytes
      before the question, and `require_memory=True` raises when that is violated.
  E2  `mode='permute'` is a genuine, seeded, content-level permutation that is
      deterministic given a seed and auditable per pair.
  E3  `evaluate_tree_counterfactual_swap` performs a REAL activation patch and
      reports an unrelated-donor control.
  E4  the selection suite and the report suite are DISJOINT.
  E5  the primary metric is `first_byte_acc`, with `unrelated_key_acc` as a control.
  E6  `fact_switch_selection_key` uses the first-byte quantities.
  E7  the generalisation split is reported in its own section.
"""
import random
import unittest

import torch

from fact_switch import (
    ALL_FAMILIES,
    GENERALIZATION_SPLITS,
    MEMORY_CHUNK_BYTES,
    MEMORY_SPLITS,
    MIN_MEMORY_DISTANCE_BYTES,
    OOV_VOCAB,
    PERMUTE_MODE,
    build_fact_switch_suite,
    check_memory_precondition,
    evaluate_fact_switch_suite,
    evaluate_pairs,
    evaluate_tree_counterfactual_swap,
    fact_chunk_offset,
    fact_question_distance,
    fact_switch_selection_key,
    generate_pairs,
    permutation_seed_for,
    require_memory_on,
    resolve_tree_mode,
    suite_pair_ids,
    suites_are_disjoint,
    supported_tree_modes,
)
from model import LM, Shape


def tiny_model(seed: int = 5) -> LM:
    torch.manual_seed(seed)
    return LM(
        'C1', slots=4,
        s=Shape(width=32, heads=2, slot_dim=16, chunk=8, target=120_000),
        read_window=16,
    ).eval()


class MemoryPreconditionTests(unittest.TestCase):
    """E1 -- the far_* splits must genuinely require the tree memory."""

    def test_far_splits_exist_and_are_memory_valid(self):
        suite = build_fact_switch_suite(pairs_per_split=3, require_memory=True)
        for split in MEMORY_SPLITS:
            self.assertIn(split, suite, f'missing memory-required split {split!r}')
            self.assertTrue(suite[split], f'{split!r} is empty')
            # The precondition check must not raise.
            stats = check_memory_precondition(suite[split], split_name=split)
            self.assertEqual(stats['pairs'], len(suite[split]))

    def test_far_splits_have_fact_in_an_earlier_chunk_and_far_enough_away(self):
        suite = build_fact_switch_suite(pairs_per_split=3)
        for split in MEMORY_SPLITS:
            for pair in suite[split]:
                for prompt_key, answer_key in (('prompt_a', 'answer_a'), ('prompt_b', 'answer_b')):
                    prompt = pair[prompt_key]
                    answer = pair[answer_key]
                    fact_chunk, q_chunk = fact_chunk_offset(prompt, answer)
                    self.assertLess(
                        fact_chunk, q_chunk,
                        f'{split}/{pair["id"]}: the fact is in chunk {fact_chunk} and the '
                        f'question in chunk {q_chunk}; the fact is inside the local '
                        f'attention window, so the tree could not be load-bearing.',
                    )
                    gap, _start_to_start = fact_question_distance(prompt, answer)
                    self.assertGreaterEqual(
                        gap, MIN_MEMORY_DISTANCE_BYTES,
                        f'{split}/{pair["id"]}: the fact is only {gap} bytes before the '
                        f'question; {MIN_MEMORY_DISTANCE_BYTES} are required.',
                    )

    def test_far_splits_are_much_longer_than_the_short_splits(self):
        suite = build_fact_switch_suite(pairs_per_split=3)
        for split in MEMORY_SPLITS:
            far_len = min(len(p['prompt_a'].encode()) for p in suite[split])
            self.assertGreater(far_len, MIN_MEMORY_DISTANCE_BYTES)
        for split in ('oov_clean', 'oov_qswap'):
            short_len = max(len(p['prompt_a'].encode()) for p in suite[split])
            self.assertLess(short_len, MIN_MEMORY_DISTANCE_BYTES)

    def test_require_memory_raises_on_a_violating_split(self):
        # A deliberately-bad suite: the fact sits in the same chunk as the question.
        bad = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        bad['far_clean'] = bad['oov_clean']
        # The memory precondition guard must reject it.
        with self.assertRaises(ValueError) as ctx:
            require_memory_on(bad)
        self.assertIn('Memory precondition', str(ctx.exception))

    def test_require_memory_accepts_a_valid_far_split(self):
        good = build_fact_switch_suite(pairs_per_split=2, require_memory=True)
        # The guard above must not have raised; check the far splits directly too.
        for split in MEMORY_SPLITS:
            check_memory_precondition(good[split], split_name=split)

    def test_check_memory_precondition_raises_on_same_chunk_fact(self):
        pairs = generate_pairs(1, seed=3, vocab=OOV_VOCAB, distractor_prob=0.0)
        with self.assertRaises(ValueError) as ctx:
            check_memory_precondition(pairs, split_name='bad')
        self.assertIn('Memory precondition violated', str(ctx.exception))

    def test_check_memory_precondition_raises_when_too_close(self):
        pairs = generate_pairs(1, seed=3, vocab=OOV_VOCAB, distractor_prob=0.0)
        # Require an absurd distance so even the short splits fail.
        with self.assertRaises(ValueError) as ctx:
            check_memory_precondition(pairs, min_distance=10_000_000, split_name='bad')
        self.assertIn('Memory precondition violated', str(ctx.exception))

    def test_check_memory_precondition_rejects_an_empty_split(self):
        with self.assertRaises(ValueError):
            check_memory_precondition([], split_name='empty')

    def test_fact_chunk_offset_rejects_an_absent_answer(self):
        suite = build_fact_switch_suite(pairs_per_split=1)
        pair = suite['oov_clean'][0]
        with self.assertRaises(ValueError):
            fact_chunk_offset(pair['prompt_a'], 'definitely-not-in-the-prompt')


class SelectionReportSplitTests(unittest.TestCase):
    """E4 -- the checkpoint must not be selected on the suite it is reported on."""

    def test_selection_and_report_suites_are_disjoint(self):
        select_suite = build_fact_switch_suite(pairs_per_split=3, seed_offset=0)
        report_suite = build_fact_switch_suite(pairs_per_split=3, seed_offset=100_000)
        self.assertTrue(
            suites_are_disjoint(select_suite, report_suite),
            'the select suite and the report suite share pair ids or content',
        )
        ids_a = set().union(*suite_pair_ids(select_suite).values())
        ids_b = set().union(*suite_pair_ids(report_suite).values())
        self.assertEqual(ids_a & ids_b, set())
        # The CONTENT must differ too, not just the ids.
        from fact_switch import suite_pair_keys
        self.assertEqual(suite_pair_keys(select_suite) & suite_pair_keys(report_suite), set())

    def test_a_suite_is_not_disjoint_from_itself(self):
        suite = build_fact_switch_suite(pairs_per_split=2)
        self.assertFalse(suites_are_disjoint(suite, suite))

    def test_run_two_stage_reports_the_disjoint_flag(self):
        from fact_switch import run_two_stage_fact_switch
        model = tiny_model()
        select_suite = build_fact_switch_suite(pairs_per_split=1, include_far=False)
        report_suite = build_fact_switch_suite(pairs_per_split=1, seed_offset=100_000, include_far=False)
        _model, res = run_two_stage_fact_switch(
            model, select_suite, 'cpu',
            max_steps=1, quads_per_step=1, eval_interval_seconds=0.0,
            select_suite=select_suite, report_suite=report_suite,
        )
        self.assertTrue(res['selection_is_disjoint'])
        self.assertIn('report_suite_splits', res)
        self.assertIn('selection_suite_splits', res)

    def test_single_suite_is_flagged_as_not_disjoint(self):
        from fact_switch import run_two_stage_fact_switch
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=1, include_far=False)
        _model, res = run_two_stage_fact_switch(
            model, suite, 'cpu',
            max_steps=1, quads_per_step=1, eval_interval_seconds=0.0,
        )
        self.assertFalse(res['selection_is_disjoint'])
        self.assertIn('selection_note', res)
        self.assertIn('upward-biased', res['selection_note'])


class FirstByteMetricTests(unittest.TestCase):
    """E5 -- the primary metric is the first-byte decision."""

    def test_evaluate_pairs_returns_the_new_keys_with_sane_types(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=3, include_far=False)
        res = evaluate_pairs(model, suite['oov_clean'], 'cpu', mode='on', do_generate=False)
        # The RATE metrics must all be in [0, 1].
        for key in (
            'first_byte_acc', 'first_byte_indiv_acc', 'first_byte_a_acc', 'first_byte_b_acc',
            'unrelated_key_acc',
        ):
            self.assertIn(key, res, f'missing new metric {key!r}')
            self.assertIsInstance(res[key], float)
            self.assertTrue(0.0 <= res[key] <= 1.0, f'{key} out of range: {res[key]}')
        # The DIAGNOSTICS are log-prob sums; they are finite but not bounded to [0, 1].
        for key in ('sum_sens_diagnostic', 'fb_sum_sens_diagnostic'):
            self.assertIn(key, res, f'missing new metric {key!r}')
            self.assertIsInstance(res[key], float)
            self.assertTrue(
                res[key] == res[key] and abs(res[key]) != float('inf'),
                f'{key} is not finite: {res[key]}',
            )

    def test_legacy_keys_are_still_present(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        res = evaluate_pairs(model, suite['oov_clean'], 'cpu', mode='on', do_generate=False)
        # Other code reads these; they must not be deleted.
        for key in ('pair_acc', 'indiv_acc', 'mean_sens', 'fb_pair_acc', 'fb_mean_sens',
                    'fb_mean_abs_bias', 'by_direction', 'rows'):
            self.assertIn(key, res, f'legacy key {key!r} was removed')

    def test_first_byte_acc_matches_the_per_row_flags(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=4, include_far=False)
        pairs = suite['oov_clean']
        res = evaluate_pairs(model, pairs, 'cpu', mode='on', do_generate=False)
        expected = sum(r['first_byte_pair_correct'] for r in res['rows']) / len(pairs)
        self.assertAlmostEqual(res['first_byte_acc'], expected, places=9)
        # The pair metric is the conjunction of the two bindings.
        for r in res['rows']:
            self.assertEqual(
                r['first_byte_pair_correct'],
                bool(r['first_byte_a_correct'] and r['first_byte_b_correct']),
            )

    def test_unrelated_control_prompts_do_not_contain_the_answer(self):
        from fact_switch import build_unrelated_control_prompts
        suite = build_fact_switch_suite(pairs_per_split=3, include_far=False)
        pairs = suite['oov_clean']
        prompts = build_unrelated_control_prompts(pairs)
        self.assertEqual(len(prompts), len(pairs))
        for prompt, pair in zip(prompts, pairs):
            self.assertNotIn(pair['answer_a'], prompt)

    def test_unrelated_control_can_be_disabled(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        res = evaluate_pairs(
            model, suite['oov_clean'], 'cpu', mode='on', do_generate=False,
            unrelated_control=False,
        )
        self.assertNotIn('unrelated_key_acc', res)


class SuiteEvaluationTests(unittest.TestCase):
    """E7 -- the generalisation split is reported separately and prominently."""

    def test_evaluate_fact_switch_suite_returns_the_new_sections(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2)
        res = evaluate_fact_switch_suite(model, suite, 'cpu', do_generate=False)
        self.assertIn('generalization_splits', res)
        self.assertIn('memory_splits', res)
        for split in GENERALIZATION_SPLITS:
            self.assertIn(split, res['generalization_splits'])
            section = res['generalization_splits'][split]
            self.assertIn('first_byte_acc', section)
            self.assertIn('pair_acc', section)
        for split in MEMORY_SPLITS:
            self.assertIn(split, res['memory_splits'])

    def test_format_table_prints_the_generalisation_caveat(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2)
        res = evaluate_fact_switch_suite(model, suite, 'cpu', do_generate=False)
        from fact_switch import format_fact_switch_table
        text = format_fact_switch_table('T', res)
        self.assertIn('BINDING GENERALISATION', text)
        self.assertIn('nonce_qswap', text)
        self.assertIn('MEMORY-REQUIRED', text)
        self.assertIn('far_clean', text)

    def test_selection_key_uses_the_first_byte_quantities(self):
        from fact_switch import fact_switch_selection_key as key
        base = {
            'oov_qswap': dict(pair_acc=0.5, first_byte_acc=0.25, gen_indiv_exact=0.5, fb_mean_sens=1.0),
            'user_qswap': dict(pair_acc=0.5, first_byte_acc=0.25, gen_indiv_exact=0.5, fb_mean_sens=1.0),
            'oov_clean': dict(pair_acc=0.5, first_byte_acc=0.5, gen_indiv_exact=0.5, fb_mean_sens=1.0),
        }
        improved = {
            'oov_qswap': dict(base['oov_qswap'], first_byte_acc=0.75),
            'user_qswap': dict(base['user_qswap'], first_byte_acc=0.75),
            'oov_clean': base['oov_clean'],
        }
        self.assertGreater(key(improved), key(base))

    def test_selection_key_is_a_tuple_of_floats(self):
        k = fact_switch_selection_key({'oov_qswap': {}, 'oov_clean': {}})
        self.assertIsInstance(k, tuple)
        self.assertTrue(all(isinstance(v, float) for v in k))


class CounterfactualSwapTests(unittest.TestCase):
    """E3 -- the swap must be a real activation patch, with an unrelated control."""

    def test_swap_reports_the_swap_and_the_unrelated_control(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=3, include_far=False)
        res = evaluate_tree_counterfactual_swap(model, suite['oov_clean'], 'cpu')
        self.assertTrue(res['available'])
        self.assertEqual(res['pairs'], len(suite['oov_clean']))
        for key in (
            'base_first_byte_acc',
            'swapped_first_byte_acc',
            'unrelated_control_first_byte_acc',
            'fact_swap_drop',
            'unrelated_control_drop',
            'fact_specific_excess',
        ):
            self.assertIn(key, res, f'missing swap result {key!r}')
            value = res[key]
            self.assertIsInstance(value, float)
            self.assertTrue(
                value == value and abs(value) != float('inf'),
                f'{key} is not finite: {value}',
            )

    def test_swap_rows_record_the_unrelated_donor(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=3, include_far=False)
        res = evaluate_tree_counterfactual_swap(model, suite['oov_clean'], 'cpu')
        for row in res['rows']:
            self.assertIn('base_prefers_gold', row)
            self.assertIn('swapped_prefers_gold', row)
            self.assertIsNotNone(row['control'])
            self.assertIn('unrelated_pair_id', row['control'])

    def test_the_unrelated_control_uses_a_different_family(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=4, include_far=False)
        pairs = suite['oov_clean']
        res = evaluate_tree_counterfactual_swap(model, pairs, 'cpu')
        for row in res['rows']:
            self.assertNotEqual(row['control']['unrelated_family'], row['family'])

    def test_swap_is_available_only_when_the_model_has_a_tree(self):
        torch.manual_seed(1)
        model = LM('A0', slots=0, s=Shape(width=32, heads=2, chunk=8, target=120_000)).eval()
        suite = build_fact_switch_suite(pairs_per_split=1, include_far=False)
        res = evaluate_tree_counterfactual_swap(model, suite['oov_clean'], 'cpu')
        self.assertFalse(res['available'])
        self.assertIn('note', res)


class PermutationTests(unittest.TestCase):
    """E2 -- 'permute' is a genuine, seeded, deterministic permutation."""

    def test_mode_resolution_falls_back_with_a_note(self):
        model = tiny_model()
        supported = supported_tree_modes(model)
        resolved, note = resolve_tree_mode(PERMUTE_MODE, model)
        self.assertIn(resolved, supported)
        if PERMUTE_MODE not in supported:
            self.assertIsNotNone(note)
            self.assertIn('permute', note)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            resolve_tree_mode('not-a-mode', tiny_model())

    def test_permutation_seed_is_deterministic(self):
        self.assertEqual(permutation_seed_for('p0001', 7), permutation_seed_for('p0001', 7))
        self.assertNotEqual(permutation_seed_for('p0001', 7), permutation_seed_for('p0002', 7))
        self.assertNotEqual(permutation_seed_for('p0001', 7), permutation_seed_for('p0001', 8))

    def test_permutation_is_deterministic_given_a_seed(self):
        """A seeded permutation of the same content must give the same result twice."""
        from model import tree_permutation
        p1 = tree_permutation(16, seed=1234)
        p2 = tree_permutation(16, seed=1234)
        self.assertTrue(torch.equal(p1, p2))
        p3 = tree_permutation(16, seed=4321)
        self.assertFalse(torch.equal(p1, p3))

    def test_permutation_actually_reorders_content(self):
        from model import tree_permutation
        perm = tree_permutation(16, seed=99)
        self.assertEqual(sorted(perm.tolist()), list(range(16)))
        self.assertFalse(torch.equal(perm, torch.arange(16)))

    def test_permutation_audit_records_the_donor_content(self):
        from fact_switch import audit_permutation_donors
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        audit = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=7)
        self.assertTrue(audit['available'])
        self.assertEqual(audit['pairs'], len(suite['oov_clean']))
        for row in audit['rows']:
            self.assertIn('own_source_signature', row)
            self.assertIn('permuted_source_signature', row)
            self.assertIn('donor_differs', row)
            self.assertIn('permutation_seed', row)

    def test_permutation_audit_is_reproducible_for_a_fixed_seed(self):
        from fact_switch import audit_permutation_donors
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        a = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=11)
        b = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=11)
        self.assertEqual(
            [r['permuted_source_signature'] for r in a['rows']],
            [r['permuted_source_signature'] for r in b['rows']],
        )

    def test_tree_ablations_use_permute_not_shuffle(self):
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=1)
        res = evaluate_fact_switch_suite(
            model, suite, 'cpu', do_generate=False, run_tree_ablations=True,
        )
        for split, abl in res['tree_ablations'].items():
            self.assertIn('on', abl)
            self.assertIn('off', abl)
            self.assertIn(PERMUTE_MODE, abl, f'{split} is missing the permute ablation')

    def test_permutation_audit_detects_that_the_donor_changed(self):
        """The audit must actually SEE the permutation.

        It used to wrap `Tree.write`, which runs BEFORE the permutation is
        applied (the permutation happens inside `Tree.build`, on the already
        written leaves), so `own_source_signature == permuted_source_signature`
        and `donor_differs` was `False` for every pair -- the audit reported
        "no difference" while a permutation had in fact been applied.
        """
        from fact_switch import audit_permutation_donors
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=3, include_far=False)
        audit = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=7)
        self.assertTrue(audit['available'])
        self.assertGreater(
            audit['donor_differs_fraction'], 0.0,
            'the permutation audit did not detect ANY change in the memory content',
        )
        for row in audit['rows']:
            self.assertNotEqual(
                row['own_source_signature'], row['permuted_source_signature'],
                f"pair {row['id']}: the permuted memory is identical to the own memory",
            )
            self.assertTrue(row['donor_differs'])

    def test_permutation_audit_uses_the_seed_it_records(self):
        """The recorded per-pair seed must be the seed the ablation USED.

        `Tree.build` used to call `tree_permutation(C)` with the module-level
        constant seed, so every pair used the SAME permutation no matter what
        seed the audit recorded.
        """
        from fact_switch import audit_permutation_donors
        model = tiny_model()
        suite = build_fact_switch_suite(pairs_per_split=2, include_far=False)
        audit = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=7)
        seeds = {r['permutation_seed'] for r in audit['rows']}
        self.assertEqual(len(seeds), len(audit['rows']), 'the per-pair seeds are not distinct')
        # Different base seeds must produce different memories.
        other = audit_permutation_donors(model, suite['oov_clean'], 'cpu', permutation_seed=8)
        self.assertNotEqual(
            [r['permuted_source_signature'] for r in audit['rows']],
            [r['permuted_source_signature'] for r in other['rows']],
            'changing the base seed did not change the permutation',
        )

    def test_permute_mode_changes_the_output_and_stays_causal(self):
        """`permute` must be a real ablation AND must not leak the future."""
        model = tiny_model()
        torch.manual_seed(11)
        ids = torch.randint(0, 256, (2, 160))
        with torch.no_grad():
            on = model(ids, mode='on')
            perm = model(ids, mode=PERMUTE_MODE)
        self.assertGreater(
            (on - perm).abs().max().item(), 1e-6,
            "mode='permute' produced the same output as mode='on'",
        )
        # Causality: perturbing a LATE token must not change EARLY outputs.
        for mode in ('on', PERMUTE_MODE, 'off'):
            perturbed = ids.clone()
            perturbed[:, 120:] = (perturbed[:, 120:] + 7) % 256
            with torch.no_grad():
                a = model(ids, mode=mode)
                b = model(perturbed, mode=mode)
            diff = (a - b).abs().max(dim=0).values.max(dim=1).values
            changed = (diff > 1e-9).nonzero().flatten()
            self.assertTrue(
                len(changed) == 0 or int(changed[0]) >= 120,
                f"mode={mode!r} leaked the future: output at position "
                f"{int(changed[0]) if len(changed) else -1} changed when a later token changed",
            )

    def test_permutation_is_a_real_permutation(self):
        from model import tree_permutation
        for n in (1, 2, 5, 16, 33):
            perm = tree_permutation(n)
            self.assertEqual(sorted(perm.tolist()), list(range(n)))
            self.assertEqual(perm.numel(), n)

    def test_tree_permute_seed_is_an_attribute_not_a_constant(self):
        """`Tree.build` must honour the tree's `permute_seed`.

        It used to call `tree_permutation(C)` with the module-level constant
        seed, so every pair used the SAME permutation no matter what seed the
        audit recorded -- the recorded seed was a fiction.
        """
        model = tiny_model()
        tree = model.tree
        self.assertTrue(hasattr(tree, 'permute_seed'))
        ids = torch.randint(0, 256, (2, 160))

        def out_for(seed):
            tree.permute_seed = seed
            with torch.no_grad():
                return model(ids, mode=PERMUTE_MODE).clone()

        a = out_for(1234)
        b = out_for(4321)
        self.assertGreater(
            (a - b).abs().max().item(), 1e-9,
            'changing Tree.permute_seed did not change the permuted output',
        )
        # The same seed must reproduce the same output.
        self.assertTrue(torch.equal(out_for(1234), a))
        # Restoring the default seed must restore the default output.
        default = tree.permute_seed
        tree.permute_seed = default
        with torch.no_grad():
            base = model(ids, mode=PERMUTE_MODE)
        self.assertTrue(torch.allclose(base, a, atol=1e-6))

    def test_donor_signature_distinguishes_different_content(self):
        """The signature must change when the content changes.

        It used to hash only the first 16 floats of the memory row, which
        land on the `C = 1` frontier node -- a single leaf that is identical
        under every permutation.  Every pair therefore reported
        "no difference".
        """
        from fact_switch import donor_signature
        a = torch.zeros(4, 32)
        b = torch.zeros(4, 32)
        # A difference LATE in row 0, well outside the first 16 floats.
        b[0, 31] = 1.0
        self.assertNotEqual(donor_signature(a, 0), donor_signature(b, 0))
        # The ROW index selects which row is hashed.
        d = torch.zeros(4, 32)
        d[2, 3] = 1.0
        self.assertNotEqual(donor_signature(d, 0), donor_signature(d, 2))


if __name__ == '__main__':
    unittest.main()
