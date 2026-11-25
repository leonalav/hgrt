"""Offline regression checks for the self-contained Colab notebook."""
import ast
import codecs
import json
import re
import unittest
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import torch
from torch.nn import functional as F

from model import LM, Shape


def notebook_functions():
    notebook = json.loads(Path(__file__).with_name('run.ipynb').read_text(encoding='utf8'))
    names = {'world_key', 'make_probe', 'ending_scores', 'utf8_mask', 'sample_byte', 'sample',
             'choose_batch', 'backward_with_backoff', 'compatible_config', 'window_offsets', 'windows',
             'sentence_boundary', 'sentence_cuts', 'continuation_batch', 'continuation_nll', 'choose_paragraphs',
             'display_wiki_sample', 'probe_model', 'setup_profiles', 'finished_sentence', 'prose_prompt_pairs'}
    scope = dict(torch=torch, F=F, codecs=codecs, np=np, device='cpu', lru_cache=lru_cache,
                 re=re, precision=lambda device: nullcontext(), release_memory=lambda: None)
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            tree = ast.parse(''.join(cell['source']))
            functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
            exec(compile(ast.Module(functions, []), cell['id'], 'exec'), scope)
    return scope


class Notebook(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = notebook_functions()

    def test_portable_notebook_embeds_tested_model_and_data(self):
        folder = Path(__file__).parent
        cells = json.loads((folder/'run.ipynb').read_text(encoding='utf8'))['cells']
        sources = {cell['id']: ''.join(cell['source']) for cell in cells}
        self.assertEqual(sources['775ed5c3'].split('\n', 1)[1].rstrip(),
                         (folder/'model.py').read_text(encoding='utf8').rstrip())
        self.assertEqual(sources['dialog-data'].rstrip(),
                         (folder/'dialogue.py').read_text(encoding='utf8').rstrip())
        self.assertEqual(sources['fact-switch-lib'].split('\n', 1)[1].rstrip(),
                         (folder/'fact_switch.py').read_text(encoding='utf8').rstrip())
        for cell in cells:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']), filename=cell.get('id', 'cell'))

    def test_colab_standalone_isolated_execution(self):
        """Simulate opening run.ipynb from a GitHub link in Colab with zero repo .py files on sys.path."""
        import sys
        folder = Path(__file__).parent.resolve()
        cells = json.loads((folder/'run.ipynb').read_text(encoding='utf8'))['cells']
        sources = {cell['id']: ''.join(cell['source']) for cell in cells}
        saved_path = list(sys.path)
        saved_mods = {k: sys.modules.pop(k) for k in ('model', 'dialogue', 'fact_switch') if k in sys.modules}
        try:
            sys.path = [p for p in sys.path if Path(p or '.').resolve() != folder]
            ns = {'__name__': '__main__'}  # Exact Colab cell __name__ without __file__!
            for cid in ('775ed5c3', '9a302c8e', 'dialog-data', 'fact-switch-lib'):
                exec(compile(sources[cid], cid, 'exec'), ns)
            # Exercise the embedded model, dialogue prep, 2-stage fact-switch trainer & Tree ablation.
            shape = ns['Shape'](width=32, heads=2, slot_dim=16, chunk=8, target=120000)
            net = ns['LM']('C1', 4, s=shape, read_window=256, mid_window=256)
            suite = ns['build_fact_switch_suite'](qa_style='qa', pairs_per_split=2)
            self.assertIn('Question:', suite['user_6pairs'][0]['prompt_a'])
            self.assertIn('Answer: ', suite['user_6pairs'][0]['prompt_a'])
            net, res = ns['run_two_stage_fact_switch'](
                net, suite, 'cpu', minutes=0.0, max_steps=2, quads_per_step=1,
                lr=1e-3, qa_style='qa', seed=7, eval_interval_seconds=999.0,
            )
            self.assertEqual(res['steps'], 2)
            self.assertIn('tree_ablations', res['selected'])
            self.assertIn('oov_qswap', res['selected']['tree_ablations'])
        finally:
            sys.path = saved_path
            sys.modules.update(saved_mods)

    def test_qa_style_separation_and_sustained_prose(self):
        from dialogue import chat_prompt, is_extractive_qa, long_story_continuations, prepare_dialogues, target_batch
        prompt = chat_prompt('Where?', [('Who?', 'Mira')], 'Mira went to the library.', 512, qa_style='qa')
        self.assertIn('Passage: Mira went to the library.\n', prompt)
        self.assertIn('Question: Who?\nAnswer: Mira\n', prompt)
        self.assertTrue(prompt.endswith('Question: Where?\nAnswer: '))
        self.assertTrue(is_extractive_qa(prompt, 'library\n'))
        self.assertFalse(is_extractive_qa(prompt, 'yes\n'))
        _, train_y = target_batch([(prompt, 'library\n')], 'cpu', history_targets=True)
        self.assertEqual(bytes(train_y[train_y != -100].tolist()), b'Mira\nlibrary\n')

        story = ' '.join(f'Sentence number {i} describes the expedition moving steadily along the river bank.' for i in range(1, 9))
        long_rows = list(long_story_continuations(story, 1024, min_prefix_bytes=180, target_bytes=240))
        self.assertTrue(long_rows)
        self.assertGreaterEqual(len(long_rows[0][0].encode()), 180)

        item = dict(
            story=story,
            questions=[dict(turn_id=1, input_text='Where?')],
            answers=[dict(turn_id=1, input_text='river bank')],
        )
        dialogs = [dict(utterances=[dict(speaker='USER', text='Hi.'), dict(speaker='ASSISTANT', text='Hello.')])]
        pools = prepare_dialogues(
            {'coqa-dev-v1.0.json': {'data': []}, 'coqa-train-v1.0.json': {'data': [item]}, 'self-dialogs.json': dialogs},
            1024, qa_style='qa',
        )
        qa_rows = pools['train']['qa'] + pools['validation']['qa']
        self.assertTrue(qa_rows and qa_rows[0][0].endswith('Answer: '))
        chat_rows = pools['train']['chat'] + pools['validation']['chat'] + pools['heldout']['chat']
        self.assertTrue(chat_rows and chat_rows[0][0].endswith('Assistant: '))

    def test_mid_and_read_window_stream_equivalence(self):
        torch.manual_seed(5)
        shape = Shape(width=32, heads=2, slot_dim=16, chunk=8, target=200000)
        base = LM(s=shape)
        both = LM(s=shape, read_window=24, mid_window=24).eval()
        both.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(both.blocks[3].mix.window, 24)
        self.assertEqual(both.blocks[7].mix.window, 24)
        seq = torch.randint(256, (2, 35))
        with torch.inference_mode():
            full = both(seq)
            state, parts = None, []
            for block in seq.split(8, dim=1):
                logits, state = both.stream_chunk(block, state)
                parts.append(logits)
            torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-5, rtol=2e-4)

    def test_reply_mask_and_complete_turns(self):
        from dialogue import chat_prompt, target_batch
        history = [('Earlier?', 'A long answer about an earlier event.'), ('Who?', 'Jo.')]
        recent = chat_prompt('Where?', history[-1:], 'A café.', 200)
        prompt = chat_prompt('Where?', history, 'A café.', len(recent.encode()))
        self.assertEqual(prompt, recent)
        self.assertIn('User: Who?\nAssistant: Jo.\n', prompt)
        self.assertIsNone(chat_prompt('Where?', [], 'A café.', 4))
        x, y = target_batch([(prompt, 'Lyon.\n'), ('X', 'é\n')], 'cpu')
        for i, (p, a) in enumerate([(prompt, 'Lyon.\n'), ('X', 'é\n')]):
            start = len(p.encode()) - 1
            self.assertTrue((y[i, :start] == -100).all())
            self.assertEqual(y[i, start:start+len(a.encode())].tolist(), list(a.encode()))
            self.assertTrue((y[i, start+len(a.encode()):] == -100).all())
            self.assertEqual(x[i, :start+1].tolist(), list(p.encode()))
        _, train_y = target_batch([(prompt, 'Lyon.\n')], 'cpu', history_targets=True)
        self.assertEqual(bytes(train_y[train_y != -100].tolist()), b'Jo.\nLyon.\n')

    def test_reserved_passage_and_followup_history(self):
        from dialogue import prepare_dialogues
        story = 'Lina met Ray. They walked home.'
        item = dict(story=story, questions=[
            dict(turn_id=1, input_text='Who did Lina meet?'),
            dict(turn_id=2, input_text='Where did they go?')],
            answers=[dict(turn_id=1, input_text='Ray.'), dict(turn_id=2, input_text='Home.')])
        dialogs = [dict(utterances=[dict(speaker=s, text=t) for s, t in [
            ('USER', 'Hello.'), ('USER', 'Is a table free?'), ('ASSISTANT', 'It is.'),
            ('USER', 'For two?'), ('ASSISTANT', 'Yes.')]])]
        data = {'coqa-dev-v1.0.json': {'data': [item]}, 'coqa-train-v1.0.json': {'data': [item]},
                'self-dialogs.json': dialogs}
        pools = prepare_dialogues(data, 512)
        self.assertEqual(pools['train']['qa'] + pools['validation']['qa'], [])
        self.assertEqual(pools['heldout']['prose'], [('Lina met Ray. ', 'They walked home.')])
        followup, target = pools['heldout']['qa'][1]
        self.assertEqual(target, 'Home.\n')
        self.assertEqual(followup, 'Passage: Lina met Ray. They walked home.\n'
                         'User: Who did Lina meet?\nAssistant: Ray.\nUser: Where did they go?\nAssistant: ')
        groups = [data['chat'] for data in pools.values() if data['chat']]
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0][1], ('User: Hello. Is a table free?\nAssistant: It is.\n'
                                       'User: For two?\nAssistant: ', 'Yes.\n'))
        self.assertNotIn('qa_excerpt', pools['heldout'])
        self.assertNotIn('qa_excerpt', pools['validation'])

    def test_short_evidence_is_for_extractive_training_answers(self):
        from dialogue import training_excerpt
        story = 'An earlier event. ' * 18 + 'The blue book was in Leeds. ' + 'A later event. ' * 18
        start = story.index('The blue')
        answer = dict(input_text='Leeds', span_start=start, span_end=start+27)
        excerpt = training_excerpt(story, answer)
        self.assertIn('The blue book was in Leeds.', excerpt)
        self.assertLess(len(excerpt), len(story))
        for other in ('yes', 'no', 'Dover'):
            self.assertEqual(training_excerpt(story, dict(answer, input_text=other)), '')

    def test_long_final_attention_streams_across_chunks(self):
        torch.manual_seed(3)
        shape = Shape(width=32, heads=2, slot_dim=16, chunk=8, target=200000)
        base = LM(s=shape)
        wider = LM(s=shape, read_window=24).eval()
        wider.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(wider.blocks[-1].mix.window, 24)
        self.assertEqual(wider.blocks[3].mix.window, 8)
        seq = torch.randint(256, (2, 35))
        with torch.inference_mode():
            full = wider(seq)
            state, parts = None, []
            for block in seq.split(8, dim=1):
                logits, state = wider.stream_chunk(block, state)
                parts.append(logits)
            torch.testing.assert_close(full, torch.cat(parts, 1), atol=2e-5, rtol=2e-4)
            torch.testing.assert_close(wider(seq[:, :17]), full[:, :17], atol=2e-5, rtol=2e-4)

    def test_held_out_worlds_and_labels(self):
        make = self.api['make_probe']
        held_out = make(64)
        worlds = {item['world'] for item in held_out}
        # Reuse the evaluation seed deliberately to exercise the exclusion.
        practice = make(64, seed=719, excluded_worlds=worlds)
        self.assertTrue(worlds.isdisjoint(item['world'] for item in practice))
        for left, right in zip(held_out[::2], held_out[1::2]):
            self.assertEqual(left['endings'], right['endings'])
            self.assertNotEqual(left['answer'], right['answer'])
            for item in (left, right):
                facts, query = item['prompt'].rsplit('. ', 1)
                if item['template'] == 0:
                    in_box = dict(re.findall(r'The (\w+) is inside the (\w+)', facts))
                    in_place = dict(re.findall(r'The (\w+) is in the (\w+)', facts))
                else:
                    in_box = {obj: box for box, obj in re.findall(r'Inside the (\w+) is the (\w+)', facts)}
                    in_place = {box: place for place, box in re.findall(r'The (\w+) holds the (\w+)', facts)}
                noun = query.split()[1]
                place = in_place[in_box[noun] if item['task'] == 'two_hop' else noun]
                self.assertEqual(item['endings'][item['answer']], f' {place}.')

    def test_short_lessons_teach_the_corresponding_links(self):
        make = self.api['make_probe']
        for lesson in ('one_link', 'support_only'):
            for item in make(12, lesson=lesson):
                facts, query = item['prompt'].rsplit('. ', 1)
                answer = item['endings'][item['answer']].strip(' .')
                noun = query.split()[1]
                if lesson == 'one_link' and item['task'] == 'two_hop':
                    statements = (f'The {noun} is inside the {answer}', f'Inside the {answer} is the {noun}')
                else:
                    box = noun
                    if item['task'] == 'two_hop':
                        match = (re.search(r'is inside the (\w+)', facts) if item['template'] == 0
                                 else re.search(r'Inside the (\w+)', facts))
                        box = match[1]
                    statements = (f'The {box} is in the {answer}', f'The {answer} holds the {box}')
                self.assertIn(statements[item['template']], facts)

    def test_prose_continuation_uses_only_its_own_paragraph(self):
        boundary = self.api['sentence_boundary']
        example = b'The highway reached Sault Ste. Marie. Another road continued east.'
        self.assertEqual(example[boundary(example, 8, 5):], b'Another road continued east.')
        self.assertIsNone(boundary(b'Just one sentence.', 5, 5))
        row = b'The boat crossed the river. The crew tied it up.'
        anchor = boundary(row, 12, 12)
        self.assertEqual(row[anchor:anchor+3], b'The')
        self.assertIsNone(boundary(row, anchor+1, 12))
        self.assertIsNone(boundary(row, 12, len(row)-anchor+1))
        raw = np.frombuffer(b'aaaaABCDbbbbWXYZcccc1234', dtype=np.uint8)
        anchors = np.array([4, 12, 20])
        real = np.arange(3)
        other = np.roll(real, 1)
        ids, y = self.api['continuation_batch'](raw, anchors, real, other, 4, 4)
        good, bad = ids.chunk(2)
        torch.testing.assert_close(good[:, 4:], bad[:, 4:])
        torch.testing.assert_close(good[:, :4].roll(1, 0), bad[:, :4])
        torch.testing.assert_close(y[:, 3:], torch.tensor([list(b'ABCD'),list(b'WXYZ'),list(b'1234')]))
        pos, neg = self.api['choose_paragraphs'](np.random.default_rng(4), 3, 100)
        self.assertTrue(np.all(pos != neg))
        logits = torch.zeros(6, 7, 256)
        logits[:3].scatter_(2, y.unsqueeze(-1), 5.)
        true_nll, wrong_nll = self.api['continuation_nll'](logits.requires_grad_(), y, 4)
        self.assertEqual(true_nll.shape, (3, 7))
        self.assertEqual(wrong_nll.shape, (3, 4))
        self.assertTrue((true_nll[:, 3:] < wrong_nll).all())
        wrong_nll.sum().backward()
        self.assertFalse(logits.grad[3:, :3].any())

    def test_display_of_wikitext_punctuation(self):
        raw = 'At dawn , the well @-@ known crew crossed .\nIt took 3 @.@ 5 days !'
        self.assertEqual(self.api['display_wiki_sample'](raw),
                         'At dawn, the well-known crew crossed.\nIt took 3.5 days!')

    def test_generation_prompts_share_a_boundary_and_end_before_the_reference(self):
        make = self.api['prose_prompt_pairs']
        start = 'An earlier paragraph.\nA long introduction mentions the café. '
        history = 'The captain lowered the sail while the crew stood ready. '
        short = 'The ship reached the harbour. '
        prefix = (start + history + short).encode()
        raw = np.frombuffer(prefix + b'The secret continuation follows.', dtype=np.uint8)
        # The context budget starts inside the UTF-8 encoding of the é in café.
        width = len(prefix) - prefix.index('é'.encode()) - 1
        rows = make(raw, [len(prefix)], width, 3)
        self.assertEqual(rows, [dict(offset=len(prefix), short=short, long=history+short)])
        self.assertEqual(rows, make(raw, [len(prefix), len(prefix)], width, 3))
        self.assertEqual(make(raw, [len(prefix)], len(short.encode()), 1), [])
        self.assertEqual(make(raw, [len(prefix)], width, 0), [])

    def test_sample_stops_at_a_generated_sentence_or_hard_limit(self):
        class ScriptedLM:
            s = Shape(chunk=64)
            def __init__(self):
                self.next_bytes = iter(b'First. Second! More words follow. ')
            def stream_chunk(self, ids, state):
                logits = torch.full((1, 1, 256), -1000.)
                logits[0, 0, next(self.next_bytes)] = 0.
                return logits, None
        sample = self.api['sample']
        self.assertEqual(sample(ScriptedLM(), 'Start. ', 30, 1., 'cpu', finish_after=8), 'Start. First. Second! ')
        self.assertEqual(sample(ScriptedLM(), 'Start. ', 11, 1., 'cpu', finish_after=8), 'Start. First. Seco')

    def test_container_diagnostic_compares_scores_per_byte(self):
        api = notebook_functions()
        api['ending_scores'] = lambda *args: torch.tensor([[-5., -6.], [-5., -6.]])
        rows = [dict(task='object_to_container', answer=a, endings=[' bag.', ' drawer.']) for a in (0, 1)]
        _, scored = api['probe_model'](Mock(), rows, 'cpu')
        self.assertEqual(scored[0]['predicted'], 1)
        self.assertEqual(scored[0]['logp_left'], -5.)
        self.assertEqual(scored[0]['score_unit'], 'logp_per_byte')

    def test_practice_only_skips_prose_training_benchmarks(self):
        api = notebook_functions()
        api.update(TRAIN_MINUTES=0, DEVICE='cpu', profile_prose=Mock(side_effect=AssertionError))
        profiles, tuning = api['setup_profiles']('torch')
        self.assertFalse(profiles)
        self.assertEqual(tuning['trials'], [])
        self.assertGreaterEqual(tuning['selected_batch'], 2)

    def test_short_place_lessons_require_choosing_between_both_places(self):
        for item in self.api['make_probe'](32, lesson='support_only'):
            facts = item['prompt'].rsplit('. ', 1)[0]
            for ending in item['endings']:
                self.assertRegex(facts, r'\b' + ending.strip(' .') + r'\b')

    def test_hop_diagnostics_use_the_original_world_and_correct_link(self):
        make = self.api['make_probe']
        items = [r for r in make(8) if r['task'] == 'two_hop']
        for stage in ('object_to_container', 'container_to_place'):
            hops = [r for r in make(8, hop=stage) if r['task'] == stage]
            self.assertEqual(len(hops), len(items))
            for row, hop in zip(items, hops):
                facts, query = row['prompt'].rsplit('. ', 1)
                hop_facts, hop_query = hop['prompt'].rsplit('. ', 1)
                self.assertEqual(facts, hop_facts)
                noun = query.split()[1]
                box = (re.search(r'The '+noun+r' is inside the (\w+)', facts)[1] if row['template']==0
                       else re.search(r'Inside the (\w+) is the '+noun, facts)[1])
                if stage == 'object_to_container':
                    self.assertEqual(hop['endings'][hop['answer']], ' '+box+'.')
                    self.assertIn('The '+noun+' ', hop_query)
                    self.assertNotEqual(hop_query, query)
                    self.assertIn('container', hop_query)
                else:
                    self.assertIn('The '+box+' ', hop_query)
                    self.assertEqual(hop['endings'][hop['answer']], row['endings'][row['answer']])

    def test_practice_uses_the_same_container_question_as_the_diagnostic(self):
        make = self.api['make_probe']
        short = [r for r in make(4, lesson='one_link') if r['task'] == 'two_hop']
        full = [r for r in make(4, hop='object_to_container') if r['task'] == 'object_to_container']
        for practice, probe in zip(short, full):
            self.assertIn('container', practice['prompt'].rsplit('. ', 1)[1])
            self.assertEqual(practice['prompt'].rsplit('. ', 1)[1], probe['prompt'].rsplit('. ', 1)[1])
            self.assertEqual(practice['endings'], probe['endings'])

    def test_batched_scores_match_each_byte_and_gradient(self):
        torch.manual_seed(14)
        model = LM(s=Shape(width=16, heads=2, slot_dim=8, chunk=8, target=50000))
        items = [{'prompt': 'At dawn', 'endings': [' é.', ' east.']},
                 {'prompt': 'A', 'endings': [' x.', ' y.']}]
        actual = self.api['ending_scores'](model, items, 'cpu')
        # Reference the full vocabulary projection at every input position.
        sequences = [torch.tensor(list((item['prompt'] + ending).encode())) for item in items for ending in item['endings']]
        ids = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        logp = model(ids[:, :-1]).log_softmax(-1)
        expected = []
        for item in items:
            for ending in item['endings']:
                start = len(item['prompt'].encode()) - 1
                expected.append(sum(logp[len(expected), start+i, byte] for i, byte in enumerate(ending.encode())))
        oracle = torch.stack(expected).reshape_as(actual)
        torch.testing.assert_close(actual, oracle)
        torch.testing.assert_close(torch.autograd.grad(actual.sum(), model.embed.weight)[0],
                                   torch.autograd.grad(oracle.sum(), model.embed.weight)[0],
                                   atol=2e-5, rtol=1e-4)

    def test_batch_search_throughput_and_memory(self):
        tune = self.api['choose_batch']
        rates = {2: 100, 4: 180, 5: 160, 7: 150, 8: 170, 16: 160}
        def measure(n):
            return dict(batch=n, rate=rates.get(n, 155), peak_allocated_bytes=10*n if n < 7 else 10*n+5)
        self.assertEqual(tune(measure, 2, 2, 16, 200, 'rate')[0]['batch'], 4)
        best, trials = tune(measure, 2, 2, 16, 70, 'rate')
        self.assertEqual(best['batch'], 4)
        self.assertIn('over_memory_budget', [t['status'] for t in trials])
        def limited(n):
            if n > 2:
                raise torch.OutOfMemoryError()
            return measure(n)
        best, trials = tune(limited, 8, 2, 16, 200, 'rate')
        self.assertEqual(best['batch'], 2)
        self.assertEqual(trials[0]['status'], 'out_of_memory')
        with self.assertRaises(torch.OutOfMemoryError):
            tune(limited, 4, 4, 16, 200, 'rate')

    def test_oom_retry_rewinds_random_windows(self):
        rng = np.random.default_rng(101)
        opt = Mock()
        def backward(size):
            draws = rng.integers(1000, size=size)
            if size > 2:
                raise torch.OutOfMemoryError()
            return draws
        draws, size = self.api['backward_with_backoff'](backward, opt, rng, 8, 2, True)
        reference = np.random.default_rng(101)
        self.assertEqual(size, 2)
        np.testing.assert_array_equal(draws, reference.integers(1000, size=2))
        self.assertEqual(rng.bit_generator.state, reference.bit_generator.state)
        self.assertEqual(opt.zero_grad.call_count, 2)
        opt.step.assert_not_called()

    def test_batch_change_on_resume_and_window_storage(self):
        match = self.api['compatible_config']
        old = dict(seed=42, batch=16, corpus_hashes={'train': 'abc'})
        self.assertTrue(match(old, dict(old, batch=32), True))
        self.assertFalse(match(old, dict(old, batch=32), False))
        self.assertFalse(match(old, dict(old, batch=32, corpus_hashes={}), True))
        starts = np.array([0, 64, 191], dtype=np.int64)
        raw = np.arange(256, dtype=np.uint8)
        actual = self.api['windows'](torch.from_numpy(raw), starts, 64)
        expected = self.api['windows'](raw, starts, 64)
        torch.testing.assert_close(actual, expected)

    @torch.inference_mode()
    def test_generation_across_chunk_boundaries(self):
        torch.manual_seed(6)
        model = LM(s=Shape(width=16, heads=2, slot_dim=8, chunk=8, target=50000)).eval()
        for prompt in ('a' * 7, 'b' * 8, 'c' * 17):
            tokens = list(prompt.encode())
            generator = torch.Generator().manual_seed(123)
            decoder = codecs.getincrementaldecoder('utf8')()
            for remaining in range(12, 0, -1):
                logits = model(torch.tensor([tokens]))[0, -1]
                valid = self.api['utf8_mask'](torch.arange(256), decoder.getstate()[0], remaining)
                byte = self.api['sample_byte'](logits.masked_fill(~valid, -torch.inf), .75, .9, generator)
                decoder.decode(bytes([byte]))
                tokens.append(byte)
            expected = bytes(tokens).decode('utf8')
            self.assertEqual(self.api['sample'](model, prompt, 12, .75, 'cpu', .9), expected)

    def test_utf8_mask_matches_standard_decoder(self):
        prefixes = [b'', b'\xc2', b'\xdf', b'\xe0', b'\xed', b'\xef\x80',
                    b'\xf0', b'\xf3\x91', b'\xf4', b'\xf4\x80\x80']
        for prefix in prefixes:
            allowed = self.api['utf8_mask'](torch.arange(256), prefix, 4).tolist()
            for byte in range(256):
                valid = False
                for suffix in (b'', b'\x80', b'\x80\x80', b'\xa0\x80', b'\x80\x80\x80', b'\x90\x80\x80'):
                    try:
                        (prefix + bytes([byte]) + suffix).decode('utf8')
                    except UnicodeDecodeError:
                        continue
                    valid = True
                    break
                self.assertEqual(allowed[byte], valid, (prefix, byte))
        for remaining, maximum in ((1, 0x7f), (2, 0xdf), (3, 0xef)):
            mask = self.api['utf8_mask'](torch.arange(256), b'', remaining)
            self.assertEqual(int(torch.where(mask)[0][-1]), maximum)


if __name__ == '__main__':
    unittest.main()
