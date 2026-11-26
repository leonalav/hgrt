"""Phase 0 cell source for run.ipynb (greedy vs sampled decode hypothesis test).

Kept in sync_notebook.py so the cell is reproducible from source control.
"""

PHASE0_CODE = '''
"""
PHASE 0: HYPOTHESIS VALIDATION
Compare greedy vs. sampled decoding on the trained best_lm.pt checkpoint.

Decision gate:
- If greedy >> sampled: problem is sampling, not training -> inference fix is enough
- If greedy ~ sampled: problem is the training objective -> proceed to Phase 1 retrain

This cell uses the notebook's own LM class and sampler (defined in earlier cells).
It does not import model.py, because model.py is embedded in this notebook and is
never written to disk.
"""

import json, re, torch

# ---- Locate the checkpoint produced by this run -----------------------------
ck_path = run_dir / 'best_lm.pt'
if not ck_path.exists():
    raise FileNotFoundError(
        f'Checkpoint not found at {ck_path}. '
        'Run the training cells above first, then re-run this cell.'
    )

print(f'Loading checkpoint from: {ck_path}')
ck = torch.load(ck_path, map_location='cpu', weights_only=True)
print(f'Checkpoint: kind={ck["kind"]}, slots={ck.get("slots")}, '
      f'validation_bpb={ck.get("validation_bpb"):.4f}')

# ---- Rebuild the model ------------------------------------------------------
model = LM(ck['kind'], ck.get('slots', 8), read_window=ck.get('read_window')).to(device)
model.load_state_dict(ck['state_dict'])
model.eval()
print(f'Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters\\n')

# ---- Prompts to probe -------------------------------------------------------
TEST_PROMPTS = [
    'The expedition reached the river at dawn. ',
    'Gauthier and Knoote married on May 22 , 1911 . ',
    'Income from open days at the mill came to \\u00a3 151 . ',
    'There is disagreement among music scholars as to its importance . ',
]

def gen_greedy(prompt, max_bytes=160):
    """Greedy decode: always take the highest-probability valid byte."""
    tokens = list(prompt.encode('utf8'))
    decoder = codecs.getincrementaldecoder('utf8')()
    byte_ids = torch.arange(256, device=device)
    chunk = model.s.chunk
    state = None
    start = 0
    with torch.inference_mode():
        for remaining in range(max_bytes, 0, -1):
            while len(tokens) - start > chunk:
                ids = torch.tensor([tokens[start:start+chunk]], device=device)
                _, state = model.stream_chunk(ids, state)
                start += chunk
            ids = torch.tensor([tokens[start:]], device=device)
            logits, pending = model.stream_chunk(ids, state)
            valid = utf8_mask(byte_ids, decoder.getstate()[0], remaining)
            masked = logits[0, -1].float().masked_fill(~valid, -torch.inf)
            byte = int(masked.argmax(-1).item())
            decoder.decode(bytes([byte]))
            tokens.append(byte)
            if byte == 10:
                break
            if ids.size(1) == chunk:
                state = pending
                start += chunk
    return bytes(tokens[len(prompt.encode()):]).decode('utf8', errors='replace')

def gen_sampled(prompt, max_bytes=160, temp=0.75, top_p=0.9):
    """Sampled decode, matching the notebook's own sampling settings."""
    tokens = list(prompt.encode('utf8'))
    generator = torch.Generator(device=device).manual_seed(123)
    decoder = codecs.getincrementaldecoder('utf8')()
    byte_ids = torch.arange(256, device=device)
    chunk = model.s.chunk
    state = None
    start = 0
    with torch.inference_mode():
        for remaining in range(max_bytes, 0, -1):
            while len(tokens) - start > chunk:
                ids = torch.tensor([tokens[start:start+chunk]], device=device)
                _, state = model.stream_chunk(ids, state)
                start += chunk
            ids = torch.tensor([tokens[start:]], device=device)
            logits, pending = model.stream_chunk(ids, state)
            valid = utf8_mask(byte_ids, decoder.getstate()[0], remaining)
            masked = logits[0, -1].float().masked_fill(~valid, -torch.inf)
            byte = sample_byte(masked, temp, top_p, generator)
            decoder.decode(bytes([byte]))
            tokens.append(byte)
            if byte == 10:
                break
            if ids.size(1) == chunk:
                state = pending
                start += chunk
    return bytes(tokens[len(prompt.encode()):]).decode('utf8', errors='replace')

def score(text):
    t = text.replace('@-@', '-').replace('@,@', ',')
    words = len(re.findall(r"[A-Za-z']+", t))
    sents = [s for s in re.split(r'(?<=[.!?])\\s+', t.strip()) if s]
    complete = sum(1 for s in sents if re.search(r'[.!?]$', s))
    dangling = bool(re.search(r'\\b(and|or|of|in|on|at|to|with|the|a|an|but)$', t, re.I))
    return dict(text=t[:100], words=words, sents=len(sents), complete=complete, dangling=dangling)

print('=' * 70)
print('PHASE 0: GREEDY vs SAMPLED DECODE')
print('=' * 70 + '\\n')

results = {}
for strat, gen_fn in [('greedy', gen_greedy), ('sampled', gen_sampled)]:
    results[strat] = [score(gen_fn(p)) for p in TEST_PROMPTS]
    print(f'{strat} generation complete')

for i, prompt in enumerate(TEST_PROMPTS):
    print(f'\\nPrompt {i}: {prompt[:60]!r}')
    for strat in ('greedy', 'sampled'):
        r = results[strat][i]
        print(f'  {strat:8s}: {r["words"]:3d}w {r["complete"]}/{r["sents"]}s '
              f'dangling={str(r["dangling"]):5s} | {r["text"]!r}')

print('\\n' + '=' * 70)
print('AGGREGATE METRICS')
print('=' * 70)
rates = {}
for strat in ('greedy', 'sampled'):
    rr = results[strat]
    comp = sum(r['complete'] for r in rr) / max(1, sum(r['sents'] for r in rr))
    dang = sum(r['dangling'] for r in rr) / len(rr)
    rates[strat] = comp
    print(f'\\n{strat.upper()}:')
    print(f'  complete-sentence rate: {100*comp:.0f}%')
    print(f'  dangling-end rate:      {100*dang:.0f}%')
    print(f'  mean words/gen:         {sum(r["words"] for r in rr)/len(rr):.0f}')

greedy_comp, sampled_comp = rates['greedy'], rates['sampled']
print('\\n' + '=' * 70)
print('HYPOTHESIS TEST RESULT')
print('=' * 70)

if greedy_comp > sampled_comp + 0.15:
    print(f'\\nGREEDY >> SAMPLED (greedy {100*greedy_comp:.0f}% vs sampled {100*sampled_comp:.0f}%)')
    print('  => Problem is sampling strategy, NOT the training objective.')
    print('  => RECOMMENDATION: use greedy / low-temperature decoding (inference-only fix).')
    print('     Phases 1-3 of the prose plan are likely unnecessary.')
elif greedy_comp < sampled_comp - 0.05:
    print(f'\\nSAMPLED > GREEDY (sampled {100*sampled_comp:.0f}% vs greedy {100*greedy_comp:.0f}%)')
    print('  => Unexpected. Investigate further before changing the objective.')
else:
    print(f'\\nGREEDY ~ SAMPLED (greedy {100*greedy_comp:.0f}% vs sampled {100*sampled_comp:.0f}%)')
    print('  => Problem is the training objective / weights, NOT sampling strategy.')
    print('  => RECOMMENDATION: proceed to Phase 1 (retrain with a sentence-aware objective).')

# Persist the result for later phases and for the final report.
phase0 = {
    'checkpoint': str(ck_path),
    'validation_bpb': ck.get('validation_bpb'),
    'prompts': len(TEST_PROMPTS),
    'greedy': {
        'complete_sentence_rate': greedy_comp,
        'dangling_end_rate': sum(r['dangling'] for r in results['greedy']) / len(results['greedy']),
        'mean_words': sum(r['words'] for r in results['greedy']) / len(results['greedy']),
        'samples': [r['text'] for r in results['greedy']],
    },
    'sampled': {
        'complete_sentence_rate': sampled_comp,
        'dangling_end_rate': sum(r['dangling'] for r in results['sampled']) / len(results['sampled']),
        'mean_words': sum(r['words'] for r in results['sampled']) / len(results['sampled']),
        'samples': [r['text'] for r in results['sampled']],
    },
    'decision': ('greedy_better' if greedy_comp > sampled_comp + 0.15
                 else 'sampled_better' if greedy_comp < sampled_comp - 0.05
                 else 'objective_is_bottleneck'),
}
out_path = run_dir / 'phase0_results.json'
out_path.write_text(json.dumps(phase0, indent=2, ensure_ascii=False), encoding='utf8')
print(f'\\nSaved: {out_path}')
'''
