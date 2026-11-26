"""Replacement resume-selection block for run.ipynb (cell d1eb119d).

Replaces the old `matching_checkpoint` logic, which only globbed
`wiki-*/latest.pt` and therefore could never see a run directory whose only
prose artefact was `best_lm.pt` (for example a dialogue-only continuation run).

New behaviour:
- Survey every run directory and report why each candidate was kept or skipped.
- `latest.pt` (has optimizer state) -> resume training, newest compatible wins.
- `best_lm.pt` (weights only) -> seed INITIAL_WEIGHTS, lowest selection loss wins.
- The best weights are used whenever they are better than the weights that the
  resume checkpoint would load, so a superior `best_lm.pt` is never ignored.
"""

RESUME_BLOCK = '''
def _selection_loss(weights):
    """Rank prose checkpoints the same way the trainer does: continuation BPB, else validation BPB."""
    continuation = weights.get('continuation') or {}
    matched = continuation.get('matched_bpb')
    if isinstance(matched, (int, float)) and math.isfinite(matched):
        return float(matched)
    bpb = weights.get('validation_bpb')
    if isinstance(bpb, (int, float)) and math.isfinite(bpb):
        return float(bpb)
    return float('inf')


def _config_diff(saved, current):
    """Explain a config mismatch in human-readable terms."""
    if not isinstance(saved, dict):
        return 'no readable saved config'
    if not isinstance(current, dict):
        return 'no current config'
    diffs = [f'{key}: saved={saved.get(key)!r} current={current.get(key)!r}'
             for key in sorted(set(saved) | set(current))
             if saved.get(key) != current.get(key)]
    return '; '.join(diffs) if diffs else 'identical dictionaries'


def survey_checkpoints(roots, config, adaptive):
    """Survey every run directory for usable prose checkpoints.

    Returns (best_resume, best_weights, report):
      best_resume  - newest compatible `latest.pt` (can resume training exactly)
      best_weights - `best_lm.pt` with the lowest selection loss (weights only)
      report       - one entry per directory, explaining its fate
    """
    resumes, weights, report = [], [], []

    for root in roots:
        for run in sorted(root.glob('wiki-*')):
            if not run.is_dir() or run == run_dir:
                continue
            entry = {'run': str(run)}

            latest = run/'latest.pt'
            if latest.is_file():
                entry['latest'] = str(latest)
                try:
                    saved = json.loads((run/'config.json').read_text(encoding='utf8')).get('config')
                except (OSError, ValueError):
                    saved = None
                if compatible_config(saved, config, adaptive):
                    entry['latest_status'] = 'compatible'
                    resumes.append((latest.stat().st_mtime, latest))
                else:
                    entry['latest_status'] = 'config_mismatch'
                    entry['latest_detail'] = _config_diff(saved, config)

            best = run/'best_lm.pt'
            if best.is_file():
                entry['best_lm'] = str(best)
                try:
                    ck = torch.load(best, map_location='cpu', weights_only=True)
                except Exception as exc:
                    entry['best_lm_status'] = 'unreadable'
                    entry['best_lm_detail'] = str(exc)
                else:
                    if ck.get('kind') != 'C1' or ck.get('slots') != 8:
                        entry['best_lm_status'] = 'wrong_architecture'
                    elif ck.get('config', {}).get('corpus_hashes') != config['corpus_hashes']:
                        entry['best_lm_status'] = 'different_corpus'
                    else:
                        loss = _selection_loss(ck)
                        entry['best_lm_status'] = 'usable'
                        entry['selection_loss'] = loss
                        weights.append((loss, -best.stat().st_mtime, best))
                    del ck

            report.append(entry)

    best_resume = max(resumes)[1] if resumes else None
    best_weights = min(weights)[2] if weights else None
    return best_resume, best_weights, report


def print_resume_report(report, chosen_resume, chosen_weights):
    """Log every candidate so a silent fallback can never happen again."""
    if not report:
        print('Resume survey: no other run directories found.')
        return
    print('Resume survey of candidate run directories:')
    for entry in report:
        print(f'  {entry["run"]}')
        if entry.get('latest'):
            detail = entry.get('latest_detail')
            print(f'      latest.pt : {entry.get("latest_status", "?")}'
                  + (f' ({detail})' if detail else ''))
        if entry.get('best_lm'):
            detail = entry.get('best_lm_detail')
            loss = entry.get('selection_loss')
            loss_text = f', selection_loss={loss:.4f}' if isinstance(loss, float) and math.isfinite(loss) else ''
            print(f'      best_lm.pt : {entry.get("best_lm_status", "?")}{loss_text}'
                  + (f' ({detail})' if detail else ''))
    print(f'  -> resume training from : {chosen_resume or "(none)"}')
    print(f'  -> best prose weights   : {chosen_weights or "(none)"}')


resume_path = Path(RESUME_FROM) if RESUME_FROM else None
best_weights_path = None

if AUTO_RESUME and not (resume_path or INITIAL_WEIGHTS):
    roots = {parent} if OUTPUT_ROOT else {parent, local_root/'HGRT-10M'/'runs'}
    if not OUTPUT_ROOT:
        roots = roots | {Path.cwd()/'runs'}
    adaptive = dynamic or TRAIN_MINUTES == 0
    resume_path, best_weights_path, survey = survey_checkpoints(roots, config, adaptive)
    print_resume_report(survey, resume_path, best_weights_path)

    # A `latest.pt` resumes training exactly but may hold older weights than the
    # best `best_lm.pt` in another directory. When the best weights are strictly
    # better, seed from them instead, so superior prose is never discarded.
    if best_weights_path is not None:
        if resume_path is None:
            INITIAL_WEIGHTS = str(best_weights_path)
        else:
            try:
                latest_weights = torch.load(resume_path, map_location='cpu', weights_only=True)
                latest_loss = _selection_loss(latest_weights)
                del latest_weights
            except Exception:
                latest_loss = float('inf')
            try:
                best_ck = torch.load(best_weights_path, map_location='cpu', weights_only=True)
                best_loss = _selection_loss(best_ck)
                del best_ck
            except Exception:
                best_loss = float('inf')
            if best_loss < latest_loss:
                print(f'Best prose weights are better than the resume checkpoint '
                      f'({best_loss:.4f} < {latest_loss:.4f}); seeding weights from best_lm.pt.')
                INITIAL_WEIGHTS = str(best_weights_path)
                resume_path = None

if TRAIN_MINUTES == 0 and not (resume_path or INITIAL_WEIGHTS):
    raise ValueError('For practice without prose training, supply an existing prose checkpoint.')
print('Continuing from ' + str(resume_path) if resume_path else
      'Starting from ' + str(INITIAL_WEIGHTS) if INITIAL_WEIGHTS else 'Starting a fresh prose model.')
save_json('config.json', dict(config=config, hardware=hardware, backend=backend, minutes=TRAIN_MINUTES,
                              initial_weights=str(INITIAL_WEIGHTS), resume_from=str(resume_path or ''),
                              prose_objective='paragraph-continuation-v2', continuation_active=use_continuations))
model = new_model(backend)
optimizer = make_optimizer(model) if TRAIN_MINUTES else None
train_rng = np.random.default_rng(SEED)
step, bytes_seen = 0, 0
if resume_path or INITIAL_WEIGHTS:
    ck = torch.load(resume_path or INITIAL_WEIGHTS, map_location='cpu', weights_only=True)
    if ck['kind'] != 'C1' or ck['slots'] != 8:
        raise ValueError('Select a C1-S8 checkpoint.')
    model.load_state_dict(ck['state_dict'])
    model.blocks[-1].mix.window = ck.get('read_window') or model.s.chunk
    model.blocks[3].mix.window = ck.get('mid_window') or model.s.chunk
    if resume_path:
        if not compatible_config(ck.get('config'), config, dynamic or TRAIN_MINUTES == 0):
            raise ValueError('Resume requires matching data, model and optimizer settings. Use INITIAL_WEIGHTS for transfer.')
        previous_batch = ck.get('batch_at_save', ck['config']['batch'])
        if TRAIN_MINUTES and previous_batch != batch_size:
            print(f'Resuming with {batch_size} sequences per update (previously {previous_batch}).')
        if TRAIN_MINUTES:
            optimizer.load_state_dict(ck['optimizer'])
        step, bytes_seen = ck['step'], ck['bytes_seen']
        train_rng.bit_generator.state = json.loads(ck['train_rng'])
    else:
        step, bytes_seen = ck.get('step', 0), ck.get('bytes_seen', 0)
        if ck.get('validation_bpb') is not None:
            print(f'Seeded weights from {INITIAL_WEIGHTS} '
                  f'(validation_bpb={ck["validation_bpb"]:.4f}); starting a fresh optimizer.')
    del ck
'''
