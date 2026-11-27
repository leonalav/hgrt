"""Execute run.ipynb end-to-end on CPU with tiny budgets.

This is the strongest pre-flight check available before spending an A100
session: it actually RUNS every cell, so a NameError, a bad import, a shape
mismatch or a runtime crash in the training/eval path shows up here rather
than 30 minutes into a Colab session.

The budgets are tiny (a few training steps per stage), so this validates
plumbing, not learning.

Usage:
    python _smoke_nb.py            # execute and report
    python _smoke_nb.py --keep     # keep the executed notebook for inspection
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
NB = ROOT / 'run.ipynb'
OUT = ROOT / 'runs' / 'smoke'
SMOKE = OUT / 'run_smoke.ipynb'

# Settings overrides for the smoke run.
OVERRIDES = {
    'DEVICE': "'cpu'",
    'TRAIN_MINUTES': '0.05',
    'FACT_SWITCH_MINUTES': '0.05',
    'DIALOGUE_MINUTES': '0.05',
    'SAVE_TO_DRIVE': 'False',
    'AUTO_RESUME': 'False',
    'OWT_DOCUMENTS': '0',
    'PROBE_PAIRS': '4',
    'VALID_WINDOWS': '4',
    'TEST_WINDOWS': '4',
    'FACT_SWITCH_EVAL_PAIRS': '2',
    'DIALOGUE_EVAL_ROWS': '4',
    'MAX_UPDATES': '3',
    'AUTO_BATCH': 'False',
    'TRY_FUSED': 'False',
    'SENTENCE_EVAL_GROUPS': '16',
}


def patch(nb):
    cells = {c.get('id'): c for c in nb['cells']}
    src = cells['3adb75ac']
    text = ''.join(src['source'])
    for key, value in OVERRIDES.items():
        pattern = rf'^{re.escape(key)} = .*$'
        new, n = re.subn(pattern, f'{key} = {value}', text, count=1, flags=re.M)
        if n == 0:
            print(f'  WARNING: setting {key} not found')
        text = new
    src['source'] = text.splitlines(keepends=True)
    # Neutralise the Drive mount: it is Colab-only and would hang locally.
    c = cells.get('4761bdeb')
    if c:
        t = ''.join(c['source'])
        t = t.replace('from google.colab import drive',
                      "raise ImportError('no Drive on this runtime')")
        c['source'] = t.splitlines(keepends=True)
    return nb


def main():
    keep = '--keep' in sys.argv
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    nb = json.loads(NB.read_text(encoding='utf8'))
    nb = patch(nb)
    SMOKE.write_text(json.dumps(nb, indent=1), encoding='utf8')
    print(f'wrote {SMOKE}')

    cmd = [
        sys.executable, '-m', 'jupyter', 'nbconvert', '--to', 'notebook',
        '--execute', '--ExecutePreprocessor.timeout=3600',
        '--ExecutePreprocessor.kernel_name=python3',
        f'--output-dir={OUT}', '--output=run_smoke.ipynb',
        str(SMOKE),
    ]
    print('running:', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4000)
    (OUT / 'nbconvert.stdout.txt').write_text(proc.stdout, encoding='utf8')
    (OUT / 'nbconvert.stderr.txt').write_text(proc.stderr, encoding='utf8')
    print('nbconvert exit code:', proc.returncode)
    if proc.returncode != 0:
        print('--- stdout tail ---')
        print(proc.stdout[-4000:])
        print('--- stderr tail ---')
        print(proc.stderr[-6000:])
        return 1

    out_nb = json.loads((OUT / 'run_smoke.ipynb').read_text(encoding='utf8'))
    code_cells = [c for c in out_nb['cells'] if c.get('cell_type') == 'code']
    errors = []
    for c in code_cells:
        for o in c.get('outputs', []):
            if o.get('output_type') == 'error':
                errors.append((c.get('id'), '\n'.join(o.get('traceback', []))))
    print(f'\n=== {len(code_cells)} code cells, {len(errors)} errors ===')
    for cid, tb in errors:
        print(f'--- ERROR in cell {cid} ---')
        print(tb[-4000:])
    if errors:
        return 1

    print('\n=== stage summary ===')
    for c in out_nb['cells']:
        for o in c.get('outputs', []):
            t = ''.join(o.get('text', []))
            if not t.strip():
                continue
            for line in t.splitlines():
                if any(k in line for k in (
                        'MEMORY-REQUIRED', 'precondition_ok', 'BINDING GENERALISATION',
                        'fact-switch stage:', 'Dialogue stage:', 'Selected dialogue',
                        'Selected fact-switch', 'counterfactual swap:',
                        'permutation', 'Validation BPB', 'validation_bpb')):
                    print(f'  [{c.get("id")}] {line.strip()[:150]}')

    print(f'\nOK -- no cell errors. Executed notebook: {OUT / "run_smoke.ipynb"}')
    if not keep:
        shutil.rmtree(OUT)
        print(f'(removed {OUT}; pass --keep to retain)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
