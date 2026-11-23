# Paste the whole file into one code cell in the A100 Colab notebook.
# This portable snapshot includes data.py, model.py and run.py at the end.
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from zipfile import ZipFile

MINUTES = 25
TASK = 'controls'  # 'recall' runs the full gated comparison when time permits.
SAVE_TO_DRIVE = True


def install(python, log, *packages, required=True):
    return subprocess.run([
        str(python), '-m', 'pip', '--log', str(log), '--disable-pip-version-check',
        'install', '--retries', '1', '--timeout', '10', *packages
    ], check=required).returncode


def prepare_python(work):
    smi = shutil.which('nvidia-smi')
    smi_ok = smi and subprocess.run([
        smi, '--query-gpu=name,memory.total,driver_version', '--format=csv'
    ]).returncode == 0
    # Packages installed during this run go in a disposable virtual environment.
    env = work / 'env'
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages',
                    '--without-pip', str(env)], check=True)
    python = env / 'bin/python'
    log = work / 'pip.log'
    check_build = "import torch; from packaging.version import Version; assert torch.version.cuda and Version(torch.__version__) >= Version('2.5')"
    if subprocess.run([python, '-c', check_build], capture_output=True).returncode:
        if not smi_ok:
            raise RuntimeError('Select a GPU runtime in Colab and rerun this cell.')
        # Official PyTorch CUDA 12.8 wheel, compatible with the observed 580 driver.
        install(python, log, '--index-url', 'https://download.pytorch.org/whl/cu128', 'torch==2.9.1+cu128')

    compute_probe = """
import torch
if not torch.cuda.is_available():
    raise SystemExit('PyTorch cannot open a GPU in this kernel. Check the Colab accelerator setting.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('Select the BF16-capable A100 for this experiment.')
x = torch.ones((16, 16), device='cuda', dtype=torch.bfloat16)
assert (x @ x)[0, 0].item() == 16
print('CUDA compute verified:', torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, flush=True)
"""
    if subprocess.run([python, '-c', compute_probe]).returncode:
        raise RuntimeError('GPU compute check failed in this notebook runtime. See the CUDA message above.')
    install(python, log, 'numpy>=1.24')
    fla_import = 'from fla.ops.gated_delta_rule import chunk_gated_delta_rule'
    if subprocess.run([python, '-c', fla_import], capture_output=True).returncode:
        # Preserve the verified Torch/Triton versions when adding the optional kernel.
        freeze = "import importlib.metadata as m; print('torch=='+m.version('torch')); print('triton=='+m.version('triton'))"
        versions = subprocess.run([python, '-c', freeze], capture_output=True, text=True)
        if versions.returncode == 0:
            constraints = work / 'constraints.txt'
            constraints.write_text(versions.stdout)
            install(python, log, '-c', str(constraints), 'fla-core', required=False)
    have_fla = subprocess.run([python, '-c', fla_import], capture_output=True).returncode == 0
    backend = 'fla' if have_fla else 'torch'
    print('Gated delta backend:', backend, flush=True)
    return python, backend


def results_directory():
    root = Path('/content') if Path('/content').is_dir() else Path.cwd()
    root = root / 'HGRT-10M' / 'runs'
    if SAVE_TO_DRIVE:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            my_drive = Path('/content/drive/MyDrive')
            if my_drive.is_dir():
                root = my_drive / 'HGRT-10M' / 'runs'
        except Exception as exc:
            # Colab authentication and mount exceptions vary by runtime version.
            print('Drive mount unavailable. Saving to the runtime.', type(exc).__name__, flush=True)
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f'{TASK}-{time.strftime("%Y%m%d-%H%M%S")}-', dir=root))


def run_milestone():
    with tempfile.TemporaryDirectory(prefix='hgrt-a100-') as tmp:
        work = Path(tmp)
        raw = base64.b64decode(''.join(SOURCE.split()), validate=True)
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError('Source checksum mismatch. Copy the complete cell.')
        with ZipFile(io.BytesIO(raw)) as archive:
            for filename in ('run.py', 'model.py', 'data.py'):
                (work / filename).write_bytes(archive.read(filename))
        python, backend = prepare_python(work)
        output = results_directory()
        print('Results directory:', output, flush=True)
        deadline = time.monotonic() + 60 * MINUTES
        batch_tokens = 8192
        while time.monotonic() < deadline:
            out = output / f'{backend}-{batch_tokens}'
            result = subprocess.run([
                python, '-u', 'run.py', TASK, '--device', 'cuda', '--backend', backend,
                '--minutes', str((deadline-time.monotonic())/60),
                '--batch-tokens', str(batch_tokens), '--examples', '2000', '--out', str(out)
            ], cwd=work)
            report_file = out / 'report.json'
            report = json.loads(report_file.read_text()) if report_file.is_file() else {}
            if result.returncode == 0:
                profiles = report.get('profiles', [])
                print(json.dumps(dict(status=report['status'], first_profile=profiles[:1],
                                      gates=report.get('gates', {}), report=str(report_file)), indent=2))
                return
            error = report.get('error', '')
            if not report.get('models') and batch_tokens > 4096 and 'out of memory' in error.lower():
                batch_tokens = 4096
                print('Retrying at 4,096 batch tokens within the original deadline.', flush=True)
                continue
            if backend == 'fla' and not report.get('profiles') and any(
                name in error for name in ('CompilationError', 'OutOfResources', 'ImportError', 'ModuleNotFoundError')
            ):
                backend = 'torch'
                print('Fused kernel could not initialize. Retrying with CUDA PyTorch.', flush=True)
                continue
            raise RuntimeError(f'Runner exited with code {result.returncode}. Results: {out}')
        print('Time budget exhausted. Results:', output)


# Portable source snapshot (ZIP data, standard Base64).
SOURCE_SHA256 = 'd55400f574eaa1ecfa3f32425569c57de47381eb220ac812fed4bcd3d18c0853'
SOURCE = '''
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
q8NVAu6y70AqG30Pmuy7j7OJTuD/+vj5D2rwNRCl6ez15k+/b369tQVFvssOXo6YkwR2SH//u+4jHsysxsKTvWcO7tHd7ZMk
Om8TkH7ba1RqVwaE2S3RemtoTqmLebgX41hKIw/PNEKJoD8CQctQYxcfPj7HKNaILbOTt5MTA2Bnr/Fhxi+xcCda99hnJ+wN
J88zGsQRIsvBhO4RTn4bg/zJG9qEtuo1EQYZlBb3wa0xBaY7I1OmT905anPD6QGnfkcfqREDa+E3Zg60NC1m1VckBrYyDYNM
SKXxLBz3ATIteLYlw2gUx96rtrVsuBf/AOaAaPxGYDP9QNgbbLqP+nnYDtEFrblrdS5S+1ffr/1rN5pi90K/moN0W0BUbybn
byemZXqjqp1SGQ9Xu5zftGJUjoRXZ4C27f06TsQ6qDfeNszqjUo9vfWGbY32v2qFGbRCC8RIi+Oap29ppl4TNtuVVOWdiskr
269u9qVXweYbrpetwEdtLUK+0XwZjaaWhjM+8c+njaiN7A920HUY4Vc/9t6LQLwAii0gBWRiFA38rtUZS+HM8uSfe69HTYt9
9MLvX7Tql2i01WiX/IBU5unYTYE5cLc24mT5sMB61PituJkuTpvKs2NF8DKnPH4iDFY5zKbU4Hp2kmdwT1Q/Cdlo2S8g9jZg
sVI9vdTvvnSBOMyv9YFtRXZRSFOymeP7A2CdELM/c/JuD/ZkP7jDB031+hVCZzoG55Q2iCeaqRM3kmnvBt7l5WK4Y4PFO0VD
I6in2eXoHzJ2m4NlqromjoQH8b0plBqH4TQziEtOWtM5Z66zRlep+qKvp8Xj4+NPRY56oxR0r22xrJBpsszg1UwooPgjBZxx
Q1eGFAroklEPNu/lMSinhIO9FvRqS3PbQRriW0OQ8a4YqI6QVf7z3079Ahma6T+aEG86IlhF8hyj3NJnGdR15cHKblCHucIR
QMpQQNk9qVt2fcPA6f0kfdwDbxv9/YbPI1qsh32gAQO0UqPOfeBg1iu9+OpWx9pbfu8yL1C3eHrEvyZEO/AYi3gar9GHvkqu
W6TTba52pWNrFv72V3xMc8nNUTiO4Xlwq3NQbj2zmX3NbDIu/h1RurWqf7Dc6fZndb2oxyDjp3Q22EfXS3p6uM23unTydYdl
mPP1BqdupdpK6YpH6YqAFv/puIJqo1XkwKtPe3L1j/4HUEsDBBQAAAAIAAAAIQBQlu0hGQcAAK0QAAAHAAAAZGF0YS5weZVX
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
'''

if __name__ == '__main__':
    run_milestone()
