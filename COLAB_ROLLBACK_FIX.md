# Colab Notebook Rollback — Permanent Fix

## The Problem

Colab's **Drive autosave** copies the notebook to `My Drive/Colab Notebooks/` and, on
"Save checkpoint" / runtime disconnect / tab close, it may **overwrite** that copy.
Because the workflow is "open notebook from GitHub → run all → Colab autosaves over
its Drive copy", the GitHub version and the Colab copy can drift, and Colab can
**restore the older Drive copy** over your newer edits. Net effect: *your cell edits
keep reverting to a previous unwanted version.*

There are **three distinct mechanisms** causing this, and each needs its own fix.

| # | Mechanism | Symptom | Fix |
|---|---|---|---|
| **1** | Drive autosave overwrite | Colab restores old Drive copy on reopen | Stop editing the Colab copy — treat GitHub as source of truth |
| **2** | Stale Drive copy shadows GitHub URL | Opening "from GitHub" loads old cached Drive file | Rename/clear the stale Drive copy once |
| **3** | `runtime.disconnect` / idle-timeout | Kernel dies, notebook state lost | Save checkpoints to Drive outside the notebook dir |

---

## The Permanent Workflow

### Rule 1 — GitHub is the single source of truth

**Never edit cells in the Colab UI.** Edit `.py` files locally → `python sync_notebook.py`
→ commit → push. Colab only ever *runs* the notebook.

This eliminates mechanism #1 permanently: if Colab overwrites its Drive copy with stale
content, it doesn't matter — the next `Runtime → Run all` after re-opening from GitHub
gets the correct version.

### Rule 2 — Open from GitHub every time, never from Drive

In Colab: **File → Open notebook → GitHub**, or paste the raw URL:

```
https://github.com/leonalav/hgrt/blob/main/run.ipynb?raw=1
```

Do **not** click the notebook in `My Drive/Colab Notebooks/`. That copy is Colab's
autosave scratch space and will contain stale cells.

> Tip: Bookmark the raw URL. Colab's "Open from GitHub" form wants `owner/repo` +
> `path`, not a raw URL, so use `File → Open notebook → GitHub` → type `leonalav/hgrt`
> → path `run.ipynb`.

### Rule 3 — Clear the stale Drive copy once (mechanism #2)

Run this **once** in a Colab cell to stop the old Drive copy from shadowing your work:

```python
from google.colab import drive
drive.mount('/content/drive')
import shutil, os, glob

stale = glob.glob('/content/drive/MyDrive/Colab Notebooks/*.ipynb')
for p in stale:
    backup = p + '.colab-backup'
    if not os.path.exists(backup):
        shutil.copy2(p, backup)
    print('archived', p, '->', backup)
```

This renames the offending copy to `*.ipynb.colab-backup` (which Colab ignores), so it
can never be restored over your GitHub version again. The originals are preserved as
`.colab-backup` in case you need them.

### Rule 4 — Keep checkpoints out of the notebook directory

Already handled: `run_dir` is created under `MyDrive/HGRT-10M/runs/wiki-<timestamp>/`,
**not** in `Colab Notebooks/`. Verify with:

```python
print(run_dir)
# expected: /content/drive/MyDrive/HGRT-10M/runs/wiki-YYYYMMDD-HHMMSS-xxxxx
```

If `run_dir` points into `Colab Notebooks/`, set `OUTPUT_ROOT` explicitly in the
Run-settings cell:

```python
OUTPUT_ROOT = '/content/drive/MyDrive/HGRT-10M/runs'
```

---

## Already Built Into the Notebook (defence in depth)

These are already committed, so even if Colab reverts to an older Drive copy the
notebook self-heals:

| Guard | Cell | What it does |
|---|---|---|
| **Settings bootstrap** | cell 4 | Re-executes the "Run settings" cell if `DEVICE`/`CONTEXT` are missing → no more `NameError` when cells run out of order |
| **`CONTEXT` default** | cell 9 | `CONTEXT = globals().get('CONTEXT', 512)` → corpus prep never crashes on a missing setting |
| **Phase 0 at the end** | cell 33 | Uses `run_dir / 'best_lm.pt'` (defined in cell 4) instead of a hard-coded path |
| **Embedded modules** | cells 6/27/28 | `model.py`, `dialogue.py`, `fact_switch.py` are embedded verbatim → zero external files needed on Colab |
| **`__file__` guard** | `fact_switch.py` | `if __name__ == '__main__' and '__file__' in globals()` → argparse never eats Jupyter kernel flags |

---

## Preventing Future Reverts (the "for good" part)

### A. Add a notebook self-check cell

Paste this **as the first cell** (or run it any time) to detect drift:

```python
# ── Drift check ────────────────────────────────────────────────────────────
import hashlib, json, urllib.request
GITHUB_RAW = 'https://github.com/leonalav/hgrt/raw/main/run.ipynb'
try:
    remote = urllib.request.urlopen(GITHUB_RAW, timeout=30).read()
    local = open('run.ipynb','rb').read() if __import__('os').path.exists('run.ipynb') else b''
    print('remote sha256:', hashlib.sha256(remote).hexdigest()[:16])
    if local:
        print('local  sha256:', hashlib.sha256(local).hexdigest()[:16])
        print('MATCH' if local == remote else 'DIFFERS — you may be on a stale copy')
except Exception as e:
    print('drift check skipped:', e)
```

### B. Make the notebook refuse to run a stale copy

Add to the **top of the Run-settings cell (cell 2)** a hard gate that compares the
in-memory notebook against GitHub and warns loudly. (Implemented in
`COLAB_DRIFT_GUARD` below — see "Optional hardening".)

### C. Disable Colab's Drive autosave noise

`Edit → Settings → Notebook → "Save the notebook as you type"` → turn **off** if you
prefer Colab never to touch Drive with your notebook. Checkpoints still save fine
because the notebook writes them explicitly via `torch.save` to `run_dir`.

### D. Keep a local mirror

```bash
git clone https://github.com/leonalav/hgrt.git
cd hgrt10m
python sync_notebook.py     # after any .py edit
git commit -am "..." && git push
```

Never edit in Colab. This is the whole fix.

---

## Troubleshooting Checklist

| Symptom | Cause | Fix |
|---|---|---|
| Cell edits disappear after closing tab | Colab Drive autosave restored old copy | Rule 3 (clear stale copy once) + Rule 1 (GitHub is truth) |
| `NameError: CONTEXT is not defined` | Cell 9 ran without cell 2 | Fixed by bootstrap in cell 4; if it recurs, `Runtime → Run all` |
| `FileNotFoundError: runs/best_lm.pt` | Phase 0 ran before training, or stale notebook copy | Phase 0 now at the end + uses `run_dir`; re-open from GitHub |
| `argparse` error about `-f kernel-xxx.json` | Missing `__file__` guard | Already fixed in `fact_switch.py`; re-sync notebook |
| Checkpoint missing after disconnect | Saved only in ephemeral `/content` | Set `SAVE_TO_DRIVE = True` (default) and verify `run_dir` is under `MyDrive` |
| Changes to `model.py` not reflected | Notebook embeds a stale copy | `python sync_notebook.py` then commit+push |

---

## One-Time Recovery Checklist (do these in order)

1. **Local:** `git pull` → confirm you're on `b25c3e1` or later.
2. **Colab cell:** run the Rule 3 archive script (clears the shadowing Drive copy).
3. **Colab:** `File → Open notebook → GitHub` → `leonalav/hgrt` → `run.ipynb`.
   *(Do not open the Drive copy.)*
4. **Colab:** `Runtime → Run all`.
5. **Verify:** the last cell prints `PHASE 0: GREEDY vs SAMPLED DECODE` and a verdict.
6. **Never edit in the Colab UI again** — edit `.py` locally, sync, commit, push.

---

## Optional Hardening (I can implement on request)

1. **Drift guard in cell 2** — refuse to proceed if the running notebook differs from
   GitHub `main`, with a one-line "re-open from GitHub" instruction.
2. **`sync_notebook.py` CI check** — a GitHub Action that runs the 43 unit tests and
   fails the build if `run.ipynb` is out of sync with the `.py` files.
3. **Colab badge cell** — a first cell that prints the commit SHA of the running
   notebook, so you always know which version you're on.
