"""Deterministic byte-text splits and conflict-checked, first-query recall examples."""
import hashlib
from pathlib import Path

import numpy as np
import torch


def digest(data):
    return hashlib.sha256(data).hexdigest()


class Text:
    def __init__(self, path=None):
        if path:
            raw = Path(path).read_bytes()
            a, b = int(.8*len(raw)), int(.9*len(raw))
            splits = [raw[:a], raw[a:b], raw[b:]]
            self.source = str(path)
        else:
            from pydoc_data.topics import topics
            # Deduplicate aliases before partitioning whole help topics.
            docs = sorted({v.encode() for v in topics.values()}, key=digest)
            a, b = int(.8*len(docs)), int(.9*len(docs))
            splits = [b'\n\n'.join(group) for group in (docs[:a], docs[a:b], docs[b:])]
            self.source = "Python pydoc_data.topics"
        if min(map(len, splits)) < 1024:
            raise ValueError("Each text split needs at least 1024 bytes; supply a larger corpus")
        self.hashes = [digest(z) for z in splits]
        self.train, self.valid = [np.frombuffer(z, dtype=np.uint8) for z in splits[:2]]
        counts = np.bincount(self.train, minlength=256) + 1
        self.unigram_nll = -np.log(counts / counts.sum())

    @staticmethod
    def windows(raw, starts, length):
        piece = torch.from_numpy(raw[starts[:, None] + np.arange(length+1)])
        return piece[:, :-1], piece[:, 1:]

    def sample(self, rng, batch, length):
        return self.windows(self.train, rng.integers(len(self.train)-length, size=batch), length)

    def validation(self, length=256):
        # Fixed, nonoverlapping target positions.
        starts = np.arange(0, len(self.valid)-length, length)
        starts = starts[np.linspace(0, len(starts)-1, min(len(starts), 128), dtype=int)]
        return self.windows(self.valid, starts, length)


# length, load, min and max distance (value token to scored query key), anchor
STRATA = {
    'easy': (256, 4, 65, 176, 224),
    'short16': (256, 16, 65, 176, 224),
    'middle': (512, 8, 128, 416, 480),
    'intermediate': (1024, 8, 384, 928, 992),
    'carry15': (1024, 16, 704, 928, 992),
    'carry16': (1088, 16, 704, 928, 1056),
    'primary31': (2048, 16, 1024, 1984, 2016),
    'carry32': (2112, 16, 1024, 1984, 2080),
}
TRAIN_STAGES = ['easy', 'easy', 'middle', 'intermediate', 'primary31']
SCREEN_SEEDS = {'easy': 3100, 'short16': 3101, 'carry15': 3102, 'carry16': 3102,
                'primary31': 3103, 'carry32': 3103}
FINAL_SEED = 910000  # reserved for a subsequent study


def recall(rng, n, name):
    length, load, near, far, q = STRATA[name]
    # Facts occupy shuffled cells within a randomly shifted compact cluster.
    width = load*4
    low, high = max(0, q-far-2), q-near-width
    if not (low <= high and q+2*(load-1) < length and near > 64):
        raise ValueError(f"Infeasible placement for {name}")
    start = rng.integers(low, high+1, size=(n, 1))
    facts = start + 4*np.arange(load) + rng.integers(0, 2, (n, load))
    keys = rng.random((n, 32)).argsort(1)[:, :load] + 16
    values = rng.random((n, 32)).argsort(1)[:, :load] + 64
    order = rng.random((n, load)).argsort(1)
    pos = q + 2*np.arange(load)
    rows = np.arange(n)[:, None]
    x = rng.integers(128, 256, (n, length), dtype=np.uint8)
    x[rows, facts] = 1
    x[rows, facts+1] = keys
    x[rows, facts+2] = values
    x[:, pos-1] = 2
    x[:, pos] = keys[rows, order]
    distances = q - facts[rows, order[:, :1]] - 2
    assert np.all((distances >= near) & (distances <= far))
    assert np.all(facts[:, -1]+2 < q-64)
    return torch.from_numpy(x), torch.from_numpy(pos), torch.from_numpy(values[rows, order])


def screen_sets(n):
    sets, manifest = {}, {}
    for name, seed in SCREEN_SEEDS.items():
        data = recall(np.random.default_rng(seed), n, name)
        sets[name] = data
        h = hashlib.sha256()
        for tensor in data:
            h.update(tensor.numpy().tobytes())
        length, load, _, _, q = STRATA[name]
        chunks = q // 64
        manifest[name] = dict(seed=seed, sha256=h.hexdigest(), n=n, load=load, length=length,
                              query=q, chunks=chunks, popcount=chunks.bit_count())
    return sets, manifest
