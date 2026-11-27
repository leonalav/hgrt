"""Tests for the hierarchical tree memory in `model.py`.

The tree is a BINARY-COUNTER FRONTIER.  After `C` chunks have been written
the memory holds exactly the nodes whose spans tile `[0, C)`:

    C = 1  ->  [ leaf0 ]
    C = 2  ->  [ merge(0,1) ]
    C = 3  ->  [ merge(0,1), leaf2 ]
    C = 4  ->  [ merge(0..3) ]

The level of a node is the position of the set bit in `C`, so the readable
levels for a given `C` are exactly the set bits of `C`.

These tests lock in:

  T1  the frontier nodes of every completion count tile `[0, C)` exactly,
      with no gap and no overlap;
  T2  the visible levels are exactly the set bits of `C`;
  T3  no readable node may cover a chunk at or after the query chunk
      (no future leak);
  T4  the tree must run at any context length, not just 512;
  T5  causality: changing the bytes of chunk k must not change the outputs
      of any chunk before k;
  T6  `delta_reference` must match flash-linear-attention's reference
      recurrence, with `DELTA_QUERY_SCALE = 1.0`;
  T7  `mode='permute'` must be a genuine, deterministic, causal ablation
      (the old `shuffle` was a BATCH-AXIS roll, which is not an ablation);
  T8  `mode='off'` must return the input unchanged;
  T9  the streaming path must agree with the batch path.
"""
import unittest

import torch
import torch.nn.functional as F

from model import DELTA_QUERY_SCALE, LM, Shape, delta_reference


def small_model(**kw):
    torch.manual_seed(0)
    shape = Shape(width=64, heads=2, slot_dim=32, chunk=16)
    return LM('C1', slots=4, s=shape, backend='torch', **kw)


class FrontierTests(unittest.TestCase):
    """T1-T3: the shape of the frontier."""

    def setUp(self):
        self.model = small_model()
        self.tree = self.model.tree
        self.chunk = self.tree.chunk
        self.chunks = 8

    def _frontier(self, completed):
        """The (level, node_index) pairs of the frontier for `completed`."""
        return [(level, self.tree.frontier_index(completed, level))
                for level in range(len(self.tree.plan(self.chunks, torch.device('cpu'))[0]))
                if (completed >> level) & 1]

    def test_T1_frontier_spans_tile_the_completed_range(self):
        """For every completion count the frontier spans tile [0, C)."""
        for C in range(1, self.chunks):
            spans = []
            for level in range(8):
                if (C >> level) & 1:
                    lo, hi = self.tree.frontier_span(C, level)
                    spans.append((lo, hi))
            covered = sorted(ch for lo, hi in spans for ch in range(lo, hi))
            self.assertEqual(covered, list(range(C)),
                             f'C={C}: frontier spans {spans} do not tile [0,{C})')

    def test_T2_visible_levels_are_the_set_bits(self):
        """The readable levels are exactly the set bits of the completion count."""
        _, mask = self.tree.plan(self.chunks, torch.device('cpu'))
        nlev = mask.size(1)
        for c in range(self.chunks - 1):
            C = c + 1
            visible = [l for l in range(nlev) if bool(mask[c, l])]
            expected = [l for l in range(nlev) if (C >> l) & 1]
            self.assertEqual(visible, expected,
                             f'query chunk {c} (C={C}): {visible} != {expected}')

    def test_T3_no_readable_node_covers_a_future_chunk(self):
        """No readable node may cover a chunk at or after the query chunk."""
        _, mask = self.tree.plan(self.chunks, torch.device('cpu'))
        nlev = mask.size(1)
        for c in range(self.chunks - 1):
            C = c + 1
            for level in range(nlev):
                if not bool(mask[c, level]):
                    continue
                lo, hi = self.tree.frontier_span(C, level)
                # The query chunk is `C` (0-based): chunks 0..C-1 are behind
                # it, so no readable node may cover chunk >= C.
                for chunk in range(lo, hi):
                    self.assertLess(
                        chunk, C,
                        f'query chunk {c} reads level {level} covering chunk {chunk}')

    def test_T3_the_first_leaf_is_reachable(self):
        """Chunk 0 must reach the query that follows it."""
        _, mask = self.tree.plan(self.chunks, torch.device('cpu'))
        # Query chunk 1 has C = 2, whose frontier is [merge(0,1)].
        # Query chunk 2 has C = 3, whose frontier includes leaf 2.
        # The leaf of chunk 0 is inside the level-1 merge for C = 2.
        self.assertTrue(bool(mask[1, 1]),
                        'query chunk 1 must read the merged (0,1) node')


class TreeRuntimeTests(unittest.TestCase):
    """T4, T7, T8: the tree runs and its modes behave."""

    def setUp(self):
        self.model = small_model()
        self.model.eval()

    def test_T4_tree_runs_at_any_context_length(self):
        model = self.model
        for chunks in (2, 4, 8, 16, 32):
            ids = torch.randint(0, 256, (2, model.tree.chunk * chunks))
            with torch.no_grad():
                out = model(ids, 'on')
            self.assertEqual(tuple(out.shape), (2, model.tree.chunk * chunks, 256))

    def test_T4_tree_handles_a_context_that_is_not_a_multiple_of_the_chunk(self):
        model = self.model
        ids = torch.randint(0, 256, (2, model.tree.chunk * 5 + 7))
        with torch.no_grad():
            out = model(ids, 'on')
        self.assertEqual(tuple(out.shape), (2, model.tree.chunk * 5 + 7, 256))

    def test_T7_permute_is_a_real_ablation(self):
        ids = torch.randint(0, 256, (1, self.model.tree.chunk * 8))
        with torch.no_grad():
            on = self.model(ids, 'on')
            permuted = self.model(ids, 'permute')
        self.assertGreater((on - permuted).abs().max().item(), 1e-8,
                           "'permute' is a no-op")

    def test_T7_permute_is_deterministic(self):
        ids = torch.randint(0, 256, (1, self.model.tree.chunk * 8))
        with torch.no_grad():
            a = self.model(ids, 'permute')
            b = self.model(ids, 'permute')
        self.assertTrue(torch.equal(a, b))

    def test_T7_permute_does_not_swap_batch_rows(self):
        """`permute` must NOT be a batch-axis roll.

        The legacy `shuffle` was `source.roll(1, 0)`, which swaps BATCH
        ITEMS.  That is a batch-level control, not a memory ablation: it
        changes which batch row is read, so it is not "same content, broken
        memory".  `permute` must break the CONTENT ORDER instead, so two
        identical batch rows stay identical to each other while differing
        from the unpermuted result.
        """
        model = self.model
        ids = torch.randint(0, 256, (3, model.tree.chunk * 8))
        dup = ids[:1].repeat(3, 1)
        with torch.no_grad():
            plain = model(dup, 'on')
            permuted = model(dup, 'permute')
        # Identical rows must give identical rows under 'permute'.
        self.assertTrue(torch.allclose(permuted[0], permuted[1], atol=1e-6),
                        "'permute' is not batch-row independent")
        # And the permutation must actually change the content order.
        self.assertGreater((plain - permuted).abs().max().item(), 1e-8)

    def test_T7_shuffle_remains_the_legacy_batch_roll(self):
        """`shuffle` keeps its legacy batch-axis semantics.

        The legacy test suite pins this behaviour, and it is a different
        control from `permute`: it swaps memory between batch rows instead
        of breaking the content order within a row.
        """
        model = self.model
        ids = torch.randint(0, 256, (3, model.tree.chunk * 8))
        dup = ids[:1].repeat(3, 1)
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(dup, 'on'), model(dup, 'shuffle'),
                                           atol=1e-6))
        with self.assertRaises(ValueError):
            model(ids[:1], 'shuffle')

    def test_T8_off_returns_the_input_unchanged(self):
        tree = self.model.tree
        x = torch.randn(2, 64, self.model.embed.weight.shape[1])
        out = tree(x, x, 'off')
        self.assertTrue(torch.equal(out, x))


class CausalityTests(unittest.TestCase):
    """T5: the tree must not leak the future."""

    def setUp(self):
        self.model = small_model()
        self.model.eval()
        self.chunk = self.model.tree.chunk
        self.chunks = 8
        self.ids = torch.randint(0, 256, (1, self.chunk * self.chunks))

    def _chunk_deltas(self, perturbed_chunk, mode='on'):
        lo = perturbed_chunk * self.chunk
        hi = lo + self.chunk
        pert = self.ids.clone()
        pert[:, lo:hi] = (pert[:, lo:hi] + 37) % 256
        with torch.no_grad():
            base = self.model(self.ids, mode)
            other = self.model(pert, mode)
        return [(base[:, k * self.chunk:(k + 1) * self.chunk]
                 - other[:, k * self.chunk:(k + 1) * self.chunk]).abs().max().item()
                for k in range(self.chunks)]

    def test_T5_causality_no_output_depends_on_a_later_chunk(self):
        for k in range(self.chunks - 1):
            deltas = self._chunk_deltas(k)
            for earlier in range(k):
                self.assertEqual(
                    deltas[earlier], 0.0,
                    f'perturbing chunk {k} changed chunk {earlier}')

    def test_T5_causality_holds_for_permute(self):
        for k in range(self.chunks - 1):
            deltas = self._chunk_deltas(k, mode='permute')
            for earlier in range(k):
                self.assertEqual(deltas[earlier], 0.0,
                                 f'perturbing chunk {k} changed chunk {earlier}')


class DeltaReferenceTests(unittest.TestCase):
    """T6: `delta_reference` must implement exactly the FLA recurrence.

    The reference is flash-linear-attention's `naive_recurrent_gated_delta_rule`
    (`fla/ops/gated_delta_rule/naive.py`), the upstream ground truth:

        h = h * exp(g)
        b_v = (v - h^T k) * beta
        h = h + k (b_v)^T
        o = q^T h

    with `h` key-major `(B, H, K, V)`.
    """

    @staticmethod
    def _fla_reference(q, k, v, g, beta, scale=1.0):
        q = q.transpose(1, 2).contiguous().float() * scale
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()
        b, h, t, d_k = k.shape
        d_v = v.shape[-1]
        state = torch.zeros(b, h, d_k, d_v)
        out = torch.zeros(b, h, t, d_v)
        for i in range(t):
            state = state * g[:, :, i].exp()[..., None, None]
            b_v = v[:, :, i].clone()
            b_v = b_v - (state * k[:, :, i][..., None]).sum(-2)
            b_v = b_v * beta[:, :, i][..., None]
            state = state + k[:, :, i].unsqueeze(-1) * b_v.unsqueeze(-2)
            out[:, :, i] = torch.einsum('bhd,bhdm->bhm', q[:, :, i], state)
        return out.transpose(1, 2).contiguous(), state

    def test_T6_delta_reference_matches_the_fla_reference(self):
        torch.manual_seed(1)
        b, length, h, d = 2, 12, 3, 8
        q = torch.randn(b, length, h, d)
        k = torch.randn(b, length, h, d)
        v = torch.randn(b, length, h, d)
        g = -torch.rand(b, length, h)
        beta = torch.rand(b, length, h)

        out, state = delta_reference(q, k, v, g, beta, None)
        ref, ref_state = self._fla_reference(q, k, v, g, beta, scale=DELTA_QUERY_SCALE)

        self.assertTrue(torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
                        f'output max abs diff {(out - ref).abs().max().item():.2e}')
        self.assertTrue(torch.allclose(state, ref_state, atol=1e-4, rtol=1e-4),
                        f'state max abs diff {(state - ref_state).abs().max().item():.2e}')

    def test_T6_delta_reference_matches_with_normalized_inputs(self):
        """The model normalises q and k before the call; check that path too."""
        torch.manual_seed(1)
        b, length, h, d = 2, 12, 3, 8
        q = F.normalize(torch.randn(b, length, h, d), dim=-1)
        k = F.normalize(torch.randn(b, length, h, d), dim=-1)
        v = torch.randn(b, length, h, d)
        g = -torch.rand(b, length, h)
        beta = torch.rand(b, length, h)

        out, _ = delta_reference(q, k, v, g, beta, None)
        ref, _ = self._fla_reference(q, k, v, g, beta, scale=DELTA_QUERY_SCALE)
        self.assertTrue(torch.allclose(out, ref, atol=1e-4, rtol=1e-4),
                        f'output max abs diff {(out - ref).abs().max().item():.2e}')

    def test_T6_delta_query_scale_is_one(self):
        """FLA's default scale is 1/sqrt(d); the torch backend uses no scaling.

        If `DELTA_QUERY_SCALE` drifted to 1/sqrt(d) the two backends would
        silently disagree, so the constant is pinned here.
        """
        self.assertEqual(DELTA_QUERY_SCALE, 1.0)

    def test_T6_delta_reference_is_causal(self):
        """Changing a late key must not change an early output."""
        torch.manual_seed(2)
        b, length, h, d = 1, 16, 2, 8
        q = torch.randn(b, length, h, d)
        k = torch.randn(b, length, h, d)
        v = torch.randn(b, length, h, d)
        g = -torch.rand(b, length, h)
        beta = torch.rand(b, length, h)
        out, _ = delta_reference(q, k, v, g, beta, None)

        k2 = k.clone()
        k2[:, 10:] += 5.0
        out2, _ = delta_reference(q, k2, v, g, beta, None)
        for t in range(10):
            self.assertTrue(torch.allclose(out[:, t], out2[:, t], atol=1e-5))


class StreamingTests(unittest.TestCase):
    """T9: the streaming path must agree with the batch path."""

    def test_stream_chunk_matches_forward(self):
        """`sample.py` streams; training and evaluation do not.

        A mismatch means the model that is evaluated is not the model that
        generates.
        """
        model = small_model()
        model.eval()
        chunk = model.tree.chunk
        ids = torch.randint(0, 256, (1, chunk * 3))
        with torch.no_grad():
            batched = model(ids, 'on')
            state = None
            pieces = []
            for i in range(0, ids.size(1), chunk):
                piece, state = model.stream_chunk(ids[:, i:i + chunk], state)
                pieces.append(piece)
            streamed = torch.cat(pieces, dim=1)
        self.assertTrue(torch.allclose(batched, streamed, atol=1e-4, rtol=1e-4),
                        f'max abs diff {(batched - streamed).abs().max().item():.2e}')


if __name__ == '__main__':
    unittest.main()
