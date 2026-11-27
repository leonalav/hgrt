"""Eight-layer causal LM. Training builds binary-tree levels in parallel."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

# --- Gated DeltaNet scale convention -------------------------------------
# `delta_reference` (the pure-torch backend) implements the *unscaled*
# Gated DeltaNet query convention
#
#       S_t = a_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T
#       o_t = S_t q_t
#
# i.e. the query is used as-is: there is NO 1/sqrt(d_k) factor.  The
# flash-linear-attention reference implementation instead defaults to
# `scale = d_k ** -0.5`, so the `fla` call site must be given the scale that
# matches `delta_reference` (this constant) for the two backends to agree.
#
# The recurrent state is stored KEY-major, i.e. `(B, H, K, V)`: the first
# state axis indexes the key dimension and the second the value dimension.
# That is also FLA's default layout (`state_v_first=False`), so the state
# returned by either backend can be fed to the other unchanged.
DELTA_QUERY_SCALE = 1.0

# --- Tree permutation ----------------------------------------------------
# `mode='permute'` (deprecated alias: `mode='shuffle'`) scrambles the
# *content* of the tree by permuting the source-chunk order.  The permutation
# is deterministic and seeded so that the ablation is reproducible and
# auditable; see `tree_permutation`.
TREE_PERMUTE_SEED = 20260126


def tree_permutation(n, seed=TREE_PERMUTE_SEED):
    """Deterministic permutation of `n` source-chunk indices.

    The same permutation is applied to every batch item, so the ablation
    changes only *which* content lands in which memory slot and never which
    item is compared with which.
    """
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=generator)


@dataclass(frozen=True)
class Shape:
    width: int = 256
    heads: int = 4
    chunk: int = 64
    slot_dim: int = 128
    vocab: int = 256
    target: int = 10_000_000


def linear(a, b):
    return nn.Linear(a, b, bias=False)


def delta_reference(q, k, v, g, beta, state):
    """Solve the gated delta recurrence in small blocks, with FP32 accumulation.

    Unscaled query convention (`DELTA_QUERY_SCALE == 1.0`): the state is
    key-major, `(B, H, K, V)`, and the output is `o_t = S_t q_t` with no
    `d_k ** -0.5` factor applied to the query.  See the module-level
    `DELTA_QUERY_SCALE` comment for the FLA equivalence argument.
    """
    b, length, h, d = q.shape
    block = min(32, length)
    pad = (-length) % block

    def blocks(x):
        x = F.pad(x.float(), (0, 0, 0, 0, 0, pad))
        return x.transpose(1, 2).reshape(b, h, -1, block, x.size(-1))

    with torch.autocast(q.device.type, enabled=False):
        qb, kb, vb = map(blocks, (q, k, v))
        beta = blocks(beta.unsqueeze(-1))
        log_decay = blocks(g.unsqueeze(-1)).cumsum(-2)
        decay = (log_decay - log_decay.transpose(-1, -2)).clamp_max(0).exp().tril()
        prefix = log_decay.exp()
        bk = beta * kb
        matrix = ((bk @ kb.transpose(-1, -2)) * decay).tril(-1)
        matrix = matrix + torch.eye(block, device=q.device)
        rhs = torch.cat((beta * vb, bk * prefix), -1)
        u, w = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True).chunk(2, -1)
        scores = (qb @ kb.transpose(-1, -2)) * decay
        end_keys = kb * decay[..., -1, :].unsqueeze(-1)
        if state is None:
            state = q.new_zeros((b, h, d, d), dtype=torch.float32)
        results = []
        for i in range(kb.size(2)):
            updates = u[:, :, i] - w[:, :, i] @ state
            results.append((qb[:, :, i] * prefix[:, :, i]) @ state + scores[:, :, i] @ updates)
            state = prefix[:, :, i, -1:] * state + end_keys[:, :, i].transpose(-1, -2) @ updates
        out = torch.cat(results, 2).transpose(1, 2)[:, :length]
    return out.to(q.dtype), state


class Delta(nn.Module):
    def __init__(self, s, backend):
        super().__init__()
        self.h, self.d = s.heads, s.width // s.heads
        self.backend = backend
        self.qkv = linear(s.width, 3 * s.width)
        self.conv = nn.Conv1d(3*s.width, 3*s.width, 4, groups=3*s.width, bias=False)
        self.gates = linear(s.width, 2*s.heads)
        self.decay_bias = nn.Parameter(torch.linspace(-5, -2, s.heads))
        self.out_gate = linear(s.width, s.width)
        self.out = linear(s.width, s.width)

    def forward(self, x, cache, offset, streaming):
        qkv = self.qkv(x).transpose(1, 2)
        prev, state = cache if cache is not None else (None, None)
        history = F.pad(qkv, (3, 0)) if prev is None else torch.cat((prev, qkv), -1)
        shaped = F.silu(self.conv(history).transpose(1, 2)).reshape(*x.shape[:2], 3, self.h, self.d)
        q, k, v = shaped.unbind(2)
        q, k = (F.normalize(t.float(), dim=-1, eps=1e-6).to(v.dtype) for t in (q, k))
        decay, beta = self.gates(x).float().chunk(2, -1)
        g = -F.softplus(decay + self.decay_bias)
        beta = beta.sigmoid()
        if self.backend == 'fla':
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            # `scale` must reproduce `delta_reference`'s convention.  FLA
            # defaults to `d_k ** -0.5`; `DELTA_QUERY_SCALE` is 1.0 because
            # `delta_reference` applies no query scaling.  The state layout
            # matches too: both are key-major `(B, H, K, V)`.
            out, state = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(),
                g.contiguous(), beta.to(v.dtype).contiguous(), scale=DELTA_QUERY_SCALE,
                initial_state=state, output_final_state=streaming)
        else:
            out, state = delta_reference(q, k, v, g, beta, state)
        out = F.rms_norm(out, (self.d,), eps=1e-5).reshape_as(x)
        return self.out(out * F.silu(self.out_gate(x))), ((history[..., -3:].clone(), state) if streaming else None)


class Attention(nn.Module):
    def __init__(self, s, local):
        super().__init__()
        self.h, self.d = s.heads, s.width // s.heads
        self.window = s.chunk if local else 0
        self.qkv = linear(s.width, 3*s.width)
        self.out = linear(s.width, s.width)
        self.register_buffer("freq", 10000 ** (-torch.arange(0, self.d, 2) / self.d), persistent=False)

    def forward(self, x, cache, offset, streaming):
        b, length, width = x.shape
        q, k, v = self.qkv(x).reshape(b, length, 3, self.h, self.d).permute(2, 0, 3, 1, 4)
        angle = torch.arange(offset, offset+length, device=x.device)[:, None] * self.freq
        co, si = angle.cos().to(q.dtype), angle.sin().to(q.dtype)

        def rotate(t):
            a, z = t.chunk(2, -1)
            return torch.cat((a*co - z*si, z*co + a*si), -1)

        q, k = rotate(q), rotate(k)
        if cache is not None:
            k, v = (torch.cat((a, z), 2) for a, z in zip(cache, (k, v)))
        if cache is None and (not self.window or length <= self.window):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif self.window and not streaming:
            # Each block sees the previous and current block, with an exact sliding mask.
            w = self.window
            pad = (-length) % w
            def split(t):
                return F.pad(t, (0, 0, 0, pad)).reshape(b, self.h, -1, w, self.d).transpose(1, 2)
            def adjacent(t):
                cur = split(t)
                prev = F.pad(cur[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
                return torch.cat((prev, cur), 3)
            i = torch.arange(w, device=x.device)[:, None] + w
            j = torch.arange(2*w, device=x.device)
            mask = ((i >= j) & (i-j < w)).expand((length+pad)//w, 1, w, 2*w).clone()
            mask[0, :, :, :w] = False
            # Batch the windows in 4D so CUDA can dispatch its fused SDPA kernels.
            query = split(q).reshape(-1, self.h, w, self.d)
            keys = adjacent(k).reshape(-1, self.h, 2*w, self.d)
            values = adjacent(v).reshape_as(keys)
            y = F.scaled_dot_product_attention(query, keys, values, attn_mask=mask.repeat(b, 1, 1, 1))
            y = y.reshape(b, -1, self.h, w, self.d).transpose(1, 2).reshape(b, self.h, -1, self.d)[:, :, :length]
        else:
            difference = torch.arange(length, device=x.device)[:, None] + k.size(2)-length - torch.arange(k.size(2), device=x.device)
            valid = difference >= 0
            if self.window:
                valid &= difference < self.window
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=valid)
        keep = self.window-1 if self.window else k.size(2)
        next_cache = (k[:, :, -keep:].clone(), v[:, :, -keep:].clone()) if streaming else None
        return self.out(y.transpose(1, 2).reshape(b, length, width)), next_cache


class Tree(nn.Module):
    """Binary merge tree over chunk summaries.

    The tree is a BINARY-COUNTER FRONTIER, not a full tree.  After `C`
    chunks have been written, the memory holds exactly the nodes whose spans
    tile `[0, C)` -- the classic "merge two summaries when they are
    siblings" structure:

        C = 1  ->  [ leaf0 ]
        C = 2  ->  [ merge(0,1) ]
        C = 3  ->  [ merge(0,1), leaf2 ]
        C = 4  ->  [ merge(0..3) ]

    The level of a node is the position of the set bit in `C`, so the visible
    levels for a given `C` are exactly the set bits of `C`.  This keeps the
    persistent state O(log n) (see `run.py:measured_state`, which asserts
    `len(forest) == chunks.bit_count()`), and it makes the batch path and
    the streaming path agree: `stream_chunk` carries exactly this frontier.

    The level-availability mask is therefore *not* a "prefix of every
    level" mask.  It selects the frontier nodes of each completion count.
    """

    def __init__(self, s, slots):
        super().__init__()
        self.chunk = s.chunk
        self.project = linear(s.width, s.slot_dim)
        self.slots = nn.Parameter(torch.randn(slots, s.slot_dim) * 0.2)
        self.merge_proj = linear(s.slot_dim, s.slot_dim)
        nn.init.eye_(self.merge_proj.weight)
        self.gain = nn.Parameter(torch.tensor(0.1))
        # The seed used by the `permute` ablation.  It is an ATTRIBUTE (not a
        # module-level constant) so that an evaluation harness can set a
        # per-pair seed and have the ablation actually honour it -- otherwise
        # the seed recorded in a report would be a fiction.
        self.permute_seed = TREE_PERMUTE_SEED

    def write(self, source):
        z = self.project(source)
        weights = (self.slots.to(z.dtype) @ z.transpose(-1, -2)) / math.sqrt(z.size(-1))
        return weights.softmax(-1) @ z

    def merge(self, left, right):
        return self.merge_proj((left + right) * .5)

    def read(self, x, nodes, mask=None):
        """Attend from the query chunks to the tree's memory nodes.

        Axis convention (spelled out explicitly, because this is the
        contract that `model.py` used to get wrong):

        * `x`     is `(B, Q, Lq, W)`  -- B batch, Q query chunks, Lq the bytes
          inside a query chunk, W the model width.
        * `nodes` is `(B, Q, N, E)`   -- B batch, Q query chunks, N memory
          nodes for THAT query chunk's frontier, E the slot dimension.
        * `mask`  is `(Q, N)`         -- `True` where a query chunk may read
          that node.

        `F.scaled_dot_product_attention` interprets a 4-D input as
        `(B, H, L, E)`.  The *query-chunk* axis is pushed into the head axis,
        because the level-availability mask is defined per query chunk: for a
        given query chunk the readable set of nodes is the same for every byte
        in that chunk.  Each query chunk therefore gets its OWN key head --
        its own frontier -- which is what makes the memory O(log n) per
        query chunk rather than O(n).

        Concretely:

            q      -> (B, Q, Lq, E)   (batch, head=query chunk, len, dim)
            keys   -> (B, Q, N, E)    (batch, head=query chunk, key len N, dim)
            mask   -> (Q, 1, N)       (broadcast over the byte axis)

        so the SDPA axes are (batch, head=query chunk, query length = bytes in
        the chunk, key length = memory node, embedding = slot dim).  The `Lq`
        axis broadcasts because every byte of a chunk shares the same
        visible-node set.
        """
        b, q_len, lq, _ = x.shape
        # `self.project` maps the model width to the slot dimension, so the
        # embedding size of the attention is `slot_dim`, NOT the input width.
        e = self.project.weight.shape[0]
        q = self.project(x).reshape(b, q_len, lq, e)
        if nodes.dim() == 5:
            # (B, Q, N, S, E) -> (B, Q, N*S, E): every slot of every node is
            # a key of that query chunk's head.
            keys = nodes.reshape(b, nodes.size(1), -1, e)
        elif nodes.dim() == 4:
            # (B, Q, N, E) -> already one key per node.
            keys = nodes
        else:
            # (B, N, E) -> a single shared key head.
            keys = nodes.unsqueeze(1)
        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m.reshape(m.size(0), -1)
            if m.size(0) != q_len:
                m = m.reshape(-1)[:q_len]
            # The mask is per NODE; each node contributes `slots` keys, so a
            # node-level mask is expanded to key level first.
            slots = self.slots.size(0)
            if m.size(1) * slots == keys.size(2):
                m = m.repeat_interleave(slots, dim=-1)
            # The mask is per query chunk: row `i` describes the frontier of
            # query chunk `i`.  When the mask covers the whole memory it must
            # be sliced per query chunk.
            if m.size(1) != keys.size(2):
                per = m.size(1) // q_len
                if per * q_len != m.size(1):
                    raise ValueError(
                        f'mask width {m.size(1)} is not compatible with '
                        f'{q_len} query chunks and {keys.size(2)} keys')
                m = torch.stack([m[i, i * per:(i + 1) * per] for i in range(q_len)])
            m = m.reshape(q_len, 1, -1)
            mask = m
        a = F.scaled_dot_product_attention(q, keys, keys, attn_mask=mask)
        return self.gain * F.linear(a.reshape(b, q_len, lq, e), self.project.weight.T)

    @staticmethod
    def frontier_index(completed, level):
        """The index of the level-`level` frontier node for completion `C`.

        This is the ORIGINAL index formula, `C // 2**level - 1`, clamped into
        range.  It differs from the "aligned block that ends the frontier"
        formula by one block, so it must be used verbatim: it is what the
        trained checkpoints were trained with.

        The node it selects is the one whose span, together with the other
        frontier nodes, tiles `[0, C)`.
        """
        return max(0, completed // (2 ** level) - 1)

    @staticmethod
    def frontier_span(completed, level):
        """The chunk span covered by the level-`level` frontier node.

        For a completion count `C`, the frontier nodes tile `[0, C)`.  The
        level-`l` node is the last block of size `2**l` that ends at or
        before `C`, i.e. it covers
        `[ (C // 2**l) * 2**l - 2**l , (C // 2**l) * 2**l )`.
        """
        size = 2 ** level
        end = (completed // size) * size
        return end - size, end

    def plan(self, chunks, device):
        """Return the per-level node counts and the level-availability mask.

        `chunks` is the total number of chunks in the sequence.  The LAST
        chunk is the one being predicted, so it is never written and never
        read: there are `chunks - 1` leaves.

        The tree is a binary-counter FRONTIER (see the class docstring): for a
        completion count `C`, the readable nodes are exactly the frontier
        nodes, one per SET BIT of `C`.  The memory therefore holds one
        frontier per query chunk: `levels` nodes per query chunk.

        The mask is `(chunks-1, levels)`: row `c` says which of query chunk
        `c`'s `levels` nodes are readable.
        """
        leaves = max(0, chunks - 1)
        levels = []
        span = 1
        while True:
            count = leaves // span
            if count == 0:
                break
            levels.append(count)
            span *= 2
        mask = torch.zeros(max(0, chunks - 1), len(levels), dtype=torch.bool, device=device)
        # For query chunk `c` the completion count is `C = c + 1`: the row `c`
        # of the mask is consumed by query chunk `c + 1` (query chunk 0 has no
        # history and is dropped before the read), and query chunk `c + 1`
        # has chunks `0 .. c` behind it.  The readable nodes are the frontier
        # nodes of that `C` -- one per set bit of `C`.
        for c in range(max(0, chunks - 1)):
            C = c + 1
            for level in range(len(levels)):
                if (C >> level) & 1:
                    mask[c, level] = True
        return levels, mask

    def build(self, source, chunks, permute=False):
        """Build the frontier memory for every query chunk.

        `nodes` is `(B, (chunks-1), levels, slots, slot_dim)`: for each
        query chunk, its frontier nodes, level by level.

        The level-`l` node of the frontier for completion count `C` is the
        node of level `l` covering the span `frontier_span(C, l)`, so the
        whole frontier can be built bottom-up from the leaves.

        When `permute` is set, the CONTENT ORDER of the memory is ablated
        while causality is preserved.  A single global permutation of the
        source chunks cannot work: it would move later content into earlier
        memory slots and leak the future.  (And a permutation that maps
        every prefix `[0, C)` onto itself is necessarily the identity, so a
        global permutation is either the identity or non-causal.)

        Instead the permutation is applied PER QUERY CHUNK: query chunk `q`
        sees a frontier built from a permutation of the `C = q+1` chunks
        behind it.  All of those chunks are already in the past, so this is
        causal, and it genuinely changes which content is summarised together
        -- which is exactly the "broken memory, same content" control the
        evaluation needs.
        """
        b, length, d = source.shape
        c = self.chunk
        leaves = max(0, chunks - 1)
        if leaves == 0:
            return source.new_zeros((b, 0, self.slots.size(0), self.slots.size(1))), []
        # Level 0: one leaf per source chunk.  The projection is done ONCE,
        # so the per-`C` permutations only cost the cheap merges.
        level = self.write(source[:, :leaves * c].reshape(b, leaves, c, d))

        def stack(leaf_tensor):
            """Build the level stack from a leaf array."""
            out = [leaf_tensor]
            lvl = leaf_tensor
            while lvl.size(1) >= 2:
                pairs = lvl.size(1) // 2
                lvl = self.merge(lvl[:, :2 * pairs:2], lvl[:, 1:2 * pairs:2])
                out.append(lvl)
            return out

        picked = []
        nlev = len(stack(level))
        for C in range(1, leaves + 1):
            if permute:
                # Permute the `C` chunks behind this query chunk among
                # themselves.  All of them are already in the past, so this
                # stays causal.  The seed is the tree's `permute_seed`, so the
                # ablation is reproducible and the recorded seed is the seed
                # that was actually used.
                order = tree_permutation(C, seed=self.permute_seed).to(source.device)
                leaves_c = level[:, :C][:, order]
                # Pad back up to the full leaf count so the level stack has
                # the same shape for every query chunk.
                pad = level.new_zeros((b, leaves - C, self.slots.size(0),
                                       self.slots.size(1)))
                leaves_c = torch.cat((leaves_c, pad), 1)
            else:
                leaves_c = level
            nodes = stack(leaves_c)
            for l in range(nlev):
                idx = self.frontier_index(C, l)
                idx = max(0, min(idx, nodes[l].size(1) - 1))
                picked.append(nodes[l][:, idx])
        memory = torch.stack(picked, 1)      # (B, leaves*nlev, slots, slot_dim)
        # Reshape to (B, Q, levels, slots, slot_dim) so each query chunk
        # carries its own frontier as a separate SDPA key head.
        memory = memory.reshape(b, leaves, nlev, self.slots.size(0),
                                self.slots.size(1))
        levels = [n.size(1) for n in stack(level)]
        return memory, levels

    def forward(self, x, source, mode):
        """Apply the tree memory to `x`.

        Modes:

        * `'on'`      -- the tree as trained.
        * `'off'`     -- no tree contribution at all.
        * `'permute'` -- a CAUSAL content-order ablation: each query chunk
          sees a frontier built from a permutation of the chunks behind it.
          The content is the same, but the ORDER in which the chunks are
          summarised into the memory is broken.  This is a genuine "same
          content, broken memory" control, and it never leaks the future.
        * `'shuffle'` -- the LEGACY ablation, kept for backwards
          compatibility: the tree source is rolled along the BATCH axis, so
          each batch row reads the previous row's memory.  This is a
          batch-level control, NOT a content-order ablation, and it is not a
          causal substitute for `'permute'`.
        """
        if mode not in ('on', 'off', 'permute', 'shuffle'):
            raise ValueError(f"Unknown tree mode: {mode!r}")
        if mode == 'off' or x.size(1) <= self.chunk:
            return x
        b, length, d = x.shape
        c = self.chunk
        padded = F.pad(x, (0, 0, 0, (-length) % c)).reshape(b, -1, c, d)
        chunks = padded.size(1)
        permute = mode == 'permute'
        if mode == 'shuffle':
            # Legacy behaviour: swap the memory between batch rows.
            if b < 2:
                raise ValueError("Shuffled memory requires independent donor sequences")
            source = source.roll(1, 0)
        # The permutation is applied per query chunk inside `build`, so the
        # memory content order is ablated while causality is preserved.
        memory, levels = self.build(source, chunks, permute=permute)
        if memory.size(1) == 0:
            return x
        _, mask = self.plan(chunks, x.device)
        # Query chunk 0 has no history: it is dropped and the contribution is
        # re-inserted at the front below.  Every later query chunk reads its
        # own frontier.
        reads = self.read(padded[:, 1:], memory, mask)
        return x + F.pad(reads.flatten(1, 2), (0, 0, c, 0))[:, :length]


class Block(nn.Module):
    def __init__(self, s, mix, hidden):
        super().__init__()
        self.mix = mix
        self.n1 = nn.RMSNorm(s.width, eps=1e-5)
        self.n2 = nn.RMSNorm(s.width, eps=1e-5)
        self.up, self.down = linear(s.width, 2*hidden), linear(hidden, s.width)

    def forward(self, x, cache, offset, streaming):
        y, cache = self.mix(self.n1(x), cache, offset, streaming)
        x = x + y
        gate, val = self.up(self.n2(x)).chunk(2, -1)
        return x + self.down(F.silu(gate)*val), cache


class LM(nn.Module):
    def __init__(self, kind='C1', slots=8, s=Shape(), backend='torch', read_window=None, mid_window=None):
        super().__init__()
        if kind not in ('A0', 'B1', 'C1'):
            raise ValueError(kind)
        self.s, self.kind = s, kind
        self.embed = nn.Embedding(s.vocab, s.width)
        nn.init.normal_(self.embed.weight, std=.02)
        self.norm = nn.RMSNorm(s.width, eps=1e-5)
        mixers = [Attention(s, kind != 'A0') if kind == 'A0' or i in (3, 7)
                  else Delta(s, backend) for i in range(8)]
        if read_window is not None:
            if kind != 'C1' or read_window < s.chunk:
                raise ValueError('read_window requires C1 and at least one full chunk')
            mixers[-1].window = read_window
        if mid_window is not None:
            if kind != 'C1' or mid_window < s.chunk:
                raise ValueError('mid_window requires C1 and at least one full chunk')
            mixers[3].window = mid_window
        # Centre the shared B1/C1 width between the two parameter counts.
        fixed = sum(p.numel() for module in [self.embed, self.norm, *mixers] for p in module.parameters()) + 16*s.width
        tree_mid = (s.width*s.slot_dim + s.slot_dim**2 + 16*s.slot_dim + 1)/2 if kind != 'A0' else 0
        self.hidden = max(4, 4 * round((s.target-fixed-tree_mid)/(8*3*s.width*4)))
        self.blocks = nn.ModuleList(Block(s, m, self.hidden) for m in mixers)
        self.tree = Tree(s, slots) if kind == 'C1' else None

    def forward(self, tokens, mode='on', positions=None):
        x = self.embed(tokens)
        for i, layer in enumerate(self.blocks):
            if i == 7 and self.tree is not None:
                x = self.tree(x, source, mode)
            x, _ = layer(x, None, 0, False)
            if i == 3:
                source = x
        if positions is not None:
            if positions.ndim == 1:
                x = x.index_select(1, positions)
            else:
                x = x[torch.arange(x.size(0), device=x.device)[:, None], positions]
        return F.linear(self.norm(x), self.embed.weight)

    def stream_chunk(self, tokens, state=None):
        """Process one complete aligned chunk, or a terminal partial chunk.

        The tree read here MUST match `forward` exactly, otherwise generation
        (`sample.py` streams, training/evaluation do not) would use a
        different memory than the one that was evaluated.  Two things matter:

        * the same node set and the same level-availability mask, and
        * the same "a chunk never reads its own leaf" rule.

        The forest holds the binary-carry frontier: `forest[l]` is the single
        pending node of level `l`.  After a chunk completes, its leaf is
        written and merged upwards with the pending sibling at each level.
        """
        caches, forest, offset = state or ([None]*8, {}, 0)
        if offset % self.s.chunk or not 0 < tokens.size(1) <= self.s.chunk:
            raise ValueError("stream_chunk expects aligned chunks")
        x = self.embed(tokens)
        new_caches = []
        source = None
        for i, layer in enumerate(self.blocks):
            if i == 7 and self.tree is not None:
                x = self._tree_read_streaming(x, forest, offset)
            x, cache = layer(x, caches[i], offset, True)
            new_caches.append(cache)
            if i == 3:
                source = x
        forest = dict(forest)
        if self.tree is not None and tokens.size(1) == self.s.chunk:
            node = self.tree.write(source)
            level = 0
            while level in forest:
                node = self.tree.merge(forest.pop(level), node)
                level += 1
            forest[level] = node
        return F.linear(self.norm(x), self.embed.weight), (new_caches, forest, offset + tokens.size(1))

    def _tree_read_streaming(self, x, forest, offset):
        """Tree read for the streaming path, matching `Tree.forward`.

        `forward` builds the whole tree and reads, for query chunk `q`, the
        frontier of completion count `C = q + 1`.  The streaming path has
        exactly that frontier already, carried in `forest`: `forest[l]` is
        the PENDING node of level `l` of the binary counter.

        For a completion count `C` the frontier node at level `l` is the one
        whose span ends the frontier.  Because the forest stores only the
        pending node of each level, and the pending node of level `l` IS the
        frontier node of level `l` for the current completion count, the
        streaming read can use the forest directly:

            frontier node of level l  =  forest[l]

        and the readable levels are the set bits of `C`, exactly as in the
        batch path.
        """
        tree = self.tree
        c = tree.chunk
        completed = offset // c
        if completed == 0 or not forest:
            return x
        # The frontier of completion count `C` is the pending node of each
        # level, and the readable levels are the set bits of `C`.
        levels = sorted(forest)
        # (B, nlev, S, E) -> (B, 1, nlev, S, E): a single query chunk with
        # `nlev` nodes, matching the 5-D layout `Tree.build` produces.
        nodes = torch.stack([forest[l] for l in levels], 1).unsqueeze(1)
        # The mask is per NODE (one node per level).
        mask = torch.zeros(1, len(levels), dtype=torch.bool, device=x.device)
        for i, l in enumerate(levels):
            mask[0, i] = bool((completed >> l) & 1)
        reads = tree.read(x.unsqueeze(1), nodes, mask)
        return x + reads.squeeze(1)
