"""Eight-layer causal LM. Training builds binary-tree levels in parallel."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
    """Solve the gated delta recurrence in small blocks, with FP32 accumulation."""
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
            out, state = chunk_gated_delta_rule(q.contiguous(), k.contiguous(), v.contiguous(),
                g.contiguous(), beta.to(v.dtype).contiguous(), scale=1., initial_state=state,
                output_final_state=streaming)
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
        if self.window and not streaming:
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
            y = F.scaled_dot_product_attention(split(q), adjacent(k), adjacent(v), attn_mask=mask)
            y = y.transpose(1, 2).reshape(b, self.h, -1, self.d)[:, :, :length]
        elif not self.window and cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
    def __init__(self, s, slots):
        super().__init__()
        self.chunk = s.chunk
        self.project = linear(s.width, s.slot_dim)
        self.slots = nn.Parameter(torch.randn(slots, s.slot_dim) * 0.2)
        self.merge_proj = linear(s.slot_dim, s.slot_dim)
        nn.init.eye_(self.merge_proj.weight)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def write(self, source):
        z = self.project(source)
        weights = (self.slots.to(z.dtype) @ z.transpose(-1, -2)) / math.sqrt(z.size(-1))
        return weights.softmax(-1) @ z

    def merge(self, left, right):
        return self.merge_proj((left + right) * .5)

    def read(self, x, nodes, mask=None):
        q = self.project(x)
        a = F.scaled_dot_product_attention(q, nodes, nodes, attn_mask=mask)
        return self.gain * F.linear(a, self.project.weight.T)

    def forward(self, x, source, mode):
        if mode == 'off' or x.size(1) <= self.chunk:
            return x
        if mode == 'shuffle':
            if len(x) < 2:
                raise ValueError("Shuffled memory requires independent donor sequences")
            source = source.roll(1, 0)
        b, length, d = x.shape
        c = self.chunk
        padded = F.pad(x, (0, 0, 0, (-length) % c)).reshape(b, -1, c, d)
        chunks = padded.size(1)
        level = self.write(source[:, :(chunks-1)*c].reshape(b, chunks-1, c, d))
        completed = torch.arange(chunks, device=x.device)
        nodes, masks = [], []
        span = 1
        while True:
            index = completed // span
            nodes.append(level[:, (index-1).clamp(0, level.size(1)-1)])
            masks.append((index % 2 == 1)[:, None].expand(-1, level.size(2)))
            pairs = level.size(1) // 2
            if not pairs:
                break
            level = self.merge(level[:, :2*pairs:2], level[:, 1:2*pairs:2])
            span *= 2
        # Chunk zero has no history. B,C act as batch and head axes in SDPA.
        memory = torch.cat(nodes, 2)[:, 1:]
        mask = torch.cat(masks, 1)[1:, None, :]
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
    def __init__(self, kind='C1', slots=8, s=Shape(), backend='torch'):
        super().__init__()
        if kind not in ('A0', 'B1', 'C1'):
            raise ValueError(kind)
        self.s, self.kind = s, kind
        self.embed = nn.Embedding(s.vocab, s.width)
        nn.init.normal_(self.embed.weight, std=.02)
        self.norm = nn.RMSNorm(s.width, eps=1e-5)
        mixers = [Attention(s, kind != 'A0') if kind == 'A0' or i in (3, 7)
                  else Delta(s, backend) for i in range(8)]
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
            x = x.index_select(1, positions)
        return F.linear(self.norm(x), self.embed.weight)

    def stream_chunk(self, tokens, state=None):
        """Process one complete aligned chunk, or a terminal partial chunk."""
        caches, forest, offset = state or ([None]*8, {}, 0)
        if offset % self.s.chunk or not 0 < tokens.size(1) <= self.s.chunk:
            raise ValueError("stream_chunk expects aligned chunks")
        x = self.embed(tokens)
        new_caches = []
        for i, layer in enumerate(self.blocks):
            if i == 7 and self.tree is not None and forest:
                x = x + self.tree.read(x, torch.cat([forest[j] for j in sorted(forest)], 1))
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
