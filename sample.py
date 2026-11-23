"""Sample a continuation from a trained byte-language-model checkpoint."""
import argparse
from contextlib import nullcontext

import torch

from model import LM


@torch.inference_mode()
def sample(model, prompt, count, temperature, device):
    tokens = list(prompt.encode('utf8'))
    if not tokens:
        raise ValueError('Supply a nonempty prompt')
    chunk = model.s.chunk
    state = None
    # A full chunk can be committed once. Only recompute the unfinished suffix.
    start = 0
    for _ in range(count):
        while len(tokens)-start > chunk:
            ids = torch.tensor([tokens[start:start+chunk]], device=device)
            _, state = model.stream_chunk(ids,state)
            start += chunk
        ids = torch.tensor([tokens[start:]], device=device)
        logits, pending = model.stream_chunk(ids,state)
        probs = (logits[0,-1].float()/temperature).softmax(0)
        tokens.append(int(torch.multinomial(probs,1).item()))
        if ids.size(1) == chunk:
            state = pending
            start += chunk
    return bytes(tokens).decode('utf8', errors='replace')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint')
    p.add_argument('prompt')
    p.add_argument('--bytes', type=int, default=160)
    p.add_argument('--temperature', type=float, default=.8)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    a = p.parse_args()
    if a.temperature <= 0 or a.bytes < 0:
        p.error('temperature must be positive and byte count must be nonnegative')
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    model = LM(ck['kind'], ck['slots'])
    model.load_state_dict(ck['state_dict'])
    model.to(a.device).eval()
    torch.manual_seed(a.seed)
    amp = torch.autocast('cuda', torch.bfloat16) if a.device == 'cuda' else nullcontext()
    with amp:
        print(sample(model, a.prompt, a.bytes, a.temperature, a.device))
