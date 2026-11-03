"""
Attention mask of a block (blocks.py). The same mask is used at every layer.

    query token of ...   may attend to
    a context student    every context token (any student, any position)
    a target student     every context token + the tokens of its own student up to itself

Nobody attends to another target's tokens, and context never attends to a target.
So whatever a token reads was computed only from context tokens or from the same
target's earlier tokens: a target's out_t can reach only later tokens of that same
target, at any depth. The prediction for event t is read at ev_t (before out_t).

Padding queries attend only to themselves (a row with nothing allowed would give NaN);
padding is never attended to by a real token.

Two forms of the same rule:
  - dense_mask(batch)            (B, L, L) bool, for tests, printing and small runs
  - flex_block_mask(batch)       FlexAttention BlockMask, for training
"""

import numpy as np
import torch

from learner_model.training.blocks import KIND_NAMES, PAD


def allowed(q, kv, student_q, student_kv, target_q, target_kv, valid_q, valid_kv):
    """The rule, elementwise on tensors (broadcasting works). q, kv: token positions."""
    read = ~target_kv | ((student_q == student_kv) & (kv <= q))
    return (valid_q & valid_kv & read) | (~valid_q & (q == kv))


def mask_inputs(batch, device=None):
    """The three per-token tensors the mask needs, from collate() output."""
    student = torch.as_tensor(np.asarray(batch["student"]), dtype=torch.int32, device=device)
    target = torch.as_tensor(np.asarray(batch["target"]), dtype=torch.bool, device=device)
    valid = torch.as_tensor(np.asarray(batch["kind"]) != PAD, dtype=torch.bool, device=device)
    return student, target, valid


def dense_mask(batch, device=None):
    """(B, L, L) bool, True = query (row) may attend to key (column)."""
    student, target, valid = mask_inputs(batch, device)
    L = student.shape[1]
    q = torch.arange(L, device=student.device)[:, None]
    kv = torch.arange(L, device=student.device)[None, :]
    return allowed(q, kv, student[:, :, None], student[:, None, :], target[:, :, None],
                   target[:, None, :], valid[:, :, None], valid[:, None, :])


def flex_block_mask(batch, device="cuda"):
    """BlockMask for torch.nn.attention.flex_attention (same rule as dense_mask)."""
    from torch.nn.attention.flex_attention import create_block_mask

    student, target, valid = mask_inputs(batch, device)
    B, L = student.shape

    def mask_mod(b, h, q, kv):
        return allowed(q, kv, student[b, q], student[b, kv], target[b, q], target[b, kv],
                       valid[b, q], valid[b, kv])

    return create_block_mask(mask_mod, B, None, L, L, device=device)


def reachable(mask):
    """Transitive closure of one (L, L) mask: [q, kv] True if information at kv can reach q
    through any number of layers (with the residual connection, every token keeps its own past)."""
    m = mask.clone() | torch.eye(mask.shape[0], dtype=torch.bool, device=mask.device)
    while True:
        nxt = (m.float() @ m.float()) > 0
        if torch.equal(nxt, m):
            return m
        m = nxt


def render_mask(batch, i=0, max_tokens=80):
    """Text picture of the mask of block i: one row per query token, '#' = may attend."""
    kind, target = batch["kind"][i], batch["target"][i]
    m = dense_mask({k: batch[k][i:i + 1] for k in ("student", "target", "kind")})[0]
    n = min(int((kind != PAD).sum()), max_tokens)
    name = [KIND_NAMES[k].upper() if t else KIND_NAMES[k] for k, t in zip(kind[:n], target[:n])]
    lines = ["   " + "".join(name)]
    for q in range(n):
        lines.append(f"{name[q]:>2} " + "".join("#" if m[q, k] else "." for k in range(n)))
    return "\n".join(lines)
