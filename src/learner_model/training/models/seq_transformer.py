"""
Sequence transformer over blocks of students (blocks.py, block_mask.py).

Input: one batch from blocks.collate(), turned into tensors by to_tensors().
Output: one logit per token; the ones that matter are on the EV tokens of targets
(label >= 0), the predicted probability that the student answers correctly.

Token vectors (width d):
  SEP   a learned vector
  EX    Linear(exercise embedding)   (optionally only its first ex_dim values, L2-normalized again)
        + a learned vector when the exercise has no embedding
  EV    Linear(event embedding) + Linear([log_dt, is_first])  (time since the previous event)
  OUT   Linear([result embedding ; data_correct])   (data_correct only if correct_vector)
  + kind embedding (SEP / EX / EV / OUT) + position embedding (event index in the student's window)

Body: pre-norm transformer layers. Attention uses the block mask (block_mask.allowed) and
adds a learned per-layer, per-head bias to the score when query and key belong to the
same student. Two implementations of the same attention:
  "flex"   FlexAttention (training on GPU; memory grows linearly with the length)
  "dense"  a full (B, heads, L, L) score matrix (tests, CPU, short blocks)
Head: a 2-layer MLP on every token -> 1 logit.
"""

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from learner_model.training.block_mask import allowed
from learner_model.training.blocks import EV, EX, OUT, PAD, SEP


@dataclass
class ModelConfig:
    d: int = 256
    n_layers: int = 6
    n_heads: int = 8
    ff_mult: int = 4
    dropout: float = 0.1
    max_events: int = 64          # size of the position embedding
    ex_dim: int | None = None     # keep only the first ex_dim values of the exercise vector
    correct_vector: bool = True   # data_correct as an extra input of the outcome projection

    def to_dict(self):
        return asdict(self)


def to_tensors(batch, device):
    """collate() output (numpy) -> dict of tensors on device."""
    out = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            out[k] = t.to(device, non_blocking=True)
    return out


class Layer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d
        self.n_heads = cfg.n_heads
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, cfg.ff_mult * d), nn.GELU(), nn.Linear(cfg.ff_mult * d, d))
        self.drop = nn.Dropout(cfg.dropout)
        self.same_student_bias = nn.Parameter(torch.zeros(cfg.n_heads))

    def forward(self, x, attend):
        B, L, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(B, L, 3, self.n_heads, d // self.n_heads).unbind(2)
        o = attend(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), self.same_student_bias)
        x = x + self.drop(self.proj(o.transpose(1, 2).reshape(B, L, d)))
        return x + self.drop(self.ff(self.norm2(x)))


class SeqTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, ex_dim, ev_dim, res_dim, attention="flex"):
        super().__init__()
        self.cfg, self.attention = cfg, attention
        d = cfg.d
        self.ex_dim = min(cfg.ex_dim or ex_dim, ex_dim)
        self.proj_ex = nn.Linear(self.ex_dim, d)
        self.unknown_ex = nn.Parameter(torch.zeros(d))
        self.proj_ev = nn.Linear(ev_dim, d)
        self.proj_time = nn.Linear(2, d)
        self.proj_out = nn.Linear(res_dim + int(cfg.correct_vector), d)
        self.sep = nn.Parameter(torch.randn(d) * 0.02)
        self.kind_emb = nn.Embedding(4, d)
        self.pos_emb = nn.Embedding(cfg.max_events, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(Layer(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    # ---------------------------------------------------------------- inputs

    def embed(self, b):
        """Token vectors (B, L, d) from the batch tensors."""
        dtype = self.proj_ev.weight.dtype
        ex = b["ex"][..., :self.ex_dim].to(dtype)
        if self.ex_dim < b["ex"].shape[-1]:
            ex = F.normalize(ex, dim=-1)   # Matryoshka: re-normalize the truncated vector
        p_ex = self.proj_ex(ex) + (~b["ex_known"])[..., None] * self.unknown_ex
        log_dt = b["log_dt"].to(dtype)
        time = torch.stack([log_dt.clamp(min=0), (log_dt < 0).to(dtype)], -1)
        p_ev = self.proj_ev(b["ev"].to(dtype)) + self.proj_time(time)
        res = b["res"].to(dtype)
        if self.cfg.correct_vector:
            res = torch.cat([res, b["correct"].to(dtype)[..., None]], -1)
        p_out = self.proj_out(res)

        kind, event = b["kind"].long(), b["event"].long().clamp(min=0)
        idx = event[..., None].expand(-1, -1, p_ex.shape[-1])
        x = torch.zeros(*kind.shape, p_ex.shape[-1], dtype=dtype, device=kind.device)
        for k, p in ((EX, p_ex), (EV, p_ev), (OUT, p_out)):
            x = torch.where((kind == k)[..., None], p.gather(1, idx), x)
        x = torch.where((kind == SEP)[..., None], self.sep.to(dtype), x)
        x = x + self.kind_emb(kind.clamp(min=0)) + self.pos_emb(b["pos"].long())
        return x * (kind != PAD)[..., None]

    # ---------------------------------------------------------------- attention

    def _attend_fn(self, b):
        student, target = b["student"].int(), b["target"]
        valid = b["kind"] != PAD
        if self.attention == "dense":
            L = student.shape[1]
            pos = torch.arange(L, device=student.device)
            mask = allowed(pos[:, None], pos[None, :], student[:, :, None], student[:, None, :],
                           target[:, :, None], target[:, None, :], valid[:, :, None], valid[:, None, :])
            same = (student[:, :, None] == student[:, None, :])[:, None]   # (B, 1, L, L)

            def attend(q, k, v, bias):
                s = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                s = s + same * bias.to(s.dtype)[None, :, None, None]
                s = s.masked_fill(~mask[:, None], float("-inf"))
                return torch.softmax(s, -1) @ v
            return attend

        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        def mask_mod(bi, h, q, kv):
            return allowed(q, kv, student[bi, q], student[bi, kv], target[bi, q], target[bi, kv],
                           valid[bi, q], valid[bi, kv])

        B, L = student.shape
        # _compile: on GPU the mask is built without materializing a (B, L, L) tensor
        block_mask = create_block_mask(mask_mod, B, None, L, L, device=student.device,
                                       _compile=student.is_cuda)
        flex = _compiled_flex() if student.is_cuda else flex_attention

        def attend(q, k, v, bias):
            def score_mod(score, bi, h, q_idx, kv_idx):
                return score + torch.where(student[bi, q_idx] == student[bi, kv_idx], bias[h], 0.0)
            return flex(q, k, v, score_mod=score_mod, block_mask=block_mask)
        return attend

    # ---------------------------------------------------------------- forward

    def forward(self, b):
        """b: to_tensors(collate(...)). Returns logits (B, L)."""
        x = self.drop(self.embed(b))
        attend = self._attend_fn(b)
        for layer in self.layers:
            x = layer(x, attend)
        return self.head(self.norm(x)).squeeze(-1)


def loss_fn(logits, b):
    """Mean binary cross-entropy over the labelled tokens (EV tokens of targets)."""
    has = b["label"] >= 0
    return F.binary_cross_entropy_with_logits(logits[has].float(), b["label"][has].float())


_FLEX = None


def _compiled_flex():
    """FlexAttention needs torch.compile to run as a fused (fast, memory-light) kernel."""
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX
