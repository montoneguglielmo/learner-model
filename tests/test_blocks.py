"""Tests of blocks.py and block_mask.py on synthetic data.

    uv run --with pytest pytest tests
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
import torch

from learner_model.training.block_mask import dense_mask, reachable, render_mask
from learner_model.training.blocks import (EV, EX, OUT, PAD, SEP, BlockConfig, BlockSampler, EventStore,
                                           batches, build_block, collate, render)

N_EX, EX_DIM, DIM = 30, 16, 8


def synthetic_store(n_students=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_students):
        n = int(rng.integers(1, 150))
        t0 = datetime(2025, 1, 1) + timedelta(days=int(rng.integers(0, 100)))
        times = sorted(t0 + timedelta(seconds=int(s)) for s in rng.integers(0, 10**6, n))
        for t in times:
            rows.append({"user_id": f"u{u:03d}", "created_at": t,
                         "exercise_id": f"e{int(rng.integers(0, N_EX + 3))}",  # a few have no embedding
                         "data_correct": bool(rng.random() < 0.6)})
    df = pl.DataFrame(rows).sample(fraction=1.0, shuffle=True, seed=seed)  # unsorted on purpose
    df = df.with_row_index("event_id").with_columns(
        pl.Series("event_embedding", rng.normal(size=(len(rows), DIM)).astype(np.float32),
                  dtype=pl.Array(pl.Float32, DIM)),
        pl.Series("result_embedding", rng.normal(size=(len(rows), DIM)).astype(np.float32),
                  dtype=pl.Array(pl.Float32, DIM)),
    )
    ex_ids = [f"e{i}" for i in range(N_EX)]
    return EventStore(df, ex_ids, rng.normal(size=(N_EX, EX_DIM))), df


@pytest.fixture(scope="module")
def store():
    return synthetic_store()[0]


@pytest.fixture(scope="module")
def sampler(store):
    return BlockSampler(store, BlockConfig(n_targets=3, n_context=5, max_events=16), store.users[:30])


def test_store_order_and_fields(store):
    df = synthetic_store()[1].sort(["user_id", "created_at", "event_id"])
    assert store.n_events == df.height
    assert store.users == sorted(df["user_id"].unique().to_list())
    assert np.array_equal(store.correct, df["data_correct"].cast(pl.Int8).to_numpy())
    # log_dt = -1 exactly at each student's first event
    assert np.array_equal(np.flatnonzero(store.log_dt == -1), store.start)
    assert store.n_unknown_exercises > 0


def test_epoch_targets_once_and_context_from_train(store, sampler):
    seen = []
    for windows in sampler.epoch(0):
        tg = [w.student for w in windows if w.target]
        ctx = [w.student for w in windows if not w.target]
        assert len(ctx) == 5 and not set(tg) & set(ctx)
        assert set(ctx) <= set(sampler.train.tolist())
        for w in windows:
            assert w.n == min(16, store.length[w.student])
            assert store.start[w.student] <= w.first
            assert w.first + w.n <= store.start[w.student] + store.length[w.student]
        seen += tg
    assert sorted(seen) == sorted(sampler.targets.tolist())


def test_random_windows_change_between_epochs(sampler):
    first = lambda e: {w.student: w.first for b in sampler.epoch(e) for w in b if w.target}
    assert first(0) == first(0)
    assert first(0) != first(1)


def test_evaluation_uses_last_events_and_train_context(store, sampler):
    val = store.users[30:]
    blocks = list(sampler.evaluation(val))
    assert blocks == list(sampler.evaluation(val))  # one fixed draw
    for windows in blocks:
        for w in windows:
            if w.target:
                assert store.users[w.student] in val
                assert w.first + w.n == store.start[w.student] + store.length[w.student]
            else:
                assert store.users[w.student] not in val


def test_build_block_tokens(store, sampler):
    windows = next(sampler.epoch(0))
    b = build_block(store, windows)
    assert len(b["kind"]) == sum(1 + 3 * w.n for w in windows)
    # each student: SEP then (EX, EV, OUT) * n, positions 0..n-1
    for s, w in enumerate(windows):
        idx = np.flatnonzero(b["student"] == s)
        assert b["kind"][idx[0]] == SEP
        assert list(b["kind"][idx[1:]]) == [EX, EV, OUT] * w.n
        assert list(b["pos"][idx[1:]]) == list(np.repeat(np.arange(w.n), 3))
        assert (b["target"][idx] == w.target).all()
    # labels: only on EV tokens of targets, equal to data_correct of that event
    has = b["label"] >= 0
    assert (b["kind"][has] == EV).all() and b["target"][has].all()
    assert has.sum() == sum(w.n for w in windows if w.target)
    assert np.array_equal(b["label"][has], store.correct[b["rows"][b["event"][has]]])
    # embeddings of the right events
    assert np.array_equal(b["ev"], store.ev[b["rows"]])
    assert np.array_equal(b["ex_known"], store.ex_row[b["rows"]] >= 0)


def test_collate_padding(store, sampler):
    blocks = [build_block(store, w) for w in next(batches(sampler.epoch(0), 4))]
    batch = collate(blocks, pad_to=128)
    B, L = batch["kind"].shape
    assert B == len(blocks) and L % 128 == 0
    for i, b in enumerate(blocks):
        n = len(b["kind"])
        assert (batch["kind"][i, n:] == PAD).all() and (batch["label"][i, n:] == -1).all()
        assert np.array_equal(batch["kind"][i, :n], b["kind"])


def test_mask_rules(store, sampler):
    batch = collate([build_block(store, next(sampler.epoch(0)))], pad_to=32)
    m = dense_mask(batch)[0].numpy()
    student, target, kind = batch["student"][0], batch["target"][0], batch["kind"][0]
    valid = kind != PAD
    for q in np.flatnonzero(valid):
        for k in np.flatnonzero(valid):
            if not target[k]:
                expect = True                       # context is readable by everyone
            else:
                expect = bool(target[q]) and student[q] == student[k] and k <= q
            assert m[q, k] == expect, (q, k)
    assert not m[np.ix_(valid, ~valid)].any()        # padding is never read


def test_no_leak_at_any_depth(store, sampler):
    """However many layers: a target's ev_t never receives information from its out_t
    (or anything later of itself), nor from another target."""
    batch = collate([build_block(store, next(sampler.epoch(0)))], pad_to=1)
    r = reachable(dense_mask(batch)[0]).numpy()
    student, target, kind, event = (batch[k][0] for k in ("student", "target", "kind", "event"))
    for q in np.flatnonzero(target & (kind == EV)):
        own = student == student[q]
        assert not r[q, own & (np.arange(len(kind)) > q)].any()      # own future, incl. out_t
        assert not r[q, target & ~own].any()                          # other targets
        out_t = np.flatnonzero(own & (kind == OUT) & (event == event[q]))
        assert len(out_t) == 1 and not r[q, out_t[0]]
    # and the model can use the past: ev_t sees out_{t-1}
    q = np.flatnonzero(target & (kind == EV))[1]
    assert r[q, q - 2] and kind[q - 2] == OUT


def test_render(store, sampler):
    batch = collate([build_block(store, next(sampler.epoch(0)))])
    assert render({k: batch[k][0] for k in ("kind", "target")}).startswith("Sxeo")
    assert render_mask(batch, max_tokens=20).count("\n") == 20


def test_flex_attention_matches_dense(store, sampler):
    """FlexAttention with the BlockMask gives the same output as dense attention with dense_mask."""
    import warnings

    from torch.nn.attention.flex_attention import flex_attention

    from learner_model.training.block_mask import flex_block_mask

    batch = collate([build_block(store, w) for w in next(batches(sampler.epoch(0), 2))])
    q = torch.randn(2, 1, batch["kind"].shape[1], 16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # "called without torch.compile": fine for a test
        flex = flex_attention(q, q, q, block_mask=flex_block_mask(batch, device="cpu"))
    dense = torch.nn.functional.scaled_dot_product_attention(q, q, q, attn_mask=dense_mask(batch)[:, None])
    assert torch.allclose(flex, dense, atol=1e-5)
