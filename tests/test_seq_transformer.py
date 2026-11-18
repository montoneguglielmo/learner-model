"""Tests of the sequence transformer and its training script on synthetic data (CPU).

    uv run --with pytest pytest tests
"""

import numpy as np
import pytest
import torch
from test_blocks import synthetic_store

from learner_model.training.blocks import EV, OUT, BlockConfig, BlockSampler, build_block, collate
from learner_model.training.models.seq_transformer import ModelConfig, SeqTransformer, loss_fn, to_tensors
from learner_model.training.seq_transformer import TRAIN_DEFAULTS, apply_overrides, evaluate, load_config, train

BLOCK = BlockConfig(n_targets=2, n_context=3, max_events=8)
MODEL = ModelConfig(d=32, n_layers=3, n_heads=2, dropout=0.0, max_events=8)  # head size 16: FlexAttention minimum


@pytest.fixture(scope="module")
def store():
    return synthetic_store(n_students=40)[0]


def a_batch(store, n_blocks=2):
    sampler = BlockSampler(store, BLOCK, store.users[:30])
    it = sampler.epoch(0)
    return collate([build_block(store, next(it)) for _ in range(n_blocks)], pad_to=16)


def make_model(store, attention, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = ModelConfig(**{**MODEL.to_dict(), **kw})
    m = SeqTransformer(cfg, store.ex_dim, store.ev_dim, store.res_dim, attention).eval()
    for layer in m.layers:   # non-zero biases, so the test also covers them
        torch.nn.init.normal_(layer.same_student_bias, std=1.0)
    return m


@pytest.mark.parametrize("attention", ["dense", "flex"])
def test_own_outcome_never_reaches_its_prediction(store, attention):
    """Change the result (embedding + data_correct) of event t of a target: the prediction
    for t must not move, the prediction for t+1 must, and other students must not move."""
    batch = a_batch(store)
    model = make_model(store, attention)
    with torch.no_grad():
        base = model(to_tensors(batch, "cpu"))

    kind, target, student, event = (batch[k][0] for k in ("kind", "target", "student", "event"))
    ev_tokens = np.flatnonzero(target & (kind == EV))
    q = ev_tokens[2]                       # third event of the first target
    e = event[q]
    changed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in batch.items()}
    changed["correct"][0, e] = 1 - changed["correct"][0, e]
    changed["res"][0, e] = -changed["res"][0, e]
    with torch.no_grad():
        new = model(to_tensors(changed, "cpu"))

    assert torch.allclose(base[0, q], new[0, q], atol=1e-5)               # its own prediction
    assert not torch.allclose(base[0, q + 3], new[0, q + 3], atol=1e-5)   # the next one sees it
    others = np.flatnonzero((student != student[q]) & (kind >= 0))
    assert torch.allclose(base[0, others], new[0, others], atol=1e-5)    # nobody else
    assert torch.allclose(base[1], new[1], atol=1e-5)                    # other blocks
    assert kind[q + 1] == OUT and event[q + 1] == e


def test_flex_matches_dense(store):
    batch = to_tensors(a_batch(store), "cpu")
    with torch.no_grad():
        dense = make_model(store, "dense")(batch)
        flex = make_model(store, "flex")(batch)
    has = batch["label"] >= 0
    assert torch.allclose(dense[has], flex[has], atol=1e-4)


def test_gradients_reach_every_parameter(store):
    model = make_model(store, "dense").train()
    batch = to_tensors(a_batch(store), "cpu")
    loss_fn(model(batch), batch).backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None or not p.grad.abs().sum() > 0]
    # unknown_ex: only if an exercise in the batch has no embedding
    assert set(no_grad) <= {"unknown_ex"}, no_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward needs a GPU")
def test_flex_gradients_match_dense_on_gpu(store):
    """Compiled FlexAttention (as in training) gives the same loss and gradients as dense,
    including the gradient of the same-student bias used inside score_mod."""
    batch = to_tensors(a_batch(store), "cuda")
    grads = {}
    for attention in ("dense", "flex"):
        model = make_model(store, attention).cuda()
        loss = loss_fn(model(batch), batch)
        loss.backward()
        grads[attention] = (loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()
                                          if p.grad is not None})
    (l_d, g_d), (l_f, g_f) = grads["dense"], grads["flex"]
    assert abs(l_d - l_f) < 1e-4 and g_d.keys() == g_f.keys()
    for n in g_d:
        assert torch.allclose(g_d[n], g_f[n], atol=1e-3, rtol=1e-2), n


def test_options(store):
    batch = to_tensors(a_batch(store), "cpu")
    for kw in ({"ex_dim": 8}, {"correct_vector": False}, {"use_event": False}, {"use_time": False},
               {"use_event": False, "use_time": False}, {"use_result": False}):
        with torch.no_grad():
            out = make_model(store, "dense", **kw)(batch)
        assert torch.isfinite(out).all()


def test_train_and_evaluate(store, tmp_path):
    users = store.users
    split = {"train": set(users[:28]), "val": set(users[28:34]), "test": set(users[34:])}
    stores, splits = {store.name: store}, {store.name: split}
    tc = {**TRAIN_DEFAULTS, "batch_size": 4, "warmup_steps": 2, "max_epochs": 2, "log_every": 1}
    model, history = train(stores, splits, BLOCK, MODEL, tc, tmp_path, device="cpu", attention="dense")
    assert len(history) == 2 and (tmp_path / "model.pt").exists()
    assert all(np.isfinite(h["train_loss"]) for h in history)

    pred, metrics = evaluate(model, stores, splits, BLOCK, "val", 4, "cpu")
    # every val student with >= min_events events: its last max_events events
    expected = sum(min(store.length[k], BLOCK.max_events) for k in store.students(split["val"])
                   if store.length[k] >= BLOCK.min_events)
    assert pred.height == expected
    assert set(pred["user_id"].unique().to_list()) <= set(split["val"])
    assert pred["p_model"].is_between(0, 1).all()
    assert metrics["model"]["n"] == pred.filter(pred["n_past"] >= 1).height
    assert "auc" in metrics["baseline"] and "1-9" in metrics["by_history"]
    assert list(metrics["by_group"]) == ["dataset/dataset"]  # no source column: one group
    assert set(pred["dataset"].unique()) == {"dataset"}


def test_switches_drop_their_input(store):
    """use_event / use_result: the model output no longer depends on that embedding."""
    batch = to_tensors(a_batch(store), "cpu")
    for kw, key in (({"use_event": False}, "ev"), ({"use_time": False}, "log_dt"),
                    ({"use_result": False}, "res")):
        model = make_model(store, "dense", **kw)
        changed = {**batch, key: torch.randn_like(batch[key].float()).to(batch[key].dtype)}
        with torch.no_grad():
            assert torch.allclose(model(batch), model(changed))
    with torch.no_grad():   # and with the switch on, it does depend on it
        model = make_model(store, "dense")
        changed = {**batch, "ev": torch.randn_like(batch["ev"].float()).to(batch["ev"].dtype)}
        assert not torch.allclose(model(batch), model(changed))


def test_overrides(tmp_path):
    cfg = {"block": {"n_context": 12, "max_events": 64}, "model": {"max_events": 64}}
    apply_overrides(cfg, ["block.n_context=24", "model.use_event=false", "train.lr=1.0e-4", "block.eval_last=null"])
    assert cfg["block"] == {"n_context": 24, "max_events": 64, "eval_last": None}
    assert cfg["model"]["use_event"] is False and cfg["train"]["lr"] == 1e-4
    with pytest.raises(ValueError, match="key=value"):
        apply_overrides(cfg, ["block.n_context"])

    path = tmp_path / "c.yaml"
    path.write_text("block: {max_events: 64}\nmodel: {max_events: 64}\n")
    _, block_cfg, model_cfg, _ = load_config(path, overrides=["block.max_events=256", "model.max_events=256"])
    assert block_cfg.max_events == model_cfg.max_events == 256
    with pytest.raises(ValueError, match="model.max_events"):
        load_config(path, overrides=["block.max_events=128"])


def test_eval_last(store):
    """eval_last: evaluation targets are labelled on their last eval_last events only, the same
    events whatever max_events is (as long as max_events >= eval_last)."""
    with pytest.raises(ValueError, match="eval_last"):
        BlockConfig(max_events=8, eval_last=16)
    users, val = store.users[:30], store.users[30:]

    def labelled_rows(cfg):
        out = set()
        for windows in BlockSampler(store, cfg, users).evaluation(val):
            b = build_block(store, windows)
            out |= set(b["rows"][b["event"][b["label"] >= 0]].tolist())
        return out

    small = labelled_rows(BlockConfig(n_targets=2, n_context=3, max_events=8, eval_last=4))
    big = labelled_rows(BlockConfig(n_targets=2, n_context=3, max_events=32, eval_last=4))
    assert small == big
    ks = store.students(val)
    ks = ks[store.length[ks] >= 2]
    assert len(small) == sum(min(int(store.length[k]), 4) for k in ks)
    # training windows are never shortened
    cfg = BlockConfig(n_targets=2, n_context=3, max_events=8, eval_last=4)
    windows = next(BlockSampler(store, cfg, users).epoch(0))
    b = build_block(store, windows)
    assert (b["label"] >= 0).sum() == sum(w.n for w in windows if w.target)
