"""Tests of multi.py (training on several datasets) on small synthetic stores.

    uv run --with pytest pytest tests
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from learner_model.training.blocks import BlockConfig, EventStore
from learner_model.training.multi import (DatasetEntry, MultiSampler, check_compatible, dataset_entries)

DIM, EX_DIM = 8, 16
CFG = BlockConfig(n_targets=2, n_context=3, max_events=8)


def make_store(name, sources, n_per_source=20, seed=0, dim=DIM):
    """Students s<i> of each source, each with 3..30 events, all from that source."""
    rng = np.random.default_rng(seed)
    rows = []
    for src in sources:
        for i in range(n_per_source):
            t0 = datetime(2025, 1, 1)
            for j in range(int(rng.integers(3, 30))):
                rows.append({"user_id": f"{src}_u{i}", "created_at": t0 + timedelta(minutes=j),
                             "source": src, "exercise_id": "e0", "data_correct": bool(rng.random() < 0.7)})
    df = pl.DataFrame(rows).with_row_index("event_id").with_columns(
        pl.Series("event_embedding", rng.normal(size=(len(rows), dim)).astype(np.float32),
                  dtype=pl.Array(pl.Float32, dim)),
        pl.Series("result_embedding", rng.normal(size=(len(rows), dim)).astype(np.float32),
                  dtype=pl.Array(pl.Float32, dim)))
    return EventStore(df, ["e0"], rng.normal(size=(1, EX_DIM)), source_column="source", name=name)


def make_split(store, seed=0):
    users = sorted(store.users)
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    return {"train": set(users[:int(.7 * n)]), "val": set(users[int(.7 * n):int(.8 * n)]),
            "test": set(users[int(.8 * n):])}


@pytest.fixture(scope="module")
def data():
    stores = {"v2": make_store("v2", ["am", "world"]), "v1": make_store("v1", ["mia"], seed=1)}
    return stores, {k: make_split(s) for k, s in stores.items()}


def test_store_sources(data):
    stores, _ = data
    assert sorted(set(stores["v2"].sources)) == ["am", "world"]
    assert all(u.startswith(s) for u, s in zip(stores["v2"].users, stores["v2"].sources))


def test_one_group_per_dataset_source(data):
    stores, splits = data
    sampler = MultiSampler(stores, splits, CFG)
    assert [g.name for g in sampler.groups] == ["v2/am", "v2/world", "v1/mia"]


def test_blocks_stay_in_one_group_and_targets_once(data):
    stores, splits = data
    sampler = MultiSampler(stores, splits, CFG)
    seen, groups_in_order = [], []
    for gb in sampler.epoch(0):
        g = sampler.groups[gb.group]
        srcs = {g.store.sources[w.student] for w in gb.windows}
        assert srcs == {g.source}                                   # one source per block
        assert all(g.store.users[w.student] in splits[g.dataset]["train"] for w in gb.windows)
        seen += [(g.name, w.student) for w in gb.windows if w.target]
        groups_in_order.append(gb.group)
    assert len(seen) == len(set(seen)) == sampler.n_targets        # every target exactly once
    assert groups_in_order != sorted(groups_in_order)               # groups are mixed


def test_evaluation_per_group(data):
    stores, splits = data
    sampler = MultiSampler(stores, splits, CFG)
    for gb in sampler.evaluation(splits, "val"):
        g = sampler.groups[gb.group]
        for w in gb.windows:
            u = g.store.users[w.student]
            assert g.store.sources[w.student] == g.source
            assert u in (splits[g.dataset]["val"] if w.target else splits[g.dataset]["train"])


def test_train_fraction(data):
    stores, splits = data
    full = MultiSampler(stores, splits, CFG)
    half = MultiSampler(stores, splits, CFG, {"v1": 0.5})
    n = {g.name: len(g.sampler.train) for g in full.groups}
    m = {g.name: len(g.sampler.train) for g in half.groups}
    assert m["v1/mia"] == round(n["v1/mia"] / 2) and m["v2/am"] == n["v2/am"]


def test_same_source_in_two_datasets_fails():
    stores = {"v2": make_store("v2", ["am"]), "v1": make_store("v1", ["am", "mia"])}
    with pytest.raises(ValueError, match="Source 'am' is selected from both"):
        check_compatible(stores)


def test_vector_sizes_must_match():
    stores = {"a": make_store("a", ["x"]), "b": make_store("b", ["y"], dim=4)}
    with pytest.raises(ValueError, match="vector sizes"):
        check_compatible(stores)


def test_too_few_students_names_the_group(data):
    stores, splits = data
    with pytest.raises(ValueError, match="v2/am: .*a block needs 20"):
        MultiSampler(stores, splits, BlockConfig(n_targets=10, n_context=10))


def test_student_with_two_sources_fails():
    rows = pl.DataFrame({"user_id": ["u", "u"], "created_at": [datetime(2025, 1, 1), datetime(2025, 1, 2)],
                         "source": ["a", "b"], "exercise_id": ["e0", "e0"], "data_correct": [True, False]})
    rows = rows.with_row_index("event_id").with_columns(
        pl.Series("event_embedding", np.zeros((2, DIM), np.float32), dtype=pl.Array(pl.Float32, DIM)),
        pl.Series("result_embedding", np.zeros((2, DIM), np.float32), dtype=pl.Array(pl.Float32, DIM)))
    with pytest.raises(ValueError, match="several sources"):
        EventStore(rows, ["e0"], np.zeros((1, EX_DIM)), source_column="source")


def test_config_entries():
    old = dataset_entries({"events_id": "e", "exercises_id": "x", "split_id": "s"})
    assert old == [DatasetEntry("e", "x", "s")]
    new = dataset_entries({"datasets": [{"events_id": "e", "exercises_id": "x", "split_id": "s",
                                         "sources": ["mia"], "train_fraction": 0.5}]})
    assert new[0].sources == ["mia"] and new[0].train_fraction == 0.5
    with pytest.raises(ValueError, match="either"):
        dataset_entries({"datasets": [], "events_id": "e"})
    with pytest.raises(ValueError, match="train_fraction"):
        DatasetEntry("e", "x", "s", train_fraction=0)
