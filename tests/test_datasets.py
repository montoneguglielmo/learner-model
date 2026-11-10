"""Tests of the dataset configs, the text builders and the exercise join, on small fake
MIAAM-like tables (V2: the same exercise_id used by two sources).

    uv run --with pytest pytest tests
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from learner_model.common.dataset_config import EventColumns, ExerciseColumns, load_dataset_config
from learner_model.preprocessing.event_text import EventTextConfig, build_event_texts
from learner_model.preprocessing.exercise_text import TextConfig, build_texts
from learner_model.training.blocks import EventStore

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [ROOT / "configs/encoders/miaam_v1.yaml", ROOT / "configs/encoders/miaam_v2.yaml"]


def content(q):
    return json.dumps({"instruction": "Calcule.", "question": q, "correct_answer": "4",
                       "exercise_type": "INPUT", "feedback": {}, "images": []})


@pytest.fixture
def exercises():
    # "e1" exists in two sources with different contents; "e2" is listed twice (two activities)
    return pl.DataFrame({
        "exercise_id": ["e1", "e1", "e2", "e2"],
        "source": ["am", "adaptiv_world", "am", "am"],
        "content": [content("2+2"), content("3+1"), content("1+3"), content("1+3")],
        "gameplay_type": ["INPUT", "INPUT", "QCM", "QCM"],
        "module_name": ["Nombres", "Nombres", "Nombres", "Nombres"],
        "activity_name": ["A", "B", "C", "D"],
    })


@pytest.fixture
def events():
    t0 = datetime(2025, 1, 1)
    rows = [("u1", "e1", "am", 0, True), ("u1", "e2", "am", 30, False), ("u2", "e1", "adaptiv_world", 5, True),
            ("u1", "e1", "am", 10, False), ("u2", "e9", "adaptiv_world", 50, True)]
    return pl.DataFrame({
        "student": [r[0] for r in rows], "exercise_id": [r[1] for r in rows], "source": [r[2] for r in rows],
        "time": [t0 + timedelta(seconds=r[3]) for r in rows], "correct": [r[4] for r in rows],
        "work_mode": ["zpdes"] * len(rows), "data_answer": ["4"] * len(rows),
    })


# ------------------------------------------------------------------ configs

@pytest.mark.parametrize("path", CONFIGS)
def test_config_loads(path):
    ds = load_dataset_config(path)
    assert ds.exercise_columns == ExerciseColumns(exercise_id=("source", "exercise_id"))
    assert ds.event_columns == EventColumns(exercise_id=("source", "exercise_id"))  # MIAAM names
    assert "columns" not in ds.exercise and "columns" not in ds.event
    assert ds.events_path.name == "maths_data_filtered.parquet"
    TextConfig(**ds.exercise["text"])
    EventTextConfig(**ds.event["text"])


def test_configs_use_the_same_models():
    """Vectors of different datasets must live in the same space."""
    v1, v2 = (load_dataset_config(p) for p in CONFIGS)
    for part in ("exercise", "event"):
        assert getattr(v1, part)["model"] == getattr(v2, part)["model"], part


def test_v2_event_text_has_no_module():
    ds = load_dataset_config(CONFIGS[1])
    assert "module_name" not in ds.event["text"]["embed_event_columns"]


def test_config_exercise_id_lengths_must_match(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("dataset: bad\ndata_dir: .\nevents_file: e.parquet\nexercises_file: x.parquet\n"
                   "exercise:\n  columns:\n    exercise_id: [source, exercise_id]\n"
                   "event:\n  columns:\n    exercise_id: exercise_id\n")
    with pytest.raises(ValueError, match="same length"):
        load_dataset_config(cfg)


def test_config_exercise_id_single_column(tmp_path):
    cfg = tmp_path / "one.yaml"
    cfg.write_text("dataset: one\ndata_dir: .\nevents_file: e.parquet\nexercises_file: x.parquet\n"
                   "exercise:\n  columns:\n    exercise_id: item\n"
                   "event:\n  columns:\n    exercise_id: item_id\n    student_id: student\n")
    ds = load_dataset_config(cfg)
    assert ds.exercise_columns.exercise_id == ("item",)
    assert ds.event_columns.exercise_id == ("item_id",) and ds.event_columns.student_id == "student"


# ------------------------------------------------------------------ exercise texts

def test_exercise_ids_with_source(exercises):
    texts = build_texts(exercises, TextConfig(text_columns=["module_name", "activity_name", "content"]),
                        ["source", "exercise_id"])
    assert texts.height == 3
    assert texts.columns[:3] == ["source", "exercise_id", "gameplay_type"]
    e1 = texts.filter(pl.col("exercise_id") == "e1").sort("source")
    assert "3+1" in e1["text"][0] and "2+2" in e1["text"][1]   # adaptiv_world, am
    e2 = texts.filter(pl.col("exercise_id") == "e2")["text"][0]
    assert "C ; D" in e2                                         # both activities kept


def test_exercise_id_alone_fails_when_not_unique(exercises):
    with pytest.raises(ValueError, match="different contents"):
        build_texts(exercises, TextConfig(text_columns=["content"]), ["exercise_id"])


# ------------------------------------------------------------------ event texts

ROLES = EventColumns(student_id="student", timestamp="time", outcome="correct",
                     exercise_id=["source", "exercise_id"])


def test_event_texts_roles(events):
    cfg = EventTextConfig(keep_columns=["work_mode"], embed_event_columns=["work_mode", "source"],
                          embed_result_columns=["correct", "data_answer"])
    t = build_event_texts(events, cfg, ROLES)
    assert t.columns == ["event_id", "student", "time", "correct", "source", "exercise_id", "work_mode",
                         "event_text", "result_text"]
    # sorted by student, then time; event_id = row number in the input
    assert t["student"].to_list() == ["u1", "u1", "u1", "u2", "u2"]
    assert t["event_id"].to_list() == [0, 3, 1, 2, 4]
    assert t["event_text"][0] == "work_mode: zpdes\nsource: am"
    assert t["result_text"][0] == "correct: true\ndata_answer: 4"


def test_event_id_column(events):
    ev = events.with_columns(pl.Series("my_id", [10, 11, 12, 13, 14]))
    cfg = EventTextConfig(embed_event_columns=["work_mode"], embed_result_columns=["correct"])
    roles = EventColumns(student_id="student", timestamp="time", outcome="correct", event_id="my_id",
                         exercise_id=["source", "exercise_id"])
    t = build_event_texts(ev, cfg, roles)
    assert t["event_id"].to_list() == [10, 13, 11, 12, 14]


def test_event_texts_missing_role_column(events):
    cfg = EventTextConfig(embed_event_columns=["work_mode"], embed_result_columns=[])
    with pytest.raises(ValueError, match="not in the table"):
        build_event_texts(events, cfg, EventColumns(exercise_id=["source", "exercise_id"]))  # no user_id


# ------------------------------------------------------------------ exercise join in training

def test_store_joins_on_all_id_columns(events, exercises):
    cfg = EventTextConfig(embed_event_columns=["work_mode"], embed_result_columns=["correct"])
    t = build_event_texts(events, cfg, ROLES)
    t = t.rename({"student": "user_id", "time": "created_at", "correct": "data_correct"})
    rng = np.random.default_rng(0)
    for col in ("event_embedding", "result_embedding"):
        t = t.with_columns(pl.Series(col, rng.normal(size=(t.height, 4)).astype(np.float32),
                                     dtype=pl.Array(pl.Float32, 4)))
    ex_ids = [("am", "e1"), ("adaptiv_world", "e1"), ("am", "e2")]
    store = EventStore(t, ex_ids, np.eye(3, 5), ["source", "exercise_id"])
    rows = dict(zip(zip(t["source"], t["exercise_id"]), store.ex_row.tolist()))
    assert rows[("am", "e1")] == 0 and rows[("adaptiv_world", "e1")] == 1 and rows[("am", "e2")] == 2
    assert rows[("adaptiv_world", "e9")] == -1 and store.n_unknown_exercises == 1


# ------------------------------------------------------------------ split

def test_split_per_dataset(tmp_path, monkeypatch):
    from learner_model.preprocessing.stages import split

    n = 200
    pl.DataFrame({"user_id": [f"u{i}" for i in range(n) for _ in range(3)],
                  "data_correct": [True, False, True] * n,
                  "source": ["am"] * (3 * n)}).write_parquet(tmp_path / "events.parquet")
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(f"dataset: toy\ndata_dir: {tmp_path}\nevents_file: events.parquet\n"
                   f"exercises_file: x.parquet\n")
    monkeypatch.setattr(sys, "argv", ["split", "--config", str(cfg), "--root", str(tmp_path / "ds"),
                                      "--allow-dirty"])
    split.main()
    (folder,) = (tmp_path / "ds").iterdir()
    assert folder.name.startswith("split_students_toy__")
    sets = json.loads((folder / "split.json").read_text())
    assert [len(sets[s]) for s in ("train", "val", "test")] == [140, 20, 40]
    assert len(set().union(*map(set, sets.values()))) == n
