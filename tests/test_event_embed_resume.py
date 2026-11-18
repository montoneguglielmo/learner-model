"""The event encoder saves each chunk and resumes after a crash (fake model, no GPU).

    uv run --with pytest pytest tests
"""

import json
import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from learner_model.preprocessing.encoders import event_embed


class FakeTokenizer:
    def __call__(self, docs):
        return {"input_ids": [d.split() for d in docs]}


class FakeModel:
    """Deterministic vectors from the text; can crash after a number of encode calls."""
    tokenizer = FakeTokenizer()

    def __init__(self, crash_after=None):
        self.calls, self.crash_after = 0, crash_after

    def get_embedding_dimension(self):
        return 8

    def encode(self, docs, **kw):
        self.calls += 1
        if self.crash_after is not None and self.calls > self.crash_after:
            raise RuntimeError("simulated crash")
        return np.stack([np.random.default_rng(abs(hash(d)) % 2**32).normal(size=8) + 0.1 for d in docs])


@pytest.fixture
def setup(tmp_path, monkeypatch):
    n = 50
    t0 = datetime(2025, 1, 1)
    pl.DataFrame({
        "user_id": [f"u{i % 7}" for i in range(n)], "created_at": [t0 + timedelta(minutes=i) for i in range(n)],
        "data_correct": [i % 3 > 0 for i in range(n)], "source": ["am"] * n,
        "exercise_id": [f"e{i % 5}" for i in range(n)], "work_mode": ["zpdes"] * n,
        "data_answer": [str(i) for i in range(n)],
    }).write_parquet(tmp_path / "events.parquet")
    cfg = tmp_path / "toy.yaml"
    cfg.write_text(f"""dataset: toy
data_dir: {tmp_path}
events_file: events.parquet
exercises_file: x.parquet
exercise:
  columns: {{exercise_id: [source, exercise_id]}}
event:
  columns: {{exercise_id: [source, exercise_id]}}
  model: {{name: fake, dtype: float32, max_length: 512, batch_size: 4, output_dim: 4}}
  text: {{embed_event_columns: [work_mode], embed_result_columns: [data_correct, data_answer]}}
""")
    monkeypatch.setattr(event_embed, "resolve_revision", lambda name, rev: "abc")
    monkeypatch.setattr(event_embed, "gpu_name", lambda: None)
    return tmp_path, cfg, monkeypatch


def run(cfg, root, monkeypatch, model):
    monkeypatch.setattr(event_embed, "load_model", lambda m, rev: model)
    monkeypatch.setattr(sys, "argv", ["event_embed", "--config", str(cfg), "--root", str(root),
                                      "--chunk-size", "10", "--allow-dirty"])
    event_embed.main()


def test_resume_after_crash(setup):
    tmp_path, cfg, mp = setup
    run(cfg, tmp_path / "clean", mp, FakeModel())
    with pytest.raises(RuntimeError, match="simulated crash"):
        run(cfg, tmp_path / "crash", mp, FakeModel(crash_after=5))   # 2 kinds per chunk: dies in chunk 3
    (folder,) = (tmp_path / "crash").iterdir()
    assert not (folder / "manifest.json").exists()
    assert len(list((folder / "parts").glob("part_*.parquet"))) == 2

    second = FakeModel()
    run(cfg, tmp_path / "crash", mp, second)
    assert second.calls == 3 * 2                                     # only the 3 missing chunks
    (clean,) = (tmp_path / "clean").iterdir()
    a = pl.read_parquet(clean / "embeddings.parquet")
    b = pl.read_parquet(folder / "embeddings.parquet")
    assert a.equals(b) and a.height == 50
    m = json.loads((folder / "manifest.json").read_text())
    assert m["stats"] == {**json.loads((clean / "manifest.json").read_text())["stats"],
                          "encode_seconds": m["stats"]["encode_seconds"]}
    assert not (folder / "parts").exists() and not (folder / "pending_config.json").exists()


def test_other_config_does_not_resume(setup):
    tmp_path, cfg, mp = setup
    with pytest.raises(RuntimeError):
        run(cfg, tmp_path / "ds", mp, FakeModel(crash_after=1))
    cfg.write_text(cfg.read_text().replace("output_dim: 4", "output_dim: 2"))
    with pytest.raises(RuntimeError, match="earlier run crashed"):
        run(cfg, tmp_path / "ds", mp, FakeModel())
