"""Read an embedding dataset made by an encoder stage. Every model uses this."""

import json

import polars as pl

from learner_model.common.dataset_config import EventColumns
from learner_model.common.datasets import MANIFEST, open_dataset


def _manifest(path):
    return json.loads((path / MANIFEST).read_text())


def exercise_id_columns(dataset_id, root="datasets"):
    """The columns that identify an exercise in this embedding dataset
    (exercise.columns.exercise_id of its dataset config).
    Datasets made before the config had columns (exercise_embed_v1, text_embed_v1): ["exercise_id"]."""
    m = _manifest(open_dataset(dataset_id, root))
    return list((m.get("columns") or {}).get("exercise_id", ["exercise_id"]))


def event_columns(dataset_id, root="datasets"):
    """EventColumns of an event embedding dataset: what the student, timestamp, outcome and
    exercise id columns are called (event.columns of its dataset config).
    Datasets made before the config had columns (event_embed_v1): the MIAAM names."""
    return EventColumns(**(_manifest(open_dataset(dataset_id, root)).get("columns") or {}))


def load_exercise_embeddings(dataset_id, root="datasets"):
    """Return (exercise_ids, matrix).

    exercise_ids: list, length N. One value per exercise when the dataset has one id column
                  (e.g. "exercise_id"), a tuple of values when it has several
                  (e.g. ("am", "1234") for exercise_id [source, exercise_id]).
    matrix:       numpy float32 array, shape (N, dim); row i belongs to exercise_ids[i]

    dataset_id is the folder name, e.g. "exercise_embed_miaam_v2__2026-11-12__a3f9c1".
    Log it in W&B with every run, so you know which embeddings a result used.
    """
    path = open_dataset(dataset_id, root)
    ids = exercise_id_columns(dataset_id, root)
    df = pl.read_parquet(path / "embeddings.parquet")
    keys = df[ids[0]].to_list() if len(ids) == 1 else list(df.select(ids).iter_rows())
    return keys, df["embedding"].to_numpy()


def load_event_embeddings(dataset_id, root="datasets"):
    """Return a polars DataFrame: event_id, <key and keep columns>,
    event_embedding, result_embedding.

    One row per event, sorted by student, timestamp. The embeddings are Array columns;
    df["event_embedding"].to_numpy() gives a float32 (N, dim) matrix.
    event_embedding describes the event, result_embedding what the student answered:
    use result_embedding only for past events (absent if the encoder had no result columns).
    event_columns() says what the key columns are called.
    dataset_id is the folder name, e.g. "event_embed_miaam_v2__2026-11-12__a3f9c1".
    """
    path = open_dataset(dataset_id, root)
    return pl.read_parquet(path / "embeddings.parquet")
