"""Read an embedding dataset made by an encoder stage. Every model uses this."""

import polars as pl

from learner_model.common.datasets import open_dataset


def load_exercise_embeddings(dataset_id, root="datasets"):
    """Return (exercise_ids, matrix).

    exercise_ids: list of str, length N
    matrix:       numpy float32 array, shape (N, dim); row i belongs to exercise_ids[i]

    dataset_id is the folder name, e.g. "exercise_embed_v1__2026-10-19__a3f9c1".
    Log it in W&B with every run, so you know which embeddings a result used.
    """
    path = open_dataset(dataset_id, root)
    df = pl.read_parquet(path / "embeddings.parquet")
    return df["exercise_id"].to_list(), df["embedding"].to_numpy()

def load_event_embeddings(dataset_id, root="datasets"):
    """Return a polars DataFrame: event_id, <keep columns: user_id, created_at, ...>,
    event_embedding, result_embedding.

    One row per event, sorted by user_id, created_at. The embeddings are Array columns;
    df["event_embedding"].to_numpy() gives a float32 (N, dim) matrix.
    event_embedding describes the event, result_embedding what the student answered:
    use result_embedding only for past events.
    dataset_id is the folder name, e.g. "event_embed_v1__2026-10-29__a3f9c1".
    """
    path = open_dataset(dataset_id, root)
    return pl.read_parquet(path / "embeddings.parquet")
