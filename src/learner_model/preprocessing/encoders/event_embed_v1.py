"""
Encoder "event_embed_v1": one vector per event (one row of the interactions table),
from a short "column: value" text.

  1. Build one text per event (event_text.py): every column except user_id,
     created_at, the target columns and the drop columns (see the "text" part
     of the config).
  2. Embed each text with a sentence-embedding model (e.g. Qwen3-Embedding).
     Events are done in chunks; inside a chunk, identical texts are embedded once.
  3. Optionally keep only the first `output_dim` values (Matryoshka), then L2-normalize.
  4. Save the result as a dataset folder, following the dataset rules.

Check the texts first (no GPU):
  uv run python scripts/preview_event_texts.py --config configs/encoders/event_qwen3_0.6b.yaml

Usage (from the repo root, after committing your code):
  uv run python -m learner_model.preprocessing.encoders.event_embed_v1 \
      --config configs/encoders/event_qwen3_0.6b.yaml

Quick test on 5000 events (creates a separate dataset folder, delete it after):
  uv run python -m learner_model.preprocessing.encoders.event_embed_v1 \
      --config configs/encoders/event_qwen3_0.6b.yaml --limit 5000 --allow-dirty

Output:
  datasets/event_embed_v1__<date>__<commit6>/
    embeddings.parquet   event_id, user_id, created_at, embedding (float32, L2-normalized),
                         sorted by user_id, created_at
    texts.parquet        event_id, user_id, created_at, text: exactly the texts that were embedded
    manifest.json        written last
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
import yaml

from learner_model.common.datasets import DatasetWriter
from learner_model.preprocessing.encoders.text_embed_v1 import (gpu_name, library_versions, load_model,
                                                                resolve_revision)
from learner_model.preprocessing.event_text import EventTextConfig, build_event_texts

PIPELINE = "event_embed_v1"


def embed(model, docs, m, dim):
    """(N, dim) float32, first `dim` values, L2-normalized."""
    emb = model.encode(docs, batch_size=m["batch_size"], convert_to_numpy=True,
                       normalize_embeddings=False, show_progress_bar=False)
    emb = np.asarray(emb, dtype=np.float32)
    if dim > emb.shape[1]:
        raise ValueError(f"output_dim={dim} but the model gives {emb.shape[1]} values")
    emb = emb[:, :dim]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(emb).all():
        raise ValueError("Some embeddings are zero or not finite (NaN/inf)")
    return emb / norms


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML file, e.g. configs/encoders/event_qwen3_0.6b.yaml")
    p.add_argument("--limit", type=int, default=None, help="only the first N events (quick tests)")
    p.add_argument("--chunk-size", type=int, default=100_000, help="events per chunk (memory only)")
    p.add_argument("--root", default="datasets", help="where dataset folders are created")
    p.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (quick tests only)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    events_path = Path(cfg["events"]).resolve()
    text_cfg = EventTextConfig(**cfg.get("text", {}))

    # Texts first: a wrong column name fails here, before loading the model.
    texts = build_event_texts(pl.read_parquet(events_path), text_cfg)
    if args.limit:
        texts = texts.head(args.limit)
    print(f"{texts.height} event texts built")

    revision = resolve_revision(m["name"], m.get("revision", "main"))
    print(f"Model {m['name']} at commit {revision}")

    # Everything that changes the RESULT goes in the config (with the exact model commit).
    config = {
        "config_file": Path(args.config).name,
        "events": str(events_path),
        "model": {**m, "revision": revision},
        "text": text_cfg.to_dict(),
        "limit": args.limit,
    }
    out = DatasetWriter(PIPELINE, config, root=args.root, allow_dirty=args.allow_dirty)
    if out.reuse:
        return

    model = load_model(m, revision)
    prompt = m.get("prompt") or ""
    dim = m.get("output_dim") or model.get_sentence_embedding_dimension()

    # Chunks, so that all the embeddings never need to be in memory at once.
    writer = None
    n_tokens, n_unique = [], 0
    t0 = time.time()
    for start in range(0, texts.height, args.chunk_size):
        chunk = texts.slice(start, args.chunk_size)
        uniq = chunk.select(pl.col("text").unique(maintain_order=True)).with_row_index("k")
        docs = [prompt + t for t in uniq["text"].to_list()]
        n_tokens.extend(len(ids) for ids in model.tokenizer(docs)["input_ids"])
        n_unique += len(docs)

        emb = embed(model, docs, m, dim)
        k = chunk.join(uniq, on="text", how="left", maintain_order="left")["k"].to_numpy()
        table = chunk.select("event_id", "user_id", "created_at").with_columns(
            embedding=pl.Series(emb[k], dtype=pl.Array(pl.Float32, dim))).to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(out.path / "embeddings.parquet", table.schema)
        writer.write_table(table)

        done = start + chunk.height
        rate = done / (time.time() - t0)
        print(f"\r  {done}/{texts.height} events | ETA {(texts.height - done) / rate / 60:.1f} min   ",
              end="", flush=True)
    print()
    if writer is not None:
        writer.close()
    seconds = round(time.time() - t0, 1)
    texts.write_parquet(out.path / "texts.parquet")

    n_tokens = np.array(n_tokens)
    n_truncated = int((n_tokens > m["max_length"]).sum())
    if n_truncated:
        print(f"WARNING: {n_truncated} texts were longer than max_length={m['max_length']} tokens and were cut")

    full_dim = model.get_sentence_embedding_dimension()
    out.finish(
        input_files=[events_path],
        extra={
            "llm": {
                "name": m["name"], "revision": revision, "dtype": m["dtype"],
                "batch_size": m["batch_size"], "max_length": m["max_length"],
                "full_dim": full_dim, "output_dim": dim, "prompt": prompt or None,
            },
            "text_config": text_cfg.to_dict(),
            "stats": {
                "n_events": texts.height,
                "n_texts_embedded": n_unique,  # unique texts per chunk, summed
                "tokens_median": int(np.median(n_tokens)),
                "tokens_max": int(n_tokens.max()),
                "n_truncated": n_truncated,
                "encode_seconds": seconds,
                "gpu": gpu_name(),
            },
            "libraries": library_versions(),
        },
    )
    print(f"Created {out.path}")
    print(f"  {texts.height} events ({n_unique} texts embedded) | dim {dim} (model gives {full_dim}) | "
          f"tokens median {int(np.median(n_tokens))}, max {n_tokens.max()} | {seconds} s")


if __name__ == "__main__":
    main()
