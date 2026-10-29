"""
Encoder "exercise_embed_v1": one vector per exercise, from its text.

  1. Build one clean text per exercise (exercise_text.py), with the settings
     in the "text" part of the config.
  2. Embed each text with a sentence-embedding model (e.g. Qwen3-Embedding).
  3. Optionally keep only the first `output_dim` values (Matryoshka models
     are trained for this), then L2-normalize.
  4. Save the result as a dataset folder, following the dataset rules.

Usage (from the repo root, after committing your code):
  uv run python -m learner_model.preprocessing.encoders.exercise_embed_v1 \
      --config configs/encoders/exercise_qwen3_4b.yaml

Quick test on 50 exercises (creates a separate dataset folder, delete it after):
  uv run python -m learner_model.preprocessing.encoders.exercise_embed_v1 \
      --config configs/encoders/exercise_qwen3_0.6b.yaml --limit 50 --allow-dirty

Output:
  datasets/exercise_embed_v1__<date>__<commit6>/
    embeddings.parquet   exercise_id, embedding (float32, L2-normalized)
    texts.parquet        exactly the texts that were embedded (+ source, gameplay_type, parse_status)
    manifest.json        written last
"""

import argparse
import time
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from learner_model.common.datasets import DatasetWriter
from learner_model.common.sentence_models import gpu_name, library_versions, load_model, resolve_revision
from learner_model.preprocessing.exercise_text import TextConfig, build_texts

PIPELINE = "exercise_embed_v1"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="YAML file, e.g. configs/encoders/exercise_qwen3_4b.yaml")
    p.add_argument("--limit", type=int, default=None, help="only the first N exercises (quick tests)")
    p.add_argument("--root", default="datasets", help="where dataset folders are created")
    p.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (quick tests only)")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    m = cfg["model"]
    exercises_path = Path(cfg["exercises"]).resolve()
    text_cfg = TextConfig(**cfg.get("text", {}))

    revision = resolve_revision(m["name"], m.get("revision", "main"))
    print(f"Model {m['name']} at commit {revision}")

    # Everything that changes the RESULT goes in the config (with the exact model commit).
    config = {
        "config_file": Path(args.config).name,
        "exercises": str(exercises_path),
        "model": {**m, "revision": revision},
        "text": text_cfg.to_dict(),
        "limit": args.limit,
    }
    out = DatasetWriter(PIPELINE, config, root=args.root, allow_dirty=args.allow_dirty)
    if out.reuse:
        return

    # 1. Texts
    texts = build_texts(pl.read_parquet(exercises_path), text_cfg)
    if args.limit:
        texts = texts.head(args.limit)
    print(f"{texts.height} exercise texts built")

    # 2. Embeddings
    model = load_model(m, revision)
    prompt = m.get("prompt") or ""
    docs = [prompt + t for t in texts["text"].to_list()]
    n_tokens = np.array([len(ids) for ids in model.tokenizer(docs)["input_ids"]])
    n_truncated = int((n_tokens > m["max_length"]).sum())
    if n_truncated:
        print(f"WARNING: {n_truncated} texts are longer than max_length={m['max_length']} tokens and will be cut")

    t0 = time.time()
    emb = model.encode(docs, batch_size=m["batch_size"], convert_to_numpy=True,
                       normalize_embeddings=False, show_progress_bar=True)
    seconds = round(time.time() - t0, 1)

    # 3. Keep the first output_dim values, then L2-normalize
    emb = np.asarray(emb, dtype=np.float32)
    full_dim = emb.shape[1]
    dim = m.get("output_dim") or full_dim
    if dim > full_dim:
        raise ValueError(f"output_dim={dim} but the model gives {full_dim} values")
    emb = emb[:, :dim]
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    if (norms == 0).any() or not np.isfinite(emb).all():
        raise ValueError("Some embeddings are zero or not finite (NaN/inf)")
    emb = emb / norms

    # 4. Save
    pl.DataFrame({
        "exercise_id": texts["exercise_id"],
        "embedding": pl.Series(emb, dtype=pl.Array(pl.Float32, dim)),
    }).write_parquet(out.path / "embeddings.parquet")
    texts.write_parquet(out.path / "texts.parquet")

    out.finish(
        input_files=[exercises_path],
        extra={
            "llm": {
                "name": m["name"], "revision": revision, "dtype": m["dtype"],
                "batch_size": m["batch_size"], "max_length": m["max_length"],
                "full_dim": full_dim, "output_dim": dim, "prompt": prompt or None,
            },
            "text_config": text_cfg.to_dict(),
            "stats": {
                "n_exercises": texts.height,
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
    print(f"  {texts.height} exercises | dim {dim} (model gives {full_dim}) | "
          f"tokens median {int(np.median(n_tokens))}, max {n_tokens.max()} | {seconds} s")


if __name__ == "__main__":
    main()