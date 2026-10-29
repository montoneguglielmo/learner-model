"""
Build the event texts and print some of them, to check them by eye
before spending GPU time. Only reads and prints.

Usage (from the repo root):
  uv run python scripts/preview_event_texts.py --config configs/encoders/event_qwen3_0.6b.yaml
  uv run python scripts/preview_event_texts.py --config <file> --n 10 --seed 1
"""

import argparse
from pathlib import Path

import polars as pl
import yaml

from learner_model.preprocessing.event_text import EventTextConfig, build_event_texts, columns_in_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=5, help="examples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    text_cfg = EventTextConfig(**cfg.get("text", {}))
    events = pl.read_parquet(cfg["events"])
    print(f"{events.height} events | all columns: {events.columns}")
    used = columns_in_text(events.columns, text_cfg)
    print(f"in the text:  {used}")
    print(f"own columns:  user_id, created_at, {text_cfg.keep_columns}")
    print(f"targets:      {text_cfg.target_columns}")
    others = set(used) | {"user_id", "created_at"} | set(text_cfg.keep_columns + text_cfg.target_columns)
    print(f"not used:     {[c for c in events.columns if c not in others]}")

    texts = build_event_texts(events, text_cfg)
    chars = texts["text"].str.len_chars()
    print(f"text length (characters): median {chars.median():.0f}, p95 {chars.quantile(0.95):.0f}, max {chars.max()}")
    print(f"distinct texts: {texts['text'].n_unique()} of {texts.height}")

    for r in texts.sample(min(args.n, texts.height), seed=args.seed).iter_rows(named=True):
        print(f"\n{'=' * 80}\nevent {r['event_id']} | user {r['user_id']} | {r['created_at']}\n{'-' * 80}")
        print(r["text"])


if __name__ == "__main__":
    main()
