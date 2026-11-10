"""
Build the event texts and print some of them, to check them by eye
before spending GPU time. Only reads and prints.

Usage (from the repo root):
  uv run python scripts/preview_event_texts.py --config configs/encoders/miaam_v2.yaml
  uv run python scripts/preview_event_texts.py --config <file> --n 10 --seed 1
"""

import argparse

import polars as pl
from learner_model.common.dataset_config import load_dataset_config
from learner_model.preprocessing.event_text import (EventTextConfig, build_event_texts, not_used, output_columns,
                                                    text_columns)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n", type=int, default=5, help="examples")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ds = load_dataset_config(args.config)
    text_cfg = EventTextConfig(**ds.event.get("text", {}))
    roles = ds.event_columns
    ids = roles.exercise_id
    events = pl.read_parquet(ds.events_path)
    print(f"dataset {ds.dataset}: {ds.events_path}")
    print(f"{events.height} events | all columns: {events.columns}")
    cols = text_columns(events.columns, text_cfg, roles)
    print(f"key columns:  {roles.to_dict()}")
    print(f"kept as is:   {output_columns(roles, text_cfg)}")
    print(f"event text:   {cols['event']}")
    print(f"result text:  {cols.get('result', 'none (embed_result_columns is empty)')}")
    print(f"not used:     {not_used(events.columns, text_cfg, roles)}")

    texts = build_event_texts(events, text_cfg, roles)
    for kind in cols:
        t = texts[f"{kind}_text"]
        chars = t.str.len_chars()
        print(f"{kind} text length (characters): median {chars.median():.0f}, p95 {chars.quantile(0.95):.0f}, "
              f"max {chars.max()} | distinct texts: {t.n_unique()} of {texts.height}")

    for r in texts.sample(min(args.n, texts.height), seed=args.seed).iter_rows(named=True):
        print(f"\n{'=' * 80}\nevent {r['event_id']} | student {r[roles.student_id]} | {r[roles.timestamp]}"
              f" | exercise {', '.join(str(r[c]) for c in ids)}\n{'-' * 80}")
        print(r["event_text"])
        if "result" in cols:
            print(f"{'- ' * 10}result")
            print(r["result_text"])


if __name__ == "__main__":
    main()
