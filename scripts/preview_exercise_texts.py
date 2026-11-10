"""
Build the exercise texts and print some of them, to check them by eye
before spending GPU time. Only reads and prints (or writes to --out).

Usage (from the repo root). The settings come from the "text" part of the config:
copy the config to try other settings. Examples are picked per gameplay_type (keep that column).
--id matches the last id column, e.g. exercise_id (all sources sharing that id are shown).
  uv run python scripts/preview_exercise_texts.py --config configs/encoders/miaam_v2.yaml
  uv run python scripts/preview_exercise_texts.py --config <file> --n 2      # 2 examples per gameplay_type
  uv run python scripts/preview_exercise_texts.py --config <file> --type GRADUATED_RULER --n 5
  uv run python scripts/preview_exercise_texts.py --config <file> --id b266c5b4-57a5-4384-99b2-cdc4f543e522
  uv run python scripts/preview_exercise_texts.py --config <file> --out /tmp/texts.txt  # all texts in one file
"""

import argparse

import polars as pl
from learner_model.common.dataset_config import load_dataset_config
from learner_model.preprocessing.exercise_text import TextConfig, build_texts, not_used


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="e.g. configs/encoders/miaam_v2.yaml")
    p.add_argument("--n", type=int, default=1, help="examples per gameplay_type")
    p.add_argument("--type", help="only this gameplay_type")
    p.add_argument("--id", help="only this exercise_id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write ALL texts to this file")
    args = p.parse_args()

    ds = load_dataset_config(args.config)
    cfg = TextConfig(**ds.exercise.get("text", {}))
    ids = list(ds.id_columns)
    exercises = pl.read_parquet(ds.exercises_path)
    print(f"dataset {ds.dataset}: {ds.exercises_path}")
    print(f"{exercises.height} rows | all columns: {exercises.columns}")
    print(f"exercise id: {ids}")
    print(f"in the text: {cfg.text_columns}")
    print(f"kept as is:  {cfg.keep_columns}")
    print(f"not used:    {not_used(exercises.columns, cfg, ids)}")
    texts = build_texts(exercises, cfg, ids)
    name = lambda r: " / ".join(str(r[c]) for c in ids)  # noqa: E731

    # ---- summary
    chars = texts["text"].str.len_chars()
    print(f"{texts.height} exercises | parse status: {dict(texts['parse_status'].value_counts().iter_rows())}")
    print(f"text length (characters): median {chars.median():.0f}, p95 {chars.quantile(0.95):.0f}, max {chars.max()}")
    print(f"  (about 1 token per 3-4 characters for French)")
    print(f"exercises sharing exactly the same text as another one: "
          f"{texts.height - texts['text'].n_unique()}")
    print(f"texts with '[description partielle]' (text does not fully describe the screen): "
          f"{texts['text'].str.contains('[description partielle]', literal=True).sum()}")
    print(f"settings: {cfg.to_dict()}")

    # Most frequent explanations: generic messages show up here.
    lines = texts["text"].str.split("\n").explode().str.strip_chars()
    expl = lines.filter(lines.str.starts_with("Explication :")).value_counts(sort=True).head(15)
    print("\nMost frequent 'Explication' lines (generic ones can be added to GENERIC_INCORRECT):")
    for text, count in expl.iter_rows():
        print(f"  {count:5d}  {text[:110]}")

    if args.out:
        with open(args.out, "w") as f:
            for r in texts.iter_rows(named=True):
                f.write(f"##### {name(r)} [{r.get('gameplay_type')}, {r['parse_status']}]\n")
                f.write(r["text"] + "\n\n")
        print(f"\nAll texts written to {args.out}")

    # ---- examples
    if args.id:
        sample = texts.filter(pl.col(ids[-1]) == args.id)
    else:
        pool = texts.filter(pl.col("gameplay_type") == args.type) if args.type else texts
        sample = (pool.sample(fraction=1.0, shuffle=True, seed=args.seed)
                      .group_by("gameplay_type", maintain_order=True).head(args.n)
                      .sort("gameplay_type")
                      .select(texts.columns))
        # the Python-dict contents too, if not already shown
        if not args.type:
            sample = pl.concat([sample, texts.filter(pl.col("parse_status") != "json").head(1)]).unique(
                ids, maintain_order=True)

    for r in sample.iter_rows(named=True):
        print(f"\n{'=' * 80}\n{name(r)}  [{r.get('gameplay_type')}, {r['parse_status']}]\n{'-' * 80}")
        print(r["text"])


if __name__ == "__main__":
    main()