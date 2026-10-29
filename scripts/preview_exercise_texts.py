"""
Build the exercise texts and print some of them, to check them by eye
before spending GPU time. Only reads and prints (or writes to --out).

Usage (from the repo root):
  uv run python scripts/preview_exercise_texts.py ../MIAAM/data/maths_exercises_table.parquet
  uv run python scripts/preview_exercise_texts.py <file> --n 2                 # 2 examples per gameplay_type
  uv run python scripts/preview_exercise_texts.py <file> --type GRADUATED_RULER --n 5
  uv run python scripts/preview_exercise_texts.py <file> --id b266c5b4-57a5-4384-99b2-cdc4f543e522
  uv run python scripts/preview_exercise_texts.py <file> --no-hierarchy        # try other settings
  uv run python scripts/preview_exercise_texts.py <file> --out /tmp/texts.txt  # all texts in one file
"""

import argparse

import polars as pl

from learner_model.preprocessing.exercise_text import TextConfig, build_texts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--n", type=int, default=1, help="examples per gameplay_type")
    p.add_argument("--type", help="only this gameplay_type")
    p.add_argument("--id", help="only this exercise_id")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", help="write ALL texts to this file")
    p.add_argument("--no-hierarchy", action="store_true")
    p.add_argument("--no-answer", action="store_true")
    p.add_argument("--no-feedback", action="store_true")
    p.add_argument("--with-intent", action="store_true")
    args = p.parse_args()

    cfg = TextConfig(
        include_hierarchy=not args.no_hierarchy,
        include_correct_answer=not args.no_answer,
        include_feedback_incorrect=not args.no_feedback,
        include_pedagogical_intent=args.with_intent,
    )
    texts = build_texts(pl.read_parquet(args.path), cfg)

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
                f.write(f"##### {r['exercise_id']} [{r['gameplay_type']}, {r['source']}, {r['parse_status']}]\n")
                f.write(r["text"] + "\n\n")
        print(f"\nAll texts written to {args.out}")

    # ---- examples
    if args.id:
        sample = texts.filter(pl.col("exercise_id") == args.id)
    else:
        pool = texts.filter(pl.col("gameplay_type") == args.type) if args.type else texts
        sample = (pool.sample(fraction=1.0, shuffle=True, seed=args.seed)
                      .group_by("gameplay_type", maintain_order=True).head(args.n)
                      .sort("gameplay_type")
                      .select(texts.columns))
        # the Python-dict contents too, if not already shown
        if not args.type:
            sample = pl.concat([sample, texts.filter(pl.col("parse_status") != "json").head(1)]).unique(
                "exercise_id", maintain_order=True)

    for r in sample.iter_rows(named=True):
        print(f"\n{'=' * 80}\n{r['exercise_id']}  [{r['gameplay_type']}, {r['source']}, {r['parse_status']}]\n{'-' * 80}")
        print(r["text"])


if __name__ == "__main__":
    main()