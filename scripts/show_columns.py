"""
Print every column of a parquet file, with a few example values.
Only reads and prints.

Usage (from the repo root):
  uv run python scripts/show_columns.py ../MIAAM/data/maths_exercises_table.parquet
  uv run python scripts/show_columns.py <file> --n 5 --width 300
"""

import argparse

import polars as pl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="parquet file")
    p.add_argument("--n", type=int, default=3, help="examples per column")
    p.add_argument("--width", type=int, default=150, help="max characters per example")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    df = pl.read_parquet(args.path)
    print(f"{args.path}\n{df.height} rows, {df.width} columns")

    for col in df.columns:
        s = df[col]
        print(f"\n=== {col}  ({s.dtype})")
        print(f"    distinct values: {s.n_unique()} | empty (null): {s.null_count()}")
        # Random distinct examples, so we don't always see the first rows.
        values = s.drop_nulls().unique()
        examples = values.sample(min(args.n, len(values)), seed=args.seed)
        for v in examples:
            text = str(v).replace("\n", " ")
            if len(text) > args.width:
                text = text[: args.width] + " [...]"
            print(f"    - {text}")


if __name__ == "__main__":
    main()