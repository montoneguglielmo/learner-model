"""
Count duplicated events in a MIAAM interactions file, per source. Only reads and prints.

Three levels, from strict to loose:
  full row          every column identical
  key + attempt     same user, exercise, created_at and attempt_index
  key               same user, exercise and created_at (what tabular_per_student dedups on)
For the "key" groups it also shows which columns differ inside a group, so you can tell
real copies (nothing differs) from retries logged with the same time (attempt_index differs).

Usage (from the repo root):
  uv run python scripts/check_duplicates.py ../MIAAM/data/maths_data_filtered.parquet ../MIAAM-v2/data/maths_data_filtered.parquet
"""

import argparse

import polars as pl

KEY = ["user_id", "exercise_id", "created_at"]


def pct(n, total):
    return f"{n / total:.2%}" if total else "-"


def extra_rows(df, cols):
    """Rows beyond the first of each group of identical `cols`."""
    return df.height - df.select(cols).unique().height


def report(df):
    n = df.height
    print(f"  rows {n:,}")
    levels = [("full row", df.columns), ("key + attempt_index", [*KEY, "attempt_index"]), ("key", KEY)]
    for name, cols in levels:
        if all(c in df.columns for c in cols):
            e = extra_rows(df, cols)
            print(f"  duplicated rows ({name}): {e:,} ({pct(e, n)})")

    dup = df.filter(pl.len().over(KEY) > 1)
    if dup.height == 0:
        return
    groups = dup.group_by(KEY).agg(pl.exclude(KEY).n_unique())
    print(f"  {groups.height:,} key groups with more than one row | group sizes: "
          f"{dict(sorted(dup.group_by(KEY).len()['len'].value_counts().iter_rows()))}")
    print("  share of key groups where the column differs:")
    for c in groups.columns:
        if c in KEY:
            continue
        print(f"    {c:24} {pct((groups[c] > 1).sum(), groups.height)}")
    print("  example group:")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=40):
        first = dup.sort(KEY).head(1).select(KEY)
        print(dup.join(first, on=KEY))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+", help="maths_data_filtered.parquet (one or more versions)")
    args = p.parse_args()
    for f in args.files:
        df = pl.read_parquet(f)
        print(f"\n{'=' * 100}\n{f}\n\nALL SOURCES")
        report(df)
        if "source" in df.columns:
            for src in sorted(df["source"].unique()):
                print(f"\nsource {src}")
                report(df.filter(pl.col("source") == src))


if __name__ == "__main__":
    main()
