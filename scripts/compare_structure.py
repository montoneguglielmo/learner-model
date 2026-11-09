"""
Compare the structure of two versions of the same MIAAM file (e.g. V1 and V2).
Only reads and prints.

For each file it shows:
  - row counts
  - columns only in A, only in B, and common columns whose type differs
  - per common column: empty (null) share and number of distinct values in A and B
  - for columns with few values (source, work_mode, gameplay_type, ...): values only in A / only in B
  - whether the ids are unique on their own or only together with `source`
  - for the exercise table: whether `content` parses the same way (JSON keys used by exercise_text.py)

Usage (from the repo root):
  uv run python scripts/compare_structure.py ../MIAAM/data ../MIAAM-V2/data
  uv run python scripts/compare_structure.py <dir A> <dir B> --files maths_data_filtered.parquet
"""

import argparse
from collections import Counter
from pathlib import Path

import polars as pl

from learner_model.preprocessing.exercise_text import parse_content

FILES = ["maths_data_filtered.parquet", "maths_exercises_table.parquet"]
FEW_VALUES = 30  # columns with at most this many distinct values get their values compared


def pct(n, total):
    return f"{n / total:.1%}" if total else "-"


def compare_columns(a, b):
    sa, sb = a.schema, b.schema
    only_a = [c for c in sa if c not in sb]
    only_b = [c for c in sb if c not in sa]
    common = [c for c in sa if c in sb]
    print(f"\nRows: A {a.height:,} | B {b.height:,}")
    print(f"Columns only in A: {only_a or 'none'}")
    print(f"Columns only in B: {only_b or 'none'}")
    diff_type = [(c, sa[c], sb[c]) for c in common if sa[c] != sb[c]]
    print(f"Common columns with a different type: "
          f"{[f'{c}: {ta} -> {tb}' for c, ta, tb in diff_type] or 'none'}")

    print(f"\n{'column':32} {'type A':>14} {'null A':>7} {'null B':>7} {'distinct A':>11} {'distinct B':>11}")
    for c in common:
        na, nb = a[c].n_unique(), b[c].n_unique()
        print(f"{c:32} {str(sa[c]):>14} {pct(a[c].null_count(), a.height):>7} "
              f"{pct(b[c].null_count(), b.height):>7} {na:>11,} {nb:>11,}")

    for c in common:
        if a[c].n_unique() > FEW_VALUES or b[c].n_unique() > FEW_VALUES or sa[c] != sb[c]:
            continue
        va, vb = set(a[c].to_list()), set(b[c].to_list())
        if va != vb:
            print(f"\n  {c}: only in A {sorted(va - vb, key=str)} | only in B {sorted(vb - va, key=str)}")
    for c in [*only_a, *only_b]:
        df = a if c in sa else b
        if df[c].n_unique() <= FEW_VALUES:
            print(f"\n  {c} (only in {'A' if c in sa else 'B'}): values {sorted(set(df[c].to_list()), key=str)}")


def check_ids(name, df, ids):
    """Is each id unique on its own, or only together with source?"""
    if "source" not in df.columns:
        return
    for col in ids:
        if col not in df.columns:
            continue
        bare = df[col].n_unique()
        with_source = df.select(col, "source").n_unique()
        shared = (df.group_by(col).agg(pl.col("source").n_unique().alias("n"))
                  .filter(pl.col("n") > 1).height)
        print(f"  {name} {col}: {bare:,} bare ids, {with_source:,} (source, id) pairs, "
              f"{shared:,} ids used by more than one source")


def content_summary(df, n=2000):
    """Parse status and top-level JSON keys of `content` (sample of n rows)."""
    if "content" not in df.columns:
        return None
    sample = df["content"].drop_nulls()
    sample = sample.sample(min(n, len(sample)), seed=0)
    status, keys = Counter(), Counter()
    for raw in sample:
        obj, s = parse_content(raw)
        status[s] += 1
        if isinstance(obj, dict):
            keys.update(obj.keys())
    return len(sample), status, keys


def compare_content(a, b):
    ca, cb = content_summary(a), content_summary(b)
    if ca is None or cb is None:
        return
    print("\nContent (random sample):")
    for name, (n, status, _) in (("A", ca), ("B", cb)):
        print(f"  {name}: {n} rows, parse status {dict(status)}")
    keys = sorted(set(ca[2]) | set(cb[2]))
    print(f"  {'top-level key':30} {'share A':>8} {'share B':>8}")
    for k in keys:
        print(f"  {k:30} {pct(ca[2][k], ca[0]):>8} {pct(cb[2][k], cb[0]):>8}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a", help="folder of version A (e.g. ../MIAAM/data)")
    p.add_argument("dir_b", help="folder of version B (e.g. ../MIAAM-V2/data)")
    p.add_argument("--files", nargs="+", default=FILES)
    args = p.parse_args()

    for f in args.files:
        pa, pb = Path(args.dir_a) / f, Path(args.dir_b) / f
        print(f"\n{'=' * 100}\n{f}\n  A = {pa}\n  B = {pb}")
        missing = [str(x) for x in (pa, pb) if not x.exists()]
        if missing:
            print(f"  missing: {missing}")
            continue
        a, b = pl.read_parquet(pa), pl.read_parquet(pb)
        compare_columns(a, b)
        print("\nIds:")
        for name, df in (("A", a), ("B", b)):
            check_ids(name, df, ["user_id", "exercise_id"])
        compare_content(a, b)


if __name__ == "__main__":
    main()
