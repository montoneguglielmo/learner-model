"""
Do two versions of MIAAM (e.g. V1 and V2) contain the same students, events and exercises?
Checked per source present in both (e.g. "am"). Only reads and prints.

Usage (from the repo root):
  uv run python scripts/compare_overlap.py ../MIAAM/data ../MIAAM-v2/data
  uv run python scripts/compare_overlap.py <dir A> <dir B> --source am
"""

import argparse
from pathlib import Path

import polars as pl

EVENTS = "maths_data_filtered.parquet"
EXERCISES = "maths_exercises_table.parquet"
EVENT_KEY = ["user_id", "exercise_id", "created_at"]


def pct(n, total):
    return f"{n / total:.1%}" if total else "-"


def overlap(name, a, b):
    """a, b: sets. Prints sizes and shared share."""
    shared = len(a & b)
    print(f"  {name:40} A {len(a):>10,} | B {len(b):>10,} | in both {shared:>10,} "
          f"({pct(shared, len(a))} of A, {pct(shared, len(b))} of B)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir_a")
    p.add_argument("dir_b")
    p.add_argument("--source", nargs="*", help="sources to check (default: those in both)")
    args = p.parse_args()

    cols = [*EVENT_KEY, "source", "data_correct", "classroom_id"]
    ea = pl.read_parquet(Path(args.dir_a) / EVENTS, columns=cols)
    eb = pl.read_parquet(Path(args.dir_b) / EVENTS, columns=cols)
    xa = pl.read_parquet(Path(args.dir_a) / EXERCISES, columns=["exercise_id", "source", "content"])
    xb = pl.read_parquet(Path(args.dir_b) / EXERCISES, columns=["exercise_id", "source", "content"])

    sources = args.source or sorted(set(ea["source"].unique()) & set(eb["source"].unique()))
    print(f"Sources checked: {sources}")
    print("Ids shared ACROSS different sources (A source x B source, user_id):")
    for sa in sorted(ea["source"].unique()):
        for sb in sorted(eb["source"].unique()):
            if sa != sb:
                n = len(set(ea.filter(pl.col("source") == sa)["user_id"].unique())
                        & set(eb.filter(pl.col("source") == sb)["user_id"].unique()))
                if n:
                    print(f"  {sa} (A) x {sb} (B): {n:,} user_ids in common")

    for src in sources:
        a, b = ea.filter(pl.col("source") == src), eb.filter(pl.col("source") == src)
        print(f"\n=== source {src}: A {a.height:,} events | B {b.height:,} events")
        overlap("students (user_id)", set(a["user_id"].unique()), set(b["user_id"].unique()))
        overlap("classrooms (classroom_id)", set(a["classroom_id"].unique()), set(b["classroom_id"].unique()))
        overlap("exercises seen (exercise_id)", set(a["exercise_id"].unique()), set(b["exercise_id"].unique()))

        # Events: unique (user, exercise, time) keys; duplicated logs counted once.
        ka, kb = a.select(EVENT_KEY).unique(), b.select(EVENT_KEY).unique()
        print(f"  duplicated (user, exercise, time) rows: A {a.height - ka.height:,} | B {b.height - kb.height:,}")
        both = ka.join(kb, on=EVENT_KEY, how="inner").height
        print(f"  {'events (user, exercise, created_at)':40} A {ka.height:>10,} | B {kb.height:>10,} | in both "
              f"{both:>10,} ({pct(both, ka.height)} of A, {pct(both, kb.height)} of B)")

        # Same answer correctness on the shared events?
        ca = a.group_by(EVENT_KEY).agg(pl.col("data_correct").first().alias("ya"))
        cb = b.group_by(EVENT_KEY).agg(pl.col("data_correct").first().alias("yb"))
        j = ca.join(cb, on=EVENT_KEY, how="inner")
        if j.height:
            print(f"  shared events with the same data_correct: {pct((j['ya'] == j['yb']).sum(), j.height)}")

        # Students in both: do they have the same number of events?
        sa = ka.group_by("user_id").len("na")
        sb = kb.group_by("user_id").len("nb")
        s = sa.join(sb, on="user_id", how="inner")
        if s.height:
            print(f"  students in both with the same number of events: {pct((s['na'] == s['nb']).sum(), s.height)} "
                  f"| more in A: {pct((s['na'] > s['nb']).sum(), s.height)} "
                  f"| more in B: {pct((s['na'] < s['nb']).sum(), s.height)}")

        # Exercise table: same ids, same content?
        xa_s = xa.filter(pl.col("source") == src).unique("exercise_id")
        xb_s = xb.filter(pl.col("source") == src).unique("exercise_id")
        overlap("exercise table ids", set(xa_s["exercise_id"]), set(xb_s["exercise_id"]))
        jx = xa_s.join(xb_s, on="exercise_id", how="inner", suffix="_b")
        if jx.height:
            print(f"  shared exercises with identical content: {pct((jx['content'] == jx['content_b']).sum(), jx.height)}")


if __name__ == "__main__":
    main()
