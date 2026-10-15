"""
Stage "split": decide once which students are in train / val / test,
starting from the raw MIAAM interactions.

  - The unit is a student (default) or a classroom (--group-col classroom_id).
    With classrooms, all students of a class go to the same set.
  - Units are sorted, then shuffled with a fixed seed: the result does not
    depend on the row order in the file.
  - Fractions are fractions of units (students or classrooms), not of rows.

Usage (from the repo root, after committing your code):
  uv run python -m learner_model.preprocessing.stages.split \
      --interactions ../MIAAM/data/maths_data_filtered.parquet

Output (a new folder, never modified later):
  datasets/split_students__<date>__<commit6>/
    split.json         {"train": [user_id, ...], "val": [...], "test": [...]}
    manifest.json      (written last)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from learner_model.common.datasets import DatasetWriter

SETS = ("train", "val", "test")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--interactions", required=True, help="maths_data_filtered.parquet")
    p.add_argument("--group-col", default="user_id", choices=["user_id", "classroom_id"])
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="datasets", help="where dataset folders are created")
    p.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (quick tests only)")
    args = p.parse_args()

    if not 0 < args.val_frac + args.test_frac < 1:
        raise ValueError("val-frac + test-frac must be between 0 and 1")

    path = Path(args.interactions).resolve()
    # Only settings that change the RESULT go in the config.
    config = {"interactions": str(path), "group_col": args.group_col,
              "val_frac": args.val_frac, "test_frac": args.test_frac, "seed": args.seed}
    out = DatasetWriter("split_students", config, root=args.root, allow_dirty=args.allow_dirty)
    if out.reuse:
        return

    cols = list(dict.fromkeys(["user_id", args.group_col, "data_correct", "source"]))
    it = pl.read_parquet(path, columns=cols)

    # With classrooms: each student must belong to exactly one class.
    if args.group_col != "user_id":
        n_groups = it.group_by("user_id").agg(pl.col(args.group_col).n_unique())
        bad = n_groups.filter(pl.col(args.group_col) > 1)
        if bad.height:
            raise ValueError(f"{bad.height} students are in more than one {args.group_col}. "
                             f"Splitting by {args.group_col} would put them in two sets.")

    # Shuffle the units with a fixed seed.
    units = np.array(sorted(it[args.group_col].drop_nulls().unique().to_list()))
    order = np.random.default_rng(args.seed).permutation(len(units))
    n_test = round(args.test_frac * len(units))
    n_val = round(args.val_frac * len(units))
    unit_set = {}
    for i, idx in enumerate(order):
        unit_set[units[idx]] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    it = it.with_columns(pl.col(args.group_col).replace_strict(unit_set, default=None).alias("set"))
    if it["set"].null_count():
        raise ValueError(f"{it['set'].null_count()} rows have no {args.group_col}")

    sets = {s: sorted(it.filter(pl.col("set") == s)["user_id"].unique().to_list()) for s in SETS}
    (out.path / "split.json").write_text(json.dumps(sets, indent=1))

    # Stats, to check that the sets look alike.
    stats = {}
    for s in SETS:
        part = it.filter(pl.col("set") == s)
        stats[s] = {
            "students": len(sets[s]),
            "units": sum(v == s for v in unit_set.values()),
            "interactions": part.height,
            "correct_rate": round(part["data_correct"].mean(), 4),
            "source": dict(part["source"].value_counts().iter_rows()),
        }
    out.finish(input_files=[path], extra={"stats": stats})

    total = sum(v["students"] for v in stats.values())
    print(f"Created {out.path}")
    for s, v in stats.items():
        src = ", ".join(f"{k} {n / v['interactions']:.0%}" for k, n in sorted(v["source"].items()))
        print(f"  {s:5s}: {v['students']:6d} students ({100 * v['students'] / total:4.1f}%) | "
              f"{v['interactions']:8d} rows | correct {v['correct_rate']:.3f} | {src}")


if __name__ == "__main__":
    main()