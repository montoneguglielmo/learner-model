"""
Stage "split": decide once which students of ONE dataset are in train / val / test,
starting from that dataset's original events table (independent of the embeddings).
The dataset config (configs/encoders/<dataset>.yaml) says where the file is and which
column is the student; the split settings are given here.

  - The unit is a student (default) or a classroom (--group-col classroom_id).
    With classrooms, all students of a class go to the same set.
  - Units are sorted, then shuffled with a fixed seed: the result does not
    depend on the row order in the file.
  - Fractions are fractions of units (students or classrooms), not of rows.

Usage (from the repo root, after committing your code):
  uv run python -m learner_model.preprocessing.stages.split --config configs/encoders/miaam_v2.yaml

Output (a new folder, never modified later):
  datasets/split_students_<dataset>__<date>__<commit6>/
    split.json         {"train": [student id, ...], "val": [...], "test": [...]}
    manifest.json      (written last)
"""

import argparse
import json

import numpy as np
import polars as pl

from learner_model.common.dataset_config import load_dataset_config
from learner_model.common.datasets import DatasetWriter

SETS = ("train", "val", "test")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="dataset config, e.g. configs/encoders/miaam_v2.yaml")
    p.add_argument("--group-col", default=None,
                   help="unit of the split: the student column (default) or e.g. classroom_id")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--root", default="datasets", help="where dataset folders are created")
    p.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (quick tests only)")
    args = p.parse_args()

    if not 0 < args.val_frac + args.test_frac < 1:
        raise ValueError("val-frac + test-frac must be between 0 and 1")

    ds = load_dataset_config(args.config)
    path = ds.events_path
    student, outcome = ds.event_columns.student_id, ds.event_columns.outcome
    group_col = args.group_col or student
    # Only settings that change the RESULT go in the config.
    config = {"dataset": ds.dataset, "events": str(path), "student_col": student, "group_col": group_col,
              "val_frac": args.val_frac, "test_frac": args.test_frac, "seed": args.seed}
    out = DatasetWriter(f"split_students_{ds.dataset}", config, root=args.root, allow_dirty=args.allow_dirty)
    if out.reuse:
        return

    available = pl.read_parquet_schema(path).keys()
    cols = list(dict.fromkeys([student, group_col, outcome, *(["source"] if "source" in available else [])]))
    it = pl.read_parquet(path, columns=cols).rename({student: "user_id", outcome: "data_correct"})
    if group_col == student:
        group_col = "user_id"
    if "source" not in it.columns:
        it = it.with_columns(source=pl.lit(ds.dataset))

    # With classrooms: each student must belong to exactly one class.
    if group_col != "user_id":
        n_groups = it.group_by("user_id").agg(pl.col(group_col).n_unique())
        bad = n_groups.filter(pl.col(group_col) > 1)
        if bad.height:
            raise ValueError(f"{bad.height} students are in more than one {group_col}. "
                             f"Splitting by {group_col} would put them in two sets.")

    # Shuffle the units with a fixed seed.
    units = np.array(sorted(it[group_col].drop_nulls().unique().to_list()))
    order = np.random.default_rng(args.seed).permutation(len(units))
    n_test = round(args.test_frac * len(units))
    n_val = round(args.val_frac * len(units))
    unit_set = {}
    for i, idx in enumerate(order):
        unit_set[units[idx]] = "test" if i < n_test else "val" if i < n_test + n_val else "train"

    it = it.with_columns(pl.col(group_col).replace_strict(unit_set, default=None).alias("set"))
    if it["set"].null_count():
        raise ValueError(f"{it['set'].null_count()} rows have no {group_col}")

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
            "correct_rate": round(part["data_correct"].mean(), 4) if part.height else None,
            "source": dict(part["source"].value_counts().iter_rows()),
        }
    out.finish(input_files=[path], extra={"stats": stats})

    total = sum(v["students"] for v in stats.values())
    print(f"Created {out.path}")
    for s, v in stats.items():
        src = ", ".join(f"{k} {n / v['interactions']:.0%}" for k, n in sorted(v["source"].items()))
        print(f"  {s:5s}: {v['students']:6d} students ({100 * v['students'] / total:4.1f}%) | "
              f"{v['interactions']:8d} rows | correct {v['correct_rate'] or 0:.3f} | {src}")


if __name__ == "__main__":
    main()