"""
Stage "split": decide once which students are in train / val / test.

Version "split_pykt" copies pyKT's split, so results stay comparable with DKT:
  - test  = students in pyKT's test_sequences.csv
  - val   = students with fold == --val-fold in train_valid_sequences.csv
  - train = all other students in train_valid_sequences.csv

Usage (from the repo root, after committing your code):
  uv run python -m learner_model.preprocessing.stages.split \
      --pykt-dir ../evaluation-miaam/pykt-toolkit/data/miaamv1

Output (a new folder, never modified later):
  datasets/split_pykt__<date>__<commit6>/
    split.json
    manifest.json      (written last)
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from learner_model.common.datasets import DatasetWriter


def read_ids(path, need_fold):
    """Read only the uid (and fold) columns of a pyKT sequences file."""
    needed = {"uid", "fold"} if need_fold else {"uid"}
    header = pd.read_csv(path, nrows=0).columns
    missing = needed - set(header)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}. Columns found: {list(header)}")
    # pyKT can cut one long student sequence into several rows: same uid, several rows.
    return pd.read_csv(path, usecols=list(needed), dtype={"uid": str})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pykt-dir", required=True, help="pyKT folder with train_valid_sequences.csv and test_sequences.csv")
    p.add_argument("--val-fold", type=int, default=0, help="which pyKT fold becomes the validation set")
    p.add_argument("--root", default="datasets", help="where dataset folders are created")
    p.add_argument("--allow-dirty", action="store_true", help="allow uncommitted changes (quick tests only)")
    args = p.parse_args()

    pykt_dir = Path(args.pykt_dir).resolve()
    tv_path = pykt_dir / "train_valid_sequences.csv"
    test_path = pykt_dir / "test_sequences.csv"

    # Only settings that change the RESULT go in the config (not --root, not --allow-dirty).
    config = {"source": "pykt", "pykt_dir": str(pykt_dir), "val_fold": args.val_fold}
    out = DatasetWriter("split_pykt", config, root=args.root, allow_dirty=args.allow_dirty)
    if out.reuse:
        return

    tv = read_ids(tv_path, need_fold=True)
    test_df = read_ids(test_path, need_fold=False)

    # Check 1: each student must be in only one fold.
    folds_per_student = tv.groupby("uid")["fold"].nunique()
    bad = folds_per_student[folds_per_student > 1]
    if len(bad):
        raise ValueError(f"{len(bad)} students are in more than one fold, e.g. {list(bad.index[:5])}")
    if args.val_fold not in set(tv["fold"]):
        raise ValueError(f"fold {args.val_fold} not found. Folds: {sorted(set(tv['fold']))}")

    val = set(tv.loc[tv["fold"] == args.val_fold, "uid"])
    train = set(tv["uid"]) - val
    test = set(test_df["uid"])

    # Check 2: no student in two sets.
    for a_name, a, b_name, b in [("train", train, "val", val), ("train", train, "test", test), ("val", val, "test", test)]:
        common = a & b
        if common:
            raise ValueError(f"{len(common)} students are in both {a_name} and {b_name}, e.g. {sorted(common)[:5]}")

    sets = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    (out.path / "split.json").write_text(json.dumps(sets, indent=1))

    counts = {k: len(v) for k, v in sets.items()}
    out.finish(input_files=[tv_path, test_path], extra={"stats": counts})

    total = sum(counts.values())
    print(f"Created {out.path}")
    for k, n in counts.items():
        print(f"  {k:5s}: {n:6d} students ({100 * n / total:.1f}%)")
    print(f"  total: {total:6d} students")


if __name__ == "__main__":
    main()