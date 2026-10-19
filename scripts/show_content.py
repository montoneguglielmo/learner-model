"""
Explore the JSON inside the "content" column of the exercise table.
Only reads and prints.

For every field found in the JSON (also fields inside lists, e.g. screens[].instruction),
it prints: how many exercises have it, its type, and a few examples.
Then it shows the "images" field and the contents that are not valid JSON.

Usage (from the repo root):
  uv run python scripts/show_content.py ../MIAAM/data/maths_exercises_table.parquet
  uv run python scripts/show_content.py <file> --n 3 --width 200
"""

import argparse
import json
import random
from collections import defaultdict

import polars as pl


def walk(obj, path, found):
    """Record every value under its path. Lists become '[]' in the path."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            walk(v, f"{path}.{k}" if path else k, found)
    elif isinstance(obj, list):
        if not obj:
            found[path + "[]"].append("<empty list>")
        for item in obj:
            walk(item, path + "[]", found)
    else:
        found[path].append(obj)


def short(v, width):
    text = str(v).replace("\n", " ")
    return text if len(text) <= width else text[:width] + " [...]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--n", type=int, default=3, help="examples per field")
    p.add_argument("--width", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    rng = random.Random(args.seed)

    ex = pl.read_parquet(args.path, columns=["exercise_id", "gameplay_type", "content"]).unique("exercise_id")
    values = defaultdict(list)       # path -> all values
    exercises = defaultdict(set)     # path -> exercise ids that have it
    not_json = []

    for ex_id, gtype, content in ex.iter_rows():
        try:
            obj = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            not_json.append((ex_id, gtype, content))
            continue
        found = defaultdict(list)
        walk(obj, "", found)
        for path, vals in found.items():
            values[path].extend(vals)
            exercises[path].add(ex_id)

    n_json = ex.height - len(not_json)
    print(f"{ex.height} exercises | valid JSON: {n_json} | not JSON: {len(not_json)}")

    print("\n######## FIELDS IN content ########")
    for path in sorted(values):
        vals = values[path]
        types = sorted({type(v).__name__ for v in vals})
        distinct = list({str(v) for v in vals})
        print(f"\n=== {path}")
        print(f"    in {len(exercises[path])} exercises | type: {', '.join(types)} | distinct values: {len(distinct)}")
        for v in rng.sample(distinct, min(args.n, len(distinct))):
            print(f"    - {short(v, args.width)}")

    print("\n######## FIELDS PER gameplay_type ########")
    by_type = defaultdict(set)
    ids_by_type = dict(ex.group_by("gameplay_type").agg("exercise_id").iter_rows())
    for gtype, ids in ids_by_type.items():
        ids = set(ids)
        top = sorted(p for p in values if "." not in p and "[]" not in p and exercises[p] & ids)
        top += sorted({p.split("[]")[0] + "[]" for p in values if "[]" in p and exercises[p] & ids
                       and "." not in p.split("[]")[0]})
        by_type[gtype] = top
    for gtype in sorted(by_type, key=lambda g: -len(ids_by_type[g])):
        print(f"  {gtype} ({len(ids_by_type[gtype])}): {', '.join(by_type[gtype]) or '-'}")

    print("\n######## NOT VALID JSON ########")
    if not_json:
        counts = defaultdict(int)
        for _, gtype, _ in not_json:
            counts[gtype] += 1
        print("by gameplay_type:", dict(sorted(counts.items(), key=lambda x: -x[1])))
        for ex_id, gtype, content in rng.sample(not_json, min(args.n, len(not_json))):
            print(f"\n  [{gtype}] {ex_id}\n    {short(content, args.width * 3)}")


if __name__ == "__main__":
    main()