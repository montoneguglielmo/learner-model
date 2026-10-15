"""Read a split made by the "split" stage. Every model uses this."""

import json

from learner_model.common.datasets import open_dataset


def load_split(dataset_id, root="datasets"):
    """Return {"train": set of uids, "val": ..., "test": ...}.

    dataset_id is the folder name, e.g. "split_pykt__2026-10-15__a3f9c1".
    Log it in W&B with every run, so you know which split a result used.
    """
    path = open_dataset(dataset_id, root)
    return {k: set(v) for k, v in json.loads((path / "split.json").read_text()).items()}