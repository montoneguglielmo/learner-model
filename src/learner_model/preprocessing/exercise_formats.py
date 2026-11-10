"""
Exercise text builders, one per format of exercise table.

The encoders are the same for every dataset; only the way an exercise becomes a text
depends on the dataset (MIAAM: a `content` JSON with screens, steps, feedback, ...).
The dataset config chooses the builder with `exercise: format: <name>`.

A builder module provides:
  TextConfig                                  the settings of "exercise: text:"
  build_texts(exercises, cfg, id_columns)     one row per exercise: <ids>, <keep>, parse_status, text
  not_used(columns, cfg, id_columns)          columns of the table it does not use

To add a dataset with another format: write preprocessing/<name>/exercise_text.py and add it here.
"""

import importlib

FORMATS = {
    "miaam": "learner_model.preprocessing.miaam.exercise_text",  # MIAAM V1 and V2
}


def exercise_format(name):
    """The builder module of this format."""
    if name not in FORMATS:
        raise ValueError(f"Unknown exercise format '{name}'. Known: {sorted(FORMATS)} "
                         f"(set exercise.format in the dataset config)")
    return importlib.import_module(FORMATS[name])
