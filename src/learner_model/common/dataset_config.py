"""
One config file per dataset (configs/encoders/<dataset>.yaml) describes:
  - where the original files are (one exercise table, one attempts/events table)
  - for each table: what its key columns are called (standard names -> the dataset's
    names), which columns are kept, which are embedded, and with which model

The split stage and the two encoders all read it, but each starts from the original files:
none of them reads another one's output.

    dataset: miaam_v2                       # name used in the output folder names
    data_dir: ../MIAAM-v2/data
    events_file: maths_data_filtered.parquet
    exercises_file: maths_exercises_table.parquet
    exercise:
      columns:
        exercise_id: [source, exercise_id]  # one column, or several that together are unique
      format: miaam                         # text builder (preprocessing/exercise_formats.py)
      model: {...}
      text: {keep_columns: [...], text_columns: [...], ...}
    event:
      columns:
        event_id: null                      # null = row number in the events file
        student_id: user_id
        timestamp: created_at               # orders the events; any encoding
        outcome: data_correct               # boolean
        exercise_id: [source, exercise_id]  # links the event to its exercise (same order)
      model: {...}
      text: {keep_columns: [...], embed_event_columns: [...], embed_result_columns: [...], ...}

Training never needs these names: each encoder writes them into its manifest, and
training reads them from there.
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9_]+$")


def _as_tuple(value):
    return (value,) if isinstance(value, str) else tuple(value)


@dataclass(frozen=True)
class ExerciseColumns:
    """What the key columns of the exercise table are called."""
    exercise_id: tuple = ("exercise_id",)

    def __post_init__(self):
        object.__setattr__(self, "exercise_id", _as_tuple(self.exercise_id))
        if not self.exercise_id:
            raise ValueError("exercise.columns.exercise_id is empty")

    def to_dict(self):
        return {"exercise_id": list(self.exercise_id)}


@dataclass(frozen=True)
class EventColumns:
    """What the key columns of the events table are called."""
    student_id: str = "user_id"
    timestamp: str = "created_at"
    outcome: str = "data_correct"
    exercise_id: tuple = ("exercise_id",)
    event_id: str | None = None   # None: the row number in the events file

    def __post_init__(self):
        object.__setattr__(self, "exercise_id", _as_tuple(self.exercise_id))
        if not self.exercise_id:
            raise ValueError("event.columns.exercise_id is empty")
        clash = {self.student_id, self.timestamp, self.outcome} & set(self.exercise_id)
        if clash:
            raise ValueError(f"{sorted(clash)} is both an exercise id column and another event column")

    def to_dict(self):
        return {**asdict(self), "exercise_id": list(self.exercise_id)}


@dataclass(frozen=True)
class DatasetConfig:
    path: Path              # the config file
    dataset: str
    events_path: Path
    exercises_path: Path
    exercise_columns: ExerciseColumns
    event_columns: EventColumns
    exercise: dict          # {"format": ..., "model": ..., "text": ...}
    event: dict             # {"model": ..., "text": ...}

    def describe(self):
        """What goes into every manifest made from this dataset."""
        return {"dataset": self.dataset, "config_file": self.path.name,
                "events": str(self.events_path), "exercises": str(self.exercises_path)}


def load_dataset_config(path) -> DatasetConfig:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    for key in ("dataset", "data_dir", "events_file", "exercises_file"):
        if key not in cfg:
            raise ValueError(f"{path}: missing '{key}'")
    if not NAME_RE.match(cfg["dataset"]):
        raise ValueError(f"{path}: dataset name '{cfg['dataset']}' must be lower case letters, digits, _")
    exercise, event = cfg.get("exercise") or {}, cfg.get("event") or {}
    ex_cols = ExerciseColumns(**(exercise.get("columns") or {}))
    ev_cols = EventColumns(**(event.get("columns") or {}))
    if len(ex_cols.exercise_id) != len(ev_cols.exercise_id):
        raise ValueError(f"{path}: exercise.columns.exercise_id {list(ex_cols.exercise_id)} and "
                         f"event.columns.exercise_id {list(ev_cols.exercise_id)} must have the same length "
                         f"(they are matched in order)")
    data_dir = Path(cfg["data_dir"])  # relative to where the command runs (the repo root)
    return DatasetConfig(
        path=path, dataset=cfg["dataset"],
        events_path=(data_dir / cfg["events_file"]).resolve(),
        exercises_path=(data_dir / cfg["exercises_file"]).resolve(),
        exercise_columns=ex_cols, event_columns=ev_cols,
        exercise={k: v for k, v in exercise.items() if k != "columns"},
        event={k: v for k, v in event.items() if k != "columns"},
    )
