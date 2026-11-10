"""
One config file per dataset (configs/encoders/<dataset>.yaml) describes:
  - where the original files are (one exercise table, one attempts/events table)
  - what the columns mean (student, timestamp, outcome, exercise id)
  - how to embed the exercises (exercise: model + text) and the events (event: model + text)

The split stage and the two encoders all read it, but each starts from the original files:
none of them reads another one's output.

    dataset: miaam_v2                       # name used in the output folder names
    data_dir: ../MIAAM-v2/data
    events_file: maths_data_filtered.parquet
    exercises_file: maths_exercises_table.parquet
    id_columns: [source, exercise_id]       # what makes an exercise unique (in both tables)
    columns:
      event_id: null                        # null = row number in the events file
      student_id: user_id
      timestamp: created_at                 # orders the events; any encoding
      outcome: data_correct                 # boolean
    exercise: {model: {...}, text: {...}}
    event:    {model: {...}, text: {...}}
"""

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

NAME_RE = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True)
class Columns:
    """What each role is called in this dataset's events table."""
    student_id: str = "user_id"
    timestamp: str = "created_at"
    outcome: str = "data_correct"
    event_id: str | None = None   # None: the row number in the events file

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig:
    path: Path              # the config file
    dataset: str
    events_path: Path
    exercises_path: Path
    id_columns: tuple
    columns: Columns
    exercise: dict          # {"model": ..., "text": ...}
    event: dict             # {"model": ..., "text": ...}

    def describe(self):
        """What goes into every manifest made from this dataset."""
        return {"dataset": self.dataset, "config_file": self.path.name,
                "events": str(self.events_path), "exercises": str(self.exercises_path),
                "id_columns": list(self.id_columns), "columns": self.columns.to_dict()}


def load_dataset_config(path) -> DatasetConfig:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    for key in ("dataset", "data_dir", "events_file", "exercises_file", "id_columns"):
        if key not in cfg:
            raise ValueError(f"{path}: missing '{key}'")
    if not NAME_RE.match(cfg["dataset"]):
        raise ValueError(f"{path}: dataset name '{cfg['dataset']}' must be lower case letters, digits, _")
    id_columns = cfg["id_columns"]
    if isinstance(id_columns, str):
        id_columns = [id_columns]
    if not id_columns:
        raise ValueError(f"{path}: id_columns is empty")
    columns = Columns(**(cfg.get("columns") or {}))
    clash = {columns.student_id, columns.timestamp, columns.outcome} & set(id_columns)
    if clash:
        raise ValueError(f"{path}: {sorted(clash)} is both an exercise id column and an event role")
    data_dir = Path(cfg["data_dir"])  # relative to where the command runs (the repo root)
    return DatasetConfig(
        path=path, dataset=cfg["dataset"],
        events_path=(data_dir / cfg["events_file"]).resolve(),
        exercises_path=(data_dir / cfg["exercises_file"]).resolve(),
        id_columns=tuple(id_columns), columns=columns,
        exercise=cfg.get("exercise") or {}, event=cfg.get("event") or {},
    )
