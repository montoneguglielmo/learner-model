"""
Turn each event (one row of the MIAAM interactions table) into TWO short texts:
  event text   what the event is (exercise context, attempt, time, ...)
  result text  what the student answered (data_correct, data_answer)
so a model can use the event text of the event to predict, and the result text
only for past events.

Used by the event encoder (encoders/event_embed.py).
The result has exactly one row per event, in the same order as the output
embeddings: sorted by student, timestamp (then event_id).

Each text is one "column: value" line per column, in the order of the config, e.g.
  attempt_index: 2
  data_duration: 13.42
  source: v1.0

Columns (see columns.py):
  - the role columns (student, timestamp, outcome; see common/dataset_config.py) and the
    exercise id columns: always in the output, so training can order the events and
    find each event's exercise
  - keep_columns: other columns copied as they are into the output
  - embed_event_columns: written into the event text
  - embed_result_columns: written into the result text (never also in the event text).
    Empty list: no result text and no result embedding at all.
  - any other column: not used
  - empty values (null, NaN, ""): the line is skipped, unless include_nulls
"""

from dataclasses import asdict, dataclass, field

import polars as pl

from learner_model.common.dataset_config import Columns
from learner_model.preprocessing.columns import check_columns, unused_columns


@dataclass
class EventTextConfig:
    """What goes into the text. Saved in the manifest of the embedding dataset."""
    keep_columns: list = field(default_factory=list)  # besides the role and exercise id columns
    embed_event_columns: list = field(default_factory=list)
    embed_result_columns: list = field(default_factory=lambda: ["data_correct", "data_answer"])
    include_nulls: bool = False   # True: write "column: null" instead of skipping the line
    float_digits: int = 3         # floats are rounded to this many decimals
    max_value_chars: int = 300    # longer values are cut (long free text would dominate)

    def to_dict(self):
        return asdict(self)


KINDS = ("event", "result")  # one text and one embedding of each kind per event


def required_columns(roles: Columns, id_columns):
    """Columns that are always in the output, in this order."""
    cols = [roles.student_id, roles.timestamp, roles.outcome, *id_columns]
    if roles.event_id:
        cols.insert(0, roles.event_id)
    return list(dict.fromkeys(cols))


def output_columns(roles: Columns, id_columns, cfg):
    """Required columns, then the other keep columns."""
    req = required_columns(roles, id_columns)
    return req + [c for c in cfg.keep_columns if c not in req]


def text_columns(columns, cfg, roles=Columns(), id_columns=("exercise_id",)):
    """{"event": [...], "result": [...]}, without "result" when embed_result_columns is empty.
    Fails if the config lists do not match the table."""
    if not cfg.embed_event_columns:
        raise ValueError("embed_event_columns is empty")
    both = [c for c in cfg.embed_result_columns if c in cfg.embed_event_columns]
    if both:
        raise ValueError(f"Columns {both} are in both embed_event_columns and embed_result_columns")
    check_columns(columns, output_columns(roles, id_columns, cfg),
                  cfg.embed_event_columns + cfg.embed_result_columns)
    cols = {"event": list(cfg.embed_event_columns), "result": list(cfg.embed_result_columns)}
    return {k: v for k, v in cols.items() if v}


def not_used(columns, cfg, roles=Columns(), id_columns=("exercise_id",)):
    """Columns of the table that are neither kept nor embedded."""
    return unused_columns(columns, output_columns(roles, id_columns, cfg),
                          cfg.embed_event_columns + cfg.embed_result_columns)


def _text(cols, schema, cfg):
    lines = [pl.concat_str(pl.lit(f"{c}: "), _value(c, schema[c], cfg)) for c in cols]
    text = pl.concat_str(lines, separator="\n", ignore_nulls=True) if lines else pl.lit("")
    return text.fill_null("")


def _value(name, dtype, cfg):
    """Expression giving the value of one column as a string (null = no line)."""
    c = pl.col(name)
    if dtype.is_float():
        s = pl.when(c.is_nan()).then(None).otherwise(c.round(cfg.float_digits)).cast(pl.String)
    elif isinstance(dtype, pl.List | pl.Array):
        s = c.cast(pl.List(dtype.inner)).list.eval(pl.element().cast(pl.String)).list.join(", ")
    elif isinstance(dtype, pl.Datetime):
        s = c.dt.to_string("%Y-%m-%d %H:%M:%S")  # to the second: microseconds only add noise
    elif isinstance(dtype, pl.Struct):
        s = c.struct.json_encode()
    else:  # strings, categoricals, integers, booleans, dates, ...
        s = c.cast(pl.String)
    s = s.str.replace_all(r"\s+", " ").str.strip_chars()
    s = pl.when(s.str.len_chars() > 0).then(s.str.slice(0, cfg.max_value_chars))
    if cfg.include_nulls:
        s = s.fill_null("null")
    return s


def build_event_texts(events: pl.DataFrame, cfg: EventTextConfig, roles=Columns(),
                      id_columns=("exercise_id",)) -> pl.DataFrame:
    """
    One row per event, sorted by student, timestamp, event_id, with columns:
      event_id    roles.event_id copied, or the row number in the input table
                  (student + timestamp may not be unique)
      <role columns, exercise id columns, other keep columns>, event_text, result_text (if any)
    The role and id columns keep the dataset's own names.
    """
    cols = text_columns(events.columns, cfg, roles, id_columns)
    out_cols = [c for c in output_columns(roles, id_columns, cfg) if c != "event_id"]
    if roles.event_id:
        if events[roles.event_id].is_duplicated().any():
            raise ValueError(f"event_id column '{roles.event_id}' has duplicated values")
        events = events.with_columns(pl.col(roles.event_id).alias("event_id"))
    else:
        if "event_id" in events.columns:
            raise ValueError("The events table already has an 'event_id' column: "
                             "set columns.event_id: event_id in the dataset config")
        events = events.with_row_index("event_id")
    for role in ("student_id", "timestamp", "outcome"):
        n = events[getattr(roles, role)].null_count()
        if n:
            raise ValueError(f"{n} events have an empty {role} ({getattr(roles, role)})")
    return (
        events
        .select("event_id", *out_cols,
                *[_text(cols[k], events.schema, cfg).alias(f"{k}_text") for k in cols])
        .sort([roles.student_id, roles.timestamp, "event_id"])
    )
