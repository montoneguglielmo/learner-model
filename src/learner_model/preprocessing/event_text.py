"""
Turn each event (one row of the MIAAM interactions table) into ONE short text.

Used by the event encoder (encoders/event_embed_v1.py).
The result has exactly one row per event, in the same order as the output
embeddings: sorted by user_id, created_at.

The text is one "column: value" line per text column, in the order of the config, e.g.
  attempt_index: 2
  data_duration: 13.42
  source: v1.0

Columns (every column of the table must be in one list, see columns.py):
  - keep columns: the output columns, copied as they are (must include user_id, created_at)
  - text columns: written into the text, in the order of the config
  - drop columns: not used
  - target columns (e.g. data_correct, data_answer): what a model must predict.
    Not used, and never in the text.
  - empty values (null, NaN, ""): the line is skipped, unless include_nulls
"""

from dataclasses import asdict, dataclass, field

import polars as pl

from learner_model.preprocessing.columns import check_columns

KEY = ["user_id", "created_at"]  # the output is sorted by these


@dataclass
class EventTextConfig:
    """What goes into the text. Saved in the manifest of the embedding dataset."""
    keep_columns: list = field(default_factory=lambda: KEY + ["exercise_id"])
    text_columns: list = field(default_factory=list)
    drop_columns: list = field(default_factory=list)
    target_columns: list = field(default_factory=lambda: ["data_correct", "data_answer"])
    include_nulls: bool = False   # True: write "column: null" instead of skipping the line
    float_digits: int = 3         # floats are rounded to this many decimals
    max_value_chars: int = 300    # longer values are cut (long free text would dominate)

    def to_dict(self):
        return asdict(self)


def columns_in_text(columns, cfg):
    """The columns that go into the text. Fails if the config lists do not match the table."""
    check_columns(columns, cfg.keep_columns, cfg.text_columns, cfg.drop_columns,
                  target=cfg.target_columns, required_keep=KEY)
    return list(cfg.text_columns)


def _value(name, dtype, cfg):
    """Expression giving the value of one column as a string (null = no line)."""
    c = pl.col(name)
    if dtype.is_float():
        s = pl.when(c.is_nan()).then(None).otherwise(c.round(cfg.float_digits)).cast(pl.String)
    elif isinstance(dtype, pl.List | pl.Array):
        s = c.cast(pl.List(dtype.inner)).list.eval(pl.element().cast(pl.String)).list.join(", ")
    elif isinstance(dtype, pl.Struct):
        s = c.struct.json_encode()
    else:  # strings, categoricals, integers, booleans, dates, ...
        s = c.cast(pl.String)
    s = s.str.replace_all(r"\s+", " ").str.strip_chars()
    s = pl.when(s.str.len_chars() > 0).then(s.str.slice(0, cfg.max_value_chars))
    if cfg.include_nulls:
        s = s.fill_null("null")
    return s


def build_event_texts(events: pl.DataFrame, cfg: EventTextConfig) -> pl.DataFrame:
    """
    One row per event, sorted by user_id, created_at, with columns:
      event_id    row number in the input table (user_id + created_at may not be unique)
      <keep columns> (user_id, created_at, ...), text
    """
    cols = columns_in_text(events.columns, cfg)
    lines = [pl.concat_str(pl.lit(f"{c}: "), _value(c, events.schema[c], cfg)) for c in cols]
    text = pl.concat_str(lines, separator="\n", ignore_nulls=True) if lines else pl.lit("")
    return (
        events.with_row_index("event_id")
        .select("event_id", *cfg.keep_columns, text.fill_null("").alias("text"))
        .sort([*KEY, "event_id"])
    )
