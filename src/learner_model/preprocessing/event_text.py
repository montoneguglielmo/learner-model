"""
Turn each event (one row of the MIAAM interactions table) into TWO short texts:
  event text   what the event is (exercise context, attempt, time, ...)
  result text  what the student answered (data_correct, data_answer)
so a model can use the event text of the event to predict, and the result text
only for past events.

Used by the event encoder (encoders/event_embed_v1.py).
The result has exactly one row per event, in the same order as the output
embeddings: sorted by user_id, created_at.

Each text is one "column: value" line per column, in the order of the config, e.g.
  attempt_index: 2
  data_duration: 13.42
  source: v1.0

Columns (every column of the table must be in one list, see columns.py):
  - keep_columns: the output columns, copied as they are (must include user_id, created_at)
  - embed_event_columns: written into the event text
  - embed_result_columns: written into the result text (never also in the event text)
  - drop_columns: not used
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
    embed_event_columns: list = field(default_factory=list)
    embed_result_columns: list = field(default_factory=lambda: ["data_correct", "data_answer"])
    drop_columns: list = field(default_factory=list)
    include_nulls: bool = False   # True: write "column: null" instead of skipping the line
    float_digits: int = 3         # floats are rounded to this many decimals
    max_value_chars: int = 300    # longer values are cut (long free text would dominate)

    def to_dict(self):
        return asdict(self)


KINDS = ("event", "result")  # one text and one embedding of each kind per event


def text_columns(columns, cfg):
    """{"event": [...], "result": [...]}. Fails if the config lists do not match the table."""
    both = [c for c in cfg.embed_result_columns if c in cfg.embed_event_columns]
    if both:
        raise ValueError(f"Columns {both} are in both embed_event_columns and embed_result_columns")
    check_columns(columns, cfg.keep_columns, cfg.embed_event_columns + cfg.embed_result_columns,
                  cfg.drop_columns, required_keep=KEY)
    return {"event": list(cfg.embed_event_columns), "result": list(cfg.embed_result_columns)}


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


def build_event_texts(events: pl.DataFrame, cfg: EventTextConfig) -> pl.DataFrame:
    """
    One row per event, sorted by user_id, created_at, with columns:
      event_id    row number in the input table (user_id + created_at may not be unique)
      <keep columns> (user_id, created_at, ...), event_text, result_text
    """
    cols = text_columns(events.columns, cfg)
    return (
        events.with_row_index("event_id")
        .select("event_id", *cfg.keep_columns,
                *[_text(cols[k], events.schema, cfg).alias(f"{k}_text") for k in KINDS])
        .sort([*KEY, "event_id"])
    )
