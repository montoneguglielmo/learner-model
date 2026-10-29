"""
Check the column lists of an encoder config against the table.

Every column of the table must be listed exactly where it belongs:
  keep_columns    copied as they are into the output
  text_columns    written into the text (a column can be both kept and in the text)
  drop_columns    not used
  target_columns  (events only) what a model must predict: not used, never in the text
A column of the table that is listed nowhere is an error, so a new column
in the data is never used, or ignored, by surprise.
"""


def check_columns(columns, keep, text, drop, target=(), required_keep=()):
    columns = list(columns)
    listed = [*keep, *text, *drop, *target]
    unknown = [c for c in dict.fromkeys(listed) if c not in columns]
    if unknown:
        raise ValueError(f"Columns {unknown} are in the config but not in the table. "
                         f"Available columns: {columns}")
    unlisted = [c for c in columns if c not in listed]
    if unlisted:
        raise ValueError(f"Columns {unlisted} are in the table but in no list of the config: "
                         f"add each one to keep_columns, text_columns or drop_columns")
    used = set(keep) | set(text)
    for name, cols in [("drop_columns", drop), ("target_columns", target)]:
        both = [c for c in cols if c in used]
        if both:
            raise ValueError(f"Columns {both} are in {name} and also in keep_columns or text_columns")
    both = [c for c in drop if c in target]
    if both:
        raise ValueError(f"Columns {both} are in both drop_columns and target_columns")
    missing = [c for c in required_keep if c not in keep]
    if missing:
        raise ValueError(f"keep_columns must include {missing}")
