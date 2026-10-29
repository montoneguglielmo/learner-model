"""
Check the column lists of an encoder config against the table.

Every column of the table must be listed exactly where it belongs:
  keep_columns    copied as they are into the output
  text columns    written into a text (a column can be both kept and in a text)
  drop_columns    not used
A column of the table that is listed nowhere is an error, so a new column
in the data is never used, or ignored, by surprise.
"""


def check_columns(columns, keep, text, drop, required_keep=()):
    columns = list(columns)
    listed = [*keep, *text, *drop]
    unknown = [c for c in dict.fromkeys(listed) if c not in columns]
    if unknown:
        raise ValueError(f"Columns {unknown} are in the config but not in the table. "
                         f"Available columns: {columns}")
    unlisted = [c for c in columns if c not in listed]
    if unlisted:
        raise ValueError(f"Columns {unlisted} are in the table but in no list of the config: "
                         f"add each one to the keep list, a text list or drop_columns")
    both = [c for c in drop if c in set(keep) | set(text)]
    if both:
        raise ValueError(f"Columns {both} are in drop_columns and also in a list of used columns")
    missing = [c for c in required_keep if c not in keep]
    if missing:
        raise ValueError(f"keep_columns must include {missing}")
