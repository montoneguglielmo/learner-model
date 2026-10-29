"""
Check the column lists of an encoder config against the table.

  keep_columns    copied as they are into the output
  text columns    written into a text (a column can be both kept and in a text)
Any other column of the table is not used (unused_columns lists them).
"""


def check_columns(columns, keep, text, required_keep=()):
    columns = list(columns)
    unknown = [c for c in dict.fromkeys([*keep, *text]) if c not in columns]
    if unknown:
        raise ValueError(f"Columns {unknown} are in the config but not in the table. "
                         f"Available columns: {columns}")
    missing = [c for c in required_keep if c not in keep]
    if missing:
        raise ValueError(f"keep_columns must include {missing}")


def unused_columns(columns, keep, text):
    """Columns of the table that are neither kept nor in a text, in table order."""
    used = set(keep) | set(text)
    return [c for c in columns if c not in used]
