"""
Turn each exercise of the MIAAM exercise table into ONE clean text.

Used by the exercise encoder (and later by encoders that combine image + text).
The result has exactly one row per exercise_id.

Columns (see columns.py):
  - keep columns: copied into the output (must include exercise_id)
  - text columns: written into the text, in the order of the config.
    "content" is parsed (see below); any other column becomes "<label> : <value>",
    with a French label for the known columns (LABELS) and the column name otherwise.
  - any other column: not used

What it handles:
  - two content structures: flat (instruction, question, ...) and
    MULTI_GAMEPLAY (screens[] -> steps[])
  - content written as a Python dict instead of JSON (315 INPUT_LINE_GLOBAL)
  - text repeated twice ("Affirmation 1Selon ... Affirmation 1Selon ...")
  - missing spaces ("Affirmation 1Selon", "problèmes.Question")
  - HTML tags and entities
  - exercises listed several times with different module/objective/activity:
    all their distinct names are kept, joined with " ; "

Example output:
  Module : Organisation et gestion de données, fonctions
  Objectif : Lire et interpréter différents types de graphiques statistiques
  Activité : Lire et interpréter un diagramme circulaire
  Type d'exercice : MULTI_GAMEPLAY
  Consigne : Observe ce graphique. Puis indique si les affirmations suivantes sont vraies ou fausses.
  Étape 1 : Affirmation 1. Selon ce graphique, un quart des personnes ...
    Question : Choisir la réponse parmi ces options : Vrai, Faux.
    Réponse attendue : Vrai
    Explication : 25 %, soit un quart des personnes ...
"""

import ast
import html
import json
import re
from dataclasses import asdict, dataclass, field

import polars as pl

from learner_model.preprocessing.columns import check_columns, unused_columns

LABELS = {
    "module_name": "Module",
    "objective_name": "Objectif",
    "activity_name": "Activité",
    "objective_pedagogical_intent": "Intention pédagogique (objectif)",
    "activity_pedagogical_intent": "Intention pédagogique (activité)",
    "gameplay_type": "Type d'exercice",
}
CONTENT = "content"  # the column with the exercise itself (JSON)
TRIVIAL_FEEDBACK = {"bravo !", "bravo!", "bravo", "bravo !!"}
# Generic "wrong answer" messages: they say nothing about the exercise.
# Removed when they are the whole feedback, or cut when they start it ("Dommage. N'oublie pas...").
GENERIC_INCORRECT = ["oups", "dommage", "ce n'est pas le bon résultat", "essaie encore", "raté"]
IMG_RE = re.compile(r"IMG:\s*(.*?)\.(?:png|jpe?g|svg|gif|webp)", re.I)
LATEX_RE = re.compile(r"\$([^$]+)\$")


@dataclass
class TextConfig:
    """What goes into the text. Saved in the manifest of the embedding dataset."""
    keep_columns: list = field(default_factory=lambda: ["exercise_id", "source", "gameplay_type"])
    text_columns: list = field(default_factory=lambda: [
        "module_name", "objective_name", "activity_name", "gameplay_type", CONTENT])
    # What is taken from the content (the exercise itself):
    include_correct_answer: bool = True
    include_feedback_incorrect: bool = True  # often explains the skill being tested
    include_feedback_correct: bool = False   # mostly "Bravo !"; when True, trivial ones are skipped

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------- cleaning

def clean(text):
    """Normalize one piece of text. Returns "" for empty / None."""
    if text is None:
        return ""
    text = html.unescape(str(text))
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)                       # other HTML tags
    text = IMG_RE.sub(lambda m: f"[image : {m.group(1).rstrip('/').split('/')[-1].strip()}]", text)
    text = LATEX_RE.sub(lambda m: _latex(m.group(1)), text)
    text = re.sub(r"\s+", " ", text).strip()
    text = _remove_repetition(text)
    # Missing space between a number and a new capitalized word: "Affirmation 1Selon"
    text = re.sub(r"(\d)([A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ])", r"\1. \2", text)
    # Missing space after end of sentence: "problèmes.Question", "2.Complète", "B ?Tu"
    text = re.sub(r"([.!?])([A-ZÀ-ÖØ-Þ])", r"\1 \2", text)
    return text


def _latex(s):
    """'\\large{7~×~(-7)}' -> '7 × (-7)', '\\Large\\frac{108}{9}' -> '108/9'."""
    s = re.sub(r"\\(Large|large|LARGE|huge|Huge|small|normalsize|displaystyle|textstyle|text|mathrm)\b", "", s)
    s = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", s)
    s = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√\1", s)
    for cmd, sym in [(r"\times", "×"), (r"\div", "÷"), (r"\cdot", "·"), (r"\leq", "≤"),
                     (r"\geq", "≥"), (r"\neq", "≠"), (r"\approx", "≈"), (r"\%", "%")]:
        s = s.replace(cmd, sym)
    s = re.sub(r"\\[,;: ]", " ", s)
    s = s.replace("~", " ").replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()


def _drop_generic(feedback):
    """Remove generic 'wrong answer' starts. Returns '' if nothing useful is left."""
    text = feedback
    while True:
        norm = text.lower().replace("’", "'")
        for g in GENERIC_INCORRECT:
            if norm.startswith(g):
                rest = text[len(g):].lstrip(" .!…,;:")
                if len(rest) < len(text):
                    text = rest
                    break
        else:
            return text


def _remove_repetition(text):
    """'X X' or 'XX' (the same text twice) -> 'X'."""
    n = len(text)
    if n >= 20:
        half = n // 2
        if n % 2 == 1 and text[half] == " " and text[:half] == text[half + 1:]:
            return text[:half]
        if n % 2 == 0 and text[:half] == text[half:]:
            return text[:half]
    return text


def parse_content(raw):
    """Return (dict or None, status). status: 'json', 'python' or 'failed'."""
    try:
        return json.loads(raw), "json"
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, dict):
            return obj, "python"
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    return None, "failed"


# ---------------------------------------------------------- content -> lines

def _item_lines(item, cfg, indent=""):
    """Lines for one flat exercise or one step (they have the same fields)."""
    lines = []
    q = clean(item.get("question"))
    if q:
        lines.append(f"{indent}Question : {q}")
    if cfg.include_correct_answer:
        a = clean(item.get("correct_answer"))
        if a:
            lines.append(f"{indent}Réponse attendue : {a}")
    fb = item.get("feedback") or {}
    if isinstance(fb, dict):
        if cfg.include_feedback_correct:
            c = clean(fb.get("correct"))
            if c and c.lower() not in TRIVIAL_FEEDBACK:
                lines.append(f"{indent}Message si correct : {c}")
        if cfg.include_feedback_incorrect:
            i = _drop_generic(clean(fb.get("incorrect")))
            if i:
                lines.append(f"{indent}Explication : {i}")
    return lines


def content_lines(obj, cfg):
    """Lines describing the exercise content, for both structures."""
    if "screens" in obj:
        lines = []
        screens = obj.get("screens") or []
        step_no = 0
        last_instruction = None
        for s_no, screen in enumerate(screens, 1):
            instr = clean(screen.get("instruction"))
            if instr and instr != last_instruction:
                label = "Consigne" if len(screens) == 1 else f"Écran {s_no}"
                lines.append(f"{label} : {instr}")
                last_instruction = instr
            for step in screen.get("steps") or []:
                step_no += 1
                s_instr = clean(step.get("instruction"))
                head = f"Étape {step_no}"
                lines.append(f"{head} : {s_instr}" if s_instr and s_instr != instr else f"{head} :")
                lines.extend(_item_lines(step, cfg, indent="  "))
        return lines

    lines = []
    instr = clean(obj.get("instruction"))
    if instr:
        lines.append(f"Consigne : {instr}")
    lines.extend(_item_lines(obj, cfg))
    return lines


# ------------------------------------------------------------ main function

def not_used(columns, cfg):
    """Columns of the table that are neither kept nor in the text."""
    return unused_columns(columns, cfg.keep_columns, cfg.text_columns)


def build_texts(exercises: pl.DataFrame, cfg: TextConfig) -> pl.DataFrame:
    """
    One row per exercise_id, with columns:
      <keep columns> (exercise_id, ...), parse_status, text
    An exercise listed several times keeps the first value of each keep column,
    and all the distinct values of each text column, joined with " ; ".
    parse_status: 'json', 'python', 'failed', or None when content is not a text column.
    """
    check_columns(exercises.columns, cfg.keep_columns, cfg.text_columns, required_keep=["exercise_id"])
    value_cols = [c for c in cfg.text_columns if c != CONTENT]
    grouped = (
        exercises.group_by("exercise_id", maintain_order=True)
        .agg(
            *[pl.col(c).first() for c in cfg.keep_columns if c != "exercise_id"],
            *([pl.col(CONTENT).unique(maintain_order=True).alias("_content")]
              if CONTENT in cfg.text_columns else []),
            *[pl.col(c).cast(pl.String).drop_nulls().unique(maintain_order=True).alias(f"_text_{c}")
              for c in value_cols],
        )
    )

    rows = []
    for r in grouped.iter_rows(named=True):
        lines, status = [], None
        for col in cfg.text_columns:
            if col == CONTENT:
                if len(r["_content"]) > 1:
                    raise ValueError(f"exercise {r['exercise_id']} has {len(r['_content'])} different contents")
                obj, status = parse_content(r["_content"][0])
                if obj is not None:
                    lines.extend(content_lines(obj, cfg))
                else:  # last resort: the raw content, cleaned
                    lines.append(f"Contenu : {clean(r['_content'][0])}")
            else:
                names = [clean(x) for x in r[f"_text_{col}"] if clean(x)]
                if names:
                    lines.append(f"{LABELS.get(col, col)} : {' ; '.join(names)}")

        rows.append({
            **{c: r[c] for c in cfg.keep_columns},
            "parse_status": status,
            "text": "\n".join(lines),
        })
    return pl.DataFrame(rows)
