"""
Turn each exercise of the MIAAM exercise table into ONE clean text.

Used by the text encoders (and later by encoders that combine image + text).
The result has exactly one row per exercise_id.

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
from dataclasses import asdict, dataclass

import polars as pl

HIERARCHY = [  # (column, label)
    ("module_name", "Module"),
    ("objective_name", "Objectif"),
    ("activity_name", "Activité"),
]
INTENTS = [
    ("objective_pedagogical_intent", "Intention pédagogique (objectif)"),
    ("activity_pedagogical_intent", "Intention pédagogique (activité)"),
]
TRIVIAL_FEEDBACK = {"bravo !", "bravo!", "bravo", "bravo !!"}
# Generic "wrong answer" messages: they say nothing about the exercise.
# Removed when they are the whole feedback, or cut when they start it ("Dommage. N'oublie pas...").
GENERIC_INCORRECT = ["oups", "dommage", "ce n'est pas le bon résultat", "essaie encore", "raté"]
IMG_RE = re.compile(r"IMG:\s*(.*?)\.(?:png|jpe?g|svg|gif|webp)", re.I)
LATEX_RE = re.compile(r"\$([^$]+)\$")


@dataclass
class TextConfig:
    """What goes into the text. Saved in the manifest of the embedding dataset."""
    include_hierarchy: bool = True           # module, objective, activity names
    include_gameplay_type: bool = True
    include_correct_answer: bool = True
    include_feedback_incorrect: bool = True  # often explains the skill being tested
    include_feedback_correct: bool = False   # mostly "Bravo !"; when True, trivial ones are skipped
    include_pedagogical_intent: bool = False

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

def build_texts(exercises: pl.DataFrame, cfg: TextConfig) -> pl.DataFrame:
    """
    One row per exercise_id, with columns:
      exercise_id, source, gameplay_type, parse_status, text
    """
    name_cols = [c for c, _ in HIERARCHY + INTENTS]
    grouped = (
        exercises.group_by("exercise_id", maintain_order=True)
        .agg(
            pl.col("content").unique(maintain_order=True),
            pl.col("source").first(),
            pl.col("gameplay_type").first(),
            *[pl.col(c).drop_nulls().unique(maintain_order=True) for c in name_cols],
        )
    )

    rows = []
    for r in grouped.iter_rows(named=True):
        if len(r["content"]) > 1:
            raise ValueError(f"exercise {r['exercise_id']} has {len(r['content'])} different contents")
        obj, status = parse_content(r["content"][0])

        lines = []
        if cfg.include_hierarchy:
            for col, label in HIERARCHY:
                names = [clean(x) for x in r[col] if clean(x)]
                if names:
                    lines.append(f"{label} : {' ; '.join(names)}")
        if cfg.include_pedagogical_intent:
            for col, label in INTENTS:
                names = [clean(x) for x in r[col] if clean(x)]
                if names:
                    lines.append(f"{label} : {' ; '.join(names)}")
        if cfg.include_gameplay_type:
            lines.append(f"Type d'exercice : {r['gameplay_type']}")

        if obj is not None:
            lines.extend(content_lines(obj, cfg))
        else:  # last resort: the raw content, cleaned
            lines.append(f"Contenu : {clean(r['content'][0])}")

        rows.append({
            "exercise_id": r["exercise_id"],
            "source": r["source"],
            "gameplay_type": r["gameplay_type"],
            "parse_status": status,
            "text": "\n".join(lines),
        })
    return pl.DataFrame(rows)