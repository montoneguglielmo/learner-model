# learner-model

Predicting student answers from rich interaction data (MIAAM V1, MIAAM V2, later FoundationalASSIST).

## Datasets

Each dataset is processed on its own, from its original files:

- one **exercise table** (one or more rows per exercise)
- one **events table** (one row per attempt)

and described by one config file, `configs/encoders/<dataset>.yaml`:

| Key | Meaning |
| --- | --- |
| `dataset` | name used in the output folder names (`miaam_v1`, `miaam_v2`) |
| `data_dir`, `events_file`, `exercises_file` | where the original files are (relative to the repo root) |
| `exercise.columns` | what the key column of the exercise table is called: `exercise_id`, one column or several that are unique together (MIAAM: `[source, exercise_id]`) |
| `exercise.format` | how an exercise becomes a text (`miaam`: parses the MIAAM `content` JSON, `preprocessing/miaam/exercise_text.py`) |
| `exercise.model`, `exercise.text` | the exercise embedding: model, columns kept, columns written into the text, and the settings of the format |
| `event.columns` | what the key columns of the events table are called: `student_id`, `timestamp` (orders the events, any encoding), `outcome` (boolean), `exercise_id` (matched in order with the exercise one), `event_id` (`null` = row number) |
| `event.model`, `event.text` | the event and outcome embeddings: model, columns kept and columns embedded |

The key columns are always kept, with the dataset's own names. Each encoder writes them into
its manifest, so training finds them there.

All datasets must use the same models, so their vectors live in the same space
(`tests/test_datasets.py` checks it). The columns written into the texts may differ.

## From a dataset to embeddings and a split

Three independent steps; each reads the original files and writes one dataset folder
(`datasets/<name>__<date>__<commit6>/`, never modified, with a `manifest.json`):

```
# check the texts first (no GPU)
uv run python scripts/preview_exercise_texts.py --config configs/encoders/miaam_v2.yaml
uv run python scripts/preview_event_texts.py    --config configs/encoders/miaam_v2.yaml

uv run python -m learner_model.preprocessing.stages.split            --config configs/encoders/miaam_v2.yaml
uv run python -m learner_model.preprocessing.encoders.exercise_embed --config configs/encoders/miaam_v2.yaml
uv run python -m learner_model.preprocessing.encoders.event_embed    --config configs/encoders/miaam_v2.yaml
```

Outputs: `split_students_miaam_v2__...`, `exercise_embed_miaam_v2__...`, `event_embed_miaam_v2__...`.
The training config (`configs/training/seq_transformer.yaml`) takes their three ids.

## Adding a dataset

1. Make sure it has an exercise table and an events table with: a student id, a timestamp,
   a boolean outcome, and the exercise id columns in both tables. Prepare the files if not.
2. Copy `configs/encoders/miaam_v2.yaml`, change `dataset`, the paths, the `columns` of `exercise`
   and `event`, and the text columns. Keep the `model` parts identical.
   If its exercises are not in the MIAAM format, write `preprocessing/<name>/exercise_text.py`
   (same functions as the MIAAM one), add it to `preprocessing/exercise_formats.py` and set
   `exercise.format: <name>`. The encoders do not change.
3. Preview the texts, then run the three steps above.

## Tests

```
uv run --with pytest pytest tests
```
