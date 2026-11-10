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
| `id_columns` | what makes an exercise unique, in both tables (MIAAM: `[source, exercise_id]`) |
| `columns` | what the event columns are called: `student_id`, `timestamp` (orders the events, any encoding), `outcome` (boolean), `event_id` (`null` = row number) |
| `exercise` | model and text settings of the exercise embedding |
| `event` | model and text settings of the event and outcome embeddings |

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
2. Copy `configs/encoders/miaam_v2.yaml`, change `dataset`, the paths, `id_columns`, `columns`
   and the text columns. Keep the `model` parts identical.
3. Preview the texts, then run the three steps above.

## Tests

```
uv run --with pytest pytest tests
```
