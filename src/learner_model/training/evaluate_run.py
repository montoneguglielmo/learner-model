"""
Evaluate finished seq_transformer runs on another student set (usually test), from their
saved best model. Nothing is retrained.

    uv run python -m learner_model.training.evaluate_run runs/seq_transformer__2026-11-18__2e3f31* \
        --eval-set test

For each run folder it reads config.json (datasets, block settings, batch size) and model.pt,
and writes next to them:
    metrics_<eval_set>.json       same format as metrics.json
    predictions_<eval_set>.parquet

Runs that share the same datasets reuse the loaded data. Context students come from the train
set, with the same fixed draw as in training (block seed), as in the final evaluation of a run.
"""

import argparse
import json
from pathlib import Path

import torch

from learner_model.training.blocks import BlockConfig
from learner_model.training.models.seq_transformer import ModelConfig, SeqTransformer
from learner_model.training.multi import DatasetEntry, load_datasets
from learner_model.training.seq_transformer import evaluate, print_metrics


def evaluate_run(run_dir, eval_set, device="cuda", attention="flex", datasets_root="datasets", cache=None):
    """Metrics of a run's best model on eval_set; writes metrics_/predictions_<eval_set>."""
    run_dir = Path(run_dir)
    config = json.loads((run_dir / "config.json").read_text())
    entries = [DatasetEntry(**d) for d in config["datasets"]]
    key = json.dumps([e.to_dict() for e in entries], sort_keys=True)
    cache = {} if cache is None else cache
    if key not in cache:
        cache[key] = load_datasets(entries, datasets_root)
    stores, splits, fractions = cache[key]

    block_cfg = BlockConfig(**config["block"])
    saved = torch.load(run_dir / "model.pt", map_location=device)
    model_cfg = ModelConfig(**saved["model_config"])
    first = next(iter(stores.values()))
    model = SeqTransformer(model_cfg, first.ex_dim, first.ev_dim, first.res_dim, attention).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()

    pred, metrics = evaluate(model, stores, splits, block_cfg, eval_set, config["train"]["batch_size"],
                             device, fractions)
    pred.write_parquet(run_dir / f"predictions_{eval_set}.parquet")
    metrics = {"eval_set": eval_set, "best_epoch": saved.get("epoch"), **metrics}
    (run_dir / f"metrics_{eval_set}.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--eval-set", default="test", choices=["val", "test"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention", default="flex", choices=["flex", "dense"])
    p.add_argument("--datasets-root", default="datasets")
    args = p.parse_args()

    cache = {}
    for d in args.run_dirs:
        if not (Path(d) / "model.pt").exists():
            print(f"{d}: no model.pt, skipped")
            continue
        print(f"\n=== {d}")
        m = evaluate_run(d, args.eval_set, args.device, args.attention, args.datasets_root, cache)
        print(f"{args.eval_set}: {m['n_students']} students")
        print_metrics(m)


if __name__ == "__main__":
    main()
