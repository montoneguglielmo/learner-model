"""Train and evaluate the sequence transformer (plan: "multi-student transformer for answer prediction").

Each training block = n_targets target students + n_context context students, all from the
train set (blocks.py). The loss is the binary cross-entropy at every event of every target.
After each epoch the model is scored on val blocks (val students as targets, train students
as context, their last max_events events); the best epoch (val AUC) is kept, and training
stops after `patience` epochs without improvement.

At the end the best model predicts every event in the last max_events events of each student
of --eval-set (default val; keep test for the final run), and is compared with the smoothed
success-rate baseline on exactly the same events. Metrics count events with at least one
earlier event of the same student (as in tabular_per_student), overall and by history length.

Usage (from the repo root, code committed):
  # quick check that everything runs: 2 short epochs
  uv run python -m learner_model.training.seq_transformer \
      --config configs/training/seq_transformer.yaml --max-steps-per-epoch 50 --max-epochs 2

  # the real run
  uv run python -m learner_model.training.seq_transformer \
      --config configs/training/seq_transformer.yaml --wandb-project learner-model

Output:
  runs/seq_transformer__<date>__<commit6>/
    config.json          all settings + dataset ids + git commit + library versions
    model.pt             best model (state_dict + model config)
    history.json         train loss and val metrics per epoch
    predictions.parquet  one row per prediction: user_id, event_id, n_past, y, p_model, p_baseline
    metrics.json         AUC / log-loss / accuracy / accuracy_1 / accuracy_0, overall and by
                         history length, model and baseline (written last)
"""

import argparse
import importlib.metadata as md
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
import yaml
from tqdm import tqdm

from learner_model.common.datasets import git_info
from learner_model.common.splits import load_split
from learner_model.training.blocks import (BlockConfig, BlockSampler, EventStore, batches, build_block,
                                           collate, max_tokens)
from learner_model.training.models.seq_transformer import ModelConfig, SeqTransformer, loss_fn, to_tensors
from learner_model.training.tabular_per_student import HISTORY_BINS, bin_label, in_bin, score

PIPELINE = "seq_transformer"

TRAIN_DEFAULTS = {"batch_size": 16, "lr": 3e-4, "weight_decay": 0.01, "warmup_steps": 1000,
                  "max_epochs": 20, "patience": 5, "grad_clip": 1.0, "log_every": 50,
                  "val_max_students": None, "seed": 0}


# ---------------------------------------------------------------- prediction and metrics

class Runner:
    """Builds fixed-size batches from block specs and runs the model on them."""

    def __init__(self, store, block_cfg, device):
        self.store, self.device = store, device
        self.length = max_tokens(block_cfg)
        self.n_events = (block_cfg.n_targets + block_cfg.n_context) * block_cfg.max_events
        self.autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))

    def batch(self, specs):
        blocks = [build_block(self.store, w) for w in specs]
        return collate(blocks, length=self.length, n_events=self.n_events)


@torch.no_grad()
def predict(model, runner, blocks, batch_size, desc="predict"):
    """One row per labelled token (EV token of a target)."""
    model.eval()
    blocks = list(blocks)
    store, cols = runner.store, {"rows": [], "p": []}
    for specs in tqdm(batches(blocks, batch_size), total=-(-len(blocks) // batch_size), desc=desc,
                      unit="batch", leave=False):
        batch = runner.batch(specs)
        b = to_tensors(batch, runner.device)
        with runner.autocast:
            p = torch.sigmoid(model(b).float()).cpu().numpy()
        has = batch["label"] >= 0
        cols["rows"].append(np.take_along_axis(batch["rows"], np.maximum(batch["event"], 0), 1)[has])
        cols["p"].append(p[has])
    rows = np.concatenate(cols["rows"])
    return pl.DataFrame({
        "user_id": [store.users[k] for k in np.searchsorted(store.start, rows, side="right") - 1],
        "event_id": store.event_id[rows], "n_past": store.n_past[rows],
        "y": store.correct[rows].astype(np.int32),
        "p_model": np.concatenate(cols["p"]), "p_baseline": store.p_baseline[rows],
    })


def compute_metrics(pred):
    """Events with at least one earlier event of the student, overall and by history length."""
    pred = pred.filter(pl.col("n_past") >= 1)
    out = {"model": score(pred["y"], pred["p_model"]), "baseline": score(pred["y"], pred["p_baseline"]),
           "n_students": pred["user_id"].n_unique(), "by_history": {}}
    for lo, hi in HISTORY_BINS:
        b = pred.filter(in_bin(pl.col("n_past"), lo, hi))
        out["by_history"][bin_label(lo, hi)] = {"model": score(b["y"], b["p_model"]),
                                                "baseline": score(b["y"], b["p_baseline"])}
    return out


def print_metrics(m):
    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    hdr = (f"{'history':>8} {'n':>8} {'AUC model':>10} {'AUC base':>9} {'logloss model':>14} "
           f"{'acc1 model':>11} {'acc0 model':>11} {'acc1 base':>10} {'acc0 base':>10}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for name, r in [("all", m)] + list(m["by_history"].items()):
        a, b = r["model"], r["baseline"]
        print(f"{name:>8} {a['n']:>8} {f(a.get('auc')):>10} {f(b.get('auc')):>9} {f(a.get('logloss')):>14} "
              f"{f(a.get('accuracy_1')):>11} {f(a.get('accuracy_0')):>11} "
              f"{f(b.get('accuracy_1')):>10} {f(b.get('accuracy_0')):>10}")


# ---------------------------------------------------------------- training

def lr_factor(step, warmup, total):
    """Linear warmup, then cosine decay to 0."""
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def make_optimizer(model, lr, weight_decay):
    """Weight decay on weight matrices only (not on biases, norms, embeddings, learned vectors)."""
    decay = [p for n, p in model.named_parameters() if p.ndim == 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.ndim == 2 and "emb" not in n)]
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                              {"params": other, "weight_decay": 0.0}], lr=lr)


def train(store, split, block_cfg, model_cfg, tc, run_dir, device="cuda", attention="flex",
          max_steps_per_epoch=None, wb=None):
    """Train, keep the best epoch in run_dir/model.pt, return (model, history)."""
    torch.manual_seed(tc["seed"])
    sampler = BlockSampler(store, block_cfg, split["train"])
    runner = Runner(store, block_cfg, device)
    model = SeqTransformer(model_cfg, store.ex_dim, store.ev_dim, store.res_dim, attention).to(device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    val_users = sorted(split["val"])
    if tc["val_max_students"] and len(val_users) > tc["val_max_students"]:
        rng = np.random.default_rng(tc["seed"])
        val_users = sorted(rng.choice(val_users, tc["val_max_students"], replace=False).tolist())
    val_blocks = list(sampler.evaluation(val_users))    # fixed: same blocks every epoch

    steps_per_epoch = -(-len(sampler.targets) // block_cfg.n_targets // tc["batch_size"])
    if max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, max_steps_per_epoch)
    total = steps_per_epoch * tc["max_epochs"]
    opt = make_optimizer(model, tc["lr"], tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, tc["warmup_steps"], total))
    print(f"{len(sampler.targets)} train targets, {steps_per_epoch} steps per epoch, "
          f"{len(val_users)} val students ({len(val_blocks)} blocks)")

    history, best_auc, bad_epochs, step = [], -1.0, 0, 0
    for epoch in range(tc["max_epochs"]):
        model.train()
        t0, losses = time.time(), []
        bar = tqdm(total=steps_per_epoch, desc=f"epoch {epoch}", unit="step", smoothing=0.1)
        for i, specs in enumerate(batches(sampler.epoch(epoch), tc["batch_size"])):
            if i >= steps_per_epoch:
                break
            b = to_tensors(runner.batch(specs), device)
            with runner.autocast:
                loss = loss_fn(model(b), b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
            opt.step()
            sched.step()
            step += 1
            losses.append(loss.item())
            bar.update(1)
            bar.set_postfix(loss=f"{np.mean(losses[-tc['log_every']:]):.4f}", lr=f"{sched.get_last_lr()[0]:.1e}")
            if step % tc["log_every"] == 0:
                if wb:
                    wb.log({"train/loss": float(np.mean(losses[-tc["log_every"]:])),
                            "train/lr": sched.get_last_lr()[0]}, step=step)

        bar.close()
        val = compute_metrics(predict(model, runner, val_blocks, tc["batch_size"], desc=f"val {epoch}"))
        auc = val["model"].get("auc") or 0.0
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val,
                        "minutes": round((time.time() - t0) / 60, 1)})
        (run_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch}: train loss {np.mean(losses):.4f} | val AUC {auc:.4f} "
              f"(baseline {val['baseline'].get('auc', float('nan')):.4f}) | "
              f"val logloss {val['model'].get('logloss', float('nan')):.4f}", flush=True)
        if wb:
            wb.log({"val/auc": auc, "val/logloss": val["model"].get("logloss"),
                    "val/auc_baseline": val["baseline"].get("auc"), "epoch": epoch}, step=step)

        if auc > best_auc:
            best_auc, bad_epochs = auc, 0
            torch.save({"state_dict": model.state_dict(), "model_config": model_cfg.to_dict(),
                        "epoch": epoch, "val_auc": auc}, run_dir / "model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= tc["patience"]:
                print(f"no improvement for {bad_epochs} epochs: stop")
                break

    best = torch.load(run_dir / "model.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    print(f"best epoch {best['epoch']} (val AUC {best['val_auc']:.4f})")
    return model, history


def evaluate(model, store, split, block_cfg, eval_set, batch_size, device):
    """Predictions for every student of eval_set (targets) with train students as context."""
    sampler = BlockSampler(store, block_cfg, split["train"])
    runner = Runner(store, block_cfg, device)
    pred = predict(model, runner, sampler.evaluation(sorted(split[eval_set])), batch_size, desc=eval_set)
    return pred, compute_metrics(pred)


# ---------------------------------------------------------------- main

def library_versions():
    out = {}
    for lib in ["torch", "polars", "numpy", "scikit-learn"]:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    return out


def new_run_dir(root, commit):
    base = f"{PIPELINE}__{datetime.now(timezone.utc):%Y-%m-%d}__{commit[:6]}"
    path, i = Path(root) / base, 2
    while path.exists():
        path, i = Path(root) / f"{base}__{i}", i + 1
    path.mkdir(parents=True)
    return path


def load_config(path, max_epochs=None):
    cfg = yaml.safe_load(Path(path).read_text())
    block_cfg = BlockConfig(**cfg.get("block", {}))
    model_cfg = ModelConfig(**cfg.get("model", {}))
    tc = {**TRAIN_DEFAULTS, **cfg.get("train", {})}
    if max_epochs:
        tc["max_epochs"] = max_epochs
    if model_cfg.max_events < block_cfg.max_events:
        raise ValueError("model.max_events must be >= block.max_events")
    return cfg, block_cfg, model_cfg, tc


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="e.g. configs/training/seq_transformer.yaml")
    p.add_argument("--eval-set", default="val", choices=["val", "test"],
                   help="final evaluation; compare settings on val, keep test for the end")
    p.add_argument("--max-epochs", type=int, default=None, help="override train.max_epochs (quick checks)")
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="quick checks only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--attention", default="flex", choices=["flex", "dense"])
    p.add_argument("--datasets-root", default="datasets")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    commit, clean = git_info()
    if not clean and not args.allow_dirty:
        raise SystemExit("Uncommitted changes: commit first (or use --allow-dirty for a test).")
    cfg, block_cfg, model_cfg, tc = load_config(args.config, args.max_epochs)

    t0 = time.time()
    store = EventStore.load(cfg["events_id"], cfg["exercises_id"], args.datasets_root)
    split = load_split(cfg["split_id"], args.datasets_root)
    print(f"{store.n_events} events, {len(store.users)} students, loaded in {time.time() - t0:.0f}s")

    run_dir = new_run_dir(args.runs_root, commit)
    config = {**vars(args), "config_file": cfg, "block": block_cfg.to_dict(), "model": model_cfg.to_dict(),
              "train": tc, "git_commit": commit, "git_clean": clean, "libraries": library_versions(),
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"run folder: {run_dir}")

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, job_type=PIPELINE, config=config, name=run_dir.name)

    model, _ = train(store, split, block_cfg, model_cfg, tc, run_dir, args.device, args.attention,
                     args.max_steps_per_epoch, wb)
    pred, metrics = evaluate(model, store, split, block_cfg, args.eval_set, tc["batch_size"], args.device)
    pred.write_parquet(run_dir / "predictions.parquet")
    (run_dir / "metrics.json").write_text(json.dumps({"eval_set": args.eval_set, **metrics}, indent=2))
    print(f"\n{args.eval_set}: {metrics['n_students']} students")
    print_metrics(metrics)

    if wb:
        wb.summary[f"{args.eval_set}/auc"] = metrics["model"].get("auc")
        wb.summary[f"{args.eval_set}/auc_baseline"] = metrics["baseline"].get("auc")
        wb.summary[f"{args.eval_set}/acc1"] = metrics["model"].get("accuracy_1")
        wb.summary[f"{args.eval_set}/acc0"] = metrics["model"].get("accuracy_0")
        wb.finish()


if __name__ == "__main__":
    main()
