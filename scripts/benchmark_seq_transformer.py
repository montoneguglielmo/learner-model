"""Time each part of a training step of the sequence transformer, on the real data.

Prints, per part (mean over the timed steps, after a few warm-up steps):
  build     build_block + collate (CPU, numpy)
  to_gpu    numpy -> GPU tensors
  forward   embeddings + mask + layers (includes create_block_mask)
  mask      create_block_mask alone
  backward  loss.backward()
  optim     clip + optimizer step
and how many graphs torch.compile built (it should stop growing after warm-up:
growing = a recompile at every step).

Usage:
  uv run python scripts/benchmark_seq_transformer.py --config configs/training/seq_transformer.yaml
"""

import argparse
import time

import numpy as np
import torch
from torch._dynamo.utils import counters

from learner_model.common.splits import load_split
from learner_model.training.blocks import BlockSampler, EventStore, batches
from learner_model.training.models.seq_transformer import SeqTransformer, loss_fn, to_tensors
from learner_model.training.seq_transformer import Runner, load_config, make_optimizer


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--datasets-root", default="datasets")
    args = p.parse_args()

    cfg, block_cfg, model_cfg, tc = load_config(args.config)
    store = EventStore.load(cfg["events_id"], cfg["exercises_id"], args.datasets_root)
    split = load_split(cfg["split_id"], args.datasets_root)
    sampler = BlockSampler(store, block_cfg, split["train"])
    runner = Runner(store, block_cfg, "cuda")
    model = SeqTransformer(model_cfg, store.ex_dim, store.ev_dim, store.res_dim, "flex").cuda()
    opt = make_optimizer(model, tc["lr"], tc["weight_decay"])
    print(f"GPU: {torch.cuda.get_device_name()} | block length {runner.length} tokens | "
          f"batch {tc['batch_size']} blocks")

    def sync():
        torch.cuda.synchronize()
        return time.perf_counter()

    times = {k: [] for k in ("build", "to_gpu", "mask", "forward", "backward", "optim", "total")}
    graphs = []
    it = batches(sampler.epoch(0), tc["batch_size"])
    for step in range(args.warmup + args.steps):
        t0 = sync()
        batch = runner.batch(next(it))
        t1 = sync()
        b = to_tensors(batch, "cuda")
        t2 = sync()
        model._attend_fn(b)  # the mask alone (forward builds it again)
        t3 = sync()
        with runner.autocast:
            loss = loss_fn(model(b), b)
        t4 = sync()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t5 = sync()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        t6 = sync()
        graphs.append(counters["stats"]["unique_graphs"])
        label = "warm-up" if step < args.warmup else "timed"
        print(f"step {step:3d} ({label}): {t6 - t0:.3f} s | compiled graphs so far: {graphs[-1]}")
        if step >= args.warmup:
            for k, v in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t6 - t0)):
                times[k].append(v)

    print("\nmean over timed steps:")
    for k, v in times.items():
        print(f"  {k:9s} {np.mean(v):.3f} s")
    print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    grew = graphs[-1] - graphs[args.warmup - 1] if args.warmup else graphs[-1]
    print(f"compiled graphs added during the timed steps: {grew} (should be 0)")


if __name__ == "__main__":
    main()
