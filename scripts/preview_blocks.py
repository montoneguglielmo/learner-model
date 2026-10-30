"""Build a few blocks from the real embeddings and print them, to check them by eye.

Prints, for the first training block: who is in it, the token sequence (S x e o,
upper case = target) and the start of the attention mask ('#' = may attend).
Also times how long building and collating a batch takes.

Usage (from the repo root; no GPU needed):
  uv run python scripts/preview_blocks.py \
      --events-id event_embed_v1__2026-10-30__xxxxxx \
      --exercises-id exercise_embed_v1__2026-10-19__xxxxxx \
      --split-id split_students__2026-10-19__1a825a
"""

import argparse
import time

import numpy as np

from learner_model.common.splits import load_split
from learner_model.training.block_mask import render_mask
from learner_model.training.blocks import BlockConfig, BlockSampler, EventStore, batches, build_block, collate, render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events-id", required=True)
    p.add_argument("--exercises-id", required=True)
    p.add_argument("--split-id", required=True)
    p.add_argument("--root", default="datasets")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--mask-tokens", type=int, default=60, help="tokens shown in the mask picture")
    args = p.parse_args()

    t0 = time.time()
    store = EventStore.load(args.events_id, args.exercises_id, args.root)
    print(f"{store.n_events} events, {len(store.users)} students, loaded in {time.time() - t0:.0f}s; "
          f"{store.n_unknown_exercises} events have no exercise embedding")
    split = load_split(args.split_id, args.root)
    sampler = BlockSampler(store, BlockConfig(), split["train"])
    print(f"train: {len(sampler.train)} students, {len(sampler.targets)} can be targets "
          f"({len(sampler.targets) // sampler.cfg.n_targets + 1} blocks per epoch)")

    t0 = time.time()
    specs = next(batches(sampler.epoch(0), args.batch_size))
    batch = collate([build_block(store, w) for w in specs])
    print(f"one batch of {len(specs)} blocks built in {time.time() - t0:.2f}s, "
          f"tokens {batch['kind'].shape}, events {batch['ev'].shape}, "
          f"labels {(batch['label'] >= 0).sum()}")

    print("\nFirst block:")
    for w, u in zip(specs[0], batch["user_ids"][0]):
        print(f"  {'target ' if w.target else 'context'}  {u}  events {w.n} "
              f"(rows {w.first}..{w.first + w.n - 1} of {store.length[w.student]})")
    n = int((batch["kind"][0] >= 0).sum())
    print("\nTokens (S x e o, upper case = target):")
    print(render({k: batch[k][0][:n] for k in ("kind", "target")}, width=120))

    # a mask picture with one context student and one target, short windows
    small = BlockSampler(store, BlockConfig(n_targets=2, n_context=2, max_events=6), split["train"])
    b = collate([build_block(store, next(small.epoch(0)))], pad_to=1)
    print("\nMask of a small block (2 context + 2 targets, 6 events each), '#' = row may attend to column:")
    print(render_mask(b, max_tokens=args.mask_tokens))
    print(f"\nlabel rate in the first batch: {np.mean(batch['label'][batch['label'] >= 0]):.3f}")


if __name__ == "__main__":
    main()
