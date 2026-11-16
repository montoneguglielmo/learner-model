"""
Training on several datasets at once (plan: "train the sequence transformer on several datasets").

The training config lists datasets; each one has its own embeddings and split:

    datasets:
      - events_id: event_embed_miaam_v2__...
        exercises_id: exercise_embed_miaam_v2__...
        split_id: split_students_miaam_v2__...
        sources: null          # null = all its sources
        source_column: source  # null = the whole dataset is one source
        train_fraction: 1.0    # share of its train students used (val and test stay whole)
      - ...

An old config (events_id, exercises_id, split_id at the top) is one dataset with all its sources.

A GROUP is one (dataset, source) pair. Every block (targets and context) comes from one group;
an epoch shuffles the blocks of all groups together, so batches mix sources in proportion to
their number of students.

    stores, splits = load_datasets(dataset_entries(cfg), root)
    sampler = MultiSampler(stores, splits, block_cfg, fractions)
    for specs in batches(sampler.epoch(epoch), batch_size):   # specs: list of GroupBlock
        ...
"""

from dataclasses import asdict, dataclass
from typing import NamedTuple

import numpy as np

from learner_model.common.embeddings import dataset_name
from learner_model.common.splits import load_split
from learner_model.training.blocks import BlockConfig, BlockSampler, EventStore


@dataclass
class DatasetEntry:
    events_id: str
    exercises_id: str
    split_id: str
    sources: list | None = None
    source_column: str | None = "source"
    train_fraction: float = 1.0

    def __post_init__(self):
        if not 0 < self.train_fraction <= 1:
            raise ValueError(f"train_fraction must be in (0, 1], got {self.train_fraction}")
        if self.sources is not None and not self.sources:
            raise ValueError("sources is empty: use null for all sources")

    def to_dict(self):
        return asdict(self)


def dataset_entries(cfg):
    """The datasets of a training config (new `datasets:` list, or the old three ids)."""
    if "datasets" in cfg:
        if any(k in cfg for k in ("events_id", "exercises_id", "split_id")):
            raise ValueError("Use either `datasets:` or events_id/exercises_id/split_id, not both")
        return [DatasetEntry(**d) for d in cfg["datasets"]]
    return [DatasetEntry(cfg["events_id"], cfg["exercises_id"], cfg["split_id"])]


def load_datasets(entries, root="datasets"):
    """({name: EventStore}, {name: split}, {name: train_fraction}), checked:
    unique dataset names, embeddings and split of the same dataset, same vector sizes,
    and no source selected from two datasets (V1 am and V2 am are the same students)."""
    stores, splits, fractions = {}, {}, {}
    for e in entries:
        store = EventStore.load(e.events_id, e.exercises_id, root, e.sources, e.source_column)
        name = store.name
        if name in stores:
            raise ValueError(f"Dataset '{name}' is listed twice")
        split_name = dataset_name(e.split_id, root, default=name)
        if split_name != name:
            raise ValueError(f"{e.split_id} is a split of '{split_name}', but {e.events_id} is '{name}'")
        stores[name], splits[name], fractions[name] = store, load_split(e.split_id, root), e.train_fraction
    check_compatible(stores)
    return stores, splits, fractions


def check_compatible(stores):
    first = next(iter(stores.values()))
    for name, s in stores.items():
        dims = (s.ex_dim, s.ev_dim, s.res_dim)
        if dims != (first.ex_dim, first.ev_dim, first.res_dim):
            raise ValueError(f"'{name}' has vector sizes (exercise, event, result) {dims}, '{first.name}' "
                             f"{(first.ex_dim, first.ev_dim, first.res_dim)}: use the same embedding models")
    seen = {}
    for name, s in stores.items():
        for src in sorted(set(s.sources)):
            if src in seen:
                raise ValueError(f"Source '{src}' is selected from both '{seen[src]}' and '{name}'. "
                                 f"Pick it from one dataset only (e.g. V1 am and V2 am are the same "
                                 f"students, split separately: using both would leak).")
            seen[src] = name


class GroupBlock(NamedTuple):
    group: int      # index in MultiSampler.groups
    windows: list   # StudentWindow list, as BlockSampler yields it


@dataclass
class Group:
    dataset: str
    source: str
    store: EventStore
    users: set              # every student of this group (all sets)
    sampler: BlockSampler   # on its (possibly reduced) train students

    @property
    def name(self):
        return f"{self.dataset}/{self.source}"


class MultiSampler:
    """One BlockSampler per (dataset, source); blocks never mix groups."""

    def __init__(self, stores, splits, cfg: BlockConfig, fractions=None):
        self.cfg, self.groups = cfg, []
        fractions = fractions or {}
        for name in stores:
            store, split = stores[name], splits[name]
            sources = np.array(store.sources)
            for src in sorted(set(store.sources)):
                users = {store.users[k] for k in np.flatnonzero(sources == src)}
                train = sorted(users & split["train"])
                frac = fractions.get(name, 1.0)
                if frac < 1:
                    rng = np.random.default_rng([cfg.seed, 7])
                    train = sorted(rng.choice(train, max(1, round(frac * len(train))), replace=False).tolist())
                try:
                    sampler = BlockSampler(store, cfg, train)
                except ValueError as err:
                    raise ValueError(f"{name}/{src}: {err}") from None
                self.groups.append(Group(name, src, store, users, sampler))

    @property
    def n_targets(self):
        """Train targets over all groups (each one is a target once per epoch)."""
        return sum(len(g.sampler.targets) for g in self.groups)

    def epoch(self, epoch):
        """Training blocks of one epoch, all groups shuffled together."""
        blocks = [GroupBlock(i, w) for i, g in enumerate(self.groups) for w in g.sampler.epoch(epoch)]
        order = np.random.default_rng([self.cfg.seed, epoch, 1]).permutation(len(blocks))
        return [blocks[i] for i in order]

    def evaluation(self, splits, eval_set, users=None):
        """Evaluation blocks of every group: its eval_set students as targets, its train
        students as context. users: optional subset of (dataset, user_id) pairs."""
        for i, g in enumerate(self.groups):
            ev = splits[g.dataset][eval_set] & g.users
            if users is not None:
                ev = {u for u in ev if (g.dataset, u) in users}
            for w in g.sampler.evaluation(sorted(ev)):
                yield GroupBlock(i, w)

    def summary(self):
        return [{"group": g.name, "students": len(g.users), "train": len(g.sampler.train),
                 "targets": len(g.sampler.targets)} for g in self.groups]
