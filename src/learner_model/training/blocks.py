"""
Blocks of students for the sequence transformer (plan: "multi-student transformer
for answer prediction"). Data only: no model, no torch here.

A block is one input sequence: n_context CONTEXT students (only read) and n_targets
TARGET students (their answers are predicted). Each student gives a window of at most
max_events consecutive events, ordered by created_at (event_id only breaks ties),
written as tokens:

    [SEP] ex_1 ev_1 out_1 ex_2 ev_2 out_2 ...

    ex   exercise embedding (exercise_embed_v1, joined on exercise_id)
    ev   event embedding (event_embed_v1: what the event is, nothing from the answer)
    out  result embedding (event_embed_v1: data_correct, data_answer, data_duration)

Context students come first, then targets. Who may attend to whom is decided by the
mask (block_mask.py), not by this order. The label of event t of a target sits on its
ev_t token; every other token has label -1 (no loss).

Windows:
  - training: a random window of max_events consecutive events, for targets and context,
    drawn again every epoch (long histories are fully used over the epochs)
  - evaluation: the evaluated student's last max_events events; context students get a
    random window, from one fixed draw (fixed seed)

Usage:
    store = EventStore.load(events_id, exercises_id)
    sampler = BlockSampler(store, BlockConfig(), train_students)
    for specs in batches(sampler.epoch(epoch), batch_size=16):
        batch = collate([build_block(store, s) for s in specs])
"""

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from learner_model.common.datasets import open_dataset
from learner_model.common.embeddings import load_exercise_embeddings

# token kinds
PAD, SEP, EX, EV, OUT = -1, 0, 1, 2, 3
KIND_NAMES = {PAD: ".", SEP: "S", EX: "x", EV: "e", OUT: "o"}

ORDER = ["user_id", "created_at", "event_id"]  # the order of the event embeddings


@dataclass
class BlockConfig:
    n_targets: int = 3     # students whose answers get a loss
    n_context: int = 12    # students that are only read
    max_events: int = 64   # events kept per student (one window)
    min_events: int = 2    # shorter students are never targets
    seed: int = 0

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------- all events in memory

class EventStore:
    """Every event of every student, in memory, sorted by user_id, created_at, event_id.

    Per event (row i):
      ev[i], res[i]   event / result embedding, float16
      ex_row[i]       row of its exercise in ex_matrix, -1 if the exercise has no embedding
      correct[i]      data_correct as 0/1
      log_dt[i]       log(1 + seconds since the same student's previous event), -1 for the first
    Per student: rows start[k] .. start[k] + length[k] - 1, student id users[k].
    """

    def __init__(self, events: pl.DataFrame, exercise_ids, exercise_matrix):
        """events: event_id, user_id, created_at, exercise_id, data_correct,
        event_embedding, result_embedding (as written by event_embed_v1)."""
        missing = [c for c in ORDER + ["exercise_id", "data_correct", "event_embedding", "result_embedding"]
                   if c not in events.columns]
        if missing:
            raise ValueError(f"Event table has no column {missing}. data_correct must be in the "
                             f"keep_columns of the event encoder config.")
        if events["data_correct"].null_count():
            raise ValueError(f"{events['data_correct'].null_count()} events have no data_correct")

        keys = events.select(ORDER)
        if not keys.equals(keys.sort(ORDER)):  # the encoder writes them sorted: normally skipped
            events = events.sort(ORDER)
        self.n_events = events.height
        self.ev = _matrix(events["event_embedding"])
        self.res = _matrix(events["result_embedding"])
        self.correct = events["data_correct"].cast(pl.Int8).to_numpy()

        ex_pos = {e: i for i, e in enumerate(exercise_ids)}
        self.ex_matrix = np.asarray(exercise_matrix, dtype=np.float16)
        self.ex_row = np.array([ex_pos.get(e, -1) for e in events["exercise_id"].to_list()], dtype=np.int32)
        self.n_unknown_exercises = int((self.ex_row < 0).sum())

        keys = events.select(
            "user_id",
            log_dt=(pl.col("created_at") - pl.col("created_at").shift(1)).over("user_id")
                   .dt.total_seconds().clip(lower_bound=0).log1p().fill_null(-1),
        )
        self.log_dt = keys["log_dt"].cast(pl.Float32).to_numpy()
        runs = keys.select(pl.col("user_id").rle()).unnest("user_id")  # len, value, in order
        self.users = runs["value"].to_list()
        self.length = runs["len"].to_numpy().astype(np.int64)
        self.start = np.concatenate([[0], np.cumsum(self.length)[:-1]]).astype(np.int64)
        self.index = {u: k for k, u in enumerate(self.users)}

        self.ev_dim, self.res_dim, self.ex_dim = self.ev.shape[1], self.res.shape[1], self.ex_matrix.shape[1]

    @classmethod
    def load(cls, events_id, exercises_id, root="datasets"):
        """From the dataset folders made by event_embed_v1 and exercise_embed_v1."""
        path = open_dataset(events_id, root) / "embeddings.parquet"
        cols = ORDER + ["exercise_id", "data_correct", "event_embedding", "result_embedding"]
        events = pl.read_parquet(path, columns=cols)
        ex_ids, ex_matrix = load_exercise_embeddings(exercises_id, root)
        return cls(events, ex_ids, ex_matrix)

    def students(self, user_ids):
        """Indices (in this store) of the given students; students without events are skipped."""
        return np.array(sorted(self.index[u] for u in user_ids if u in self.index), dtype=np.int64)


def _matrix(col):
    """Array column -> (N, dim) float16, without a float32 copy of everything at once."""
    out = np.empty((len(col), col.dtype.size), dtype=np.float16)
    step = 1_000_000
    for i in range(0, len(col), step):
        out[i:i + step] = col.slice(i, step).to_numpy()
    return out


# ---------------------------------------------------------------- which students, which windows

@dataclass
class StudentWindow:
    student: int   # index in the EventStore
    first: int     # first event row (absolute row in the store)
    n: int         # number of events
    target: bool


class BlockSampler:
    def __init__(self, store: EventStore, cfg: BlockConfig, train_users):
        self.store, self.cfg = store, cfg
        self.train = store.students(train_users)
        self.targets = self.train[store.length[self.train] >= cfg.min_events]
        if len(self.train) < cfg.n_targets + cfg.n_context:
            raise ValueError(f"{len(self.train)} train students, a block needs "
                             f"{cfg.n_targets + cfg.n_context}")

    def _window(self, k, rng, target, last=False):
        n_all = int(self.store.length[k])
        n = min(n_all, self.cfg.max_events)
        offset = n_all - n if last else int(rng.integers(0, n_all - n + 1))
        return StudentWindow(int(k), int(self.store.start[k]) + offset, n, target)

    def _context(self, exclude, rng):
        pool = np.setdiff1d(self.train, exclude, assume_unique=True)
        chosen = rng.choice(pool, size=self.cfg.n_context, replace=False)
        return [self._window(k, rng, target=False) for k in chosen]

    def epoch(self, epoch):
        """Training blocks of one epoch: every train student (with at least min_events
        events) is a target exactly once, n_targets per block (the last block may have fewer)."""
        rng = np.random.default_rng([self.cfg.seed, epoch])
        order = rng.permutation(self.targets)
        for i in range(0, len(order), self.cfg.n_targets):
            tg = order[i:i + self.cfg.n_targets]
            yield self._context(tg, rng) + [self._window(k, rng, target=True) for k in tg]

    def evaluation(self, eval_users, seed=None):
        """Evaluation blocks: the given (val or test) students as targets, on their last
        max_events events; context students from the train set. Same seed = same blocks."""
        rng = np.random.default_rng([self.cfg.seed if seed is None else seed, 10**6])
        ks = self.store.students(eval_users)
        ks = ks[self.store.length[ks] >= self.cfg.min_events]
        for i in range(0, len(ks), self.cfg.n_targets):
            tg = ks[i:i + self.cfg.n_targets]
            # context comes from the train set, so it can never contain an evaluated student
            yield self._context(np.array([], dtype=np.int64), rng) + \
                [self._window(k, rng, target=True, last=True) for k in tg]


# ---------------------------------------------------------------- one block as arrays

def build_block(store: EventStore, windows):
    """One block as numpy arrays.

    Token arrays, length L = sum over students of (1 + 3 * n):
      kind       SEP / EX / EV / OUT
      student    index of the student in the block (0 .. S-1)
      target     True for tokens of target students
      pos        event index inside the student's window (0 .. max_events-1); 0 for SEP
      event      index into the event arrays below; -1 for SEP
      label      data_correct on the EV tokens of targets, -1 everywhere else
    Event arrays, length E = sum of n:
      ex, ex_known, ev, res, correct, log_dt
    Plus: user_ids (one per student), rows (store row of each event, for predictions).
    """
    n = np.array([w.n for w in windows])
    rows = np.concatenate([np.arange(w.first, w.first + w.n) for w in windows])
    E, L = len(rows), int((1 + 3 * n).sum())

    kind = np.empty(L, np.int8)
    student = np.empty(L, np.int16)
    target = np.empty(L, bool)
    pos = np.empty(L, np.int16)
    event = np.empty(L, np.int32)
    t = e = 0
    for s, w in enumerate(windows):
        m = 1 + 3 * w.n
        student[t:t + m], target[t:t + m] = s, w.target
        kind[t], pos[t], event[t] = SEP, 0, -1
        kind[t + 1:t + m] = np.tile([EX, EV, OUT], w.n)
        pos[t + 1:t + m] = np.repeat(np.arange(w.n), 3)
        event[t + 1:t + m] = np.repeat(np.arange(e, e + w.n), 3)
        t, e = t + m, e + w.n

    correct = store.correct[rows]
    label = np.full(L, -1, np.int8)
    is_label = target & (kind == EV)
    label[is_label] = correct[event[is_label]]

    ex_row = store.ex_row[rows]
    ex = np.zeros((E, store.ex_dim), np.float16)
    ex[ex_row >= 0] = store.ex_matrix[ex_row[ex_row >= 0]]

    return {
        "kind": kind, "student": student, "target": target, "pos": pos, "event": event, "label": label,
        "ex": ex, "ex_known": ex_row >= 0, "ev": store.ev[rows], "res": store.res[rows],
        "correct": correct, "log_dt": store.log_dt[rows],
        "user_ids": [store.users[w.student] for w in windows], "rows": rows,
    }


def batches(blocks, batch_size):
    """Group an iterator of blocks (or block specs) into lists of batch_size."""
    batch = []
    for b in blocks:
        batch.append(b)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def collate(blocks, pad_to=128):
    """Pad blocks to one length (a multiple of pad_to, which FlexAttention likes) and stack.
    Padding tokens have kind PAD, student -1, event -1, label -1. Returns numpy arrays
    of shape (B, L) for tokens and (B, E, ...) for events; torch conversion is left to the model."""
    B = len(blocks)
    L = -(-max(len(b["kind"]) for b in blocks) // pad_to) * pad_to
    E = max(len(b["rows"]) for b in blocks)
    pads = {"kind": PAD, "student": -1, "target": False, "pos": 0, "event": -1, "label": -1}
    out = {k: np.full((B, L), v, dtype=blocks[0][k].dtype) for k, v in pads.items()}
    for k in ("ex", "ev", "res"):
        out[k] = np.zeros((B, E, blocks[0][k].shape[1]), np.float16)
    out["ex_known"] = np.zeros((B, E), bool)
    out["correct"] = np.zeros((B, E), np.int8)
    out["log_dt"] = np.full((B, E), -1, np.float32)
    out["rows"] = np.full((B, E), -1, np.int64)
    for i, b in enumerate(blocks):
        n_tok, n_ev = len(b["kind"]), len(b["rows"])
        for k in pads:
            out[k][i, :n_tok] = b[k]
        for k in ("ex", "ev", "res", "ex_known", "correct", "log_dt", "rows"):
            out[k][i, :n_ev] = b[k]
    out["user_ids"] = [b["user_ids"] for b in blocks]
    return out


def render(block, width=None):
    """The token sequence of a block as one line: S/x/e/o, upper case for targets."""
    s = "".join(KIND_NAMES[k].upper() if t and k != PAD else KIND_NAMES[k]
                for k, t in zip(block["kind"], block["target"]))
    return s if width is None else "\n".join(s[i:i + width] for i in range(0, len(s), width))
