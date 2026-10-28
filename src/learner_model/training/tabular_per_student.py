"""Per-student in-context evaluation with TabICL ("option 1").

Words: an "interaction" is one row of the data (one answer to one exercise).
An "attempt" keeps its MIAAM meaning: one try on a given exercise (retries = attempts 2, 3...).

For each sampled student, every past interaction is one table row:
    features of the interaction  ->  correct / wrong
To predict interaction t, TabICL gets the student's rows before t as context
(at most --max-context rows, the most recent ones) and predicts row t.
Nothing is learned from other students.

Features of an interaction (only information known BEFORE the answer):
  - exercise embedding: first `dim` values of the text embedding, L2-normalized again
    (Qwen3-Embedding is Matryoshka-trained, so the first values are the most important)
  - number of past interactions, log time since the previous one, log time since the first one
  - result and log duration of the previous interaction
  - success rate so far
  - number of previous attempts on the same exercise, result of the last attempt on it
The duration of the interaction itself is NOT used (it is only known after the answer).

Baseline: success rate so far, smoothed: (correct_so_far + 1) / (interactions_so_far + 2).
It is also used when the context has only one class (all correct or all wrong):
TabICL would answer 1.0 or 0.0 there, which is too confident. These cases are counted.

Students are sampled from one split set (default: val; keep test for the final run),
in equal numbers from 4 bins of history length, so short histories are well represented.

Usage (from the repo root, code committed):
  # 1. speed test: 20 students, one size
  uv run python -m learner_model.training.tabular_per_student \
      --split-id split_students__2026-10-19__1a825a \
      --embeddings-id text_embed_v1__2026-10-19__97f653 \
      --n-students 20 --dims 64

  # 2. the real sweep (defaults: 300 val students, dims 32 64 128 256)
  uv run python -m learner_model.training.tabular_per_student \
      --split-id split_students__2026-10-19__1a825a \
      --embeddings-id text_embed_v1__2026-10-19__97f653 \
      --wandb-project learner-model

Output:
  runs/tabular_per_student__<date>__<commit6>/
    config.json          all settings + git commit + library versions
    predictions.parquet  one row per prediction: user_id, t, y, p_model, p_baseline, dim, ...
    metrics.json         AUC / log-loss / accuracy / accuracy_1 / accuracy_0, overall and by history length (written last)
"""

import argparse
import importlib.metadata as md
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from learner_model.common.datasets import git_info
from learner_model.common.embeddings import load_embeddings
from learner_model.common.splits import load_split

PIPELINE = "tabular_per_student"
KEY = ["user_id", "exercise_id", "created_at"]

# bins of the number of rows of a student (for sampling)
STUDENT_BINS = [(2, 10), (10, 50), (50, 200), (200, None)]
# bins of the number of past interactions at prediction time (for reporting)
HISTORY_BINS = [(1, 10), (10, 50), (50, 200), (200, None)]

FEATURES = [
    "f_n_past", "f_log_dt_prev", "f_log_dt_first", "f_prev_correct",
    "f_prev_log_duration", "f_past_rate", "f_ex_n_attempts", "f_ex_last_correct",
]


def bin_label(lo, hi):
    return f"{lo}+" if hi is None else f"{lo}-{hi - 1}"


def in_bin(x, lo, hi):
    return (x >= lo) if hi is None else ((x >= lo) & (x < hi))


# ---------------------------------------------------------------- data

def load_interactions(path, students, dedup):
    cols = KEY + ["attempt_index", "data_correct", "data_duration"]
    df = (pl.scan_parquet(path)
            .select(cols)
            .filter(pl.col("user_id").is_in(list(students)))
            .collect())
    df = df.sort(["user_id", "created_at", "attempt_index"])
    n0 = df.height
    if dedup:
        # same student + exercise + time = one attempt (duplicated logs in MIAAM v1.0)
        df = df.unique(subset=KEY, keep="first", maintain_order=True)
    return df, n0 - df.height


def add_features(df):
    """One row per interaction. Every feature uses only rows BEFORE the current one."""
    u, ue = "user_id", ["user_id", "exercise_id"]
    ts = pl.col("created_at")
    df = df.with_columns(
        y=pl.col("data_correct").cast(pl.Int32),
        pos=pl.int_range(pl.len()).over(u),
    )
    past_correct = pl.col("y").cum_sum().over(u) - pl.col("y")
    df = df.with_columns(
        f_n_past=pl.col("pos").cast(pl.Float32),
        f_log_dt_prev=(ts - ts.shift(1)).over(u).dt.total_seconds().clip(lower_bound=0).log1p(),
        f_log_dt_first=(ts - ts.first()).over(u).dt.total_seconds().clip(lower_bound=0).log1p(),
        f_prev_correct=pl.col("y").shift(1).over(u),
        f_prev_log_duration=pl.col("data_duration").shift(1).over(u).clip(lower_bound=0).log1p(),
        f_past_rate=past_correct / pl.col("pos"),           # NaN at pos 0
        f_ex_n_attempts=pl.int_range(pl.len()).over(ue),
        f_ex_last_correct=pl.col("y").shift(1).over(ue),
        p_baseline=(past_correct + 1) / (pl.col("pos") + 2),
    )
    # "no information" = -1 (first interaction, first time on an exercise, missing duration)
    return df.with_columns([pl.col(f).cast(pl.Float32).fill_nan(-1).fill_null(-1) for f in FEATURES])


def sample_students(df, n_students, seed):
    counts = df.group_by("user_id").len().sort("user_id")
    rng = np.random.default_rng(seed)
    per_bin = n_students // len(STUDENT_BINS)
    chosen, info = [], {}
    for lo, hi in STUDENT_BINS:
        ids = counts.filter(in_bin(pl.col("len"), lo, hi))["user_id"].to_list()
        take = min(per_bin, len(ids))
        chosen += list(rng.choice(ids, size=take, replace=False)) if take else []
        info[bin_label(lo, hi)] = {"available": len(ids), "sampled": take}
    return chosen, info


def prediction_positions(n, max_preds):
    """Attempts to predict: 1..n-1, evenly thinned to at most max_preds."""
    if n - 1 <= max_preds:
        return np.arange(1, n)
    return np.unique(np.linspace(1, n - 1, max_preds).round().astype(int))


# ---------------------------------------------------------------- model

def make_tabicl(device, n_estimators, seed):
    from tabicl import TabICLClassifier

    class CachedTabICL(TabICLClassifier):
        """TabICL reloads its checkpoint from disk at every fit(). We fit thousands of
        tiny tables, so load it once and reuse it (the weights never change)."""
        _cache = {}

        def _load_model(self):
            key = (self.checkpoint_version, str(self.model_path))
            if key not in CachedTabICL._cache:
                super()._load_model()
                CachedTabICL._cache[key] = (self.model_, self.model_config_, self.model_path_)
            self.model_, self.model_config_, self.model_path_ = CachedTabICL._cache[key]

    kw = {"device": device, "random_state": seed}
    if n_estimators:
        kw["n_estimators"] = n_estimators
    return lambda: CachedTabICL(**kw)


def predict_one(new_model, X_ctx, y_ctx, x_query):
    m = new_model().fit(X_ctx, y_ctx)
    proba = m.predict_proba(x_query[None, :])[0]
    return float(proba[list(m.classes_).index(1)])


# ---------------------------------------------------------------- metrics

def score(y, p):
    from sklearn.metrics import log_loss, roc_auc_score
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    out = {"n": int(len(y)), "correct_rate": float(y.mean()) if len(y) else None}
    hit = (p >= 0.5) == y
    if len(np.unique(y)) == 2:
        out.update(auc=float(roc_auc_score(y, p)), logloss=float(log_loss(y, p)),
                   accuracy=float(hit.mean()))
    # MIAAM paper, Table 3: accuracy on the interactions the student got right [1] / wrong [0]
    out["accuracy_1"] = float(hit[y == 1].mean()) if (y == 1).any() else None
    out["accuracy_0"] = float(hit[y == 0].mean()) if (y == 0).any() else None
    return out


def compute_metrics(pred):
    res = {}
    for dim in sorted(pred["dim"].unique()):
        d = pred.filter(pl.col("dim") == dim)
        r = {"model": score(d["y"], d["p_model"]),
             "baseline": score(d["y"], d["p_baseline"]),
             "fallback_rate": float(d["fallback"].mean()),
             "by_history": {}}
        for lo, hi in HISTORY_BINS:
            b = d.filter(in_bin(pl.col("t"), lo, hi))
            r["by_history"][bin_label(lo, hi)] = {
                "model": score(b["y"], b["p_model"]),
                "baseline": score(b["y"], b["p_baseline"]),
                "fallback_rate": float(b["fallback"].mean()) if b.height else None,
            }
        res[str(dim)] = r
    return res


def print_metrics(metrics):
    def f(x):
        return "  -  " if x is None else f"{x:.3f}"
    hdr = (f"{'dim':>5} {'history':>8} {'n':>7} {'AUC model':>10} {'AUC base':>9} "
           f"{'acc1 model':>11} {'acc0 model':>11} {'acc1 base':>10} {'acc0 base':>10} {'fallback':>9}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for dim, r in metrics.items():
        rows = [("all", r["model"], r["baseline"], r["fallback_rate"])]
        rows += [(k, v["model"], v["baseline"], v["fallback_rate"]) for k, v in r["by_history"].items()]
        for name, m, b, fb in rows:
            print(f"{dim:>5} {name:>8} {m['n']:>7} {f(m.get('auc')):>10} {f(b.get('auc')):>9} "
                  f"{f(m.get('accuracy_1')):>11} {f(m.get('accuracy_0')):>11} "
                  f"{f(b.get('accuracy_1')):>10} {f(b.get('accuracy_0')):>10} {f(fb):>9}")
        print()


# ---------------------------------------------------------------- main

def library_versions():
    out = {}
    for lib in ["tabicl", "torch", "polars", "numpy", "scikit-learn"]:
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-id", required=True)
    p.add_argument("--embeddings-id", required=True)
    p.add_argument("--interactions", default="../MIAAM/data/maths_data_filtered.parquet")
    p.add_argument("--set", default="val", choices=["train", "val", "test"],
                   help="students to evaluate on; compare settings on val, keep test for the end")
    p.add_argument("--n-students", type=int, default=300, help="split equally over 4 history-length bins")
    p.add_argument("--dims", type=int, nargs="+", default=[32, 64, 128, 256])
    p.add_argument("--max-context", type=int, default=500, help="keep only the last N past interactions")
    p.add_argument("--max-preds-per-student", type=int, default=50,
                   help="long histories: predict this many interactions, evenly spaced")
    p.add_argument("--n-estimators", type=int, default=4, help="TabICL ensemble size (library default: 8)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-dedup", action="store_true", help="keep same-time duplicated rows")
    p.add_argument("--datasets-root", default="datasets")
    p.add_argument("--runs-root", default="runs")
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()

    commit, clean = git_info()
    if not clean and not args.allow_dirty:
        raise SystemExit("Uncommitted changes: commit first (or use --allow-dirty for a test).")

    # ---- data
    split = load_split(args.split_id, args.datasets_root)
    students = split[args.set]
    df, n_dups = load_interactions(args.interactions, students, dedup=not args.no_dedup)
    print(f"{args.set} students: {len(students)} | rows: {df.height} | duplicates removed: {n_dups}")

    ex_ids, E = load_embeddings(args.embeddings_id, args.datasets_root)
    if max(args.dims) > E.shape[1]:
        raise SystemExit(f"dim {max(args.dims)} > embedding size {E.shape[1]}")
    ex_index = {e: i for i, e in enumerate(ex_ids)}
    n_before = df.height
    df = df.filter(pl.col("exercise_id").is_in(ex_ids))
    if df.height < n_before:
        print(f"WARNING: dropped {n_before - df.height} rows with no embedding")

    df = add_features(df)
    chosen, sample_info = sample_students(df, args.n_students, args.seed)
    df = df.filter(pl.col("user_id").is_in(chosen)).sort(["user_id", "pos"])
    print("sampled students per history bin:", sample_info)

    # one block per student: features, embedding row index, targets, baseline
    blocks = []
    for (uid,), g in df.group_by(["user_id"], maintain_order=True):
        blocks.append({
            "user_id": uid, "n": g.height,
            "F": g.select(FEATURES).to_numpy().astype(np.float32),
            "emb_idx": np.array([ex_index[e] for e in g["exercise_id"]]),
            "y": g["y"].to_numpy(), "base": g["p_baseline"].to_numpy(),
            "positions": prediction_positions(g.height, args.max_preds_per_student),
        })
    n_preds = sum(len(b["positions"]) for b in blocks)
    print(f"{len(blocks)} students, {n_preds} predictions per dim, dims {args.dims}")

    # ---- run folder + config
    run_dir = new_run_dir(args.runs_root, commit)
    config = {**vars(args), "git_commit": commit, "git_clean": clean,
              "features": FEATURES, "sample_info": sample_info, "n_duplicates_removed": n_dups,
              "n_predictions_per_dim": n_preds, "libraries": library_versions(),
              "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    print(f"run folder: {run_dir}")

    wb = None
    if args.wandb_project:
        import wandb
        wb = wandb.init(project=args.wandb_project, job_type=PIPELINE, config=config,
                        name=run_dir.name)

    new_model = make_tabicl(args.device, args.n_estimators, seed=42)

    # ---- evaluation
    rows = []
    for dim in args.dims:
        Ek = E[:, :dim].astype(np.float32)
        Ek /= np.linalg.norm(Ek, axis=1, keepdims=True)
        t0, done = time.time(), 0
        for b in blocks:
            X = np.hstack([b["F"], Ek[b["emb_idx"]]])
            for t in b["positions"]:
                lo = max(0, t - args.max_context)
                y_ctx = b["y"][lo:t]
                fallback = len(np.unique(y_ctx)) < 2
                p_model = b["base"][t] if fallback else predict_one(new_model, X[lo:t], y_ctx, X[t])
                rows.append({"dim": dim, "user_id": b["user_id"], "n_rows": b["n"], "t": int(t),
                             "context": int(t - lo), "y": int(b["y"][t]),
                             "p_model": p_model, "p_baseline": float(b["base"][t]),
                             "fallback": fallback})
            done += len(b["positions"])
            el = time.time() - t0
            print(f"\r  dim {dim}: {done}/{n_preds} predictions | {el / 60:.1f} min | "
                  f"ETA {el / done * (n_preds - done) / 60:.1f} min   ", end="", flush=True)
        print()

    # ---- save
    pred = pl.DataFrame(rows)
    pred.write_parquet(run_dir / "predictions.parquet")
    metrics = compute_metrics(pred)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))   # last = run finished
    print_metrics(metrics)

    if wb:
        table = [[int(dim), hist, v["model"]["n"], v["model"].get("auc"), v["baseline"].get("auc"),
                  v["model"].get("accuracy_1"), v["model"].get("accuracy_0"),
                  v["baseline"].get("accuracy_1"), v["baseline"].get("accuracy_0"),
                  v["fallback_rate"]]
                 for dim, r in metrics.items()
                 for hist, v in [("all", r)] + list(r["by_history"].items())]
        wb.log({"results": wandb.Table(columns=["dim", "history", "n", "auc_model", "auc_baseline",
                                                "acc1_model", "acc0_model", "acc1_baseline",
                                                "acc0_baseline", "fallback_rate"], data=table)})
        for dim, r in metrics.items():
            wb.summary[f"auc_model/dim{dim}"] = r["model"].get("auc")
            wb.summary[f"acc1_model/dim{dim}"] = r["model"].get("accuracy_1")
            wb.summary[f"acc0_model/dim{dim}"] = r["model"].get("accuracy_0")
        base = next(iter(metrics.values()))["baseline"]
        wb.summary["auc_baseline"] = base.get("auc")
        wb.summary["acc1_baseline"] = base.get("accuracy_1")
        wb.summary["acc0_baseline"] = base.get("accuracy_0")
        wb.finish()


if __name__ == "__main__":
    main()