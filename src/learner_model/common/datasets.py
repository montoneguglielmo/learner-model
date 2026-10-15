"""
Shared rules for every dataset a stage produces.

  1. A dataset folder is never modified after creation. A new run = a new folder.
  2. Unique folder name: datasets/<pipeline_name>__<date>__<commit6>/
  3. Each folder contains a manifest.json.
  4. manifest.json is written LAST. A folder without it is broken
     (the stage crashed) -> delete it.

Typical use inside a stage:

    out = DatasetWriter("split_pykt", config=vars(args), root=args.root)
    if out.reuse:                       # same code, same day, same config: already done
        return
    ... write files into out.path ...
    out.finish(input_files=[...], extra={...})
"""

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

MANIFEST = "manifest.json"


def file_sha256(path, chunk_size=1 << 20):
    """Hash of a file's content. Same content -> same hash."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def git_info():
    """(commit, clean). commit is None outside a git repository."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout.strip()
        return commit, not status
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None, False


class DatasetWriter:
    def __init__(self, pipeline_name, config, root="datasets", allow_dirty=False):
        commit, clean = git_info()
        if commit is None:
            raise RuntimeError("Not inside a git repository with at least one commit. "
                               "A dataset must point to a commit.")
        if not clean and not allow_dirty:
            raise RuntimeError("You have uncommitted changes. Commit first, so the dataset points "
                               "to the exact code. (For a quick test only: --allow-dirty)")

        self.pipeline_name = pipeline_name
        self.config = config
        self.commit, self.clean = commit, clean
        self.created_at = datetime.datetime.now(datetime.timezone.utc)
        self.reuse = False

        root = Path(root)
        base = f"{pipeline_name}__{self.created_at:%Y-%m-%d}__{commit[:6]}"

        # Same code, same day: compare configs. Same config -> nothing to do.
        # Different config -> add a suffix (__2, __3, ...) so the name stays unique.
        n = 1
        while True:
            dataset_id = base if n == 1 else f"{base}__{n}"
            path = root / dataset_id
            if not path.exists():
                break
            manifest_path = path / MANIFEST
            if not manifest_path.exists():
                raise RuntimeError(f"{path} has no {MANIFEST}: an earlier run crashed. "
                                   f"Delete this folder and run again.")
            if json.loads(manifest_path.read_text())["config"] == config:
                print(f"Dataset already exists with the same code and config: {path}")
                self.reuse = True
                break
            n += 1

        self.dataset_id, self.path = dataset_id, path
        if not self.reuse:
            path.mkdir(parents=True)

    def finish(self, input_files, extra=None):
        """Hash inputs and outputs, then write manifest.json. Call this LAST."""
        output_files = {
            str(p.relative_to(self.path)): file_sha256(p)
            for p in sorted(self.path.rglob("*")) if p.is_file() and p.name != MANIFEST
        }
        manifest = {
            "dataset_id": self.dataset_id,
            "created_at": self.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "git_commit": self.commit,
            "git_clean": self.clean,
            "pipeline_name": self.pipeline_name,
            "config": self.config,
            "input_files": {str(p): file_sha256(p) for p in input_files},
            "output_files": output_files,
            **(extra or {}),
        }
        (self.path / MANIFEST).write_text(json.dumps(manifest, indent=2))
        return self.path


def open_dataset(dataset_id, root="datasets"):
    """Path of a finished dataset. Refuses broken folders (no manifest)."""
    path = Path(root) / dataset_id
    if not (path / MANIFEST).exists():
        raise FileNotFoundError(f"{path} does not exist or has no {MANIFEST} (broken run).")
    return path