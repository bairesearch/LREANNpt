#!/usr/bin/env python3
"""Remove bulky/generated artifacts from a completed benchmark's archival copy.

Usage:
    python clean_population_pertubation_train_convergence.py FOLDER
    python clean_population_pertubation_train_convergence.py FOLDER --dry-run

The default command deletes immediately, without checking running experiments
or asking for confirmation. Use it on the archival copy, as intended.

Keep: scripts, frozen source, protocol, reports, figures, dataset manifests,
completed run JSON, verification records, execution/failure logs and unknown
files. Delete: prepared binary datasets (including split-index arrays), raw
downloads, model checkpoints, caches, temporary files and runtime status files.

The cleaned folder is a compact results/code archive. Run reproduce.py to
rebuild data and retrain in that folder, replacing saved results. Resuming or
re-evaluating saved models requires checkpoints retained in another copy;
cleanup removes them. See README.txt for the pinned environment setup.

Only the supplied folder is modified. Directory symlinks are never traversed;
deleting an artifact symlink removes the link, not its target. Git metadata is
left alone. CLEANUP.json records each cleanup's deletions and byte counts.
Requires Python 3.9+ and only the standard library.
"""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys


CACHE_DIRECTORIES = {"__pycache__", "pycache", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
DATA_SUFFIXES = {".npy", ".npz", ".arrow", ".parquet", ".feather", ".h5", ".hdf5"}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors"}
RAW_SOURCE_SUFFIXES = DATA_SUFFIXES | {
    ".csv", ".tsv", ".arff", ".data", ".bin", ".pkl", ".pickle",
    ".gz", ".zip", ".bz2", ".xz", ".tar", ".tgz", ".7z", ".zst",
}
RUNTIME_FILES = {
    "LAUNCH.json", "MANAGER.json", "PREPARATION.json", "PROGRESS.json",
    "SUPERVISOR.json", "CANCEL",
}
REPORT_NAME = "CLEANUP.json"
REPORT_FORMAT = "population-pertubation-archive-cleanup-v1"


@dataclass(frozen=True)
class Removal:
    path: str
    bytes: int
    reason: str


def raise_walk_error(error):
    raise error


def cache_removals(directory, root):
    """Inspect caches file-by-file so even nested .git metadata is preserved."""
    for current, dirs, files in os.walk(directory, followlinks=False, onerror=raise_walk_error):
        current = Path(current)
        dirs[:] = sorted(name for name in dirs if name != ".git")
        for name in list(dirs):
            path = current / name
            if path.is_symlink():
                yield Removal(path.relative_to(root).as_posix(), path.lstat().st_size, "cache symlink")
                dirs.remove(name)
        for name in sorted(files):
            if name == ".git":
                continue
            path = current / name
            yield Removal(path.relative_to(root).as_posix(), path.lstat().st_size, "Python/tool cache")


def removal_reason(relative):
    name, suffix = relative.name, relative.suffix.lower()
    top = relative.parts[0]
    if suffix in {".pyc", ".pyo"}:
        return "compiled Python cache"
    if suffix in {".tmp", ".part"}:
        return "temporary/incomplete file"
    if len(relative.parts) == 1 and name in RUNTIME_FILES:
        return "transient runtime status"
    if top == "data" and suffix in DATA_SUFFIXES:
        return "prepared dataset or split-index array"
    if relative.parts[:2] == ("sources", "sklearn") and name in {"samples_py3", "targets_py3"}:
        return "downloaded scikit-learn dataset cache"
    if top == "sources" and suffix in RAW_SOURCE_SUFFIXES:
        return "raw dataset download"
    if top in {"runs", "logs", "checkpoints"} and suffix in MODEL_SUFFIXES:
        return "model/resume checkpoint"
    if top == "runs" and name.endswith(".progress.json"):
        return "transient run progress"
    if top == "logs" and suffix in {".lock", ".pid"}:
        return "runtime lock/PID file"
    return None


def plan_cleanup(root):
    removals = []
    cache_dirs = set()
    for current, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        current = Path(current)
        dirs[:] = sorted(name for name in dirs if name != ".git")
        for name in list(dirs):
            path = current / name
            if name in CACHE_DIRECTORIES:
                if path.is_symlink():
                    removals.append(Removal(path.relative_to(root).as_posix(), path.lstat().st_size, "cache symlink"))
                else:
                    removals.extend(cache_removals(path, root))
                    for nested, children, _ in os.walk(path, followlinks=False, onerror=raise_walk_error):
                        children[:] = [c for c in children if c != ".git" and not (Path(nested) / c).is_symlink()]
                        cache_dirs.add(Path(nested))
                dirs.remove(name)
            elif path.is_symlink():
                dirs.remove(name)
        for name in sorted(files):
            if name == ".git":
                continue
            path = current / name
            relative = path.relative_to(root)
            reason = removal_reason(relative)
            if reason:
                removals.append(Removal(relative.as_posix(), path.lstat().st_size, reason))
    return sorted(removals, key=lambda item: item.path), cache_dirs


def readable_size(size):
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f} {unit}"
        amount /= 1024


def load_cleanup_record(root):
    path = root / REPORT_NAME
    if not path.exists() and not path.is_symlink():
        return {"format": REPORT_FORMAT, "cleanups": []}
    if path.is_symlink():
        raise ValueError(f"Cannot write cleanup record through a symlink: {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict) or record.get("format") != REPORT_FORMAT or not isinstance(record.get("cleanups"), list):
        raise ValueError(f"Preserving existing unrelated {REPORT_NAME}; rename it before cleaning")
    return record


def clean(root, removals, cache_dirs):
    record = load_cleanup_record(root)  # Validate before deleting anything.
    removed, errors = [], []
    possible_empty_dirs = set(cache_dirs)
    for item in removals:
        path = root / item.path
        try:
            if not path.parent.resolve().is_relative_to(root):
                raise ValueError("Parent directory resolves outside the supplied folder")
            path.unlink()  # Also safe for file/directory symlinks: never follows them.
            removed.append(asdict(item))
            possible_empty_dirs.update(parent for parent in path.parents if parent != root and parent.is_relative_to(root))
        except (OSError, ValueError) as error:
            errors.append({"path": item.path, "error": str(error)})

    pruned = []
    for directory in sorted(possible_empty_dirs, key=lambda p: (-len(p.parts), str(p))):
        if directory.is_symlink() or not directory.exists():
            continue
        if not directory.resolve().is_relative_to(root):
            errors.append({"path": directory.relative_to(root).as_posix(), "error": "Directory resolves outside the supplied folder"})
            continue
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                pruned.append(directory.relative_to(root).as_posix())
        except OSError as error:
            errors.append({"path": directory.relative_to(root).as_posix(), "error": str(error)})

    if removed or pruned or errors:
        record["cleanups"].append({
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "deleted_files": removed,
            "deleted_empty_directories": pruned,
            "deleted_bytes": sum(item["bytes"] for item in removed),
            "errors": errors,
            "archive_note": "Prepared data and model checkpoints are intentionally omitted; this is a results/code archive, not a self-contained training or saved-model verification package.",
        })
        # Use a unique file in the target rather than overwriting a pre-existing
        # temporary path or following a link at that path.
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, prefix=".cleanup-", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2)
            handle.write("\n")
        temporary.replace(root / REPORT_NAME)
    return removed, pruned, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path, help="Completed benchmark folder or archival copy to clean")
    parser.add_argument("--dry-run", action="store_true", help="Preview deletions without changing any files")
    args = parser.parse_args(argv)
    root = args.folder.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Not a directory: {root}")
    if root.parent == root:
        parser.error("Provide a benchmark folder, not a filesystem root")
    try:
        removals, cache_dirs = plan_cleanup(root)
        prefix = "Would delete" if args.dry_run else "Planned deletion"
        for item in removals:
            print(f"{prefix}: {item.path} ({readable_size(item.bytes)}; {item.reason})")
        if args.dry_run:
            print(f"Dry run: {len(removals)} files, {readable_size(sum(item.bytes for item in removals))}. Empty generated directories would also be removed.")
            return 0
        removed, pruned, errors = clean(root, removals, cache_dirs)
        print(f"Deleted {len(removed)} files and {len(pruned)} empty directories; removed {readable_size(sum(item['bytes'] for item in removed))}.")
        if removed or pruned or errors:
            print(f"Cleanup record: {root / REPORT_NAME}")
        for error in errors:
            print(f"ERROR: {error['path']}: {error['error']}", file=sys.stderr)
        return 1 if errors else 0
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
