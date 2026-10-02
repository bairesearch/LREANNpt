#!/usr/bin/env python3
"""Reset a stopped benchmark to its code and required bootstrap inputs.

Usage:
    python delete_population_pertubation_train_convergence_data.py [FOLDER] --dry-run
    python delete_population_pertubation_train_convergence_data.py [FOLDER]

FOLDER defaults to this script's directory, regardless of the working directory.
Without --dry-run, deletion is immediate. Stop the benchmark before running this.

Keep scripts, tests, frozen source/patches, requirements, README, configuration,
the two bundled CSV inputs, one reference manifest per dataset, and the source
split record required by reproduce.py. Everything else is removed, including
unknown files, all results, downloads, arrays, checkpoints, logs, verification
outputs, runtime state, caches and previous cleanup records. Git metadata stays.
When reference_manifests/ exists, its manifests supersede data/*/manifest.json.

Required inputs are checked before deletion. Directory symlinks are not followed;
unneeded symlinks are unlinked without changing their targets. No deletion log is
written. Afterwards, run reproduce.py without --resume to start from scratch.
Requires Python 3.9+ on Linux and only the standard library.
"""

import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
CONFIG_FILES = {
    'README.txt', 'protocol.json', 'environment_original.json', 'HOLDS.json',
    'requirements.txt', 'verification/source_split_overlap.json',
}
REQUIRED_SCRIPTS = {
    'reproduce.py', 'reproduction_support.py', 'prepare_full.py', 'audit_data.py',
    'train_full.py', 'manage.py', 'report.py', 'model_setup.py', 'batched.py',
    'fast_population.py', 'cached_graphed_population.py', 'verify_backend.py',
    'verify_preprocessing.py', 'verify_cached_graph.py', 'verify_results.py',
}
BUNDLED_INPUTS = {
    'new-thyroid': 'inputs/new-thyroid.csv',
    'titanic': 'inputs/titanic_openml_40945.csv',
}
CACHE_DIRS = {'__pycache__', 'pycache', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def required_file(root, relative):
    """Reject missing files and symlinks in bootstrap paths before any deletion."""
    path = root / relative
    if not path.is_relative_to(root) or '..' in path.parts:
        raise ValueError(f'Invalid bootstrap path: {relative}')
    if any(p.is_symlink() for p in (path, *path.parents) if p != root and p.is_relative_to(root)):
        raise ValueError(f'Required input must not use a symlink: {relative}')
    if not path.is_file():
        raise ValueError(f'Missing required bootstrap file: {relative}')
    return path


def bootstrap_files(root):
    """Find the files needed by the launcher's fresh-workspace initialization."""
    keep = set(CONFIG_FILES) | REQUIRED_SCRIPTS
    for relative in sorted(keep):
        required_file(root, relative)
    protocol = json.loads((root / 'protocol.json').read_text())
    datasets = protocol.get('datasets')
    hashes = protocol.get('source_hashes')
    if not isinstance(datasets, list) or not datasets or not isinstance(hashes, dict) or not hashes:
        raise ValueError('protocol.json must define datasets and frozen source_hashes')
    for name, expected in hashes.items():
        relative = f'source/LREANNpt/{name}'
        path = required_file(root, relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Frozen source differs from protocol.json: {relative}')
        keep.add(relative)
    for dataset in datasets:
        if not isinstance(dataset, str) or dataset in ('.', '..') or Path(dataset).name != dataset:
            raise ValueError(f'Invalid dataset name: {dataset!r}')
        relative = f'reference_manifests/{dataset}.json'
        if not (root / relative).exists() and not (root / relative).is_symlink():
            relative = f'data/{dataset}/manifest.json'
        manifest = json.loads(required_file(root, relative).read_text())
        if manifest.get('dataset') != dataset or not manifest.get('files'):
            raise ValueError(f'Invalid reference manifest: {relative}')
        keep.add(relative)
        if dataset in BUNDLED_INPUTS:
            bundled = BUNDLED_INPUTS[dataset]
            content = required_file(root, bundled).read_bytes()
            source = manifest['sources'][0]
            if len(content) != source['bytes'] or hashlib.sha256(content).hexdigest() != source['sha256']:
                raise ValueError(f'Bundled source does not match its manifest: {bundled}')
            keep.add(bundled)
    return keep


def keep_file(relative, bootstrap):
    if relative.as_posix() in bootstrap:
        return True
    if any(part in CACHE_DIRS for part in relative.parts[:-1]):
        return False
    if len(relative.parts) == 1:
        return (relative.suffix in {'.py', '.sh'}
                or relative.match('requirements*.txt')
                or relative.name in {'.gitignore', '.gitattributes', 'AGENTS.md'})
    return (relative.parts[0] == 'source' and relative.suffix in {'.py', '.patch'}
            or relative.parts[0] == 'tests' and relative.suffix == '.py')


def raise_walk_error(error):
    raise error


def plan_reset(root):
    bootstrap = bootstrap_files(root)
    removals, directories = [], []
    for current, dirs, files in os.walk(root, followlinks=False, onerror=raise_walk_error):
        current = Path(current)
        dirs[:] = sorted(name for name in dirs if name != '.git')
        for name in list(dirs):
            path = current / name
            if path.is_symlink():
                removals.append(path.relative_to(root))
                dirs.remove(name)
            else:
                directories.append(path.relative_to(root))
        for name in sorted(files):
            if name == '.git':
                continue
            path = current / name
            relative = path.relative_to(root)
            if not keep_file(relative, bootstrap):
                removals.append(relative)
    return sorted(removals), sorted(directories, key=lambda p: (-len(p.parts), str(p)))


@contextmanager
def stopped_benchmark(root):
    """Hold existing launcher/manager/report locks; never create preview artifacts."""
    with ExitStack() as stack:
        for name in ('reproduce.lock', 'manager.lock', 'report.lock'):
            path = root / 'logs' / name
            if path.parent.is_symlink() or path.is_symlink() or not path.exists():
                continue
            handle = stack.enter_context(path.open('r'))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f'Benchmark is active ({name}); stop it before resetting') from error
        yield


def delete_files(root, removals, directories):
    deleted, pruned = 0, 0
    for relative in removals:
        path = root / relative
        if not path.parent.resolve().is_relative_to(root):
            raise ValueError(f'Directory now resolves outside the benchmark: {relative}')
        path.unlink()
        deleted += 1
    for relative in directories:
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f'Directory changed during reset: {relative}')
        if not any(path.iterdir()):
            path.rmdir()
            pruned += 1
    return deleted, pruned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', nargs='?', type=Path, default=HERE,
                        help="Benchmark folder (default: this script's folder)")
    parser.add_argument('--dry-run', action='store_true', help='Preview without changing any files')
    args = parser.parse_args(argv)
    root = args.folder.expanduser().resolve()
    if not root.is_dir() or root.parent == root:
        parser.error(f'Provide a benchmark directory, not a filesystem root: {root}')
    try:
        with stopped_benchmark(root):
            removals, directories = plan_reset(root)
            total = sum((root / relative).lstat().st_size for relative in removals)
            for relative in removals:
                print(f'{"Would delete" if args.dry_run else "Deleting"}: {relative}')
            if args.dry_run:
                print(f'Dry run: {len(removals)} files/symlinks, {total:,} bytes. Empty directories would also be removed.')
            else:
                deleted, pruned = delete_files(root, removals, directories)
                print(f'Deleted {deleted} files/symlinks and {pruned} empty directories ({total:,} bytes).')
                print(f'Ready for a fresh launch: python {root / "reproduce.py"}')
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
