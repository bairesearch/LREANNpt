#!/usr/bin/env python3
"""Rerun the benchmark in this folder, or in a new --output folder."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from reproduction_support import require_workspace, sha256

HERE = Path(__file__).resolve().parent


def reset_results(output):
    """Discard previous training outputs while retaining code and input caches."""
    def remove(path):
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)

    for name in ('runs', 'checkpoints', 'figures'):
        remove(output / name)
        (output / name).mkdir()
    for path in (output / 'logs').iterdir():
        # Keep lock inodes: the launcher holds reproduce.lock during this reset.
        if path.suffix != '.lock':
            remove(path)
    for path in (output / 'verification').iterdir():
        if path.name not in ('source_split_overlap.json', 'portability', 'production_convergence_tests.json'):
            remove(path)
    names = ['REPORT.txt', 'summary.json', 'summary.csv', 'per_seed.csv',
             'LAUNCH.json', 'MANAGER.json', 'PREPARATION.json', 'PROGRESS.json',
             'SUPERVISOR.json', 'FINALISATION.json', 'CANCEL']
    names += [f'{stem}.{extension}'
              for stem in ('training_loss_grid', 'validation_loss_grid', 'test_accuracy_by_population')
              for extension in ('png', 'svg')]
    for name in names:
        remove(output / name)
        remove(output / (name + '.tmp'))


def initialise_workspace(output=None, resume=False):
    output = HERE if output is None else Path(output).expanduser().resolve()
    if resume:
        require_workspace(output)
        return output
    in_place = output == HERE
    if not in_place:
        if HERE.is_relative_to(output):
            raise ValueError('Output cannot be a parent of the benchmark folder')
        # Explicit separate output directories must still be new.
        output.mkdir(parents=True, exist_ok=False)
    files = sorted(HERE.glob('*.py')) + sorted(HERE.glob('*.sh')) + sorted(HERE.glob('requirements*.txt'))
    files += [HERE / name for name in ('protocol.json', 'environment_original.json', 'README.txt', 'HOLDS.json')]
    files += [p for p in (HERE / 'source').rglob('*') if p.is_file() and p.suffix in ('.py', '.patch')]
    files += [p for p in (HERE / 'inputs').rglob('*') if p.is_file()]
    hashes = {}
    for source in files:
        relative = source.relative_to(HERE)
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not in_place:
            shutil.copyfile(source, destination)
        hashes[relative.as_posix()] = sha256(destination)
    protocol = json.loads((output / 'protocol.json').read_text())
    for name in protocol['datasets']:
        reference = HERE / 'reference_manifests' / (name + '.json')
        if not reference.exists():
            reference = HERE / 'data' / name / 'manifest.json'
        relative = Path('reference_manifests') / (name + '.json')
        destination = output / relative
        destination.parent.mkdir(exist_ok=True)
        if reference != destination:
            shutil.copyfile(reference, destination)
        hashes[relative.as_posix()] = sha256(destination)
        # Keep expected manifests for the preparation code, never archived run results.
        data = output / 'data' / name
        data.mkdir(parents=True, exist_ok=True)
        if not (data / 'manifest.json').exists():
            shutil.copyfile(destination, data / 'manifest.json')
    for name in ('sources', 'runs', 'logs', 'figures', 'verification', 'checkpoints'):
        (output / name).mkdir(exist_ok=True)
    overlap = HERE / 'verification/source_split_overlap.json'
    if not in_place:
        shutil.copyfile(overlap, output / 'verification/source_split_overlap.json')
    hashes['verification/source_split_overlap.json'] = sha256(overlap)
    if in_place:
        reset_results(output)
    record = {
        'format': 'suann-full-reproduction-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'snapshot_sha256': hashes,
        'fresh_models': True,
        'checkpoint_directory': 'checkpoints',
        'banking_marketing_policy': protocol['banking_marketing_policy'],
        'historical_results_copied': False,
    }
    (output / 'REPRODUCTION.json').write_text(json.dumps(record, indent=2) + '\n')
    return output


def check_environment(output, need_cuda):
    if sys.version_info < (3, 11):
        raise RuntimeError('Python 3.11+ is required; the documented installation uses Python 3.12')
    requirements = []
    for line in (output / 'requirements.txt').read_text().splitlines():
        line = line.strip()
        if line and not line.startswith(('#', '--')):
            name, version = line.split('==', 1)
            requirements.append((name, version))
    installed = {}
    errors = []
    for name, version in requirements:
        try:
            installed[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed[name] = None
        if installed[name] != version:
            errors.append(f'{name}: expected {version}, installed {installed[name]}')
    if errors:
        raise RuntimeError('Install requirements.txt in the documented environment:\n' + '\n'.join(errors))
    record = {'python': sys.version, 'platform': platform.platform(), 'packages': installed,
              'cuda_required': need_cuda}
    if need_cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('This archived batched benchmark requires a CUDA GPU')
        record.update(torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                      gpu=torch.cuda.get_device_name(), gpu_memory=torch.cuda.get_device_properties(0).total_memory,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    path = output / 'environment_runs.json'
    history = json.loads(path.read_text()) if path.exists() else []
    history.append({'time_utc': datetime.now(timezone.utc).isoformat(), **record})
    path.write_text(json.dumps(history, indent=2) + '\n')


def run_script(output, script, *arguments, manager=False):
    command = [sys.executable, '-u', str(output / script), *arguments]
    print('RUN', ' '.join(command), flush=True)
    if not manager:
        subprocess.run(command, cwd=output, check=True)
        return
    # The manager's cancellation handler kills only its own process group.
    with (output / 'logs/manager.log').open('a') as log:
        process = subprocess.Popen(command, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        print(f'Manager PID {process.pid}; progress: {output / "REPORT.txt"}; '
              f'log: {output / "logs/manager.log"}', flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE,
                        help='Output folder (default: this script\'s folder, overwriting saved results; a separate fresh folder must be new)')
    parser.add_argument('--resume', action='store_true', help='Explicitly resume this output folder; verify its frozen snapshot')
    parser.add_argument('--prepare-only', action='store_true', help='Download, rebuild, and audit data without training or CUDA checks')
    parser.add_argument('--datasets', nargs='+', help='Only with --prepare-only: prepare these complete datasets; training always uses the full protocol')
    args = parser.parse_args()
    if args.datasets and not args.prepare_only:
        parser.error('--datasets is supported only with --prepare-only; training always runs the complete protocol')
    protocol = json.loads((HERE / 'protocol.json').read_text())
    if args.datasets and set(args.datasets) - set(protocol['datasets']):
        parser.error('Unknown dataset name')
    output = args.output.expanduser().resolve()
    in_place = output == HERE
    if in_place:
        if args.resume:
            require_workspace(output)
        (output / 'logs').mkdir(exist_ok=True)
    else:
        output = initialise_workspace(output, args.resume)
    # One orchestrator at a time per output, including during data preparation.
    import fcntl
    with (output / 'logs/reproduce.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume and (output / 'CANCEL').exists():
            raise RuntimeError('This run was cancelled; remove its CANCEL file explicitly before resuming')
        check_environment(output, need_cuda=not args.prepare_only)
        if in_place and not args.resume:
            # Lock and check dependencies before discarding any saved results.
            initialise_workspace(output)
        require_workspace(output)
        dataset_args = ['--datasets', *args.datasets] if args.datasets else []
        run_script(output, 'prepare_full.py', *dataset_args)
        run_script(output, 'audit_data.py', *dataset_args)
        if args.prepare_only:
            print(f'Prepared data verified. To train all datasets, repeat with --resume and without --prepare-only: {output}')
            return
        for script in ('verify_preprocessing.py', 'verify_backend.py', 'verify_cached_graph.py'):
            run_script(output, script)
        run_script(output, 'manage.py', manager=True)
        print(f'Completed. Report: {output / "REPORT.txt"}', flush=True)


if __name__ == '__main__':
    main()
