"""Exercise destructive resets only in disposable benchmark copies."""
import contextlib
import fcntl
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import delete_population_pertubation_train_convergence_data as reset
import reproduce


class DeleteDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.parent = Path(self.temporary.name)
        self.root = reproduce.initialise_workspace(self.parent / 'benchmark')

    def write(self, relative, content='generated artifact'):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def invoke(self, *args):
        output, errors = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(errors):
            code = reset.main([str(self.root), *args])
        return code, output.getvalue(), errors.getvalue()

    def tree(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                for p in self.root.rglob('*')}

    def saved_artifacts(self):
        names = [
            'data/iris/train_x.npy', 'data/iris/train_y.npy',
            'data/iris/train_source_indices.npy', 'sources/bank.zip',
            'sources/sklearn/covertype/samples_py3',
            'runs/iris_backprop_adam_seed11.json', 'runs/run.progress.json',
            'runs/run.pt', 'runs/run.best.pt', 'checkpoints/run.pt',
            'logs/manager.log', 'logs/failure.json', 'logs/reproduce.lock',
            'logs/manager.lock', 'logs/report.lock', 'figures/iris.png',
            'verification/portability/checks.json', 'verification/backend.json',
            'verification/production_convergence_tests.json',
            'REPORT.txt', 'summary.json', 'summary.csv', 'per_seed.csv',
            'training_loss_grid.png', 'validation_loss_grid.svg',
            'test_accuracy_by_population.png', 'environment_runs.json',
            'FINALISATION.json', 'MANAGER.json', 'PROGRESS.json', 'CANCEL',
            'CLEANUP.json', 'batched_verification.json', 'scratch.tmp',
            '__pycache__/module.pyc', '.pytest_cache/README.md',
            'source/LREANNpt/__pycache__/module.pyc',
            'miscellaneous/unknown-output.extension', 'REPORT-old-experiment.txt',
        ]
        for name in names:
            self.write(name)
        return names + ['REPRODUCTION.json']

    def assert_can_bootstrap(self):
        # Import the retained launcher from its own folder in a new interpreter.
        # This executes real snapshot initialization and integrity/input checks.
        process = subprocess.run([
            sys.executable, '-B', '-c',
            "import reproduce, reproduction_support as s; "
            "root = reproduce.initialise_workspace(); "
            "s.require_workspace(root); "
            "[s.expected_manifest(n) for n in __import__('json').loads((root/'protocol.json').read_text())['datasets']]; "
            "s.bundled_source('titanic', 'titanic_openml_40945.csv'); "
            "s.bundled_source('new-thyroid', 'new-thyroid.csv')",
        ], cwd=self.root, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_reset_removes_all_outputs_and_can_start_fresh(self):
        generated = self.saved_artifacts()
        self.write('tests/test_example.py', 'pass\n')
        self.write('.gitignore', '__pycache__/\n')
        self.write('.gitattributes', '*.csv text\n')
        expected = {
            p.relative_to(self.root).as_posix(): p.read_bytes()
            for p in self.root.rglob('*') if p.is_file()
            and (p.suffix in ('.py', '.sh', '.patch')
                 or p.relative_to(self.root).as_posix() in reset.CONFIG_FILES
                 or p.parent.name in ('reference_manifests', 'inputs'))
        }
        code, _, errors = self.invoke()
        self.assertEqual(code, 0, errors)
        for name in generated:
            self.assertFalse((self.root / name).exists(), name)
        for name, content in expected.items():
            self.assertEqual((self.root / name).read_bytes(), content, name)
        for directory in ('runs', 'sources', 'data', 'logs', 'checkpoints', 'figures', 'miscellaneous'):
            self.assertFalse((self.root / directory).exists(), directory)
        self.assertEqual([p.name for p in (self.root / 'verification').iterdir()], ['source_split_overlap.json'])
        self.assertTrue((self.root / '.gitignore').exists())
        self.assertTrue((self.root / '.gitattributes').exists())
        once = self.tree()
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(self.tree(), once)
        self.assert_can_bootstrap()

    def test_archive_without_reference_directory_keeps_manifest_seeds(self):
        shutil.rmtree(self.root / 'reference_manifests')
        self.saved_artifacts()
        manifests = {p.relative_to(self.root): p.read_bytes()
                     for p in (self.root / 'data').glob('*/manifest.json')}
        self.assertEqual(len(manifests), 13)
        self.assertEqual(self.invoke()[0], 0)
        for relative, content in manifests.items():
            self.assertEqual((self.root / relative).read_bytes(), content)
        self.assertEqual(list((self.root / 'data').rglob('*.npy')), [])
        self.assert_can_bootstrap()

    def test_dry_run_has_no_filesystem_changes(self):
        self.saved_artifacts()
        before = self.tree()
        code, output, errors = self.invoke('--dry-run')
        self.assertEqual(code, 0, errors)
        self.assertIn('Would delete: runs/iris_backprop_adam_seed11.json', output)
        self.assertEqual(self.tree(), before)

    def test_missing_manifest_refuses_before_deleting_results(self):
        self.saved_artifacts()
        (self.root / 'reference_manifests/iris.json').unlink()
        (self.root / 'data/iris/manifest.json').unlink()
        before = self.tree()
        code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('Missing required bootstrap file', errors)
        self.assertEqual(self.tree(), before)

    def test_corrupt_bundled_input_refuses_before_deleting_results(self):
        self.saved_artifacts()
        self.write('inputs/new-thyroid.csv', 'corrupt')
        before = self.tree()
        code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('Bundled source does not match', errors)
        self.assertEqual(self.tree(), before)

    def test_active_launcher_or_manager_or_report_refuses_reset(self):
        self.saved_artifacts()
        before = self.tree()
        for name in ('reproduce.lock', 'manager.lock', 'report.lock'):
            with self.subTest(lock=name), (self.root / 'logs' / name).open('r') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                code, _, errors = self.invoke()
                self.assertEqual(code, 1)
                self.assertIn('Benchmark is active', errors)
                self.assertEqual(self.tree(), before)

    def test_symlink_targets_and_nested_git_metadata_are_untouched(self):
        outside = self.parent / 'outside'
        outside.mkdir()
        sentinel = outside / 'sentinel'
        sentinel.write_text('must survive')
        (self.root / 'external').symlink_to(outside, target_is_directory=True)
        (self.root / 'broken-output').symlink_to(outside / 'missing')
        (self.root / 'runs' / 'external.json').symlink_to(sentinel)
        self.write('runs/.git/config', 'git fixture')
        self.write('__pycache__/.git', 'gitdir: fixture')
        self.write('.git/config', 'top-level git fixture')
        self.assertEqual(self.invoke()[0], 0)
        self.assertEqual(sentinel.read_text(), 'must survive')
        self.assertFalse((self.root / 'external').is_symlink())
        self.assertFalse((self.root / 'broken-output').is_symlink())
        self.assertFalse((self.root / 'runs/external.json').is_symlink())
        self.assertEqual((self.root / 'runs/.git/config').read_text(), 'git fixture')
        self.assertEqual((self.root / '__pycache__/.git').read_text(), 'gitdir: fixture')
        self.assertEqual((self.root / '.git/config').read_text(), 'top-level git fixture')

    def test_symlinked_bootstrap_input_refuses_reset(self):
        bundled = self.root / 'inputs/new-thyroid.csv'
        outside = self.parent / 'new-thyroid.csv'
        bundled.replace(outside)
        bundled.symlink_to(outside)
        result = self.write('runs/sentinel.json')
        code, _, errors = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('Required input must not use a symlink', errors)
        self.assertTrue(result.exists())
        self.assertTrue(outside.exists())

    def test_default_cli_uses_script_folder_from_other_working_directory(self):
        before = self.tree()
        process = subprocess.run([
            sys.executable, '-B',
            str(self.root / 'delete_population_pertubation_train_convergence_data.py'), '--dry-run',
        ], cwd=self.parent, text=True, capture_output=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertIn('Would delete: REPRODUCTION.json', process.stdout)
        self.assertEqual(self.tree(), before)


if __name__ == '__main__':
    unittest.main()
