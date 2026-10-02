"""Portability/provenance checks using disposable folders, without training."""
import io
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import reproduce
import reproduction_support as support


class ReproductionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def workspace(self, name='run'):
        return reproduce.initialise_workspace(self.root / name)

    def test_fresh_snapshot_excludes_results_and_checkpoints(self):
        output = self.workspace()
        support.require_workspace(output)
        self.assertEqual(list((output / 'runs').iterdir()), [])
        self.assertEqual(list((output / 'checkpoints').iterdir()), [])
        self.assertFalse((output / 'REPORT.txt').exists())
        self.assertFalse((output / 'verification/backend.json').exists())
        self.assertEqual(json.loads((output / 'HOLDS.json').read_text()), {})
        self.assertEqual(len(list((output / 'reference_manifests').glob('*.json'))), 13)
        self.assertEqual(list((output / 'data').rglob('*.npy')), [])
        original = ROOT / 'source/LREANNpt/ANNpt_data.py'
        copied = output / 'source/LREANNpt/ANNpt_data.py'
        self.assertFalse(copied.samefile(original))
        self.assertFalse(copied.is_symlink())

    def test_resume_explicit_and_snapshot_immutable(self):
        output = self.workspace()
        with self.assertRaises(FileExistsError):
            reproduce.initialise_workspace(output)
        self.assertEqual(reproduce.initialise_workspace(output, resume=True), output)
        (output / 'source/LREANNpt/ANNpt_data.py').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            reproduce.initialise_workspace(output, resume=True)

    def test_uninitialised_folder_requires_launcher(self):
        with self.assertRaisesRegex(RuntimeError, 'Use reproduce.py'):
            support.require_workspace(self.root)

    def saved_results(self, output):
        names = [
            'runs/iris_backprop_adam_seed11.json', 'runs/iris_backprop_adam_seed11.progress.json',
            'runs/iris_backprop_adam_seed11.best.pt', 'checkpoints/iris_backprop_adam_seed11.pt',
            'figures/iris.png', 'logs/manager.log', 'logs/failure_old.json',
            'REPORT.txt', 'REPORT.txt.tmp', 'summary.json', 'summary.csv', 'per_seed.csv',
            'training_loss_grid.png', 'validation_loss_grid.svg', 'test_accuracy_by_population.png',
            'MANAGER.json', 'PROGRESS.json', 'CANCEL', 'verification/backend.json',
            'verification/metrics_iris_backprop_adam_seed11.json', 'verification/initial_visual_review.json',
        ]
        for name in names:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('saved result or checkpoint')
        return names

    def test_in_place_fresh_run_replaces_results_and_preserves_inputs(self):
        output = self.workspace()
        # Match a cleaned original archive with only data manifests, no launch record.
        shutil.rmtree(output / 'reference_manifests')
        (output / 'REPRODUCTION.json').unlink()
        old_results = self.saved_results(output)
        preserved = ['data/iris/train_x.npy', 'sources/cached.csv',
                     'verification/portability/checks.json', 'verification/production_convergence_tests.json',
                     'REPORT-maxTrainingRows10k.txt', 'CLEANUP.json']
        for name in preserved:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('retained fixture')
        preserved += ['data/iris/manifest.json', 'source/LREANNpt/ANNpt_data.py',
                      'inputs/titanic_openml_40945.csv', 'protocol.json',
                      'verification/source_split_overlap.json']
        original = {name: support.sha256(output / name) for name in preserved}
        reference = (output / 'data/iris/manifest.json').read_bytes()
        with patch.object(reproduce, 'HERE', output):
            self.assertEqual(reproduce.initialise_workspace(), output)
            support.require_workspace(output)
            for name in old_results:
                self.assertFalse((output / name).exists(), name)
            self.assertEqual(original, {name: support.sha256(output / name) for name in preserved})
            self.assertEqual((output / 'reference_manifests/iris.json').read_bytes(), reference)
            # A subsequent fresh run must work with existing reference manifests.
            (output / 'runs/second.json').write_text('{}')
            reproduce.initialise_workspace(output)
            support.require_workspace(output)
            self.assertFalse((output / 'runs/second.json').exists())
            self.assertEqual((output / 'reference_manifests/iris.json').read_bytes(), reference)

    def test_default_cli_runs_full_pipeline_in_script_folder(self):
        output = self.workspace()
        old_results = self.saved_results(output)
        def check_environment(path, need_cuda):
            self.assertEqual(path, output)
            self.assertTrue(need_cuda)
            self.assertTrue((path / 'REPORT.txt').exists())
        def run_script(path, script, *args, **kwargs):
            self.assertEqual(path, output)
            for name in old_results:
                self.assertFalse((path / name).exists(), name)
            support.require_workspace(path)
        with patch.object(reproduce, 'HERE', output), \
             patch.object(sys, 'argv', ['reproduce.py']), \
             patch.object(reproduce, 'check_environment', side_effect=check_environment), \
             patch.object(reproduce, 'run_script', side_effect=run_script) as runner:
            reproduce.main()
        self.assertEqual([call.args[1] for call in runner.call_args_list],
                         ['prepare_full.py', 'audit_data.py', 'verify_preprocessing.py',
                          'verify_backend.py', 'verify_cached_graph.py', 'manage.py'])
        self.assertEqual(runner.call_args_list[-1].kwargs, {'manager': True})

    def test_resume_cli_preserves_state_in_place_and_in_separate_output(self):
        for separate in (False, True):
            with self.subTest(separate=separate):
                output = self.workspace('resume-' + str(separate))
                names = self.saved_results(output)
                names.remove('CANCEL')
                (output / 'CANCEL').unlink()
                names.append('REPRODUCTION.json')
                hashes = {name: support.sha256(output / name) for name in names}
                arguments = ['--output', str(output)] if separate else []
                with patch.object(reproduce, 'HERE', ROOT if separate else output), \
                     patch.object(sys, 'argv', ['reproduce.py', *arguments, '--resume']), \
                     patch.object(reproduce, 'check_environment'), \
                     patch.object(reproduce, 'run_script'):
                    reproduce.main()
                self.assertEqual(hashes, {name: support.sha256(output / name) for name in names})

    def test_default_cli_environment_failure_preserves_results(self):
        output = self.workspace()
        names = self.saved_results(output)
        hashes = {name: support.sha256(output / name) for name in names}
        with patch.object(reproduce, 'HERE', output), \
             patch.object(sys, 'argv', ['reproduce.py']), \
             patch.object(reproduce, 'check_environment', side_effect=RuntimeError('environment mismatch')), \
             patch.object(reproduce, 'run_script') as runner:
            with self.assertRaisesRegex(RuntimeError, 'environment mismatch'):
                reproduce.main()
            runner.assert_not_called()
        self.assertEqual(hashes, {name: support.sha256(output / name) for name in names})

    def test_launcher_lock_precedes_reset_and_keeps_same_inode(self):
        output = self.workspace()
        self.saved_results(output)
        lock_path = output / 'logs/reproduce.lock'
        with lock_path.open('a') as lock, \
             patch.object(reproduce, 'HERE', output), \
             patch.object(sys, 'argv', ['reproduce.py', '--prepare-only', '--datasets', 'iris']), \
             patch.object(reproduce, 'check_environment') as environment, \
             patch.object(reproduce, 'run_script') as runner:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            inode = lock_path.stat().st_ino
            with self.assertRaises(BlockingIOError):
                reproduce.main()
            self.assertTrue((output / 'REPORT.txt').exists())
            environment.assert_not_called()
            runner.assert_not_called()
            fcntl.flock(lock, fcntl.LOCK_UN)
            def verify_lock(*args, **kwargs):
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            runner.side_effect = verify_lock
            reproduce.main()
            self.assertEqual(lock_path.stat().st_ino, inode)
            self.assertEqual([call.args for call in runner.call_args_list],
                             [(output, 'prepare_full.py', '--datasets', 'iris'),
                              (output, 'audit_data.py', '--datasets', 'iris')])
            environment.assert_called_once_with(output, need_cuda=False)

    def test_cancelled_resume_preserves_existing_state(self):
        output = self.workspace()
        names = self.saved_results(output)
        hashes = {name: support.sha256(output / name) for name in names}
        with patch.object(reproduce, 'HERE', output), \
             patch.object(sys, 'argv', ['reproduce.py', '--resume']), \
             patch.object(reproduce, 'run_script') as runner:
            with self.assertRaisesRegex(RuntimeError, 'remove its CANCEL file'):
                reproduce.main()
            runner.assert_not_called()
        self.assertEqual(hashes, {name: support.sha256(output / name) for name in names})

    def test_two_fresh_runs_have_independent_files(self):
        first, second = self.workspace('a'), self.workspace('b')
        (first / 'runs/sentinel.json').write_text('{}')
        (first / 'checkpoints/sentinel.pt').write_bytes(b'checkpoint')
        self.assertEqual(list((second / 'runs').iterdir()), [])
        self.assertEqual(list((second / 'checkpoints').iterdir()), [])
        self.assertFalse((first / 'train_full.py').samefile(second / 'train_full.py'))

    def fixture_manifest(self):
        data = self.root / 'data/example'
        data.mkdir(parents=True)
        array = data / 'train_x.npy'
        array.write_bytes(b'array fixture')
        files = {array.name: {'bytes': array.stat().st_size, 'sha256': support.sha256(array)}}
        manifest = {'data_sha256': 'fixture', 'files': files}
        (data / 'manifest.json').write_text(json.dumps(manifest))
        (self.root / 'reference_manifests').mkdir()
        (self.root / 'reference_manifests/example.json').write_text(json.dumps(manifest))
        return array

    def test_cleaned_or_corrupt_arrays_are_not_skipped(self):
        array = self.fixture_manifest()
        with patch.object(support, 'HERE', self.root):
            self.assertTrue(support.prepared_data_matches('example'))
            array.write_bytes(b'wrong content')
            self.assertFalse(support.prepared_data_matches('example'))
            array.unlink()
            self.assertFalse(support.prepared_data_matches('example'))

    def test_download_verifies_content_and_does_not_accept_changed_source(self):
        destination = self.root / 'download.csv'
        original = self.root / 'expected.csv'
        original.write_bytes(b'correct dataset')
        meta = {'bytes': original.stat().st_size, 'sha256': support.sha256(original)}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(original.read_bytes())) as request:
            support.download_verified('https://example.invalid/data', destination, meta)
            self.assertEqual(destination.read_bytes(), original.read_bytes())
            self.assertEqual(request.call_count, 1)
        with patch('urllib.request.urlopen', side_effect=AssertionError('should use verified cache')):
            support.download_verified('https://example.invalid/data', destination, meta)
        destination.write_bytes(b'old corrupt cache')
        with patch('urllib.request.urlopen', return_value=io.BytesIO(b'changed source')):
            with self.assertRaisesRegex(ValueError, 'checksum/size mismatch'):
                support.download_verified('https://example.invalid/data', destination, meta)
        self.assertEqual(destination.read_bytes(), b'old corrupt cache')
        self.assertFalse(destination.with_name('download.csv.part').exists())

    def test_rebuilt_arrays_must_match_archived_reference(self):
        manifest = json.loads((ROOT / 'data/iris/manifest.json').read_text())
        with patch.object(support, 'expected_manifest', return_value=manifest):
            support.check_prepared_manifest('iris', manifest)
            wrong = dict(manifest, data_sha256='changed')
            with self.assertRaisesRegex(ValueError, 'differs from the archived experiment'):
                support.check_prepared_manifest('iris', wrong)

    def test_bundled_inputs_match_original_source_hashes(self):
        for dataset, filename in [('titanic', 'titanic_openml_40945.csv'), ('new-thyroid', 'new-thyroid.csv')]:
            self.assertEqual(support.bundled_source(dataset, filename), ROOT / 'inputs' / filename)

    def test_manager_gets_own_process_group_and_failure_propagates(self):
        output = self.root / 'manager-test'
        (output / 'logs').mkdir(parents=True)
        (output / 'manage.py').write_text(
            "import os,json,sys\n"
            "from pathlib import Path\n"
            "Path('group.json').write_text(json.dumps([os.getpid(),os.getpgrp()]))\n"
            "sys.exit(7)\n")
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            reproduce.run_script(output, 'manage.py', manager=True)
        self.assertEqual(caught.exception.returncode, 7)
        pid, group = json.loads((output / 'group.json').read_text())
        self.assertEqual(pid, group)
        self.assertNotEqual(group, os.getpgrp())

    def test_cleaner_preserves_bundled_sources_and_removes_new_checkpoints(self):
        from clean_population_pertubation_train_convergence_data import removal_reason
        self.assertIsNone(removal_reason(Path('inputs/titanic_openml_40945.csv')))
        self.assertIsNone(removal_reason(Path('reference_manifests/iris.json')))
        self.assertIsNotNone(removal_reason(Path('checkpoints/iris_population_64_seed11.pt')))
        self.assertIsNotNone(removal_reason(Path('sources/sklearn/covertype/samples_py3')))

    def test_archive_cleanup_removes_transient_status_and_preserves_evidence(self):
        import clean_population_pertubation_train_convergence_data as cleaner
        output = self.workspace()
        transient = {'FINALISATION.json', 'verification/latest_monitoring_check.json'}
        evidence = [
            'environment_runs.json', 'REPORT.txt', 'summary.json', 'summary.csv',
            'per_seed.csv', 'training_loss_grid.png', 'figures/iris.svg',
            'runs/iris_backprop_adam_seed11.json', 'logs/manager.log',
            'logs/failure_iris.json', 'verification/backend.json',
            'verification/metrics_iris_backprop_adam_seed11.json',
            'verification/initial_visual_review.json',
            'verification/portability/checks.json', 'batched_verification.json',
            # Similar names at other paths are not transient-status matches.
            'logs/FINALISATION.json', 'latest_monitoring_check.json',
            'verification/archive/latest_monitoring_check.json',
        ]
        for name in [*transient, *evidence]:
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture: ' + name)
        protected = {path.relative_to(output).as_posix(): path.read_bytes()
                     for path in output.rglob('*') if path.is_file()
                     and path.relative_to(output).as_posix() not in transient}
        previous = {'format': cleaner.REPORT_FORMAT, 'cleanups': []}
        (output / 'CLEANUP.json').write_text(json.dumps(previous))
        removals, directories = cleaner.plan_cleanup(output)
        self.assertEqual({item.path for item in removals}, transient)
        removed, _, errors = cleaner.clean(output, removals, directories)
        self.assertEqual(errors, [])
        self.assertEqual({item['path'] for item in removed}, transient)
        for name in transient:
            self.assertFalse((output / name).exists(), name)
        for name, content in protected.items():
            self.assertEqual((output / name).read_bytes(), content, name)
        record = json.loads((output / 'CLEANUP.json').read_text())
        self.assertEqual(record['format'], cleaner.REPORT_FORMAT)
        self.assertEqual(len(record['cleanups']), 1)
        self.assertEqual({item['path'] for item in record['cleanups'][0]['deleted_files']}, transient)

    def test_huggingface_uses_pinned_url(self):
        manifest = json.loads((ROOT / 'data/iris/manifest.json').read_text())
        source = manifest['sources'][0]
        with patch.object(support, 'download_verified', return_value=Path('/unused')) as download:
            support.hub_source('scikit-learn/iris', 'Iris.csv')
        url, destination, metadata = download.call_args.args
        self.assertEqual(url, source['url'])
        self.assertEqual(metadata['sha256'], source['sha256'])
        self.assertEqual(len(destination.parent.name), 40)


if __name__ == '__main__':
    unittest.main()
