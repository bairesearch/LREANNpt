"""CPU-only regression checks for validation selection, stopping, and reporting."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT.parent / 'LREANNpt'
FROZEN = ROOT / 'source/LREANNpt'
torch.set_num_threads(1)


def load_module(path, settings=None):
    spec = importlib.util.spec_from_file_location('validation_test_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    original = sys.modules.get('ANNpt_globalDefs')
    if settings is not None:
        sys.modules['ANNpt_globalDefs'] = settings
    try:
        spec.loader.exec_module(module)
    finally:
        if settings is not None:
            if original is None:
                sys.modules.pop('ANNpt_globalDefs', None)
            else:
                sys.modules['ANNpt_globalDefs'] = original
    return module


def settings(**overrides):
    module = types.ModuleType('ANNpt_globalDefs')
    values = dict(
        populationPertubationOptimiseTrainingIterations=True,
        trainSetLossOptimisation=False,
        populationPertubationTrainingPatience=2,
        populationPertubationTrainingMinDelta=0.0001,
        populationPertubationTrainingRelativeMinDelta=0.001,
        populationPertubationTrainingLossGoal=0.01,
        populationPertubationEvaluateEveryIterations=1,
        populationPertubationMinimumTrainingIterations=1,
        populationPertubationValidationPatience=2,
        populationPertubationValidationMinDelta=0.0001,
        populationPertubationValidationRelativeMinDelta=0.001,
        populationPertubationTrainingLearningRateFactor=0.2,
        populationPertubationTrainingLearningRateReductions=1,
        populationPertubationTrainingMaxNumericalRecoveries=1,
        populationPertubationValidationSplitSize=0.2,
        populationPertubationValidationSplitSeed=20260930,
        useTabularDataset=False, useImageDataset=True, useNLPDataset=False,
        batchSize=4, device=torch.device('cpu'), disableDatasetCache=False)
    values.update(overrides)
    vars(module).update(values)
    return module


def metrics(accuracy, loss):
    return {'accuracy': accuracy, 'loss': loss, 'rows': 100}


class ValidationConvergenceTests(unittest.TestCase):
    def controller(self, **overrides):
        self.module = load_module(PRODUCTION / 'LREANNpt_SUANN_trainingConvergence.py', settings(**overrides))
        self.model = torch.nn.Linear(1, 1)
        return self.module.PopulationPertubationTrainingConvergence(0.1)

    def test_selection_uses_validation_loss_regardless_of_accuracy_or_training_fit(self):
        c = self.controller(populationPertubationValidationPatience=100)
        c.observe(0, metrics(.7, .8), metrics(.8, .4), self.model)
        c.observe(1, metrics(1., .001), metrics(.7, .1), self.model)
        self.assertEqual(c.best['step'], 1)  #Lower loss wins despite worse accuracy.
        c.observe(2, metrics(.6, 1.), metrics(.9, .9), self.model)
        self.assertEqual(c.best['step'], 1)
        c.observe(3, metrics(.5, 2.), metrics(.9, .05), self.model)
        self.assertEqual(c.best['step'], 3)
        c.observe(4, metrics(1., .001), metrics(1., .05), self.model)
        self.assertEqual(c.best['step'], 3)  #Higher accuracy cannot break a loss tie.
        c.observe(5, metrics(1., .001), metrics(1., .001), self.model)
        self.assertIsNone(c.stopReason)  #Neither perfect train nor validation fit stops.

    def test_validation_loss_plateau_stops_despite_training_loss_and_accuracy_improving(self):
        c = self.controller()
        for step in range(5):
            c.observe(step, metrics(1., .001 / (step + 1)), metrics(.5 + .1 * step, .4), self.model)
        self.assertEqual(c.stopReason, 'validation_loss_plateau_after_lr_reductions')
        self.assertEqual(c.best['step'], 0)
        self.assertEqual(c.events[0]['step'], 2)
        self.assertEqual(c.events[0]['event'], 'validation_plateau_reduce_lr_restore_best')
        self.assertAlmostEqual(c.learningRate, .02)

    def test_significant_loss_improvement_resets_patience_at_fixed_accuracy(self):
        c = self.controller(populationPertubationValidationPatience=3)
        c.observe(0, metrics(.8, .4), metrics(.8, .4), self.model)
        c.observe(1, metrics(.8, .4), metrics(.8, .3998), self.model)
        self.assertEqual(c.best['step'], 1)
        self.assertEqual(c.lastSignificantIteration, 0)
        c.observe(2, metrics(.8, .4), metrics(.8, .3995), self.model)
        self.assertEqual(c.lastSignificantIteration, 2)
        c.observe(3, metrics(1., .001), metrics(.8, .2), self.model)
        self.assertEqual(c.lastSignificantIteration, 3)
        self.assertEqual(c.reductions, 0)

    def test_minimum_complete_pass_precedes_plateau_actions(self):
        self.controller(populationPertubationTrainingLearningRateReductions=0)
        c = self.module.PopulationPertubationTrainingConvergence(.1, 6)
        for step in range(6):
            self.assertIsNone(c.observe(step, metrics(1., 0.), metrics(.8, .4), self.model))
        self.assertIsNotNone(c.observe(6, metrics(1., 0.), metrics(.8, .4), self.model))

    def test_adam_weights_and_moments_restore_from_validation_checkpoint(self):
        c = self.controller()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.1)
        def update():
            optimizer.zero_grad()
            self.model(torch.ones(1, 1)).square().sum().backward()
            optimizer.step()
        update()
        c.observe(0, metrics(.7, .7), metrics(.9, .2), self.model, optimizer)
        expected = c.state_dict()['best']
        for step in (1, 2):
            update()
            c.observe(step, metrics(1., .001), metrics(.8, .3), self.model, optimizer)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, expected['model'][name], rtol=0, atol=0)
        for key, state in optimizer.state_dict()['state'].items():
            for name, tensor in state.items():
                torch.testing.assert_close(tensor, expected['optimizer']['state'][key][name], rtol=0, atol=0)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], .02)

    def test_resume_preserves_patience_and_rejects_old_policy(self):
        c = self.controller()
        c.observe(0, metrics(.7, .7), metrics(.8, .4), self.model)
        c.observe(1, metrics(.9, .2), metrics(.7, .3), self.model)
        state = c.state_dict()
        resumed = self.module.PopulationPertubationTrainingConvergence(.1)
        resumed.load_state_dict(state)
        for step in (2, 3, 4, 5):
            for controller in (c, resumed):
                controller.observe(step, metrics(1., .001), metrics(.7, .3), self.model)
        self.assertEqual(c.stopReason, 'validation_loss_plateau_after_lr_reductions')
        self.assertEqual(c.events, resumed.events)
        self.assertEqual(c.stopReason, resumed.stopReason)
        self.assertEqual(c.best['step'], resumed.best['step'])
        old = copy.deepcopy(state)
        del old['policy']['selection_metric']
        with self.assertRaisesRegex(ValueError, 'different convergence policy'):
            resumed.load_state_dict(old)

    def test_invalid_validation_and_nonfinite_recovery(self):
        c = self.controller()
        with self.assertRaisesRegex(ValueError, 'validation metrics'):
            c.observe(0, metrics(.7, .7), None, self.model)
        c.observe(0, metrics(.7, .7), metrics(.8, .4), self.model)
        with self.assertRaises(FloatingPointError):
            c.observe(1, metrics(1., 0.), metrics(.9, float('nan')), self.model)
        with torch.no_grad():
            self.model.weight.fill_(float('nan'))
        c.recoverNonfinite(1, self.model)
        self.assertTrue(torch.isfinite(self.model.weight).all())
        self.assertEqual(c.best['step'], 0)
        self.assertIsNone(c.stopReason)
        with self.assertRaisesRegex(FloatingPointError, 'exhausted'):
            c.recoverNonfinite(2, self.model)

    def test_production_loop_uses_separate_validation_and_restores_selected_model(self):
        self.controller()
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(-1.))
            def forward(self, train, x, y, optimizer, layer):
                logits = torch.stack((x[:, 0] * self.weight, torch.zeros(len(x))), dim=1)
                self.Ztrace = [logits]
                return torch.nn.functional.cross_entropy(logits, y), 0.
        model = Model()
        train = torch.utils.data.TensorDataset(torch.ones(4, 1), torch.zeros(4, dtype=torch.long))
        validation = torch.utils.data.TensorDataset(torch.ones(3, 1), torch.ones(3, dtype=torch.long))
        def update(model, is_train, x, y, optimizer, layer):
            self.assertTrue(is_train)
            self.assertTrue((y == 0).all())  #Validation labels never reach updates.
            with torch.no_grad():
                model.weight.add_(2.)
            return model(is_train, x, y, optimizer, layer)
        algorithm = types.SimpleNamespace(populationPertubationLearningRate=.1, trainOrTestModel=update)
        with contextlib.redirect_stdout(io.StringIO()):
            result = self.module.trainPopulationPertubationUntilConverged(train, validation, model, algorithm)
        self.assertEqual(result['iterations'], 4)
        self.assertEqual(result['selected_iteration'], 0)
        self.assertEqual(result['validation']['accuracy'], 1.)
        self.assertEqual(result['validation']['rows'], 3)  #Includes partial final batch.
        self.assertEqual(result['train']['accuracy'], 0.)
        self.assertEqual(model.weight.item(), -1.)
        self.assertEqual(algorithm.populationPertubationLearningRate, .1)
        for invalid in (None, train):
            with self.assertRaisesRegex(ValueError, 'separate validation'):
                self.module.trainPopulationPertubationUntilConverged(train, invalid, model, algorithm)

    def test_frozen_controller_and_source_hashes_match(self):
        protocol = json.loads((ROOT / 'protocol.json').read_text())
        import hashlib
        for name in ('LREANNpt_SUANN_trainingConvergence.py', 'ANNpt_data.py', 'ANNpt_main.py'):
            self.assertEqual((PRODUCTION / name).read_bytes(), (FROZEN / name).read_bytes())
        for name, expected in protocol['source_hashes'].items():
            self.assertEqual(hashlib.sha256((FROZEN / name).read_bytes()).hexdigest(), expected, name)


class ValidationDataTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings(useTabularDataset=True, useImageDataset=False)
        self.data = load_module(PRODUCTION / 'ANNpt_data.py', self.settings)

    def test_split_is_deterministic_disjoint_and_keeps_rare_classes_in_training(self):
        import numpy as np
        labels = np.array([0] * 19 + [1])
        train, validation = self.data.populationValidationIndices(labels)
        again = self.data.populationValidationIndices(labels)
        np.testing.assert_array_equal(train, again[0])
        np.testing.assert_array_equal(validation, again[1])
        self.assertFalse(set(train) & set(validation))
        self.assertEqual(set(train) | set(validation), set(range(20)))
        self.assertIn(19, train)
        self.assertGreater(len(validation), 0)

    def test_tabular_holdout_precedes_preprocessing_and_repetition(self):
        self.check_tabular_holdout(False)

    def test_training_loss_keeps_full_training_rows_and_preprocessing(self):
        self.check_tabular_holdout(True)

    def check_tabular_holdout(self, train_loss):
        self.settings.trainSetLossOptimisation = train_loss
        import numpy as np
        from datasets import Dataset, DatasetDict, disable_progress_bars
        disable_progress_bars()
        labels = [0, 1] * 10
        train, val = self.data.populationValidationIndices(labels)
        values = np.arange(20, dtype=float)
        values[val] = 1000.
        categories = ['known'] * 20
        for i in val:
            categories[i] = 'validation_only'
        raw = DatasetDict(train=Dataset.from_dict({'value': values, 'category': categories, 'label': labels}),
                          test=Dataset.from_dict({'value': [2000., 2000.], 'category': ['test_only'] * 2, 'label': [0, 1]}))
        options = dict(datasetName='fixture', datasetNameFull='fixture', datasetLocalFile=False,
                       datasetSpecifyDataFiles=False, datasetHasSubsetType=False, datasetHasTestSplit=True,
                       debugCullDatasetSamples=False, datasetSplitNameTrain='train', datasetSplitNameTest='test',
                       datasetConvertFeatureValues=True, datasetConvertClassValues=False,
                       datasetConvertClassTargetColumnFloatToInt=False, datasetEqualiseClassSamples=False,
                       datasetNormalise=True, datasetNormaliseMinMax=True, datasetNormaliseStdAvg=False,
                       datasetCorrectMissingValues=False, datasetRepeat=True, datasetRepeatSize=2,
                       datasetShuffle=False, datasetOrderByClass=False, classFieldName='label')
        with patch.multiple(self.data, create=True, **options), \
             patch.object(self.data, 'load_dataset', return_value=raw), \
             contextlib.redirect_stdout(io.StringIO()):
            result = self.data.loadDatasetTabular()
        if train_loss:
            self.assertNotIn('validation', result)
            self.assertEqual(len(result['train']), 40)
            self.assertTrue((result['train']['features'][:, 0] <= 1.).all())
            self.assertEqual(result['train']['features'][:, 0].max(), 1.)
            self.assertEqual(set(result['train']['features'][:, 1].tolist()), {0., 1.})
            return
        self.assertEqual(len(result['train']), len(train) * 2)
        self.assertEqual(len(result['validation']), len(val))
        self.assertEqual(len(result['test']), 4)
        self.assertTrue((result['train']['features'][:, 0] <= 1.).all())
        self.assertTrue((result['validation']['features'][:, 0] > 1.).all())
        self.assertTrue((result['validation']['features'][:, 1] == -1.).all())
        self.assertTrue((result['test']['features'][:, 1] == -1.).all())

    def test_image_validation_uses_training_rows_with_deterministic_transforms(self):
        self.check_image_holdout(False)

    def test_training_loss_keeps_full_image_training_dataset(self):
        self.check_image_holdout(True)

    def check_image_holdout(self, train_loss):
        import numpy as np
        from PIL import Image
        config = settings(trainSetLossOptimisation=train_loss, imageDatasetAugment=True, datasetName='CIFAR10', dataPathName='/unused',
                          datasetSplitNameTrain='train', datasetSplitNameTest='test')
        data = load_module(PRODUCTION / 'ANNpt_data.py', config)
        class CIFARFixture(torch.utils.data.Dataset):
            def __init__(self, root, train, download, transform):
                self.train = train
                self.targets = [0, 1] * (10 if train else 5)
                self.transform = transform
            def __len__(self):
                return len(self.targets)
            def __getitem__(self, index):
                return self.transform(Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))), self.targets[index]
        with patch.object(data.torchvision.datasets, 'CIFAR10', CIFARFixture):
            result = data.loadDatasetImage()
        if train_loss:
            self.assertNotIn('validation', result)
            self.assertEqual((len(result['train']), len(result['test'])), (20, 10))
            self.assertIsInstance(result['train'], CIFARFixture)
            self.assertIsNot(result['train'].transform, result['test'].transform)
            return
        self.assertEqual((len(result['train']), len(result['validation']), len(result['test'])), (16, 4, 10))
        self.assertFalse(set(result['train'].indices) & set(result['validation'].indices))
        self.assertTrue(result['validation'].dataset.train)
        self.assertFalse(result['test'].train)
        self.assertIs(result['validation'].dataset.transform, result['test'].transform)
        self.assertIsNot(result['train'].dataset.transform, result['validation'].dataset.transform)
        torch.testing.assert_close(result['validation'][0][0], result['validation'][0][0], rtol=0, atol=0)


class ValidationReportTests(unittest.TestCase):
    def test_validation_mode_report(self):
        self.check_report(False)

    def test_training_mode_report(self):
        self.check_report(True)

    def check_report(self, train_loss):
        report = load_module(ROOT / 'report.py')
        report.P['trainSetLossOptimisation'] = train_loss
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'runs').mkdir()
            (output / 'logs').mkdir()
            for dataset in report.P['datasets']:
                for method in report.P['methods']:
                    for i, seed in enumerate(report.P['seeds']):
                        record = dict(dataset=dataset, method=method, seed=seed, protocol=report.P,
                                      steps=100, selected_step=10, stop_reason=('training' if train_loss else 'validation') + '_loss_plateau_after_lr_reductions',
                                      training_rows=100, training_seconds=3600. + 60*i, elapsed_seconds=7200. + 120*i, learning_rate_events=[])
                        for split, accuracy in [('train', .90), ('validation', .80), ('test', .70)]:
                            record[split] = dict(metrics(accuracy + .01 * i, .5), balanced_accuracy=accuracy)
                        (output / 'runs' / f'{dataset}_{method}_seed{seed}.json').write_text(json.dumps(record))
            with patch.object(report, 'HERE', output), patch.object(sys, 'argv', ['report.py']), contextlib.redirect_stdout(io.StringIO()):
                report.main()
            text = (output / 'REPORT.txt').read_text()
            mode = 'training' if train_loss else 'validation'
            self.assertIn(f'full-data {mode}-loss convergence', text)
            self.assertIn(f'trainSetLossOptimisation={train_loss}', text)
            self.assertIn(f'minimum {mode} cross-entropy checkpoint', text)
            self.assertNotIn('validation-accuracy', text)
            for split, expected in [('train', '91.00 ± 1.00'), ('validation', '81.00 ± 1.00'), ('test', '71.00 ± 1.00')]:
                section = text.split(f'Final {split}-set accuracy (%)')[1].split('\n\n')[0]
                self.assertIn('Backprop Adam | Population 64 | 256 | 1024 | 4096', section)
                for dataset in report.P['datasets']:
                    self.assertIn('| ' + dataset + ' | ' + ' | '.join([expected] * 5) + ' |', section)
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(len(summary), 65)
            for row in summary:
                for name, expected in dict(training_seconds_mean=3660., training_seconds_sd=60., training_seconds_total=10980.,
                                           elapsed_seconds_mean=7320., elapsed_seconds_sd=120., elapsed_seconds_total=21960.).items():
                    self.assertEqual(row[name], expected)
            for dataset in report.P['datasets']:
                for method in report.P['methods']:
                    self.assertIn(f'| {dataset} | {method} | 91.00 | 81.00 | 71.00 |', text)
            timing_section = text.split('Elapsed run time (H:MM:SS)')[1].split('\n\nActive runs')[0]
            timing_rows = [line for line in timing_section.splitlines() if line.startswith('| ')]
            self.assertEqual(len(timing_rows), 14)  #Header plus one row per dataset.
            self.assertEqual(timing_rows[0], '| Dataset | Backprop Adam | Population 64 | 256 | 1024 | 4096 |')
            for dataset in report.P['datasets']:
                self.assertIn('| ' + dataset + ' | ' + ' | '.join(['2:02:00 ± 0:02:00'] * 5) + ' |', timing_rows)
            self.assertNotIn('Total elapsed time (3 seeds)', text)
            import csv
            with (output / 'summary.csv').open() as handle:
                csv_rows = list(csv.DictReader(handle))
            self.assertEqual(len(csv_rows), 65)
            self.assertEqual(float(csv_rows[0]['training_seconds_total']), 10980.)
            self.assertEqual(float(csv_rows[0]['elapsed_seconds_total']), 21960.)
            self.assertEqual(report.duration(90061.6), '25:01:02')  #No wrap after 24 hours.
            #Incomplete groups stay pending for every split, never partial final means.
            (output / 'runs/iris_population_4096_seed33.json').unlink()
            with patch.object(report, 'HERE', output), patch.object(sys, 'argv', ['report.py']), contextlib.redirect_stdout(io.StringIO()):
                report.main()
            updated = (output / 'REPORT.txt').read_text()
            accuracies, timings = updated.split('Elapsed run time (H:MM:SS)')
            self.assertEqual(accuracies.count('pending (2/3)'), 3)
            timing_rows = [line for line in timings.splitlines() if line.startswith('| ')]
            self.assertEqual(sum(line.count('pending') for line in timing_rows), 1)
            self.assertIn('| iris | ' + ' | '.join(['2:02:00 ± 0:02:00'] * 4 + ['pending']) + ' |', timing_rows)

    def test_old_results_are_rejected_and_verifier_uses_validation_selection(self):
        report = load_module(ROOT / 'report.py')
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'runs').mkdir()
            (output / 'runs/old.json').write_text(json.dumps({'protocol': {'selection': 'training_loss'}}))
            with patch.object(report, 'HERE', output), self.assertRaisesRegex(RuntimeError, 'earlier protocol'):
                report.records()
        verify = load_module(ROOT / 'verify_results.py')
        curves = [dict(step=0, train=metrics(.7, .1), validation=metrics(.9, .5)),
                  dict(step=10, train=metrics(1., .2), validation=metrics(.8, .2))]
        for train_loss in (True, False):
            index = 0 if train_loss else 1
            best = curves[index]
            result = dict(protocol={'trainSetLossOptimisation': train_loss}, curves=curves,
                          selected_step=best['step'], train=best['train'], validation=best['validation'],
                          stop_reason=('training' if train_loss else 'validation') + '_loss_plateau_after_lr_reductions')
            verify.verify_selection(result, best)
            result['selected_step'] = curves[1-index]['step']
            with self.assertRaises(AssertionError):
                verify.verify_selection(result, curves[1-index])



if __name__ == '__main__':
    unittest.main()
