"""CPU parity checks against the pre-validation controller and mode integration.

fixtures/training_convergence_before_validation.py is the unmodified production
controller from b042c5a (immediately before the validation upgrade).
"""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest

import torch

from test_validation_convergence import ROOT, PRODUCTION, load_module, metrics, settings


class TrainingParityTests(unittest.TestCase):
    def modules(self, **overrides):
        config = settings(trainSetLossOptimisation=True, **overrides)
        original = load_module(ROOT / 'tests/fixtures/training_convergence_before_validation.py', config)
        current = load_module(PRODUCTION / 'LREANNpt_SUANN_trainingConvergence.py', config)
        return original, current

    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(actual, expected)

    def assert_controller_equal(self, original, current):
        for key, value in original.state_dict().items():
            actual = current.state_dict()[key]
            if key == 'policy':
                actual = {k: actual[k] for k in value}
            elif key == 'best' and actual is not None:
                actual.pop('validation')  #Additional diagnostic metadata only.
            self.assert_nested_equal(actual, value)

    def test_original_fixture_is_unmodified(self):
        fixture = ROOT / 'tests/fixtures/training_convergence_before_validation.py'
        self.assertEqual(hashlib.sha256(fixture.read_bytes()).hexdigest(),
                         'd7a5d875636566f65b40e0b33f1cf630da752784bcc8a2f730224e091614b139')

    def test_original_training_decisions_weights_adam_state_and_resume_match(self):
        for adam in (False, True):
            for scenario in ('plateau', 'perfect', 'recovery'):
                with self.subTest(adam=adam, scenario=scenario):
                    modules = self.modules(populationPertubationMinimumTrainingIterations=3,
                                           populationPertubationTrainingLearningRateReductions=2,
                                           populationPertubationTrainingMaxNumericalRecoveries=3)
                    models = [torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)]
                    models[1].load_state_dict(models[0].state_dict())
                    optimizers = [torch.optim.Adam(m.parameters(), lr=.1) if adam else None for m in models]
                    controllers = [module.PopulationPertubationTrainingConvergence(.1) for module in modules]
                    for step in range(30):
                        train = metrics(1., .001) if scenario == 'perfect' else metrics(.7, 1. / (min(step, 1) + 1))
                        #Validation cannot affect the original training policy, even if invalid.
                        validation = None if step % 2 else metrics(float('nan'), float('nan'))
                        for i, (c, model, optimizer) in enumerate(zip(controllers, models, optimizers)):
                            if step:
                                if optimizer is not None:
                                    optimizer.zero_grad()
                                    model(torch.ones(2, 1)).square().sum().backward()
                                    optimizer.step()
                                else:
                                    with torch.no_grad():
                                        model.weight.add_(.01)
                            if scenario == 'recovery' and step == 2:
                                with torch.no_grad():
                                    model.weight.fill_(float('nan'))
                                c.recoverNonfinite(step, model, optimizer)
                            elif i == 0:
                                c.observe(step, train, model, optimizer)
                            else:
                                c.observe(step, train, validation, model, optimizer)
                        self.assert_controller_equal(*controllers)
                        self.assert_nested_equal(models[0].state_dict(), models[1].state_dict())
                        if adam:
                            self.assert_nested_equal(optimizers[0].state_dict(), optimizers[1].state_dict())
                        if step == 2:
                            for i, module in enumerate(modules):
                                state = controllers[i].state_dict()
                                controllers[i] = module.PopulationPertubationTrainingConvergence(.1)
                                controllers[i].load_state_dict(state)
                        if controllers[0].stopReason is not None:
                            break
                    self.assertIsNotNone(controllers[0].stopReason)
                    if scenario == 'perfect':
                        self.assertEqual(step, 3)
                        self.assertEqual(controllers[1].stopReason, 'perfect_train_fit')
                    for c, model, optimizer in zip(controllers, models, optimizers):
                        c.restoreBest(model, optimizer)
                    self.assert_nested_equal(models[0].state_dict(), models[1].state_dict())

    def test_training_thresholds_and_loss_ties_match_original(self):
        original, current = self.modules(populationPertubationTrainingPatience=100,
                                         populationPertubationTrainingRelativeMinDelta=.125,
                                         populationPertubationTrainingMinDelta=.0625)
        model = torch.nn.Linear(1, 1)
        a, b = [module.PopulationPertubationTrainingConvergence(.1) for module in (original, current)]
        for step, loss in enumerate((1., .875, .874, .874, .5, .4375, .4374)):
            train = metrics(.5 + .01 * step, loss)
            a.observe(step, train, model)
            b.observe(step, train, metrics(1., 0.), model)
            self.assert_controller_equal(a, b)
        self.assertEqual(b.best['step'], 6)
        self.assertEqual(b.lastSignificantIteration, 6)

    def test_production_loop_preserves_batches_parameters_and_rng_without_validation(self):
        original, current = self.modules(populationPertubationMinimumTrainingIterations=3,
                                         populationPertubationTrainingMinDelta=1.)
        class RandomDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 9
            def __getitem__(self, index):
                return torch.rand(1) + index, index % 2
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.1))
            def forward(self, train, x, y, optimizer, layer):
                self.Ztrace = [torch.cat((x * self.weight, -x * self.weight), dim=1)]
                return torch.nn.functional.cross_entropy(self.Ztrace[-1], y), 0.
        outcomes = []
        for module in (original, current):
            torch.manual_seed(20261003)
            model = Model()
            batches = []
            def update(model, is_train, x, y, optimizer, layer):
                batches.append((x.clone(), y.clone()))
                with torch.no_grad():
                    model.weight.add_(torch.rand(()) * .01)
                return model(is_train, x, y, optimizer, layer)
            algorithm = types.SimpleNamespace(populationPertubationLearningRate=.1, trainOrTestModel=update)
            args = (RandomDataset(), model, algorithm) if module is original else (RandomDataset(), None, model, algorithm)
            with contextlib.redirect_stdout(io.StringIO()):
                result = module.trainPopulationPertubationUntilConverged(*args)
            self.assertEqual(algorithm.populationPertubationLearningRate, .1)
            outcomes.append((result, model.state_dict(), batches, torch.get_rng_state()))
        old, new = outcomes
        new[0].pop('validation')
        new[0]['policy'] = {k: new[0]['policy'][k] for k in old[0]['policy']}
        self.assert_nested_equal(new, old)

    def test_resume_rejects_other_mode_and_former_accuracy_policy(self):
        _, train = self.modules()
        validation = load_module(PRODUCTION / 'LREANNpt_SUANN_trainingConvergence.py', settings())
        a = train.PopulationPertubationTrainingConvergence(.1)
        b = validation.PopulationPertubationTrainingConvergence(.1)
        for target, state in ((a, b.state_dict()), (b, a.state_dict())):
            with self.assertRaisesRegex(ValueError, 'different convergence policy'):
                target.load_state_dict(state)
        old = b.state_dict()
        old['policy']['selection_metric'] = 'validation_accuracy'
        with self.assertRaisesRegex(ValueError, 'different convergence policy'):
            b.load_state_dict(old)


class BenchmarkModeTests(unittest.TestCase):
    def test_protocol_mode_reaches_adam_and_population_before_config_loading(self):
        #Isolate imports of production globals; CUDA is disabled in each child.
        script = '''
import json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import model_setup
root = Path(sys.argv[1])
protocol = json.loads((root/'protocol.json').read_text())
protocol['trainSetLossOptimisation'] = sys.argv[2] == 'True'
original = Path.read_text
def read(path, *args, **kwargs):
    return json.dumps(protocol) if path == root/'protocol.json' else original(path, *args, **kwargs)
with patch.object(Path, 'read_text', read):
    definitions, algorithm, convergence = model_setup.load_modules(sys.argv[3], 'iris')
controller = convergence.PopulationPertubationTrainingConvergence(.1)
assert definitions.trainSetLossOptimisation is protocol['trainSetLossOptimisation']
assert definitions.useStochasticUpdates is (sys.argv[3] != 'backprop_adam')
assert controller.policy['selection_metric'] == ('train_loss' if protocol['trainSetLossOptimisation'] else 'validation_loss')
assert controller.policy['minimum_updates'] == 2000
assert controller.policy['patience'] == 2000
assert controller.policy['min_delta'] == .0001
assert controller.policy['relative_min_delta'] == .001
assert controller.policy['lr_reductions'] == 3
assert controller.policy['max_numerical_recoveries'] == 5
'''
        for train_loss in (True, False):
            for method in ('backprop_adam', 'population_64'):
                result = subprocess.run([sys.executable, '-B', '-c', script, str(ROOT), str(train_loss), method],
                                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1'},
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_invalid_boolean_rejected(self):
        module = load_module(PRODUCTION / 'LREANNpt_SUANN_trainingConvergence.py', settings(trainSetLossOptimisation='false'))
        with self.assertRaisesRegex(ValueError, 'boolean'):
            module.PopulationPertubationTrainingConvergence(.1)

    def test_verifier_accepts_training_perfect_fit_and_rejects_validation_perfect_fit(self):
        verify = load_module(ROOT / 'verify_results.py')
        point = dict(step=3, train=metrics(1., .005), validation=metrics(.5, 2.))
        result = dict(protocol={'trainSetLossOptimisation': True}, curves=[point], selected_step=3,
                      train=point['train'], validation=point['validation'], stopping_train=point['train'],
                      stop_reason='perfect_train_fit', convergence_policy={'loss_goal': .01})
        verify.verify_selection(result, point)
        result['protocol']['trainSetLossOptimisation'] = False
        with self.assertRaises(AssertionError):
            verify.verify_selection(result, point)


if __name__ == '__main__':
    unittest.main()
