"""Checks that observation does not change learning and survives an epoch pause."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import torch
from torch.utils.data import TensorDataset, DataLoader
from snn_training.train_mnist import main, _metrics_detailed
from snn_training.torch_model import HardwareQATSNN
from snn_reference import SNNConfig, NumericConfig, NeuronModel
from snn_training.observability_campaign import build_jobs, verify_history

class TrainingObservabilityTests(unittest.TestCase):

    def test_hh_training_evaluation_without_expensive_diagnostics(self):
        config = SNNConfig(3, 2, 2, NumericConfig(), hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY, output_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        model = HardwareQATSNN(config, 1)
        loader = DataLoader(TensorDataset(torch.ones(2, 1, 1, 3), torch.tensor([0, 1])), batch_size=2)
        fast = _metrics_detailed(model, loader, torch.device('cpu'), detailed_hh=False)
        detailed = _metrics_detailed(model, loader, torch.device('cpu'))
        self.assertEqual(fast['accuracy'], detailed['accuracy'])
        self.assertEqual(fast['cross_entropy'], detailed['cross_entropy'])
        self.assertNotIn('hh_lut_clamps', fast)

    def run_training(self, output, extra):
        generator = torch.Generator().manual_seed(42)
        dataset = TensorDataset(torch.rand(24, 1, 28, 28, generator=generator), torch.arange(24) % 10)
        arguments = ['train', '--output-dir', str(output), '--device', 'cpu', '--epochs', '2', '--batch-size', '8', '--hidden-neurons', '4', '--timesteps', '2', '--validation-samples', '8', '--seed', '11', '--input-encoding', 'rate', '--rate-seed', '1', *extra]
        with mock.patch('sys.argv', arguments), mock.patch('snn_training.train_mnist.datasets.MNIST', return_value=dataset), mock.patch('builtins.print'):
            main()

    def test_observation_preserves_weights_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.run_training(root / 'original', [])
            self.run_training(root / 'observed', ['--training-observability'])
            self.run_training(root / 'resumed', ['--training-observability', '--stop-after-epoch', '1'])
            self.assertFalse((root / 'resumed' / 'manifest.json').exists())
            self.run_training(root / 'resumed', ['--training-observability', '--resume-checkpoint', str(root / 'resumed' / 'checkpoint.pt')])
            original = torch.load(root / 'original' / 'checkpoint.pt', weights_only=True)
            observed = torch.load(root / 'observed' / 'checkpoint.pt', weights_only=True)
            resumed = torch.load(root / 'resumed' / 'checkpoint.pt', weights_only=True)
            for name in original['model']:
                self.assertTrue(torch.equal(original['model'][name], observed['model'][name]))
                self.assertTrue(torch.equal(observed['model'][name], resumed['model'][name]))
            for output in (root / 'observed', root / 'resumed'):
                result = verify_history(output, completed=True)
                self.assertEqual(result['epochs'], 2)
                self.assertEqual(len(list((output / 'epoch_weights').glob('*.pt'))), 3)
                rows = [json.loads(p.read_text()) for p in sorted((output / 'epoch_metrics').glob('*.json'))]
                self.assertEqual(rows[1]['train_evaluation']['samples'], 16)
                self.assertEqual(rows[1]['validation']['samples'], 8)
                self.assertGreater(rows[2]['cumulative_epoch_wall_seconds'], rows[1]['epoch_wall_seconds'])

    def test_intermediate_stage_skips_test(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_training(Path(directory), ['--training-observability', '--skip-final-test'])
            result = verify_history(Path(directory), completed=True)
            self.assertEqual(result['test'], {'not_evaluated': True})
if __name__ == '__main__':
    unittest.main()
