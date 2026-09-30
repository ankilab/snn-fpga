import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock
TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None
if TORCH_AVAILABLE:
    import numpy as np
    import torch
    from snn_reference import NeuronModel, NumericConfig, SNNConfig, TwoLayerSNN
    from snn_reference.hodgkin_huxley_fixed import HodgkinHuxleyFixedNeuron
    from snn_training.encoding import encode_events, encode_latency_events, encode_rate_events
    from snn_training.diagnostics import GradientAccumulator
    from snn_training.train_mnist import BestCheckpoint, _early_stopping_eligible, _planned_stop_reached, _clip_grad_norm_float64, _save_epoch_checkpoint, _restore_cuda_rng_states, _step_plateau_scheduler, _training_phase, _validate_initial_model, layer0_column_mean_penalty
    from snn_training.export import ExportedModel
    from snn_training.torch_model import HardwareQATSNN, latency_spikes, rate_spikes
    from snn_training.hh_qat import HHQATDynamics

@unittest.skipUnless(TORCH_AVAILABLE, 'optional PyTorch training dependencies are absent')
class TorchTrainingTests(unittest.TestCase):

    @staticmethod
    def four_timestep_image():
        return torch.tensor([[[[1.0, 170.0 / 255.0, 85.0 / 255.0, 0.0]]]])

    def test_tensor_encoder_matches_fixed_count_contract(self) -> None:
        images = torch.tensor([[[[1.0, 128.0 / 255.0, 0.0]]]])
        spikes = latency_spikes(images, 5)
        self.assertEqual(tuple(spikes.shape), (5, 1, 3))
        self.assertTrue(torch.equal(spikes.sum(dim=0), torch.ones((1, 3))))
        self.assertEqual(spikes[:, 0, :].argmax(dim=0).tolist(), [0, 1, 4])

    def test_tensor_rate_encoder_matches_event_encoder(self) -> None:
        pixels = [0, 64, 128, 192, 255]
        images = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 1, -1) / 255.0
        spikes = rate_spikes(images, 7, rate_scale=0.8, seed=91)
        tensor_events = tuple(((timestamp, neuron_id) for timestamp in range(spikes.shape[0]) for neuron_id in range(spikes.shape[2]) if spikes[timestamp, 0, neuron_id].item()))
        events = encode_rate_events(pixels, timesteps=7, rate_scale=0.8, seed=91, expected_pixels=5)
        self.assertEqual(tensor_events, tuple(((event.timestamp, event.neuron_id) for event in events)))

    def test_tensor_rate_encoder_is_batch_partition_independent(self) -> None:
        images = torch.tensor([[[[0.25, 0.5]]], [[[0.75, 1.0]]]], dtype=torch.float32)
        batched = rate_spikes(images, 6, seed=17, sample_offset=40)
        first = rate_spikes(images[:1], 6, seed=17, sample_offset=40)
        second = rate_spikes(images[1:], 6, seed=17, sample_offset=41)
        self.assertTrue(torch.equal(batched[:, :1], first))
        self.assertTrue(torch.equal(batched[:, 1:], second))

    def test_additional_tensor_encoders_match_scalar_event_contracts(self) -> None:
        pixels = [0, 1, 64, 128, 200, 255]
        images = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 1, -1) / 255.0
        config = SNNConfig(len(pixels), 1, 1, NumericConfig())
        cases = (('sparse-latency', {'encoding_threshold': 1}), ('nonlinear-latency', {'latency_gamma': 1.5}), ('deterministic-rate', {'rate_scale': 0.75}), ('top-k-rank-order', {'top_k': 3, 'encoding_threshold': 0}), ('burst', {'burst_max_spikes': 3, 'burst_interval': 1, 'encoding_threshold': 0}))
        for input_encoding, parameters in cases:
            with self.subTest(input_encoding=input_encoding):
                model = HardwareQATSNN(config, timesteps=6, input_encoding=input_encoding, **parameters)
                spikes = model.input_spikes(images)
                tensor_events = tuple(((timestamp, neuron_id) for timestamp in range(spikes.shape[0]) for neuron_id in range(spikes.shape[2]) if spikes[timestamp, 0, neuron_id].item()))
                scalar_events = encode_events(pixels, input_encoding=input_encoding, timesteps=6, expected_pixels=len(pixels), **parameters)
                self.assertEqual(tensor_events, tuple(((event.timestamp, event.neuron_id) for event in scalar_events)))

    def test_qat_model_selects_rate_encoding_and_seed(self) -> None:
        config = SNNConfig(64, 1, 1, NumericConfig())
        images = torch.full((1, 1, 1, 64), 0.5, dtype=torch.float32)
        model = HardwareQATSNN(config, timesteps=16, input_encoding='rate', rate_scale=1.0, rate_seed=12)
        self.assertTrue(torch.equal(model.input_spikes(images), rate_spikes(images, 16, rate_scale=1.0, seed=12)))
        original = model.input_spikes(images)
        model.set_rate_seed(91)
        self.assertFalse(torch.equal(original, model.input_spikes(images)))

    def test_qat_model_rejects_invalid_encoding_configuration(self) -> None:
        config = SNNConfig(1, 1, 1, NumericConfig())
        with self.assertRaisesRegex(ValueError, 'input_encoding'):
            HardwareQATSNN(config, input_encoding='unknown')
        with self.assertRaisesRegex(ValueError, 'rate_scale'):
            HardwareQATSNN(config, input_encoding='rate', rate_scale=-1.0)

    def test_qat_model_loads_exported_integer_weight_matrices(self) -> None:
        config = SNNConfig(2, 2, 1, NumericConfig())
        model = HardwareQATSNN(config, timesteps=1)
        model.load_quantized_weights([[1, -2], [3, -4]], [[5], [-6]])
        layer0, layer1 = model.quantized_weights()
        self.assertEqual(layer0.tolist(), [[1, -2], [3, -4]])
        self.assertEqual(layer1.tolist(), [[5], [-6]])
        with self.assertRaisesRegex(ValueError, 'shapes'):
            model.load_quantized_weights([[1]], [[2]])

    def test_rate_qat_forward_matches_integer_reference(self) -> None:
        config = SNNConfig(2, 1, 1, NumericConfig(mem_width=16, current_width=16, threshold=100))
        model = HardwareQATSNN(config, timesteps=4, input_encoding='rate', rate_seed=33)
        with torch.no_grad():
            model.weight_l0.copy_(torch.tensor([[60.0], [60.0]]))
            model.weight_l1.copy_(torch.tensor([[120.0]]))
        images = torch.ones((1, 1, 1, 2))
        counts = model(images)
        reference = TwoLayerSNN(config, [[60], [60]], [[120]])
        events = encode_rate_events([255, 255], timesteps=4, seed=33, expected_pixels=2)
        self.assertEqual(tuple((int(value) for value in counts[0])), reference.infer(events).class_counts)

    def test_additional_encoding_forwards_match_integer_reference(self) -> None:
        pixels = [0, 1, 64, 128, 200, 255]
        images = torch.tensor(pixels, dtype=torch.float32).reshape(1, 1, 1, -1) / 255.0
        config = SNNConfig(len(pixels), 2, 2, NumericConfig(mem_width=16, current_width=16, threshold=100))
        layer0 = [[60, -20], [50, 30], [40, 50], [30, 60], [20, 70], [10, 80]]
        layer1 = [[110, -40], [50, 120]]
        reference = TwoLayerSNN(config, layer0, layer1)
        cases = (('sparse-latency', {'encoding_threshold': 1}), ('nonlinear-latency', {'latency_gamma': 1.5}), ('deterministic-rate', {'rate_scale': 0.75}), ('top-k-rank-order', {'top_k': 3}), ('burst', {'burst_max_spikes': 3, 'burst_interval': 1}))
        for input_encoding, parameters in cases:
            with self.subTest(input_encoding=input_encoding):
                model = HardwareQATSNN(config, timesteps=6, input_encoding=input_encoding, **parameters)
                model.load_quantized_weights(layer0, layer1)
                counts = model(images)
                events = encode_events(pixels, input_encoding=input_encoding, timesteps=6, expected_pixels=len(pixels), **parameters)
                self.assertEqual(tuple((int(value) for value in counts[0])), reference.infer(events, timesteps=6).class_counts)

    def test_known_network_and_backward_pass(self) -> None:
        config = SNNConfig(2, 1, 1, NumericConfig(mem_width=16, current_width=16, threshold=100, leak_shift=2))
        model = HardwareQATSNN(config, timesteps=4)
        with torch.no_grad():
            model.weight_l0.copy_(torch.tensor([[60.0], [60.0]]))
            model.weight_l1.copy_(torch.tensor([[120.0]]))
        images = torch.ones((1, 1, 1, 2))
        counts = model(images)
        self.assertEqual(counts.tolist(), [[1.0]])
        counts.sum().backward()
        self.assertIsNotNone(model.weight_l0.grad)
        self.assertTrue(torch.isfinite(model.weight_l0.grad).all())
        quantized_l0, quantized_l1 = model.quantized_weights()
        reference = TwoLayerSNN(config, quantized_l0.tolist(), quantized_l1.tolist())
        result = reference.infer(encode_latency_events([255, 255], timesteps=4, expected_pixels=2))
        self.assertEqual(tuple((int(value) for value in counts[0].tolist())), result.class_counts)

    def test_izhikevich_hidden_layer_matches_integer_reference(self) -> None:
        config = SNNConfig(4, 1, 1, NumericConfig(mem_width=24, current_width=24, time_width=8), hidden_neuron_model=NeuronModel.IZHIKEVICH)
        model = HardwareQATSNN(config, timesteps=4)
        with torch.no_grad():
            model.weight_l0.fill_(10.0)
            model.weight_l1.fill_(120.0)
        images = self.four_timestep_image()
        counts = model(images)
        reference = TwoLayerSNN(config, [[10]] * 4, [[120]])
        result = reference.infer(encode_latency_events([255, 170, 85, 0], timesteps=4, expected_pixels=4))
        self.assertEqual(counts.tolist(), [[1.0]])
        self.assertEqual(tuple((int(value) for value in counts[0])), result.class_counts)
        counts.sum().backward()
        self.assertTrue(torch.isfinite(model.weight_l0.grad).all())

    def test_izhikevich_output_layer_matches_integer_reference(self) -> None:
        config = SNNConfig(4, 1, 1, NumericConfig(mem_width=24, current_width=24, time_width=8), output_neuron_model=NeuronModel.IZHIKEVICH)
        model = HardwareQATSNN(config, timesteps=4)
        with torch.no_grad():
            model.weight_l0.fill_(127.0)
            model.weight_l1.fill_(10.0)
        images = self.four_timestep_image()
        counts = model(images)
        reference = TwoLayerSNN(config, [[127]] * 4, [[10]])
        result = reference.infer(encode_latency_events([255, 170, 85, 0], timesteps=4, expected_pixels=4))
        self.assertEqual(counts.tolist(), [[1.0]])
        self.assertEqual(tuple((int(value) for value in counts[0])), result.class_counts)

    def test_hh_qat_step_matches_fixed_reference_and_backpropagates(self) -> None:
        dynamics = HHQATDynamics(checkpoint_interval=10)
        reference_tensor = torch.zeros((1, 1), dtype=torch.float32)
        state = dynamics.initial_state(reference_tensor, 1)
        current = torch.tensor([[10.0]], requires_grad=True)
        next_state, spikes = dynamics(state, current)
        reference = HodgkinHuxleyFixedNeuron().step(10)
        self.assertEqual(int(next_state.voltage.item()), reference.state.voltage)
        self.assertEqual(int(next_state.sodium_activation.item()), reference.state.sodium_activation)
        self.assertEqual(int(next_state.sodium_inactivation.item()), reference.state.sodium_inactivation)
        self.assertEqual(int(next_state.potassium_activation.item()), reference.state.potassium_activation)
        self.assertEqual(bool(spikes.item()), reference.spike)
        (spikes.sum() + next_state.voltage.sum() / 100000.0).backward()
        self.assertIsNotNone(current.grad)
        self.assertTrue(torch.isfinite(current.grad).all())
        self.assertNotEqual(float(current.grad.abs().sum()), 0.0)

    def test_hh_qat_random_current_sequences_match_scalar_reference(self) -> None:
        dynamics = HHQATDynamics()
        marker = torch.zeros((1, 3), dtype=torch.float32)
        state = dynamics.initial_state(marker, 3)
        references = [HodgkinHuxleyFixedNeuron() for _ in range(3)]
        for currents in ((-20, 0, 10), (50, 20, -5), (127, -128, 5)):
            state, spikes = dynamics(state, torch.tensor([currents], dtype=torch.float32))
            expected = [neuron.step(current) for neuron, current in zip(references, currents, strict=True)]
            self.assertEqual(state.voltage.to(torch.int64).tolist()[0], [result.state.voltage for result in expected])
            self.assertEqual(state.sodium_activation.to(torch.int64).tolist()[0], [result.state.sodium_activation for result in expected])
            self.assertEqual(spikes.to(torch.bool).tolist()[0], [result.spike for result in expected])

    def test_hh_checkpointing_preserves_forward_and_gradient(self) -> None:
        gradients = []
        states = []
        for interval in (0, 10):
            dynamics = HHQATDynamics(checkpoint_interval=interval)
            marker = torch.zeros((1, 2), dtype=torch.float32)
            current = torch.tensor([[10.0, 20.0]], requires_grad=True)
            state, spikes = dynamics(dynamics.initial_state(marker, 2), current)
            loss = state.voltage.sum() / 100000.0 + spikes.sum()
            loss.backward()
            states.append(state.voltage.detach())
            gradients.append(current.grad.detach())
        self.assertTrue(torch.equal(states[0], states[1]))
        self.assertTrue(torch.allclose(gradients[0], gradients[1], atol=1e-07, rtol=1e-07))

    def test_hh_surrogate_can_optimize_a_tiny_target(self) -> None:
        dynamics = HHQATDynamics(internal_steps=10)
        current = torch.nn.Parameter(torch.tensor([[0.0]]))
        optimizer = torch.optim.Adam((current,), lr=1.0)
        losses = []
        for _ in range(6):
            optimizer.zero_grad(set_to_none=True)
            state, _ = dynamics(dynamics.initial_state(current, 1), current)
            loss = ((state.voltage - 50000.0) / 100000.0).square().mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        self.assertLess(losses[-1], losses[0] * 0.5)

    def test_hodgkin_huxley_hidden_qat_matches_integer_network(self) -> None:
        config = SNNConfig(4, 1, 1, NumericConfig(mem_width=24, current_width=24), hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        model = HardwareQATSNN(config, timesteps=2, bptt_window=1)
        with torch.no_grad():
            model.weight_l0.fill_(10.0)
            model.weight_l1.fill_(120.0)
        images = torch.ones((1, 1, 1, 4))
        counts = model(images)
        reference = TwoLayerSNN(config, [[10]] * 4, [[120]])
        result = reference.infer(encode_latency_events([255] * 4, timesteps=2, expected_pixels=4))
        self.assertEqual(tuple((int(value) for value in counts[0])), result.class_counts)
        model.set_hh_diagnostics(True)
        model(images)
        diagnostics = model.forward_diagnostics()
        self.assertIn('lut_clamps', diagnostics)
        self.assertIn('voltage_saturations', diagnostics)
        self.assertIn('gate_saturations', diagnostics)
        counts.sum().backward()
        self.assertTrue(torch.isfinite(model.weight_l0.grad).all())

    def test_gradient_accumulator_reports_parameter_health(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        parameter.grad = torch.tensor([0.0, 3.0, 4.0])
        diagnostics = GradientAccumulator()
        diagnostics.add({'weight': parameter})
        report = diagnostics.summary()['weight']
        self.assertEqual(report['batches'], 1)
        self.assertEqual(report['l2_norm_mean'], 5.0)
        self.assertAlmostEqual(report['mean_abs'], 7.0 / 3.0)
        self.assertAlmostEqual(report['zero_fraction'], 1.0 / 3.0)
        self.assertEqual(report['nonfinite_elements'], 0)
        parameter.grad = torch.tensor([1e+30, 1e+30, 1e+30])
        large = GradientAccumulator()
        large.add({'weight': parameter})
        self.assertTrue(torch.isfinite(torch.tensor(large.summary()['weight']['l2_norm_mean'], dtype=torch.float64)))
        total_norm = _clip_grad_norm_float64((parameter,), 1.0)
        self.assertGreater(total_norm, 1e+30)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(parameter.grad)), 1.0, places=6)

    def test_layer0_column_mean_penalty_targets_systematic_offsets(self) -> None:
        config = SNNConfig(2, 2, 1, NumericConfig())
        model = HardwareQATSNN(config, timesteps=1)
        with torch.no_grad():
            model.weight_l0.copy_(torch.tensor([[2.0, -3.0], [2.0, 3.0]]))
        self.assertEqual(layer0_column_mean_penalty(model).item(), 2.0)

    def test_truncated_bptt_preserves_forward_counts(self) -> None:
        config = SNNConfig(4, 1, 1, NumericConfig(mem_width=24, current_width=24, time_width=8), hidden_neuron_model=NeuronModel.IZHIKEVICH, output_neuron_model=NeuronModel.IZHIKEVICH)
        full = HardwareQATSNN(config, timesteps=4)
        truncated = HardwareQATSNN(config, timesteps=4, bptt_window=1)
        truncated.load_state_dict(full.state_dict())
        images = self.four_timestep_image()
        full_counts = full(images)
        truncated_counts = truncated(images)
        self.assertTrue(torch.equal(full_counts, truncated_counts))
        truncated_counts.sum().backward()
        self.assertTrue(torch.isfinite(truncated.weight_l0.grad).all())

    def test_best_checkpoint_restores_best_epoch_and_tracks_patience(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        checkpoint = BestCheckpoint(minimum_improvement=0.01)
        with torch.no_grad():
            model.weight.fill_(1.0)
        self.assertTrue(checkpoint.update(model, epoch=1, accuracy=0.5))
        with torch.no_grad():
            model.weight.fill_(2.0)
        self.assertFalse(checkpoint.update(model, epoch=2, accuracy=0.505))
        self.assertTrue(checkpoint.should_stop(1))
        checkpoint.restore(model)
        self.assertEqual(model.weight.item(), 1.0)
        self.assertEqual(checkpoint.best_epoch, 1)

    def test_freeze_phase_cannot_stop_before_joint_training(self) -> None:
        self.assertEqual(_training_phase(5, 5), 'frozen_readout')
        self.assertEqual(_training_phase(6, 5), 'joint')
        self.assertFalse(_early_stopping_eligible(5, 5, 2))
        self.assertFalse(_early_stopping_eligible(6, 5, 2))
        self.assertTrue(_early_stopping_eligible(7, 5, 2))
        self.assertFalse(_planned_stop_reached(1, None))
        self.assertFalse(_planned_stop_reached(1, 2))
        self.assertTrue(_planned_stop_reached(2, 2))

    def test_lr_reduction_resets_early_stopping_patience(self) -> None:
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=1)
        checkpoint = BestCheckpoint()
        checkpoint.update(model, epoch=0, accuracy=0.9)
        next_lr, reduced = _step_plateau_scheduler(scheduler, optimizer, 0.8, checkpoint)
        self.assertFalse(reduced)
        self.assertEqual(next_lr, 0.01)
        checkpoint.update(model, epoch=1, accuracy=0.8)
        _step_plateau_scheduler(scheduler, optimizer, 0.7, checkpoint)
        checkpoint.update(model, epoch=2, accuracy=0.7)
        next_lr, reduced = _step_plateau_scheduler(scheduler, optimizer, 0.6, checkpoint)
        self.assertTrue(reduced)
        self.assertEqual(next_lr, 0.005)
        self.assertEqual(checkpoint.epochs_without_improvement, 0)

    def test_initial_model_contract_validation_is_strict(self) -> None:
        config = SNNConfig(2, 1, 1, NumericConfig(), hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        exported = ExportedModel(config, 8, ((1,), (2,)), ((3,),), {})
        args = type('Arguments', (), {'timesteps': 8, 'input_encoding': 'latency', 'rate_scale': 1.0, 'rate_seed': None, 'seed': 1})()
        _validate_initial_model(exported, config, args)
        args.timesteps = 7
        with self.assertRaisesRegex(ValueError, 'timestep'):
            _validate_initial_model(exported, config, args)

    def test_epoch_checkpoint_contains_resumable_state(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        generator = torch.Generator().manual_seed(9)
        best = BestCheckpoint()
        best.update(model, 0, 0.5)
        initial = (torch.tensor([[1]], dtype=torch.int8), torch.tensor([[2]], dtype=torch.int8))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'checkpoint.pt'
            _save_epoch_checkpoint(destination, contract={'test': True}, epoch=3, model=model, optimizer=optimizer, scheduler=None, best=best, history=[{'epoch': 0}], generator=generator, initial_quantized_weights=initial)
            payload = torch.load(destination, weights_only=False)
        self.assertEqual(payload['schema'], 'snn-qat-training-checkpoint-v1')
        self.assertEqual(payload['epoch'], 3)
        self.assertEqual(payload['best']['best_epoch'], 0)
        self.assertFalse(destination.with_name(destination.name + '.tmp').exists())

    def test_cuda_rng_restore_moves_mapped_states_back_to_cpu(self) -> None:
        mapped_state = torch.zeros(16, dtype=torch.uint8)
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), mock.patch.object(torch.cuda, 'set_rng_state_all') as restore:
            _restore_cuda_rng_states([mapped_state])
        restored = restore.call_args.args[0]
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0].device.type, 'cpu')
        self.assertEqual(restored[0].dtype, torch.uint8)
if __name__ == '__main__':
    unittest.main()
