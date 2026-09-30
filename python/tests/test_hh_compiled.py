"""Numerical gates for the optional CUDA HH compiler backend."""
import importlib.util
import unittest
from unittest import mock

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
CUDA_AVAILABLE = False
if TORCH_AVAILABLE:
    import torch
    from snn_reference import NeuronModel, SNNConfig
    from snn_reference.hodgkin_huxley_fixed import HodgkinHuxleyFixedNeuron
    from snn_training.hh_qat import HHQATDynamics
    from snn_training.torch_model import HardwareQATSNN
    from snn_training.train_mnist import _arguments, _training_contract
    CUDA_AVAILABLE = torch.cuda.is_available()


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is absent")
class HHBackendTests(unittest.TestCase):
    def test_backend_validation_and_legacy_resume_contract(self):
        with self.assertRaisesRegex(ValueError, "backend"):
            HHQATDynamics(backend="invalid")
        with self.assertRaisesRegex(ValueError, "HH population"):
            HardwareQATSNN(SNNConfig(), hh_backend="compiled")
        dynamics = HHQATDynamics(backend="compiled")
        marker = torch.zeros((1, 1))
        with self.assertRaisesRegex(ValueError, "CUDA"):
            dynamics(dynamics.initial_state(marker, 1), marker)
        with mock.patch("sys.argv", ["train_mnist"]):
            args = _arguments()
        config = SNNConfig()
        legacy_contract = _training_contract(args, config)
        self.assertNotIn("hh_backend", legacy_contract)
        args.hh_backend = "compiled"
        self.assertEqual(_training_contract(args, config)["hh_backend"], "compiled")
        del args.hh_backend
        self.assertEqual(_training_contract(args, config), legacy_contract)

    def test_diagnostics_preserve_eager_states_and_gradients(self):
        results = []
        for diagnostics in (False, True):
            dynamics = HHQATDynamics(checkpoint_interval=10)
            dynamics.begin_diagnostics(diagnostics)
            current = torch.tensor([[-50., 10., 127.]], requires_grad=True)
            state, spikes = dynamics(dynamics.initial_state(current, 3), current)
            (state.voltage.sum() / 100000 + spikes.sum()).backward()
            results.append([getattr(state, f).detach() for f in state.__dataclass_fields__]
                           + [spikes.detach(), current.grad])
            if diagnostics:
                self.assertIn("lut_clamps", dynamics.diagnostics())
            else:
                self.assertEqual(dynamics.diagnostics(), {})
        for first, second in zip(*results):
            self.assertTrue(torch.equal(first, second))


@unittest.skipUnless(CUDA_AVAILABLE, "compiled HH verification requires CUDA")
class HHCompiledCUDATests(unittest.TestCase):
    def test_recurrence_matches_scalar_and_eager_gradients(self):
        # Include current wrapping, LUT clamps and saturated states. The scalar
        # oracle checks every gate and spike, independently of the torch graph.
        sequences = ((-50, 0, 10, 127, 8388607, -8388608),
                     (20, -128, 50, -5, 8388608, -8388609))
        for interval in (0, 10):
            with self.subTest(checkpoint_interval=interval):
                results = []
                for backend in ("eager", "compiled"):
                    dynamics = HHQATDynamics(backend=backend, checkpoint_interval=interval).cuda()
                    current = torch.tensor(sequences, dtype=torch.float32, device="cuda", requires_grad=True)
                    state = dynamics.initial_state(current[:1], 6)
                    references = [HodgkinHuxleyFixedNeuron() for _ in range(6)]
                    states, spikes_all = [], []
                    for step in range(2):
                        state, spikes = dynamics(state, current[step:step+1])
                        expected = [neuron.step(((value + 2**23) % 2**24) - 2**23)
                                    for neuron, value in zip(references, sequences[step])]
                        for name in state.__dataclass_fields__:
                            values = getattr(state, name)
                            self.assertEqual(values.long().tolist()[0],
                                             [getattr(r.state, name) for r in expected])
                        self.assertEqual(spikes.bool().tolist()[0], [r.spike for r in expected])
                        states.append(state.voltage)
                        spikes_all.append(spikes)
                    loss = sum(s.sum() / 100000 for s in states) + sum(s.sum() for s in spikes_all)
                    loss.backward()
                    self.assertTrue(torch.isfinite(current.grad).all())
                    results.append(current.grad.detach())
                torch.testing.assert_close(*results, atol=1e-6, rtol=1e-6)

    def test_network_optimizer_partial_batch_and_diagnostic_fallback(self):
        torch.manual_seed(91)
        config = SNNConfig(16, 8, 3, hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY,
                           output_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        models = [HardwareQATSNN(config, timesteps=3, bptt_window=1,
                                input_encoding="deterministic-rate", rate_scale=0.25,
                                hh_backend=backend, hh_checkpoint_interval=10).cuda()
                  for backend in ("eager", "compiled")]
        models[1].load_state_dict(models[0].state_dict())
        optimizers = [torch.optim.Adam(model.parameters(), lr=0.0016) for model in models]
        for batch_size in (4, 1):
            images = torch.rand(batch_size, 1, 4, 4, device="cuda")
            labels = torch.arange(batch_size, device="cuda") % 3
            counts = []
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad(set_to_none=True)
                output = model(images)
                counts.append(output.detach())
                torch.nn.functional.cross_entropy(output, labels).backward()
                optimizer.step()
            torch.testing.assert_close(*counts, atol=0, rtol=0)
            for first, second in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(first.grad, second.grad, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
            for model in models:
                model.eval()
                model.set_hh_diagnostics(True)
            with torch.no_grad():
                torch.testing.assert_close(models[0](images), models[1](images), atol=0, rtol=0)
            for key, value in models[0].forward_diagnostics().items():
                torch.testing.assert_close(value, models[1].forward_diagnostics()[key], atol=0, rtol=0)
            for model in models:
                model.set_hh_diagnostics(False)
                model.train()
        self.assertEqual(models[0].state_dict().keys(), models[1].state_dict().keys())


if __name__ == "__main__":
    unittest.main()
