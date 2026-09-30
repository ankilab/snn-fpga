from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
from snn_reference import HodgkinHuxleyFixedConfig, HodgkinHuxleyFixedNeuron, NeuronModel, NumericConfig, SNNConfig, SpikeEvent, TwoLayerSNN, dequantize_state, simulate_hodgkin_huxley
from snn_verification.generate_hh_luts import TABLE_NAMES, write_tables
NUMPY_AVAILABLE = importlib.util.find_spec('numpy') is not None
if NUMPY_AVAILABLE:
    import numpy as np
    from snn_training.encoding import encode_rate_events
    from snn_training.export import ExportedModel

class HodgkinHuxleyFixedTests(unittest.TestCase):

    def test_lut_files_are_reproducible(self) -> None:
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            generated = Path(temporary)
            write_tables(generated, HodgkinHuxleyFixedConfig())
            for name in TABLE_NAMES:
                expected = repository / 'data' / 'hh_luts' / f'hh_{name}.mem'
                self.assertEqual(expected.read_bytes(), (generated / expected.name).read_bytes())

    def test_current_pulse_tracks_float64_reference(self) -> None:
        fixed = HodgkinHuxleyFixedNeuron()
        fixed_spikes: list[float] = []
        fixed_states = []
        for timestep in range(50):
            result = fixed.step(10 if 5 <= timestep < 45 else 0)
            fixed_states.append(dequantize_state(result.state))
            if result.spike:
                self.assertIsNotNone(result.spike_substep)
                fixed_spikes.append(timestep + result.spike_substep / 100.0)
        reference = simulate_hodgkin_huxley(50.0, 0.01, lambda time: 10.0 if 5.0 <= time < 45.0 else 0.0)
        self.assertEqual(len(reference.spike_times), len(fixed_spikes))
        for expected, actual in zip(reference.spike_times, fixed_spikes, strict=True):
            self.assertLess(abs(expected - actual), 0.1)
        reference_boundaries = reference.states[100::100]
        self.assertEqual(len(reference_boundaries), len(fixed_states))
        self.assertLess(max((abs(a.voltage - b.voltage) for a, b in zip(reference_boundaries, fixed_states, strict=True))), 6.1)

    def test_clear_restores_resting_state(self) -> None:
        neuron = HodgkinHuxleyFixedNeuron()
        initial = neuron.state
        neuron.step(10)
        self.assertNotEqual(initial, neuron.state)
        neuron.clear()
        self.assertEqual(initial, neuron.state)
if __name__ == '__main__':
    unittest.main()
