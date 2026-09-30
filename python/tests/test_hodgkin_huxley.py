import math
import unittest
from snn_reference import HodgkinHuxleyConfig, hodgkin_huxley_derivatives, hodgkin_huxley_rate_constants, hodgkin_huxley_resting_state, simulate_hodgkin_huxley

class HodgkinHuxleyReferenceTests(unittest.TestCase):

    def test_rate_singularities_have_finite_analytic_limits(self) -> None:
        rates_at_m_singularity = hodgkin_huxley_rate_constants(25.0)
        rates_at_n_singularity = hodgkin_huxley_rate_constants(10.0)
        self.assertAlmostEqual(rates_at_m_singularity[0], 1.0, places=12)
        self.assertAlmostEqual(rates_at_n_singularity[4], 0.1, places=12)
        self.assertTrue(all((math.isfinite(value) for value in rates_at_m_singularity)))
        self.assertTrue(all((math.isfinite(value) for value in rates_at_n_singularity)))

    def test_resting_state_matches_original_steady_gates(self) -> None:
        state = hodgkin_huxley_resting_state()
        self.assertAlmostEqual(state.sodium_activation, 0.05293249, places=7)
        self.assertAlmostEqual(state.sodium_inactivation, 0.59612075, places=7)
        self.assertAlmostEqual(state.potassium_activation, 0.31767691, places=7)
        derivative = hodgkin_huxley_derivatives(state, 0.0)
        self.assertLess(abs(derivative.voltage), 0.005)
        self.assertAlmostEqual(derivative.sodium_activation, 0.0, places=14)
        self.assertAlmostEqual(derivative.sodium_inactivation, 0.0, places=14)
        self.assertAlmostEqual(derivative.potassium_activation, 0.0, places=14)

    def test_classic_current_pulse_produces_stable_action_potentials(self) -> None:
        stimulus = lambda time: 10.0 if 5.0 <= time < 45.0 else 0.0
        trace = simulate_hodgkin_huxley(50.0, 0.01, stimulus)
        self.assertEqual(len(trace.spike_times), 3)
        self.assertAlmostEqual(trace.spike_times[0], 6.54761, places=4)
        self.assertGreater(max((state.voltage for state in trace.states)), 100.0)
        for state in trace.states:
            self.assertTrue(0.0 <= state.sodium_activation <= 1.0)
            self.assertTrue(0.0 <= state.sodium_inactivation <= 1.0)
            self.assertTrue(0.0 <= state.potassium_activation <= 1.0)

    def test_rk4_spike_times_converge_when_timestep_is_halved(self) -> None:
        stimulus = lambda time: 10.0 if 5.0 <= time < 45.0 else 0.0
        coarse = simulate_hodgkin_huxley(50.0, 0.01, stimulus)
        fine = simulate_hodgkin_huxley(50.0, 0.005, stimulus)
        self.assertEqual(len(coarse.spike_times), len(fine.spike_times))
        self.assertLess(max((abs(left - right) for left, right in zip(coarse.spike_times, fine.spike_times))), 0.001)

    def test_invalid_configuration_and_time_grid_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            HodgkinHuxleyConfig(capacitance=0.0)
        with self.assertRaisesRegex(ValueError, 'integer multiple'):
            simulate_hodgkin_huxley(1.0, 0.3, lambda _: 0.0)
if __name__ == '__main__':
    unittest.main()
