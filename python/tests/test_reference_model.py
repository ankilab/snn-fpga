import unittest

from snn_reference import (
    NeuronModel,
    IzhikevichConfig,
    NumericConfig,
    SNNConfig,
    SpikeEvent,
    TwoLayerSNN,
    signed_wrap,
)


class IntegerArithmeticTests(unittest.TestCase):
    def test_signed_wrap(self) -> None:
        self.assertEqual(signed_wrap(127, 8), 127)
        self.assertEqual(signed_wrap(128, 8), -128)
        self.assertEqual(signed_wrap(255, 8), -1)
        self.assertEqual(signed_wrap(-129, 8), 127)

    def test_invalid_width(self) -> None:
        with self.assertRaises(ValueError):
            signed_wrap(0, 0)


class ExplicitHorizonTests(unittest.TestCase):
    def setUp(self) -> None:
        config = SNNConfig(
            1,
            1,
            1,
            NumericConfig(mem_width=16, current_width=16, threshold=100),
        )
        self.model = TwoLayerSNN(config, [[0]], [[0]])

    def test_empty_stream_runs_explicit_horizon(self) -> None:
        result = self.model.infer((), trace=True, timesteps=4)
        self.assertEqual(len(result.timesteps), 4)
        self.assertIsNone(result.prediction)

    def test_empty_stream_without_horizon_remains_invalid(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit timesteps"):
            self.model.infer(())

    def test_event_outside_horizon_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the inference horizon"):
            self.model.infer((SpikeEvent(2, 0),), timesteps=2)


class ReferenceModelTests(unittest.TestCase):
    @staticmethod
    def make_model(
        input_hidden,
        hidden_output,
        *,
        threshold=100,
        width=16,
        leak_shift=2,
        count_width=8,
    ) -> TwoLayerSNN:
        config = SNNConfig(
            input_neurons=len(input_hidden),
            hidden_neurons=len(input_hidden[0]),
            output_neurons=len(hidden_output[0]),
            numeric=NumericConfig(
                mem_width=width,
                current_width=width,
                weight_width=8,
                count_width=count_width,
                time_width=8,
                threshold=threshold,
                reset_potential=0,
                leak_shift=leak_shift,
            ),
        )
        return TwoLayerSNN(config, input_hidden, hidden_output)

    def test_matches_rtl_end_to_end_vector(self) -> None:
        # Mirrors tb/tb_snn_top.sv exactly.
        model = self.make_model(
            [
                [60, 40, 0],
                [60, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ],
            [
                [0, 120],
                [0, 0],
                [0, 0],
            ],
        )
        result = model.infer([SpikeEvent(0, 0), SpikeEvent(0, 1)], trace=True)

        self.assertEqual(result.prediction, 1)
        self.assertEqual(result.class_counts, (0, 1))
        self.assertEqual(result.hidden_membrane, (0, 40, 0))
        self.assertEqual(result.output_membrane, (0, 0))
        self.assertEqual(result.timesteps[0].hidden_current, (120, 40, 0))
        self.assertEqual(result.timesteps[0].hidden_spikes, (True, False, False))

    def test_inference_state_is_cleared_between_calls(self) -> None:
        model = self.make_model([[60], [60]], [[120]])
        events = [SpikeEvent(0, 0), SpikeEvent(0, 1)]
        first = model.infer(events)
        second = model.infer(events)
        self.assertEqual(first, second)

    def test_empty_timesteps_apply_leak(self) -> None:
        model = self.make_model([[40], [0]], [[0]], threshold=100)
        result = model.infer([SpikeEvent(0, 0), SpikeEvent(2, 1)], trace=True)
        self.assertEqual([step.hidden_membrane for step in result.timesteps], [(40,), (30,), (23,)])
        self.assertEqual(result.prediction, None)

    def test_membrane_wrap_occurs_before_threshold_comparison(self) -> None:
        model = self.make_model([[120], [20]], [[0]], threshold=100, width=8, leak_shift=7)
        result = model.infer([SpikeEvent(0, 0), SpikeEvent(0, 1)])
        self.assertEqual(result.hidden_membrane, (-116,))
        self.assertEqual(result.class_counts, (0,))

    def test_negative_shift_matches_systemverilog_arithmetic_shift(self) -> None:
        model = self.make_model([[-3], [0]], [[0]], threshold=100, width=8, leak_shift=1)
        result = model.infer([SpikeEvent(0, 0), SpikeEvent(1, 1)], trace=True)
        self.assertEqual(result.timesteps[0].hidden_membrane, (-3,))
        self.assertEqual(result.timesteps[1].hidden_membrane, (-1,))

    def test_prediction_tie_selects_lowest_index(self) -> None:
        self.assertEqual(TwoLayerSNN.select_prediction((2, 2, 1)), 0)
        self.assertEqual(TwoLayerSNN.select_prediction((0, 3, 3)), 1)
        self.assertIsNone(TwoLayerSNN.select_prediction((0, 0, 0)))

    def test_class_counter_wrap_can_invalidate_prediction(self) -> None:
        model = self.make_model([[0]], [[0]], threshold=0, count_width=8)
        result = model.infer([SpikeEvent(0, 0), SpikeEvent(255, 0)])
        self.assertEqual(result.class_counts, (0,))
        self.assertIsNone(result.prediction)

    def test_rejects_unsorted_or_out_of_range_events(self) -> None:
        model = self.make_model([[0], [0]], [[0]])
        with self.assertRaises(ValueError):
            model.infer([SpikeEvent(1, 0), SpikeEvent(0, 0)])
        with self.assertRaises(ValueError):
            model.infer([SpikeEvent(0, 2)])
        with self.assertRaises(ValueError):
            model.infer([])

    def test_rejects_unrepresentable_weight(self) -> None:
        with self.assertRaises(ValueError):
            self.make_model([[128]], [[0]])

    def test_neuron_model_ids_are_stable_and_lif_is_the_default(self) -> None:
        model = self.make_model([[0]], [[0]])
        self.assertEqual(int(NeuronModel.LIF), 0)
        self.assertEqual(int(NeuronModel.IZHIKEVICH), 1)
        self.assertEqual(int(NeuronModel.HODGKIN_HUXLEY), 2)
        self.assertEqual(model.config.hidden_neuron_model, NeuronModel.LIF)
        self.assertEqual(model.config.output_neuron_model, NeuronModel.LIF)

    def test_hodgkin_huxley_population_uses_bit_exact_state(self) -> None:
        config = SNNConfig(
            1,
            1,
            1,
            NumericConfig(mem_width=24, current_width=24, time_width=8),
            hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY,
        )
        model = TwoLayerSNN(config, [[10]], [[0]])
        result = model.infer(
            [SpikeEvent(timestamp, 0) for timestamp in range(7)], trace=True
        )
        self.assertEqual(
            [step.hidden_membrane for step in result.timesteps],
            [(36511,), (340179,), (287279,), (96090,), (-41299,), (-37168,), (-31240,)],
        )
        self.assertEqual(
            [step.hidden_spikes for step in result.timesteps],
            [(False,), (True,), (False,), (False,), (False,), (False,), (False,)],
        )

    def test_hodgkin_huxley_requires_the_rtl_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "24-bit"):
            SNNConfig(
                1,
                1,
                1,
                NumericConfig(mem_width=16, current_width=16),
                hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY,
            )

    def test_izhikevich_regular_spiking_fixed_point_sequence(self) -> None:
        config = SNNConfig(
            1,
            1,
            1,
            NumericConfig(mem_width=24, current_width=24, time_width=8),
            hidden_neuron_model=NeuronModel.IZHIKEVICH,
        )
        model = TwoLayerSNN(config, [[10]], [[0]])
        result = model.infer(
            [SpikeEvent(timestamp, 0) for timestamp in range(4)], trace=True
        )
        self.assertEqual(
            [step.hidden_membrane for step in result.timesteps],
            [(-14875,), (-12717,), (-8234,), (-16640,)],
        )
        self.assertEqual(
            [step.hidden_recovery for step in result.timesteps],
            [(-3321,), (-3306,), (-3273,), (-1112,)],
        )
        self.assertEqual(
            [step.hidden_spikes for step in result.timesteps],
            [(False,), (False,), (False,), (True,)],
        )
        self.assertEqual(result.hidden_recovery, (-1112,))

    def test_izhikevich_fixed_point_config_is_validated_when_selected(self) -> None:
        with self.assertRaisesRegex(ValueError, "frac_bits"):
            SNNConfig(
                1,
                1,
                1,
                NumericConfig(mem_width=8, current_width=8),
                hidden_neuron_model=NeuronModel.IZHIKEVICH,
                hidden_izhikevich=IzhikevichConfig(frac_bits=7),
            )


if __name__ == "__main__":
    unittest.main()
