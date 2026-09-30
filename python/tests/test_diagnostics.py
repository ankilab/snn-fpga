import unittest
from dataclasses import replace
from snn_reference import NeuronModel, NumericConfig, SNNConfig, SpikeEvent, TwoLayerSNN
from snn_training.diagnostics import diagnose_reference
from snn_training.export import ExportedModel

class DiagnosticsTests(unittest.TestCase):

    def test_report_captures_ties_activity_and_weight_statistics(self) -> None:
        config = SNNConfig(2, 1, 2, NumericConfig(mem_width=16, current_width=16, threshold=100))
        model = TwoLayerSNN(config, [[60], [60]], [[120, 120]])
        report = diagnose_reference(model, [((SpikeEvent(0, 0), SpikeEvent(0, 1)), 0)])
        self.assertEqual(report['schema'], 'snn-diagnostics-v1')
        self.assertEqual(report['accuracy'], 1.0)
        self.assertEqual(report['output_ties'], 1)
        self.assertEqual(report['output_margin']['mean'], 0.0)
        self.assertEqual(report['predicted_classes'], [1, 0])
        self.assertEqual(report['layers']['hidden']['spikes_total'], 1)
        self.assertEqual(report['layers']['output']['spikes_total'], 2)
        self.assertEqual(report['weights']['input_hidden']['max'], 60)

    def test_trace_counts_current_and_izhikevich_state_wraps(self) -> None:
        lif_config = SNNConfig(2, 1, 1, NumericConfig(mem_width=8, current_width=8, threshold=100, reset_potential=0))
        lif = TwoLayerSNN(lif_config, [[127], [127]], [[0]])
        lif_trace = lif.infer((SpikeEvent(0, 0), SpikeEvent(0, 1)), trace=True).timesteps[0]
        self.assertEqual(lif_trace.hidden_current_wraps, 1)
        izh_config = SNNConfig(1, 1, 1, NumericConfig(mem_width=16, current_width=16), hidden_neuron_model=NeuronModel.IZHIKEVICH)
        izh = TwoLayerSNN(izh_config, [[-128]], [[0]])
        izh_trace = izh.infer((SpikeEvent(0, 0),), trace=True).timesteps[0]
        self.assertGreater(izh_trace.hidden_membrane_wraps, 0)
if __name__ == '__main__':
    unittest.main()
