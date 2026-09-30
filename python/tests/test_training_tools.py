import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from snn_reference import NeuronModel, NumericConfig, SNNConfig
from snn_training.encoding import encode_burst_events, encode_deterministic_rate_events, encode_latency_events, encode_nonlinear_latency_events, encode_rate_events, encode_sparse_latency_events, encode_top_k_rank_order_events, latency_timestamp, nonlinear_latency_timestamp
from snn_training.export import export_model, load_exported_model

class LatencyEncodingTests(unittest.TestCase):

    def test_timestamp_endpoints_and_midpoint(self) -> None:
        self.assertEqual(latency_timestamp(255, 16), 0)
        self.assertEqual(latency_timestamp(0, 16), 15)
        self.assertEqual(latency_timestamp(128, 5), 1)

    def test_fixed_event_count_and_sort_order(self) -> None:
        events = encode_latency_events([0, 255, 128], timesteps=5, expected_pixels=3)
        self.assertEqual(len(events), 3)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 1), (1, 2), (4, 0)])

    def test_rejects_invalid_pixels_or_shape(self) -> None:
        with self.assertRaises(ValueError):
            encode_latency_events([0, 1], expected_pixels=3)
        with self.assertRaises(ValueError):
            latency_timestamp(256, 16)

class RateEncodingTests(unittest.TestCase):

    def test_black_never_spikes_and_full_rate_white_spikes_every_timestep(self) -> None:
        events = encode_rate_events([0, 255], timesteps=4, seed=7, expected_pixels=2)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 1), (1, 1), (2, 1), (3, 1)])

    def test_seed_is_reproducible_and_stream_is_sorted(self) -> None:
        first = encode_rate_events([64, 128, 192], timesteps=20, seed=1234, expected_pixels=3)
        repeated = encode_rate_events([64, 128, 192], timesteps=20, seed=1234, expected_pixels=3)
        different = encode_rate_events([64, 128, 192], timesteps=20, seed=1235, expected_pixels=3)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, different)
        keys = [(event.timestamp, event.neuron_id) for event in first]
        self.assertEqual(keys, sorted(keys))

    def test_rate_scale_and_arguments_are_validated(self) -> None:
        self.assertEqual(encode_rate_events([255], timesteps=3, rate_scale=0.0, expected_pixels=1), ())
        with self.assertRaises(ValueError):
            encode_rate_events([1], timesteps=0, expected_pixels=1)
        with self.assertRaises(ValueError):
            encode_rate_events([1], rate_scale=float('nan'), expected_pixels=1)
        with self.assertRaises(ValueError):
            encode_rate_events([256], expected_pixels=1)

class AdditionalEncodingTests(unittest.TestCase):

    def test_sparse_latency_omits_pixels_at_or_below_threshold(self) -> None:
        events = encode_sparse_latency_events([0, 1, 128, 255], timesteps=5, encoding_threshold=1, expected_pixels=4)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 3), (1, 2)])

    def test_nonlinear_latency_uses_gamma_curve(self) -> None:
        self.assertEqual(nonlinear_latency_timestamp(0, 5, 2.0), 4)
        self.assertEqual(nonlinear_latency_timestamp(128, 5, 2.0), 2)
        self.assertEqual(nonlinear_latency_timestamp(255, 5, 2.0), 0)
        events = encode_nonlinear_latency_events([0, 128, 255], timesteps=5, latency_gamma=2.0, expected_pixels=3)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 2), (2, 1), (4, 0)])
        for timesteps in (8, 16):
            self.assertEqual([nonlinear_latency_timestamp(pixel, timesteps, 1.0) for pixel in range(256)], [latency_timestamp(pixel, timesteps) for pixel in range(256)])

    def test_deterministic_rate_has_rounded_count_and_uniform_timing(self) -> None:
        events = encode_deterministic_rate_events([0, 128, 255], timesteps=4, rate_scale=1.0, expected_pixels=3)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 2), (1, 1), (1, 2), (2, 2), (3, 1), (3, 2)])

    def test_top_k_rank_order_is_stable_for_equal_intensities(self) -> None:
        events = encode_top_k_rank_order_events([200, 255, 200, 1], timesteps=5, top_k=3, encoding_threshold=0, expected_pixels=4)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 1), (2, 0), (4, 2)])

    def test_burst_combines_first_spike_latency_and_spike_count(self) -> None:
        events = encode_burst_events([0, 1, 128, 255], timesteps=6, burst_max_spikes=3, burst_interval=1, expected_pixels=4)
        self.assertEqual([(event.timestamp, event.neuron_id) for event in events], [(0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (4, 1)])

    def test_invalid_new_encoding_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            encode_sparse_latency_events([1], encoding_threshold=256, expected_pixels=1)
        with self.assertRaises(ValueError):
            encode_nonlinear_latency_events([1], latency_gamma=0, expected_pixels=1)
        with self.assertRaises(ValueError):
            encode_top_k_rank_order_events([1], top_k=0, expected_pixels=1)
        with self.assertRaises(ValueError):
            encode_burst_events([255], timesteps=3, burst_max_spikes=4, expected_pixels=1)

class ExportTests(unittest.TestCase):

    def setUp(self) -> None:
        self.config = SNNConfig(2, 2, 2, NumericConfig(mem_width=16, current_width=16, time_width=8))

    def test_round_trip_and_reference_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, self.config, 8, [[-128, 127], [3, -4]], [[5, 6], [-7, 8]], metadata={'test': True})
            loaded = load_exported_model(directory)
            self.assertEqual(manifest, Path(directory) / 'manifest.json')
            self.assertEqual(loaded.input_hidden_weights, ((-128, 127), (3, -4)))
            self.assertEqual(loaded.hidden_output_weights, ((5, 6), (-7, 8)))
            self.assertTrue(loaded.metadata['test'])
            loaded.reference_model()
            data = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(data['encoding']['events_per_image'], 2)
            self.assertEqual(data['weights'][0]['layout'], 'pre-major-post-minor')
            self.assertEqual(data['upload']['weight_write_count'], 8)
            self.assertEqual(len(data['upload']['weight_crc32']), 8)

    def test_checksum_detects_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            export_model(directory, self.config, 8, [[0, 0], [0, 0]], [[0, 0], [0, 0]])
            path = Path(directory) / 'weights_l0.bin'
            path.write_bytes(b'\x01' + path.read_bytes()[1:])
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                load_exported_model(directory)

    def test_rate_manifest_round_trip_and_encoder_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, self.config, 4, [[1, 2], [3, 4]], [[5, 6], [7, 8]], input_encoding='rate', rate_scale=0.75, rate_seed=19)
            loaded = load_exported_model(directory)
            self.assertEqual(loaded.input_encoding, 'rate')
            self.assertEqual(loaded.rate_scale, 0.75)
            self.assertEqual(loaded.rate_seed, 19)
            self.assertEqual(loaded.encode_events([0, 255], sample_index=3), encode_rate_events([0, 255], timesteps=4, rate_scale=0.75, seed=19, sample_index=3, expected_pixels=2))
            data = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(data['encoding']['type'], 'bernoulli-rate-v1')
            self.assertIsNone(data['encoding']['events_per_image'])
            self.assertEqual(data['encoding']['maximum_events_per_image'], 8)

    def test_rate_manifest_validation_rejects_inconsistent_capacity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, self.config, 4, [[0, 0], [0, 0]], [[0, 0], [0, 0]], input_encoding='rate')
            data = json.loads(manifest.read_text(encoding='utf-8'))
            data['encoding']['maximum_events_per_image'] = 7
            manifest.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'maximum event count'):
                load_exported_model(directory)

    def test_additional_encoding_manifests_round_trip(self) -> None:
        cases = (('sparse-latency', {'encoding_threshold': 7}, 'sparse-latency-v1'), ('nonlinear-latency', {'latency_gamma': 1.5}, 'nonlinear-latency-v1'), ('deterministic-rate', {'rate_scale': 0.75}, 'deterministic-rate-v1'), ('top-k-rank-order', {'top_k': 1, 'encoding_threshold': 3}, 'top-k-rank-order-v1'), ('burst', {'burst_max_spikes': 3, 'burst_interval': 1, 'encoding_threshold': 2}, 'burst-v1'))
        for input_encoding, parameters, manifest_type in cases:
            with self.subTest(input_encoding=input_encoding):
                with tempfile.TemporaryDirectory() as directory:
                    manifest = export_model(directory, self.config, 4, [[1, 2], [3, 4]], [[5, 6], [7, 8]], input_encoding=input_encoding, **parameters)
                    loaded = load_exported_model(directory)
                    self.assertEqual(loaded.input_encoding, input_encoding)
                    self.assertEqual(loaded.encode_events([0, 255]), load_exported_model(directory).encode_events([0, 255]))
                    data = json.loads(manifest.read_text(encoding='utf-8'))
                    self.assertEqual(data['encoding']['type'], manifest_type)

    def test_v2_round_trip_preserves_izhikevich_model(self) -> None:
        config = SNNConfig(1, 1, 1, NumericConfig(mem_width=16, current_width=16, time_width=8), hidden_neuron_model=NeuronModel.IZHIKEVICH)
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, config, 1, [[0]], [[0]])
            loaded = load_exported_model(directory)
            self.assertEqual(loaded.config.hidden_neuron_model, NeuronModel.IZHIKEVICH)
            data = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(data['format'], 'snn-int8-v2')
            self.assertEqual(data['neurons']['hidden']['model_id'], 1)
            self.assertEqual(data['neurons']['hidden']['parameters']['frac_bits'], 8)

    def test_v2_round_trip_preserves_hodgkin_huxley_model(self) -> None:
        config = SNNConfig(1, 1, 1, NumericConfig(mem_width=24, current_width=24, time_width=8), hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY, output_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, config, 2, [[10]], [[10]])
            loaded = load_exported_model(directory)
            self.assertEqual(loaded.config.hidden_neuron_model, NeuronModel.HODGKIN_HUXLEY)
            self.assertEqual(loaded.config.output_neuron_model, NeuronModel.HODGKIN_HUXLEY)
            data = json.loads(manifest.read_text(encoding='utf-8'))
            self.assertEqual(data['neurons']['hidden']['model_id'], 2)
            self.assertNotIn('parameters', data['neurons']['hidden'])

    def test_v1_manifest_loads_as_lif(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_model(directory, self.config, 1, [[0, 0], [0, 0]], [[0, 0], [0, 0]])
            data = json.loads(manifest.read_text(encoding='utf-8'))
            data['format'] = 'snn-int8-v1'
            del data['neurons']
            manifest.write_text(json.dumps(data), encoding='utf-8')
            loaded = load_exported_model(directory)
            self.assertEqual(loaded.config.hidden_neuron_model, NeuronModel.LIF)
            self.assertEqual(loaded.config.output_neuron_model, NeuronModel.LIF)
if __name__ == '__main__':
    unittest.main()
