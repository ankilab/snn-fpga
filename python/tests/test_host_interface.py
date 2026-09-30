import unittest
from pathlib import Path
from snn_host import Register, SNN_ADDRESS_RANGE, encode_event_stream, event_stream_crc32, pack_weight_cursor, weight_stream_crc32
from snn_host.cli import _distribution, _parser
from snn_reference import SpikeEvent

class HostInterfaceTests(unittest.TestCase):

    def test_distribution_reports_sample_spread_and_observed_percentiles(self) -> None:
        summary = _distribution([1, 2, 3, 4, 100])
        self.assertEqual(summary['count'], 5)
        self.assertEqual(summary['median'], 3)
        self.assertEqual(summary['p95'], 100)
        self.assertAlmostEqual(summary['mean'], 22)
        self.assertGreater(summary['stdev'], 43)

    def test_control_window_contains_register_file(self):
        self.assertLess(max((register.value for register in Register)), SNN_ADDRESS_RANGE)

    def test_register_map_is_word_aligned_and_unique(self) -> None:
        addresses = [int(register) for register in Register]
        self.assertEqual(len(addresses), len(set(addresses)))
        self.assertTrue(all((address % 4 == 0 for address in addresses)))

    def test_weight_cursor_layout(self) -> None:
        self.assertEqual(pack_weight_cursor(0, 0, 0), 0)
        self.assertEqual(pack_weight_cursor(1, 4660, 22136), 2452903544)
        with self.assertRaises(ValueError):
            pack_weight_cursor(2, 0, 0)

    def test_crc32_matches_standard_vectors(self) -> None:
        self.assertEqual(weight_stream_crc32(b'123456789'), 3421780262)
        events = [SpikeEvent(67305985, 1541)]
        self.assertEqual(encode_event_stream(events), bytes([1, 2, 3, 4, 5, 6]))
        self.assertEqual(event_stream_crc32(events), 2180413220)

    def test_qualify_command_defaults_to_small_reference_comparison(self) -> None:
        args = _parser().parse_args(['--xrt-xclbin', 'snn.xclbin', 'qualify', 'model', 'images', 'labels'])
        self.assertEqual(args.command, 'qualify')
        self.assertEqual(args.model_dir, Path('model'))
        self.assertEqual(args.samples, 10)

    def test_cli_accepts_alveo_xrt_backend(self) -> None:
        args = _parser().parse_args(['--xrt-xclbin', 'snn.xclbin', '--xrt-device', '2', 'probe'])
        self.assertEqual(args.xrt_xclbin, Path('snn.xclbin'))
        self.assertEqual(args.xrt_device, 2)
        self.assertEqual(args.xrt_kernel, 'snn_accelerator:{snn_accelerator_1}')
if __name__ == '__main__':
    unittest.main()
