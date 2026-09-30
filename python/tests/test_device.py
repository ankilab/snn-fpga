import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from snn_host import HardwareMismatchError, ModelPackage, Register, SNNDevice, StatusBits
from snn_reference import NeuronModel, NumericConfig, SNNConfig, SpikeEvent
from snn_training.export import export_model

class FakeAccelerator:

    def __init__(self, *, hidden_neurons: int=2, result: int | None=1, interface_version: int=65536, hidden_neuron_model: int=0, output_neuron_model: int=0, izh_reset_voltage: int=-65, mem_width: int=16, hh_gate_frac_bits: int=18, event_capacity: int=2, hidden_executor_count: int=0, output_executor_count: int=0) -> None:
        numeric = NumericConfig(mem_width=mem_width, current_width=mem_width, time_width=8)
        self.values = {Register.ID: 1397640753, Register.VERSION: interface_version, Register.DIMENSIONS_0: hidden_neurons << 16 | 2, Register.DIMENSIONS_1: event_capacity << 16 | 2, Register.NUMERIC_0: numeric.count_width << 24 | numeric.current_width << 16 | numeric.mem_width << 8 | numeric.weight_width, Register.NUMERIC_1: numeric.leak_shift << 8 | numeric.time_width, Register.THRESHOLD: numeric.threshold, Register.RESET_POTENTIAL: numeric.reset_potential, Register.NEURON_MODELS: 8 << 24 | output_neuron_model << 8 | hidden_neuron_model, Register.IZH_AB: 83964417, Register.IZH_RESET_VOLTAGE: izh_reset_voltage & 4294967295, Register.IZH_RECOVERY_INCREMENT: 8, Register.IZH_SPIKE_THRESHOLD: 30, Register.HH_Q_FORMAT: 2 << 24 | 18 << 16 | hh_gate_frac_bits << 8 | 12, Register.HH_WIDTHS_STEPS: 1679299608, Register.HH_LUT_RANGE: 8454112, Register.HH_SPIKE_THRESHOLD: 20, Register.EXECUTOR_COUNTS: hidden_executor_count << 16 | output_executor_count, Register.ERROR: 0, Register.RESULT: 0 if result is None else result}
        self.result = result
        self.status = int(StatusBits.IDLE | StatusBits.WEIGHT_READY | StatusBits.EVENT_READY)
        self.weight_bytes = bytearray()
        self.event_bytes = bytearray()
        self.event_timestamp = 0
        self.start_count = 0
        self.post_start_status_reads: list[int] = []

    def read32(self, offset: int) -> int:
        register = Register(offset)
        if register == Register.STATUS:
            if self.post_start_status_reads:
                return self.post_start_status_reads.pop(0)
            return self.status
        if register == Register.WEIGHT_WRITE_COUNT:
            return len(self.weight_bytes)
        if register == Register.EVENT_WRITE_COUNT:
            return len(self.event_bytes) // 6
        if register == Register.WEIGHT_CRC32:
            return zlib.crc32(self.weight_bytes) & 4294967295
        if register == Register.EVENT_CRC32:
            return zlib.crc32(self.event_bytes) & 4294967295
        return self.values.get(register, 0)

    def write32(self, offset: int, value: int) -> None:
        register = Register(offset)
        if register == Register.CONTROL and value & 2:
            self.weight_bytes.clear()
            self.event_bytes.clear()
            self.status = int(StatusBits.IDLE | StatusBits.WEIGHT_READY | StatusBits.EVENT_READY)
        if register == Register.CONTROL and value & 1:
            self.start_count += 1
            self.post_start_status_reads = [int(StatusBits.BUSY | StatusBits.WEIGHT_READY)]
            self.status = int(StatusBits.IDLE | StatusBits.DONE)
            if self.result is not None:
                self.status |= int(StatusBits.RESULT_VALID)
            self.values[Register.INFERENCE_CYCLES_LO] = 1234
            self.values[Register.INFERENCE_CYCLES_HI] = 1
            self.values[Register.CONSUMED_INPUT_EVENTS] = len(self.event_bytes) // 6
            self.values[Register.EMITTED_HIDDEN_SPIKES] = 7
            self.values[Register.EMITTED_OUTPUT_SPIKES] = 3
            phase_values = {(Register.PHASE_SYNAPSE_CYCLES_LO, Register.PHASE_SYNAPSE_CYCLES_HI): 1000, (Register.PHASE_HIDDEN_UPDATE_CYCLES_LO, Register.PHASE_HIDDEN_UPDATE_CYCLES_HI): 2000, (Register.PHASE_HIDDEN_ROUTING_CYCLES_LO, Register.PHASE_HIDDEN_ROUTING_CYCLES_HI): 3000, (Register.PHASE_OUTPUT_UPDATE_CYCLES_LO, Register.PHASE_OUTPUT_UPDATE_CYCLES_HI): 4000, (Register.PHASE_CONTROL_OVERHEAD_CYCLES_LO, Register.PHASE_CONTROL_OVERHEAD_CYCLES_HI): (1 << 32 | 1234) - 10000}
            for (low, high), phase_cycles in phase_values.items():
                self.values[low] = phase_cycles & 4294967295
                self.values[high] = phase_cycles >> 32
        elif register == Register.WEIGHT_DATA:
            self.weight_bytes.append(value & 255)
        elif register == Register.EVENT_TIMESTAMP:
            self.event_timestamp = value & 4294967295
        elif register == Register.EVENT_NEURON:
            self.event_bytes.extend(struct.pack('<IH', self.event_timestamp, value & 65535))
        else:
            self.values[register] = value & 4294967295

class DeviceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.model_dir = Path(self.temporary.name) / 'model'
        self.config = SNNConfig(2, 2, 2, NumericConfig(mem_width=16, current_width=16, time_width=8))
        export_model(self.model_dir, self.config, 8, [[-1, 2], [3, -4]], [[5, -6], [7, -8]])

    def test_rate_package_requires_v1_3_and_sufficient_event_capacity(self) -> None:
        rate_dir = Path(self.temporary.name) / 'rate-model'
        export_model(rate_dir, self.config, 8, [[-1, 2], [3, -4]], [[5, -6], [7, -8]], input_encoding='rate')
        package = ModelPackage.load(rate_dir)
        with self.assertRaisesRegex(HardwareMismatchError, 'event_capacity'):
            SNNDevice(FakeAccelerator(interface_version=65539)).validate_model(package)
        with self.assertRaisesRegex(HardwareMismatchError, 'version 1.3'):
            SNNDevice(FakeAccelerator(interface_version=65538, event_capacity=16)).validate_model(package)
        SNNDevice(FakeAccelerator(interface_version=65539, event_capacity=16)).validate_model(package)

    def test_probe_decodes_hardware_contract(self) -> None:
        info = SNNDevice(FakeAccelerator()).probe()
        self.assertEqual(info.core_id, 1397640753)
        self.assertEqual(info.hidden_neurons, 2)
        self.assertEqual(info.weight_width, 8)
        self.assertEqual(info.threshold, 100)
        self.assertEqual(info.hidden_neuron_model, int(NeuronModel.LIF))

    def test_probe_decodes_v1_1_neuron_contract(self) -> None:
        info = SNNDevice(FakeAccelerator(interface_version=65537, hidden_neuron_model=1, output_neuron_model=1)).probe()
        self.assertEqual(info.hidden_neuron_model, int(NeuronModel.IZHIKEVICH))
        self.assertEqual(info.output_neuron_model, int(NeuronModel.IZHIKEVICH))
        self.assertEqual(info.izh_frac_bits, 8)
        self.assertEqual(info.izh_a_denominator, 50)
        self.assertEqual(info.izh_reset_voltage, -65)

    def test_probe_decodes_v1_4_executor_counts(self) -> None:
        info = SNNDevice(FakeAccelerator(interface_version=65541, hidden_executor_count=1, output_executor_count=1)).probe()
        self.assertEqual(info.hidden_executor_count, 1)
        self.assertEqual(info.output_executor_count, 1)

    def test_probe_accepts_hodgkin_huxley_hardware_model(self) -> None:
        info = SNNDevice(FakeAccelerator(interface_version=65538, hidden_neuron_model=int(NeuronModel.HODGKIN_HUXLEY), output_neuron_model=int(NeuronModel.HODGKIN_HUXLEY))).probe()
        self.assertEqual(info.hidden_neuron_model, int(NeuronModel.HODGKIN_HUXLEY))
        self.assertEqual(info.output_neuron_model, int(NeuronModel.HODGKIN_HUXLEY))
        self.assertEqual(info.hh_gate_frac_bits, 18)
        self.assertEqual(info.hh_internal_steps, 100)

    def test_hodgkin_huxley_model_and_hardware_contract_are_checked(self) -> None:
        model_dir = Path(self.temporary.name) / 'hh-model'
        config = SNNConfig(2, 2, 2, NumericConfig(mem_width=24, current_width=24, time_width=8), hidden_neuron_model=NeuronModel.HODGKIN_HUXLEY, output_neuron_model=NeuronModel.HODGKIN_HUXLEY)
        export_model(model_dir, config, 8, [[0, 0], [0, 0]], [[0, 0], [0, 0]])
        package = ModelPackage.load(model_dir)
        matching = FakeAccelerator(interface_version=65538, hidden_neuron_model=2, output_neuron_model=2, mem_width=24)
        SNNDevice(matching).validate_model(package)
        mismatching = FakeAccelerator(interface_version=65538, hidden_neuron_model=2, output_neuron_model=2, mem_width=24, hh_gate_frac_bits=17)
        with self.assertRaisesRegex(HardwareMismatchError, 'gate_frac_bits'):
            SNNDevice(mismatching).validate_model(package)

    def test_upload_model_verifies_count_and_crc(self) -> None:
        fake = FakeAccelerator()
        package = ModelPackage.load(self.model_dir)
        SNNDevice(fake).upload_model(package)
        self.assertEqual(bytes(fake.weight_bytes), package.layer0 + package.layer1)
        self.assertEqual(len(fake.weight_bytes), package.expected_weight_count)

    def test_model_hardware_mismatch_is_rejected_before_upload(self) -> None:
        fake = FakeAccelerator(hidden_neurons=3)
        with self.assertRaisesRegex(HardwareMismatchError, 'hidden_neurons'):
            SNNDevice(fake).upload_model(ModelPackage.load(self.model_dir))
        self.assertEqual(fake.weight_bytes, b'')

    def test_izhikevich_model_and_hardware_contract_are_checked(self) -> None:
        model_dir = Path(self.temporary.name) / 'izh-model'
        config = SNNConfig(2, 2, 2, NumericConfig(mem_width=16, current_width=16, time_width=8), hidden_neuron_model=NeuronModel.IZHIKEVICH, output_neuron_model=NeuronModel.IZHIKEVICH)
        export_model(model_dir, config, 8, [[0, 0], [0, 0]], [[0, 0], [0, 0]])
        package = ModelPackage.load(model_dir)
        matching = FakeAccelerator(interface_version=65537, hidden_neuron_model=1, output_neuron_model=1)
        SNNDevice(matching).validate_model(package)
        mismatching = FakeAccelerator(interface_version=65537, hidden_neuron_model=1, output_neuron_model=1, izh_reset_voltage=-64)
        with self.assertRaisesRegex(HardwareMismatchError, 'reset_voltage'):
            SNNDevice(mismatching).validate_model(package)

    def test_inference_uploads_event_stream_and_returns_result(self) -> None:
        fake = FakeAccelerator(result=1)
        events = (SpikeEvent(0, 1), SpikeEvent(7, 0))
        prediction = SNNDevice(fake).infer(events, poll_interval=0)
        self.assertEqual(prediction, 1)
        self.assertEqual(fake.event_bytes, struct.pack('<IH', 0, 1) + struct.pack('<IH', 7, 0))

    def test_v1_3_inference_programs_variable_count_and_horizon(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65539, event_capacity=16)
        events = (SpikeEvent(0, 1), SpikeEvent(3, 0), SpikeEvent(3, 1))
        prediction = SNNDevice(fake).infer(events, timesteps=8, poll_interval=0)
        self.assertEqual(prediction, 1)
        self.assertEqual(fake.values[Register.INFERENCE_EVENT_COUNT], 3)
        self.assertEqual(fake.values[Register.TIMESTEP_COUNT], 8)

    def test_v1_3_accepts_empty_stream_with_explicit_horizon(self) -> None:
        fake = FakeAccelerator(result=None, interface_version=65539, event_capacity=16)
        self.assertIsNone(SNNDevice(fake).infer((), timesteps=4, poll_interval=0))
        self.assertEqual(fake.values[Register.INFERENCE_EVENT_COUNT], 0)
        self.assertEqual(fake.values[Register.TIMESTEP_COUNT], 4)

    def test_v1_4_replays_prepared_stream_and_reads_telemetry(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65540, event_capacity=16)
        device = SNNDevice(fake)
        events = (SpikeEvent(0, 1), SpikeEvent(3, 0), SpikeEvent(3, 1))
        info = device.prepare_inference(events, timesteps=8)
        uploaded = bytes(fake.event_bytes)
        first = device.replay(info, poll_interval=0)
        second = device.replay(info, poll_interval=0)
        self.assertEqual(fake.start_count, 2)
        self.assertEqual(bytes(fake.event_bytes), uploaded)
        self.assertEqual(first.prediction, 1)
        self.assertEqual(first.cycles, 1 << 32 | 1234)
        self.assertEqual(first.consumed_input_events, len(events))
        self.assertEqual(first.emitted_hidden_spikes, 7)
        self.assertEqual(first.emitted_output_spikes, 3)
        self.assertEqual(first, second)

    def test_v1_6_phase_cycles_sum_to_inference_cycles(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65542, event_capacity=16)
        device = SNNDevice(fake)
        info = device.prepare_inference((SpikeEvent(0, 1),), timesteps=1)
        telemetry = device.replay(info, poll_interval=0)
        phases = (telemetry.synapse_phase_cycles, telemetry.hidden_update_phase_cycles, telemetry.hidden_routing_phase_cycles, telemetry.output_update_phase_cycles, telemetry.control_overhead_phase_cycles)
        self.assertNotIn(None, phases)
        self.assertEqual(sum((value for value in phases if value is not None)), telemetry.cycles)

    def test_start_ignores_stale_done_until_new_execution_is_acknowledged(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65540, event_capacity=16)
        original_write = fake.write32

        def write_with_stale_done(offset: int, value: int) -> None:
            original_write(offset, value)
            if Register(offset) == Register.CONTROL and value & 1:
                fake.post_start_status_reads.insert(0, int(StatusBits.IDLE | StatusBits.DONE))
        fake.write32 = write_with_stale_done
        device = SNNDevice(fake)
        info = device.prepare_inference((SpikeEvent(0, 1),), timesteps=1)
        telemetry = device.replay(info, poll_interval=0)
        self.assertEqual(telemetry.prediction, 1)

    def test_done_allows_result_pipeline_to_settle(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65540, event_capacity=16)
        original_write = fake.write32

        def write_with_delayed_result_valid(offset: int, value: int) -> None:
            original_write(offset, value)
            if Register(offset) == Register.CONTROL and value & 1:
                fake.post_start_status_reads.extend([int(StatusBits.IDLE | StatusBits.DONE)] * 8 + [int(StatusBits.IDLE | StatusBits.DONE | StatusBits.RESULT_VALID)])
        fake.write32 = write_with_delayed_result_valid
        device = SNNDevice(fake)
        info = device.prepare_inference((SpikeEvent(0, 1),), timesteps=1)
        telemetry = device.replay(info, poll_interval=0)
        self.assertEqual(telemetry.prediction, 1)

    def test_replay_rejects_pre_v1_4_hardware(self) -> None:
        fake = FakeAccelerator(result=1, interface_version=65539, event_capacity=16)
        device = SNNDevice(fake)
        info = device.prepare_inference((SpikeEvent(0, 1),), timesteps=1)
        with self.assertRaisesRegex(HardwareMismatchError, 'version 1.4'):
            device.replay(info, poll_interval=0)

    def test_no_output_spike_returns_none(self) -> None:
        fake = FakeAccelerator(result=None)
        self.assertIsNone(SNNDevice(fake).infer((SpikeEvent(0, 0), SpikeEvent(1, 1)), poll_interval=0))

    def test_manifest_upload_tampering_is_rejected(self) -> None:
        manifest_path = self.model_dir / 'manifest.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['upload']['weight_write_count'] += 1
        manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'weight count'):
            ModelPackage.load(self.model_dir)
if __name__ == '__main__':
    unittest.main()
