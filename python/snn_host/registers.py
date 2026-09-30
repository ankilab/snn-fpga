"""Canonical SNN control-register map and upload CRC helpers."""

from __future__ import annotations

import struct
import zlib
from enum import IntEnum, IntFlag
from typing import Iterable

from snn_reference import SpikeEvent

SNN_ADDRESS_RANGE = 0x0001_0000


class Register(IntEnum):
    ID = 0x00
    VERSION = 0x04
    CONTROL = 0x08
    STATUS = 0x0C
    RESULT = 0x10
    WEIGHT_CURSOR = 0x14
    WEIGHT_DATA = 0x18
    EVENT_CURSOR = 0x1C
    EVENT_TIMESTAMP = 0x20
    EVENT_NEURON = 0x24
    DIMENSIONS_0 = 0x28
    DIMENSIONS_1 = 0x2C
    NUMERIC_0 = 0x30
    NUMERIC_1 = 0x34
    THRESHOLD = 0x38
    RESET_POTENTIAL = 0x3C
    WEIGHT_CRC32 = 0x40
    EVENT_CRC32 = 0x44
    WEIGHT_WRITE_COUNT = 0x48
    EVENT_WRITE_COUNT = 0x4C
    ERROR = 0x50
    NEURON_MODELS = 0x54
    IZH_AB = 0x58
    IZH_RESET_VOLTAGE = 0x5C
    IZH_RECOVERY_INCREMENT = 0x60
    IZH_SPIKE_THRESHOLD = 0x64
    HH_Q_FORMAT = 0x68
    HH_WIDTHS_STEPS = 0x6C
    HH_LUT_RANGE = 0x70
    HH_SPIKE_THRESHOLD = 0x74
    INFERENCE_EVENT_COUNT = 0x78
    TIMESTEP_COUNT = 0x7C
    INFERENCE_CYCLES_LO = 0x80
    INFERENCE_CYCLES_HI = 0x84
    CONSUMED_INPUT_EVENTS = 0x88
    EMITTED_HIDDEN_SPIKES = 0x8C
    EMITTED_OUTPUT_SPIKES = 0x90
    EXECUTOR_COUNTS = 0x94
    PHASE_SYNAPSE_CYCLES_LO = 0x98
    PHASE_SYNAPSE_CYCLES_HI = 0x9C
    PHASE_HIDDEN_UPDATE_CYCLES_LO = 0xA0
    PHASE_HIDDEN_UPDATE_CYCLES_HI = 0xA4
    PHASE_HIDDEN_ROUTING_CYCLES_LO = 0xA8
    PHASE_HIDDEN_ROUTING_CYCLES_HI = 0xAC
    PHASE_OUTPUT_UPDATE_CYCLES_LO = 0xB0
    PHASE_OUTPUT_UPDATE_CYCLES_HI = 0xB4
    PHASE_CONTROL_OVERHEAD_CYCLES_LO = 0xB8
    PHASE_CONTROL_OVERHEAD_CYCLES_HI = 0xBC


class ControlBits(IntFlag):
    START = 1 << 0
    CLEAR_STATUS_AND_UPLOAD_CHECKS = 1 << 1


class StatusBits(IntFlag):
    IDLE = 1 << 0
    BUSY = 1 << 1
    DONE = 1 << 2
    RESULT_VALID = 1 << 3
    ERROR = 1 << 4
    WEIGHT_READY = 1 << 5
    EVENT_READY = 1 << 6


def pack_weight_cursor(layer: int, pre: int, post: int) -> int:
    if layer not in (0, 1):
        raise ValueError("layer must be 0 or 1")
    if not 0 <= pre < (1 << 15):
        raise ValueError("pre index must fit 15 bits")
    if not 0 <= post < (1 << 16):
        raise ValueError("post index must fit 16 bits")
    return (layer << 31) | (pre << 16) | post


def weight_stream_crc32(data: bytes) -> int:
    """CRC read from WEIGHT_CRC32 after writing exactly *data*."""
    return zlib.crc32(data) & 0xFFFFFFFF


def encode_event_record(event: SpikeEvent) -> bytes:
    """Encode the six CRC bytes used by EVENT_TIMESTAMP/EVENT_NEURON commits."""
    if not 0 <= event.timestamp <= 0xFFFFFFFF:
        raise ValueError("timestamp must fit 32 bits")
    if not 0 <= event.neuron_id <= 0xFFFF:
        raise ValueError("neuron_id must fit 16 bits")
    return struct.pack("<IH", event.timestamp, event.neuron_id)


def encode_event_stream(events: Iterable[SpikeEvent]) -> bytes:
    return b"".join(encode_event_record(event) for event in events)


def event_stream_crc32(events: Iterable[SpikeEvent]) -> int:
    return zlib.crc32(encode_event_stream(events)) & 0xFFFFFFFF
