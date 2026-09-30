"""Host-side register and binary-format definitions for the SNN accelerator."""

from .registers import (
    ControlBits,
    Register,
    SNN_ADDRESS_RANGE,
    StatusBits,
    encode_event_record,
    encode_event_stream,
    event_stream_crc32,
    pack_weight_cursor,
    weight_stream_crc32,
)
from .device import (
    AcceleratorError,
    HardwareInfo,
    HardwareMismatchError,
    InferenceTelemetry,
    InferenceTimeoutError,
    ModelPackage,
    SNNDevice,
    SNNHostError,
)

__all__ = [
    "ControlBits",
    "Register",
    "SNN_ADDRESS_RANGE",
    "StatusBits",
    "encode_event_record",
    "encode_event_stream",
    "event_stream_crc32",
    "pack_weight_cursor",
    "weight_stream_crc32",
    "AcceleratorError",
    "HardwareInfo",
    "HardwareMismatchError",
    "InferenceTelemetry",
    "InferenceTimeoutError",
    "ModelPackage",
    "SNNDevice",
    "SNNHostError",
]
