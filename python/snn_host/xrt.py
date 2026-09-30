"""Persistent XRT register backend for PCIe Alveo cards."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


class XrtRegisterIO:
    """Map the RegisterIO protocol onto an ``xrt::ip`` helper process."""

    def __init__(
        self,
        xclbin: str | os.PathLike[str],
        kernel: str,
        *,
        device: int = 0,
        server: str | os.PathLike[str] = "xrt-register-server",
    ) -> None:
        if device < 0:
            raise ValueError("XRT device index must be non-negative")
        self.xclbin = Path(xclbin)
        self.kernel = kernel
        self._process = subprocess.Popen(
            [str(server), str(device), str(self.xclbin), kernel],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._process.stdout.readline().strip() if self._process.stdout else ""
        if ready != "READY":
            error = self._process.stderr.read().strip() if self._process.stderr else ""
            self._process.wait()
            raise RuntimeError(f"XRT register server failed to start: {error or ready}")

    def _request(self, command: str) -> str:
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("XRT register server pipes are unavailable")
        self._process.stdin.write(command + "\n")
        self._process.stdin.flush()
        response = self._process.stdout.readline().strip()
        if not response.startswith("OK"):
            if not response and self._process.poll() is not None:
                error = self._process.stderr.read().strip() if self._process.stderr else ""
                raise RuntimeError(f"XRT register server exited: {error}")
            raise RuntimeError(f"XRT register access failed: {response}")
        return response

    @staticmethod
    def _offset(offset: int) -> int:
        offset = int(offset)
        if offset < 0 or offset % 4:
            raise ValueError(f"invalid 32-bit register offset 0x{offset:x}")
        return offset

    def read32(self, offset: int) -> int:
        response = self._request(f"r {self._offset(offset)}")
        fields = response.split()
        if len(fields) != 2:
            raise RuntimeError(f"malformed XRT read response: {response}")
        return int(fields[1], 0) & 0xFFFF_FFFF

    def write32(self, offset: int, value: int) -> None:
        self._request(f"w {self._offset(offset)} {int(value) & 0xFFFF_FFFF}")

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                if self._process.stdin is not None:
                    self._process.stdin.write("q\n")
                    self._process.stdin.flush()
            finally:
                self._process.wait(timeout=5)
        for pipe in (self._process.stdin, self._process.stdout, self._process.stderr):
            if pipe is not None:
                pipe.close()

    def __enter__(self) -> XrtRegisterIO:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["XrtRegisterIO"]
