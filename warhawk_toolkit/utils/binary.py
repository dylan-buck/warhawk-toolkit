"""Binary reading utilities for big-endian PS3 data."""

import struct
from io import BytesIO
from typing import BinaryIO, Union


class BinaryReader:
    """Helper for reading big-endian binary data (PS3 format)."""

    def __init__(self, data: Union[bytes, BinaryIO]):
        if isinstance(data, bytes):
            self._stream = BytesIO(data)
        else:
            self._stream = data

    @property
    def stream(self) -> BinaryIO:
        return self._stream

    def tell(self) -> int:
        return self._stream.tell()

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._stream.seek(offset, whence)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def read_bytes(self, size: int) -> bytes:
        data = self._stream.read(size)
        if len(data) < size:
            raise EOFError(f"Expected {size} bytes, got {len(data)}")
        return data

    def read_u8(self) -> int:
        return struct.unpack(">B", self.read_bytes(1))[0]

    def read_u16(self) -> int:
        return struct.unpack(">H", self.read_bytes(2))[0]

    def read_u32(self) -> int:
        return struct.unpack(">I", self.read_bytes(4))[0]

    def read_u64(self) -> int:
        return struct.unpack(">Q", self.read_bytes(8))[0]

    def read_i8(self) -> int:
        return struct.unpack(">b", self.read_bytes(1))[0]

    def read_i16(self) -> int:
        return struct.unpack(">h", self.read_bytes(2))[0]

    def read_i32(self) -> int:
        return struct.unpack(">i", self.read_bytes(4))[0]

    def read_i64(self) -> int:
        return struct.unpack(">q", self.read_bytes(8))[0]

    def read_f32(self) -> float:
        return struct.unpack(">f", self.read_bytes(4))[0]

    def read_f64(self) -> float:
        return struct.unpack(">d", self.read_bytes(8))[0]

    def read_u40(self) -> int:
        """Read a 40-bit (5-byte) unsigned integer, big-endian.

        PSARC uses 40-bit integers for file sizes and offsets to handle
        large archives while keeping the TOC compact.
        """
        data = self.read_bytes(5)
        return int.from_bytes(data, byteorder="big", signed=False)

    def read_cstring(self, max_length: int = 1024) -> str:
        """Read a null-terminated string."""
        chars = []
        for _ in range(max_length):
            byte = self._stream.read(1)
            if not byte or byte == b"\x00":
                break
            chars.append(byte)
        return b"".join(chars).decode("utf-8", errors="replace")

    def read_fixed_string(self, length: int, encoding: str = "utf-8") -> str:
        """Read a fixed-length string, stripping null bytes."""
        data = self.read_bytes(length)
        # Strip null bytes from the end
        data = data.rstrip(b"\x00")
        return data.decode(encoding, errors="replace")

    def skip(self, count: int) -> None:
        """Skip forward by count bytes."""
        self._stream.seek(count, 1)

    def align(self, alignment: int) -> None:
        """Align stream position to the given boundary."""
        pos = self.tell()
        remainder = pos % alignment
        if remainder:
            self.skip(alignment - remainder)

    def remaining(self) -> int:
        """Return number of bytes remaining in stream."""
        current = self.tell()
        self._stream.seek(0, 2)  # Seek to end
        end = self.tell()
        self._stream.seek(current)
        return end - current

    def peek(self, size: int) -> bytes:
        """Read bytes without advancing position."""
        data = self._stream.read(size)
        self._stream.seek(-len(data), 1)
        return data


def read_u32_be(data: bytes, offset: int = 0) -> int:
    """Read a big-endian 32-bit unsigned integer from bytes."""
    return struct.unpack_from(">I", data, offset)[0]


def read_u16_be(data: bytes, offset: int = 0) -> int:
    """Read a big-endian 16-bit unsigned integer from bytes."""
    return struct.unpack_from(">H", data, offset)[0]


def read_u64_be(data: bytes, offset: int = 0) -> int:
    """Read a big-endian 64-bit unsigned integer from bytes."""
    return struct.unpack_from(">Q", data, offset)[0]


def write_u32_le(value: int) -> bytes:
    """Write a little-endian 32-bit unsigned integer."""
    return struct.pack("<I", value)


def write_u16_le(value: int) -> bytes:
    """Write a little-endian 16-bit unsigned integer."""
    return struct.pack("<H", value)
