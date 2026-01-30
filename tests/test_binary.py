"""Tests for binary utilities."""

import pytest

from warhawk_toolkit.utils.binary import BinaryReader


class TestBinaryReader:
    """Tests for BinaryReader class."""

    def test_read_u8(self):
        reader = BinaryReader(b"\x42")
        assert reader.read_u8() == 0x42

    def test_read_u16_big_endian(self):
        reader = BinaryReader(b"\x12\x34")
        assert reader.read_u16() == 0x1234

    def test_read_u32_big_endian(self):
        reader = BinaryReader(b"\x12\x34\x56\x78")
        assert reader.read_u32() == 0x12345678

    def test_read_u64_big_endian(self):
        reader = BinaryReader(b"\x12\x34\x56\x78\x9A\xBC\xDE\xF0")
        assert reader.read_u64() == 0x123456789ABCDEF0

    def test_read_u40(self):
        """Test 40-bit integer reading (PSARC format)."""
        reader = BinaryReader(b"\x12\x34\x56\x78\x9A")
        assert reader.read_u40() == 0x123456789A

    def test_read_u40_max_value(self):
        reader = BinaryReader(b"\xFF\xFF\xFF\xFF\xFF")
        assert reader.read_u40() == 0xFFFFFFFFFF

    def test_read_f32(self):
        # IEEE 754 representation of 1.0
        reader = BinaryReader(b"\x3F\x80\x00\x00")
        assert reader.read_f32() == pytest.approx(1.0)

    def test_read_cstring(self):
        reader = BinaryReader(b"hello\x00world")
        assert reader.read_cstring() == "hello"

    def test_read_fixed_string(self):
        reader = BinaryReader(b"test\x00\x00\x00\x00")
        assert reader.read_fixed_string(8) == "test"

    def test_seek_and_tell(self):
        reader = BinaryReader(b"\x00\x01\x02\x03\x04\x05")
        assert reader.tell() == 0
        reader.seek(3)
        assert reader.tell() == 3
        assert reader.read_u8() == 0x03

    def test_skip(self):
        reader = BinaryReader(b"\x00\x01\x02\x03\x04\x05")
        reader.skip(4)
        assert reader.read_u8() == 0x04

    def test_peek(self):
        reader = BinaryReader(b"\x12\x34\x56")
        peeked = reader.peek(2)
        assert peeked == b"\x12\x34"
        assert reader.tell() == 0  # Position unchanged

    def test_remaining(self):
        reader = BinaryReader(b"\x00\x01\x02\x03\x04\x05")
        assert reader.remaining() == 6
        reader.read_u32()
        assert reader.remaining() == 2

    def test_align(self):
        reader = BinaryReader(b"\x00" * 16)
        reader.seek(1)
        reader.align(4)
        assert reader.tell() == 4

        reader.seek(4)
        reader.align(4)
        assert reader.tell() == 4  # Already aligned

    def test_eof_error(self):
        reader = BinaryReader(b"\x00\x01")
        with pytest.raises(EOFError):
            reader.read_bytes(10)
