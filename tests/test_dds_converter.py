"""Tests for DDS converter."""

import pytest

from warhawk_toolkit.converters.dds_header import (
    create_dxt_header,
    create_rgba_header,
    get_dds_fourcc,
    DDS_MAGIC,
)


class TestDDSHeader:
    """Tests for DDS header generation."""

    def test_dds_magic(self):
        header = create_dxt_header(256, 256, b"DXT1")
        assert header[:4] == DDS_MAGIC

    def test_dds_header_size(self):
        header = create_dxt_header(256, 256, b"DXT1")
        # DDS header is 128 bytes (4 magic + 124 header)
        assert len(header) == 128

    def test_dxt1_fourcc(self):
        header = create_dxt_header(256, 256, b"DXT1")
        # FourCC is at offset 84 (4 magic + 72 to pixel format + 8 to fourcc)
        assert header[84:88] == b"DXT1"

    def test_dxt5_fourcc(self):
        header = create_dxt_header(256, 256, b"DXT5")
        assert header[84:88] == b"DXT5"

    def test_dimensions_in_header(self):
        header = create_dxt_header(512, 256, b"DXT1")
        # Height at offset 12, width at offset 16 (little-endian)
        import struct

        height = struct.unpack_from("<I", header, 12)[0]
        width = struct.unpack_from("<I", header, 16)[0]
        assert height == 256
        assert width == 512

    def test_mipmap_count(self):
        header = create_dxt_header(256, 256, b"DXT1", mipmap_count=5)
        import struct

        mipmap_count = struct.unpack_from("<I", header, 28)[0]
        assert mipmap_count == 5

    def test_rgba_header(self):
        header = create_rgba_header(128, 128)
        assert header[:4] == DDS_MAGIC
        assert len(header) == 128


class TestDDSFourCC:
    """Tests for FourCC mapping."""

    def test_dxt1_fourcc(self):
        assert get_dds_fourcc(0x06) == b"DXT1"  # Changed from 0x86

    def test_dxt3_fourcc(self):
        assert get_dds_fourcc(0x07) == b"DXT3"  # Changed from 0x87

    def test_dxt5_fourcc(self):
        assert get_dds_fourcc(0x08) == b"DXT5"  # Changed from 0x88

    def test_unknown_fourcc(self):
        assert get_dds_fourcc(0x99) is None
        # Old codes should now return None
        assert get_dds_fourcc(0x86) is None
        assert get_dds_fourcc(0x87) is None
        assert get_dds_fourcc(0x88) is None
