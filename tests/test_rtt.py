"""Tests for RTT texture format."""

import pytest

from warhawk_toolkit.formats.rtt import RTTTexture, RTTCompressionType


def create_rtt_header(
    magic: int = 0x80,
    compression: int = 0x06,  # DXT1 - changed from 0x86
    width: int = 256,
    height: int = 256,
    depth: int = 1,
    mipmaps: int = 1,
) -> bytes:
    """Create a minimal RTT header for testing.

    Matches game RTT format per JMcKiern/warhawk-reversing:
    - Byte 12: 0x00
    - Byte 13: 0x01 (depth marker)
    - Byte 14: mipmap count
    - Byte 15: 0x02
    """
    header = bytearray(128)
    header[0] = magic
    header[4] = compression
    # Width (big-endian)
    header[8] = (width >> 8) & 0xFF
    header[9] = width & 0xFF
    # Height (big-endian)
    header[10] = (height >> 8) & 0xFF
    header[11] = height & 0xFF
    # Byte 12: 0x00 (already zero)
    # Byte 13: depth marker (0x01)
    header[13] = depth
    # Byte 14: Mipmaps
    header[14] = mipmaps
    # Byte 15: format marker (0x02)
    header[15] = 0x02
    return bytes(header)


class TestRTTTexture:
    """Tests for RTTTexture class."""

    def test_valid_dxt1_texture(self):
        header = create_rtt_header(compression=RTTCompressionType.DXT1)
        rtt = RTTTexture(header)

        assert rtt.is_valid()
        assert rtt.width == 256
        assert rtt.height == 256
        assert rtt.header.is_dxt1

    def test_valid_dxt5_texture(self):
        header = create_rtt_header(compression=RTTCompressionType.DXT5)
        rtt = RTTTexture(header)

        assert rtt.is_valid()
        assert rtt.header.is_dxt5
        assert rtt.header.is_compressed

    def test_rgba_texture(self):
        header = create_rtt_header(compression=RTTCompressionType.RGBA)
        rtt = RTTTexture(header)

        assert rtt.is_valid()
        assert rtt.header.is_rgba
        assert not rtt.header.is_compressed

    def test_invalid_magic(self):
        header = create_rtt_header(magic=0x00)
        rtt = RTTTexture(header)

        assert not rtt.is_valid()

    def test_mipmap_size_dxt1(self):
        header = create_rtt_header(
            compression=RTTCompressionType.DXT1,
            width=256,
            height=256,
            mipmaps=3,
        )
        rtt = RTTTexture(header)

        # DXT1: 4x4 blocks, 8 bytes per block
        # Level 0: 256x256 = 64x64 blocks = 32768 bytes
        # Level 1: 128x128 = 32x32 blocks = 8192 bytes
        # Level 2: 64x64 = 16x16 blocks = 2048 bytes
        assert rtt.get_mipmap_size(0) == 32768
        assert rtt.get_mipmap_size(1) == 8192
        assert rtt.get_mipmap_size(2) == 2048

    def test_mipmap_size_dxt5(self):
        header = create_rtt_header(
            compression=RTTCompressionType.DXT5,
            width=128,
            height=128,
        )
        rtt = RTTTexture(header)

        # DXT5: 4x4 blocks, 16 bytes per block
        # 128x128 = 32x32 blocks = 16384 bytes
        assert rtt.get_mipmap_size(0) == 16384

    def test_bits_per_pixel(self):
        dxt1 = RTTTexture(create_rtt_header(compression=RTTCompressionType.DXT1))
        dxt5 = RTTTexture(create_rtt_header(compression=RTTCompressionType.DXT5))
        rgba = RTTTexture(create_rtt_header(compression=RTTCompressionType.RGBA))

        assert dxt1.header.bits_per_pixel == 4
        assert dxt5.header.bits_per_pixel == 8
        assert rgba.header.bits_per_pixel == 32

    def test_texture_data_extraction(self):
        header = create_rtt_header()
        texture_data = b"\xAB" * 100
        full_data = header + texture_data

        rtt = RTTTexture(full_data)
        assert rtt.texture_data == texture_data

    def test_data_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            RTTTexture(b"\x80" * 10)

    def test_repr(self):
        rtt = RTTTexture(create_rtt_header())
        repr_str = repr(rtt)
        assert "256" in repr_str
        assert "0x06" in repr_str  # Changed from 0x86
