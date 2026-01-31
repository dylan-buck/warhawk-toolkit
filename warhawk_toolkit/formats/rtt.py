"""RTT texture format parser.

RTT is Warhawk's texture format. Key characteristics:
- Magic byte: 0x80 at offset 0
- Contains DXT-compressed texture data
- Header contains dimensions, mipmap count, and compression type
"""

from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Optional, Union

from ..utils.binary import BinaryReader


class RTTCompressionType(IntEnum):
    """RTT compression/format types.

    Note: Actual game files use 0x01-0x08, not 0x85-0x88 as previously assumed.
    """

    RGBA = 0x01  # Uncompressed RGBA
    RGBA_ALT = 0x05  # Uncompressed RGBA (alternate code)
    DXT1 = 0x06  # BC1 - 4bpp, 1-bit alpha
    DXT3 = 0x07  # BC2 - 8bpp, explicit alpha
    DXT5 = 0x08  # BC3 - 8bpp, interpolated alpha


# Map RTT compression to DDS FourCC
RTT_TO_DDS_FOURCC = {
    RTTCompressionType.DXT1: b"DXT1",
    RTTCompressionType.DXT3: b"DXT3",
    RTTCompressionType.DXT5: b"DXT5",
}


@dataclass
class RTTHeader:
    """RTT texture header."""

    magic: int  # Should be 0x80
    compression_type: int  # Compression format (DXT1/3/5 or RGBA)
    width: int
    height: int
    mipmap_count: int
    depth: int  # For 3D textures, usually 1

    @property
    def is_valid(self) -> bool:
        return self.magic == 0x80

    @property
    def is_dxt1(self) -> bool:
        return self.compression_type == RTTCompressionType.DXT1

    @property
    def is_dxt3(self) -> bool:
        return self.compression_type == RTTCompressionType.DXT3

    @property
    def is_dxt5(self) -> bool:
        return self.compression_type == RTTCompressionType.DXT5

    @property
    def is_rgba(self) -> bool:
        return self.compression_type in (
            RTTCompressionType.RGBA,
            RTTCompressionType.RGBA_ALT,
        )

    @property
    def is_compressed(self) -> bool:
        return self.compression_type in (
            RTTCompressionType.DXT1,
            RTTCompressionType.DXT3,
            RTTCompressionType.DXT5,
        )

    @property
    def bits_per_pixel(self) -> int:
        if self.compression_type == RTTCompressionType.DXT1:
            return 4
        elif self.compression_type in (RTTCompressionType.DXT3, RTTCompressionType.DXT5):
            return 8
        elif self.compression_type in (RTTCompressionType.RGBA, RTTCompressionType.RGBA_ALT):
            return 32
        else:
            return 32  # Default to RGBA for unknown types

    @property
    def dds_fourcc(self) -> Optional[bytes]:
        return RTT_TO_DDS_FOURCC.get(self.compression_type)


class RTTTexture:
    """Parser for RTT texture files."""

    # Header size in bytes
    HEADER_SIZE = 128

    def __init__(self, data: Union[bytes, Path]):
        if isinstance(data, Path):
            data = data.read_bytes()
        self._data = data
        self._header: Optional[RTTHeader] = None
        self._parse()

    def _parse(self) -> None:
        """Parse the RTT header."""
        if len(self._data) < self.HEADER_SIZE:
            raise ValueError(f"RTT data too small: {len(self._data)} bytes")

        reader = BinaryReader(self._data)

        # Byte 0: Magic (0x80)
        magic = reader.read_u8()

        # Bytes 1-3: Unknown/padding
        reader.skip(3)

        # Byte 4: Compression type
        compression_type = reader.read_u8()

        # Bytes 5-7: Unknown
        reader.skip(3)

        # Bytes 8-9: Width (big-endian)
        width = reader.read_u16()

        # Bytes 10-11: Height (big-endian)
        height = reader.read_u16()

        # Byte 12: Usually 0x00 in game files
        reader.skip(1)

        # Byte 13: Usually 0x01 in game files (may indicate depth=1)
        depth = reader.read_u8()

        # Byte 14: Mipmap count (confirmed via JMcKiern's rtt2dds.py)
        mipmap_count = reader.read_u8()

        self._header = RTTHeader(
            magic=magic,
            compression_type=compression_type,
            width=width,
            height=height,
            mipmap_count=mipmap_count,
            depth=depth if depth > 0 else 1,
        )

    @property
    def header(self) -> RTTHeader:
        if not self._header:
            raise RuntimeError("RTT not parsed")
        return self._header

    @property
    def width(self) -> int:
        return self.header.width

    @property
    def height(self) -> int:
        return self.header.height

    @property
    def mipmap_count(self) -> int:
        return self.header.mipmap_count

    @property
    def compression_type(self) -> int:
        return self.header.compression_type

    @property
    def texture_data(self) -> bytes:
        """Get the raw texture data (after header)."""
        return self._data[self.HEADER_SIZE :]

    def get_mipmap_size(self, level: int) -> int:
        """Calculate the size of a specific mipmap level."""
        width = max(1, self.width >> level)
        height = max(1, self.height >> level)

        if self.header.is_compressed:
            # DXT block size is 4x4
            block_width = max(1, (width + 3) // 4)
            block_height = max(1, (height + 3) // 4)
            block_size = 8 if self.header.is_dxt1 else 16
            return block_width * block_height * block_size
        else:
            # RGBA: 4 bytes per pixel
            return width * height * 4

    def get_total_data_size(self) -> int:
        """Calculate total texture data size including all mipmaps."""
        total = 0
        for level in range(self.mipmap_count):
            total += self.get_mipmap_size(level)
        return total

    def is_valid(self) -> bool:
        """Check if this is a valid RTT texture."""
        return self._header is not None and self._header.is_valid

    @classmethod
    def from_file(cls, path: Path) -> "RTTTexture":
        """Load an RTT texture from a file."""
        return cls(path)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RTTTexture":
        """Load an RTT texture from bytes."""
        return cls(data)

    def __repr__(self) -> str:
        if self._header:
            return (
                f"RTTTexture(width={self.width}, height={self.height}, "
                f"compression=0x{self.compression_type:02X}, mipmaps={self.mipmap_count})"
            )
        return "RTTTexture(invalid)"
