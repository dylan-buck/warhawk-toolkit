"""NGP file format parser.

NGP is Warhawk's container format for 3D models and textures.
Contains:
- Model geometry (vertices, faces)
- Texture metadata with pointers to data in NGP or paired VRAM file

Key characteristics:
- Big-endian throughout (PS3)
- Uses relative pointers (offset from pointer location)
- Texture data can be stored in NGP file or paired VRAM file
"""

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from ..utils.binary import BinaryReader


@dataclass
class NGPTextureHeader:
    """Texture header from NGP file (16 bytes)."""

    compression_method: int  # 0x00: 1/5=none, 6=DXT1, 7=DXT3, 8=DXT5
    data_location_flag: int  # 0x01
    image_format: int  # 0x02-0x03
    width: int  # 0x04-0x05
    height: int  # 0x06-0x07
    data_in_ngp: bool  # 0x08: True if data in NGP, False if in VRAM
    num_mipmaps: int  # 0x0A
    data_offset: int  # 0x0C-0x0F: offset to texture data

    # Calculated fields
    pointer_offset: int = 0  # Position of pointer to this header
    header_offset: int = 0  # Position of this header in NGP
    data_size: int = 0  # Calculated texture data size

    @property
    def is_dxt1(self) -> bool:
        return self.compression_method == 0x06

    @property
    def is_dxt3(self) -> bool:
        return self.compression_method == 0x07

    @property
    def is_dxt5(self) -> bool:
        return self.compression_method == 0x08

    @property
    def is_compressed(self) -> bool:
        return self.compression_method in (0x06, 0x07, 0x08)

    @property
    def fourcc(self) -> Optional[bytes]:
        if self.compression_method == 0x06:
            return b"DXT1"
        elif self.compression_method == 0x07:
            return b"DXT3"
        elif self.compression_method == 0x08:
            return b"DXT5"
        return None


def calculate_texture_size(
    width: int, height: int, num_mipmaps: int, compression: int
) -> int:
    """Calculate total texture data size including mipmaps."""
    if compression == 0x06:  # DXT1
        bytes_per_block = 8
        pixels_per_block = 16
    elif compression in (0x07, 0x08):  # DXT3/DXT5
        bytes_per_block = 16
        pixels_per_block = 16
    else:  # Uncompressed (assume RGBA)
        bytes_per_block = 4
        pixels_per_block = 1

    total_size = 0
    w, h = width, height

    for _ in range(max(1, num_mipmaps)):
        if compression in (0x06, 0x07, 0x08):
            # DXT: 4x4 blocks
            blocks_wide = max(1, (w + 3) // 4)
            blocks_high = max(1, (h + 3) // 4)
            level_size = blocks_wide * blocks_high * bytes_per_block
        else:
            level_size = w * h * bytes_per_block

        total_size += level_size
        w = max(1, w // 2)
        h = max(1, h // 2)

    return total_size


def dereference_relative_pointer(data: bytes, pointer_offset: int) -> int:
    """Dereference a relative pointer (offset from pointer location)."""
    relative_offset = struct.unpack_from(">i", data, pointer_offset)[0]
    return pointer_offset + relative_offset


class NGPFile:
    """Parser for NGP model/texture files."""

    def __init__(self, data: Union[bytes, Path], vram_data: Optional[bytes] = None):
        if isinstance(data, Path):
            data = data.read_bytes()
        self._data = bytes(data)
        self._vram_data = vram_data
        self._texture_headers: List[NGPTextureHeader] = []
        self._parse()

    def _parse(self) -> None:
        """Parse the NGP file structure."""
        if len(self._data) < 0x18:
            raise ValueError(f"NGP data too small: {len(self._data)} bytes")

        # Parse texture table at offset 0x14 (relative pointer)
        self._parse_texture_table()

    def _parse_texture_table(self) -> None:
        """Parse the texture pointer table."""
        # Texture table pointer is at offset 0x10 (relative pointer)
        texture_table_offset = dereference_relative_pointer(self._data, 0x10)

        if texture_table_offset <= 0 or texture_table_offset >= len(self._data):
            return

        # First u32 is the number of textures
        num_textures = struct.unpack_from(">I", self._data, texture_table_offset)[0]

        # Following are relative pointers to each texture header
        for i in range(num_textures):
            pointer_offset = texture_table_offset + 4 + (i * 4)
            if pointer_offset + 4 > len(self._data):
                break

            header_offset = dereference_relative_pointer(self._data, pointer_offset)
            if header_offset <= 0 or header_offset + 0x10 > len(self._data):
                continue

            header = self._parse_texture_header(header_offset, pointer_offset)
            if header:
                self._texture_headers.append(header)

    def _parse_texture_header(
        self, offset: int, pointer_offset: int = 0
    ) -> Optional[NGPTextureHeader]:
        """Parse a 16-byte texture header."""
        if offset + 0x10 > len(self._data):
            return None

        header_bytes = self._data[offset : offset + 0x10]

        compression_method = header_bytes[0x00]
        data_location_flag = header_bytes[0x01]
        image_format = struct.unpack_from(">H", header_bytes, 0x02)[0]
        width = struct.unpack_from(">H", header_bytes, 0x04)[0]
        height = struct.unpack_from(">H", header_bytes, 0x06)[0]
        data_in_ngp = header_bytes[0x08] == 0x01
        num_mipmaps = header_bytes[0x0A]
        data_offset = struct.unpack_from(">I", header_bytes, 0x0C)[0]

        # Calculate data size
        data_size = calculate_texture_size(
            width, height, num_mipmaps, compression_method
        )

        return NGPTextureHeader(
            compression_method=compression_method,
            data_location_flag=data_location_flag,
            image_format=image_format,
            width=width,
            height=height,
            data_in_ngp=data_in_ngp,
            num_mipmaps=max(1, num_mipmaps),
            data_offset=data_offset,
            pointer_offset=pointer_offset,
            header_offset=offset,
            data_size=data_size,
        )

    @property
    def texture_count(self) -> int:
        return len(self._texture_headers)

    @property
    def texture_headers(self) -> List[NGPTextureHeader]:
        return self._texture_headers

    def get_texture_data(self, index: int) -> Optional[bytes]:
        """Get raw texture data for a texture by index."""
        if index < 0 or index >= len(self._texture_headers):
            return None

        header = self._texture_headers[index]

        if header.data_in_ngp:
            # Data is in NGP file
            if header.data_offset + header.data_size > len(self._data):
                return None
            return self._data[header.data_offset : header.data_offset + header.data_size]
        else:
            # Data is in VRAM file
            if self._vram_data is None:
                return None
            if header.data_offset + header.data_size > len(self._vram_data):
                return None
            return self._vram_data[
                header.data_offset : header.data_offset + header.data_size
            ]

    def build_rtt_data(self, index: int) -> Optional[bytes]:
        """Build a complete RTT file from texture header and data.

        RTT header layout (128 bytes):
        - Byte 0: 0x80 (magic)
        - Bytes 1-3: size minus 4 (big-endian, 3 bytes)
        - Byte 4: compression_method
        - Bytes 5-7: padding
        - Bytes 8-9: width (big-endian)
        - Bytes 10-11: height (big-endian)
        - Byte 12: depth (usually 1)
        - Byte 13: mipmap count  <-- CRITICAL: must be at byte 13!
        - Bytes 14-127: padding
        """
        if index < 0 or index >= len(self._texture_headers):
            return None

        header = self._texture_headers[index]
        texture_data = self.get_texture_data(index)

        if texture_data is None:
            return None

        rtt_size = 0x80 + len(texture_data)

        rtt_header = bytearray(0x80)

        # Byte 0: Magic
        rtt_header[0] = 0x80

        # Bytes 1-3: Size field (size minus 4, big-endian)
        size_minus_4 = rtt_size - 4
        rtt_header[1] = (size_minus_4 >> 16) & 0xFF
        rtt_header[2] = (size_minus_4 >> 8) & 0xFF
        rtt_header[3] = size_minus_4 & 0xFF

        # Byte 4: Compression method
        rtt_header[4] = header.compression_method

        # Bytes 5-7: Padding (leave as 0)

        # Bytes 8-9: Width (big-endian)
        rtt_header[8] = (header.width >> 8) & 0xFF
        rtt_header[9] = header.width & 0xFF

        # Bytes 10-11: Height (big-endian)
        rtt_header[10] = (header.height >> 8) & 0xFF
        rtt_header[11] = header.height & 0xFF

        # Byte 12: Depth (usually 1)
        rtt_header[12] = 0x01

        # Byte 13: Mipmap count - CRITICAL fix!
        rtt_header[13] = header.num_mipmaps

        return bytes(rtt_header) + texture_data

    def extract_textures(
        self, output_dir: Path, prefix: Optional[str] = None
    ) -> Iterator[Tuple[int, Path]]:
        """Extract all textures as RTT files.

        Yields (index, output_path) for each extracted texture.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, header in enumerate(self._texture_headers):
            rtt_data = self.build_rtt_data(i)
            if rtt_data is None:
                continue

            # Generate filename like the original tool: 0x{pointer_offset}_0x{header_offset}.rtt
            name = f"0x{header.pointer_offset:x}_0x{header.header_offset:x}.rtt"
            if prefix:
                name = f"{prefix}_{name}"

            output_path = output_dir / name
            output_path.write_bytes(rtt_data)
            yield i, output_path

    def is_valid(self) -> bool:
        """Check if we found any textures."""
        return len(self._texture_headers) > 0

    @classmethod
    def from_file(cls, path: Path, vram_path: Optional[Path] = None) -> "NGPFile":
        """Load an NGP file, optionally with paired VRAM file."""
        path = Path(path)
        vram_data = None

        if vram_path and Path(vram_path).exists():
            vram_data = Path(vram_path).read_bytes()
        elif vram_path is None:
            # Auto-detect VRAM file (same name, .vram extension)
            auto_vram = path.with_suffix(".vram")
            if auto_vram.exists():
                vram_data = auto_vram.read_bytes()

        return cls(path, vram_data)

    @classmethod
    def from_bytes(cls, data: bytes, vram_data: Optional[bytes] = None) -> "NGPFile":
        """Load an NGP file from bytes."""
        return cls(data, vram_data)

    def __repr__(self) -> str:
        return f"NGPFile(textures={self.texture_count})"
