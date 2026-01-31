"""DDS header generation for texture conversion."""

import struct
from dataclasses import dataclass
from typing import Optional


# DDS magic number
DDS_MAGIC = b"DDS "

# DDS header flags
DDSD_CAPS = 0x1
DDSD_HEIGHT = 0x2
DDSD_WIDTH = 0x4
DDSD_PITCH = 0x8
DDSD_PIXELFORMAT = 0x1000
DDSD_MIPMAPCOUNT = 0x20000
DDSD_LINEARSIZE = 0x80000
DDSD_DEPTH = 0x800000

# DDS pixel format flags
DDPF_ALPHAPIXELS = 0x1
DDPF_FOURCC = 0x4
DDPF_RGB = 0x40

# DDS caps flags
DDSCAPS_COMPLEX = 0x8
DDSCAPS_MIPMAP = 0x400000
DDSCAPS_TEXTURE = 0x1000


@dataclass
class DDSPixelFormat:
    """DDS pixel format structure (32 bytes)."""

    size: int = 32
    flags: int = 0
    fourcc: bytes = b"\x00\x00\x00\x00"
    rgb_bit_count: int = 0
    r_bitmask: int = 0
    g_bitmask: int = 0
    b_bitmask: int = 0
    a_bitmask: int = 0

    def to_bytes(self) -> bytes:
        """Serialize to bytes (little-endian)."""
        return struct.pack(
            "<II4sIIIII",
            self.size,
            self.flags,
            self.fourcc,
            self.rgb_bit_count,
            self.r_bitmask,
            self.g_bitmask,
            self.b_bitmask,
            self.a_bitmask,
        )


@dataclass
class DDSHeader:
    """DDS file header (124 bytes + 4 byte magic)."""

    width: int
    height: int
    mipmap_count: int = 1
    depth: int = 1
    pixel_format: DDSPixelFormat = None
    pitch_or_linear_size: int = 0

    def __post_init__(self):
        if self.pixel_format is None:
            self.pixel_format = DDSPixelFormat()

    def to_bytes(self) -> bytes:
        """Serialize to bytes including magic (little-endian)."""
        # Calculate flags
        flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT

        if self.mipmap_count > 1:
            flags |= DDSD_MIPMAPCOUNT

        if self.pixel_format.flags & DDPF_FOURCC:
            flags |= DDSD_LINEARSIZE
        else:
            flags |= DDSD_PITCH

        # Calculate caps
        caps1 = DDSCAPS_TEXTURE
        if self.mipmap_count > 1:
            caps1 |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP

        caps2 = 0

        # Build header
        header = struct.pack(
            "<4sI",
            DDS_MAGIC,
            124,  # Header size (excluding magic)
        )

        header += struct.pack(
            "<IIIII",
            flags,
            self.height,
            self.width,
            self.pitch_or_linear_size,
            self.depth,
        )

        header += struct.pack("<I", self.mipmap_count)

        # Reserved1 (11 DWORDs = 44 bytes)
        header += b"\x00" * 44

        # Pixel format
        header += self.pixel_format.to_bytes()

        # Caps
        header += struct.pack("<IIII", caps1, caps2, 0, 0)

        # Reserved2
        header += struct.pack("<I", 0)

        return header


def create_dxt_header(
    width: int,
    height: int,
    fourcc: bytes,
    mipmap_count: int = 1,
) -> bytes:
    """Create a DDS header for DXT-compressed texture."""
    pixel_format = DDSPixelFormat(
        flags=DDPF_FOURCC,
        fourcc=fourcc,
    )

    # Calculate linear size (size of first mipmap)
    block_size = 8 if fourcc == b"DXT1" else 16
    blocks_wide = max(1, (width + 3) // 4)
    blocks_high = max(1, (height + 3) // 4)
    linear_size = blocks_wide * blocks_high * block_size

    header = DDSHeader(
        width=width,
        height=height,
        mipmap_count=mipmap_count,
        pixel_format=pixel_format,
        pitch_or_linear_size=linear_size,
    )

    return header.to_bytes()


def create_rgba_header(
    width: int,
    height: int,
    mipmap_count: int = 1,
) -> bytes:
    """Create a DDS header for uncompressed RGBA texture."""
    pixel_format = DDSPixelFormat(
        flags=DDPF_RGB | DDPF_ALPHAPIXELS,
        rgb_bit_count=32,
        r_bitmask=0x00FF0000,
        g_bitmask=0x0000FF00,
        b_bitmask=0x000000FF,
        a_bitmask=0xFF000000,
    )

    # Pitch = bytes per row
    pitch = width * 4

    header = DDSHeader(
        width=width,
        height=height,
        mipmap_count=mipmap_count,
        pixel_format=pixel_format,
        pitch_or_linear_size=pitch,
    )

    return header.to_bytes()


def get_dds_fourcc(compression_type: int) -> Optional[bytes]:
    """Map RTT compression type to DDS FourCC.

    Note: Actual game files use 0x06-0x08, not 0x86-0x88 as previously assumed.
    """
    mapping = {
        0x06: b"DXT1",
        0x07: b"DXT3",
        0x08: b"DXT5",
    }
    return mapping.get(compression_type)
