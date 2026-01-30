"""RTT to DDS texture converter."""

from pathlib import Path
from typing import Optional, Union

from ..formats.rtt import RTTTexture
from .dds_header import create_dxt_header, create_rgba_header, get_dds_fourcc


def convert_rtt_to_dds(
    rtt: Union[RTTTexture, bytes, Path],
    output_path: Optional[Path] = None,
) -> bytes:
    """Convert an RTT texture to DDS format.

    Args:
        rtt: RTT texture (RTTTexture instance, bytes, or file path)
        output_path: Optional path to write DDS file

    Returns:
        DDS file data as bytes
    """
    # Load RTT if needed
    if isinstance(rtt, (bytes, Path)):
        rtt = RTTTexture(rtt)

    if not rtt.is_valid():
        raise ValueError("Invalid RTT texture")

    # Generate DDS header
    if rtt.header.is_compressed:
        fourcc = get_dds_fourcc(rtt.compression_type)
        if not fourcc:
            raise ValueError(f"Unsupported compression type: 0x{rtt.compression_type:02X}")
        header = create_dxt_header(
            width=rtt.width,
            height=rtt.height,
            fourcc=fourcc,
            mipmap_count=rtt.mipmap_count,
        )
    else:
        header = create_rgba_header(
            width=rtt.width,
            height=rtt.height,
            mipmap_count=rtt.mipmap_count,
        )

    # Get texture data
    texture_data = rtt.texture_data

    # For DXT textures, data may need byte swapping (PS3 is big-endian)
    if rtt.header.is_compressed:
        texture_data = swap_dxt_endian(texture_data, rtt.compression_type)

    # Combine header and data
    dds_data = header + texture_data

    # Write to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(dds_data)

    return dds_data


def swap_dxt_endian(data: bytes, compression_type: int) -> bytes:
    """Swap endianness of DXT-compressed texture data.

    PS3 stores DXT data in big-endian format.
    DDS expects little-endian format.

    DXT1: 8-byte blocks (2x 16-bit color + 4-byte indices)
    DXT3/DXT5: 16-byte blocks (8-byte alpha + 8-byte color)
    """
    result = bytearray(len(data))
    is_dxt1 = compression_type == 0x86
    block_size = 8 if is_dxt1 else 16

    offset = 0
    while offset + block_size <= len(data):
        if is_dxt1:
            # DXT1: swap color bytes and indices
            result[offset : offset + 8] = swap_dxt1_block(data[offset : offset + 8])
        else:
            # DXT3/DXT5: swap alpha block then color block
            result[offset : offset + 8] = swap_alpha_block(
                data[offset : offset + 8], compression_type
            )
            result[offset + 8 : offset + 16] = swap_dxt1_block(
                data[offset + 8 : offset + 16]
            )
        offset += block_size

    # Copy any remaining bytes
    if offset < len(data):
        result[offset:] = data[offset:]

    return bytes(result)


def swap_dxt1_block(block: bytes) -> bytes:
    """Swap endianness of a DXT1 color block (8 bytes).

    Format:
    - 2 bytes: color0 (RGB565)
    - 2 bytes: color1 (RGB565)
    - 4 bytes: lookup table (2-bit indices)
    """
    if len(block) < 8:
        return block

    result = bytearray(8)

    # Swap 16-bit colors
    result[0] = block[1]
    result[1] = block[0]
    result[2] = block[3]
    result[3] = block[2]

    # Copy lookup table (no swap needed for indices)
    result[4:8] = block[4:8]

    return bytes(result)


def swap_alpha_block(block: bytes, compression_type: int) -> bytes:
    """Swap endianness of alpha block (8 bytes).

    DXT3: 16 4-bit alpha values (explicit alpha)
    DXT5: 2 alpha endpoints + 6 bytes of 3-bit indices (interpolated alpha)
    """
    if len(block) < 8:
        return block

    is_dxt3 = compression_type == 0x87

    if is_dxt3:
        # DXT3: swap pairs of bytes for 16-bit alpha values
        result = bytearray(8)
        for i in range(0, 8, 2):
            result[i] = block[i + 1]
            result[i + 1] = block[i]
        return bytes(result)
    else:
        # DXT5: alpha endpoints (2 bytes) + indices (6 bytes)
        # Indices are tightly packed 3-bit values, no swap needed
        return block


def convert_rtt_file(
    input_path: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """Convert an RTT file to DDS.

    Args:
        input_path: Path to RTT file
        output_path: Optional output path (defaults to same name with .dds extension)

    Returns:
        Path to the created DDS file
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".dds")

    rtt = RTTTexture.from_file(input_path)
    convert_rtt_to_dds(rtt, output_path)

    return output_path
