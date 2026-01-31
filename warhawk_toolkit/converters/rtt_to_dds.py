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

    # Texture data is already in the correct format for DDS (little-endian).
    # No byte swapping needed - confirmed by comparing with JMcKiern/warhawk-reversing.
    texture_data = rtt.texture_data

    # Combine header and data
    dds_data = header + texture_data

    # Write to file if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(dds_data)

    return dds_data


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
