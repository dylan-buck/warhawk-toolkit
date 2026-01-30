"""PSARC header and TOC structures."""

from dataclasses import dataclass
from typing import List

# PSARC magic bytes
PSARC_MAGIC = b"PSAR"


@dataclass
class PSARCHeader:
    """PSARC archive header (32 bytes)."""

    magic: bytes  # 4 bytes: "PSAR"
    version_major: int  # 2 bytes
    version_minor: int  # 2 bytes
    compression_type: str  # 4 bytes: "zlib" or "lzma"
    toc_length: int  # 4 bytes: Total TOC size including header
    toc_entry_size: int  # 4 bytes: Size of each TOC entry (30 bytes)
    toc_entry_count: int  # 4 bytes: Number of entries
    block_size: int  # 4 bytes: Block size for compression (65536 typical)
    archive_flags: int  # 4 bytes: Flags (relative paths, case-insensitive, etc.)

    @property
    def is_valid(self) -> bool:
        return self.magic == PSARC_MAGIC

    @property
    def uses_zlib(self) -> bool:
        return self.compression_type == "zlib"

    @property
    def uses_lzma(self) -> bool:
        return self.compression_type == "lzma"


@dataclass
class PSARCTOCEntry:
    """PSARC table of contents entry (30 bytes)."""

    name_digest: bytes  # 16 bytes: MD5 hash of filename
    block_index: int  # 4 bytes: Index into block size table
    uncompressed_size: int  # 5 bytes (40-bit): Original file size
    file_offset: int  # 5 bytes (40-bit): Offset in archive

    # Resolved after reading manifest
    filename: str = ""


@dataclass
class PSARCManifest:
    """Manifest containing filenames for all entries."""

    filenames: List[str]

    @classmethod
    def from_data(cls, data: bytes) -> "PSARCManifest":
        """Parse manifest from raw data (newline-separated paths)."""
        text = data.decode("utf-8", errors="replace")
        # Split by newlines and filter empty
        filenames = [line for line in text.split("\n") if line]
        return cls(filenames=filenames)
