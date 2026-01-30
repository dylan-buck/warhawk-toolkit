"""PSARC archive reader and extractor."""

import zlib
from pathlib import Path
from typing import BinaryIO, Iterator, List, Optional, Tuple

from ..utils.binary import BinaryReader
from .header import PSARC_MAGIC, PSARCHeader, PSARCManifest, PSARCTOCEntry


class PSARCReader:
    """Reader for PSARC (PlayStation Archive) files."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._file: Optional[BinaryIO] = None
        self._header: Optional[PSARCHeader] = None
        self._entries: List[PSARCTOCEntry] = []
        self._block_sizes: List[int] = []

    def __enter__(self) -> "PSARCReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def open(self) -> None:
        """Open the archive and parse headers."""
        self._file = open(self.path, "rb")
        self._read_header()
        self._read_toc()
        self._read_block_table()
        self._read_manifest()

    def close(self) -> None:
        """Close the archive file."""
        if self._file:
            self._file.close()
            self._file = None

    @property
    def header(self) -> PSARCHeader:
        if not self._header:
            raise RuntimeError("Archive not opened")
        return self._header

    @property
    def entries(self) -> List[PSARCTOCEntry]:
        return self._entries

    def _read_header(self) -> None:
        """Read the 32-byte PSARC header."""
        reader = BinaryReader(self._file)

        magic = reader.read_bytes(4)
        if magic != PSARC_MAGIC:
            raise ValueError(f"Invalid PSARC magic: {magic!r}, expected {PSARC_MAGIC!r}")

        version_major = reader.read_u16()
        version_minor = reader.read_u16()
        compression_type = reader.read_bytes(4).decode("ascii").rstrip("\x00")
        toc_length = reader.read_u32()
        toc_entry_size = reader.read_u32()
        toc_entry_count = reader.read_u32()
        block_size = reader.read_u32()
        archive_flags = reader.read_u32()

        self._header = PSARCHeader(
            magic=magic,
            version_major=version_major,
            version_minor=version_minor,
            compression_type=compression_type,
            toc_length=toc_length,
            toc_entry_size=toc_entry_size,
            toc_entry_count=toc_entry_count,
            block_size=block_size,
            archive_flags=archive_flags,
        )

    def _read_toc(self) -> None:
        """Read the table of contents entries."""
        reader = BinaryReader(self._file)

        for _ in range(self._header.toc_entry_count):
            name_digest = reader.read_bytes(16)
            block_index = reader.read_u32()
            uncompressed_size = reader.read_u40()
            file_offset = reader.read_u40()

            self._entries.append(
                PSARCTOCEntry(
                    name_digest=name_digest,
                    block_index=block_index,
                    uncompressed_size=uncompressed_size,
                    file_offset=file_offset,
                )
            )

    def _read_block_table(self) -> None:
        """Read the block size table.

        The block table stores compressed sizes for each block.
        Block count = sum of ceil(uncompressed_size / block_size) for all entries.
        """
        # Calculate total number of blocks needed
        total_blocks = 0
        for entry in self._entries:
            if entry.uncompressed_size > 0:
                blocks = (entry.uncompressed_size + self._header.block_size - 1) // self._header.block_size
                total_blocks += blocks

        # Determine block size entry width based on block_size
        # Entry width is the minimum bytes needed to represent block_size
        # A block_size of 65536 (0x10000) needs 3 bytes, but compressed sizes
        # are < block_size (or 0 for uncompressed), so 2 bytes suffice
        if self._header.block_size <= 0x10000:  # <= 65536
            entry_width = 2
        elif self._header.block_size <= 0x1000000:  # <= 16777216
            entry_width = 3
        else:
            entry_width = 4

        reader = BinaryReader(self._file)
        for _ in range(total_blocks):
            if entry_width == 2:
                self._block_sizes.append(reader.read_u16())
            elif entry_width == 3:
                # Read 3 bytes big-endian
                data = reader.read_bytes(3)
                self._block_sizes.append(int.from_bytes(data, byteorder="big"))
            else:
                self._block_sizes.append(reader.read_u32())

    def _read_manifest(self) -> None:
        """Read the manifest (first entry) to get filenames."""
        if not self._entries:
            return

        # First entry is always the manifest
        manifest_entry = self._entries[0]
        manifest_data = self._extract_entry_data(manifest_entry)

        manifest = PSARCManifest.from_data(manifest_data)

        # Assign filenames to entries (manifest itself has no name)
        self._entries[0].filename = "/manifest"
        for i, filename in enumerate(manifest.filenames):
            if i + 1 < len(self._entries):
                self._entries[i + 1].filename = filename

    def _extract_entry_data(self, entry: PSARCTOCEntry) -> bytes:
        """Extract and decompress data for a single entry."""
        if entry.uncompressed_size == 0:
            return b""

        self._file.seek(entry.file_offset)

        # Calculate number of blocks for this entry
        block_count = (entry.uncompressed_size + self._header.block_size - 1) // self._header.block_size

        result = bytearray()
        block_idx = entry.block_index

        for i in range(block_count):
            compressed_size = self._block_sizes[block_idx + i]

            # If compressed_size is 0, the block is stored uncompressed at full block_size
            if compressed_size == 0:
                block_data = self._file.read(self._header.block_size)
            else:
                block_data = self._file.read(compressed_size)

            # Check if the block is compressed (starts with zlib header)
            if len(block_data) >= 2 and block_data[0] == 0x78:
                try:
                    decompressed = zlib.decompress(block_data)
                    result.extend(decompressed)
                except zlib.error:
                    # Not actually compressed, use as-is
                    result.extend(block_data)
            else:
                result.extend(block_data)

        # Trim to exact size
        return bytes(result[: entry.uncompressed_size])

    def extract_file(self, entry: PSARCTOCEntry) -> bytes:
        """Extract a single file from the archive."""
        return self._extract_entry_data(entry)

    def extract_all(
        self, output_dir: Path, progress_callback: Optional[callable] = None
    ) -> Iterator[Tuple[str, Path]]:
        """Extract all files to the output directory.

        Yields (filename, output_path) for each extracted file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(self._entries):
            # Skip manifest
            if entry.filename == "/manifest":
                continue

            # Clean up filename
            filename = entry.filename.lstrip("/")
            if not filename:
                filename = f"unknown_{i}"

            output_path = output_dir / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)

            data = self._extract_entry_data(entry)
            output_path.write_bytes(data)

            if progress_callback:
                progress_callback(i, len(self._entries), entry.filename)

            yield entry.filename, output_path

    def list_files(self) -> List[str]:
        """List all filenames in the archive."""
        return [e.filename for e in self._entries if e.filename != "/manifest"]

    def get_entry_by_name(self, filename: str) -> Optional[PSARCTOCEntry]:
        """Find an entry by filename."""
        # Normalize path
        filename = filename.lstrip("/")
        for entry in self._entries:
            if entry.filename.lstrip("/") == filename:
                return entry
        return None
