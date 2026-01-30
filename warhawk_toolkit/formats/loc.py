"""LOC localization file format parser.

LOC files contain localized strings for Warhawk.
Supports extraction to various output formats (JSON, CSV, plain text).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

from ..utils.binary import BinaryReader


@dataclass
class LOCEntry:
    """A single localization entry."""

    id: int
    key: str
    value: str


@dataclass
class LOCHeader:
    """LOC file header."""

    magic: int
    version: int
    entry_count: int
    string_table_offset: int


class LOCFile:
    """Parser for LOC localization files."""

    # Common LOC magic values (varies by game version)
    MAGIC_VALUES = [0x4C4F4300, 0x00434F4C]  # "LOC\0" and "\0COL"

    def __init__(self, data: Union[bytes, Path]):
        if isinstance(data, Path):
            data = data.read_bytes()
        self._data = data
        self._header: Optional[LOCHeader] = None
        self._entries: List[LOCEntry] = []
        self._parse()

    def _parse(self) -> None:
        """Parse the LOC file structure."""
        if len(self._data) < 16:
            raise ValueError(f"LOC data too small: {len(self._data)} bytes")

        reader = BinaryReader(self._data)
        self._parse_header(reader)
        self._parse_entries(reader)

    def _parse_header(self, reader: BinaryReader) -> None:
        """Parse the LOC file header."""
        magic = reader.read_u32()

        # Check magic and determine byte order
        if magic not in self.MAGIC_VALUES:
            # Try as string table directly (some LOC files have no header)
            reader.seek(0)
            self._header = LOCHeader(
                magic=0,
                version=0,
                entry_count=0,
                string_table_offset=0,
            )
            self._parse_string_table(reader)
            return

        version = reader.read_u32()
        entry_count = reader.read_u32()
        string_table_offset = reader.read_u32()

        self._header = LOCHeader(
            magic=magic,
            version=version,
            entry_count=entry_count,
            string_table_offset=string_table_offset,
        )

    def _parse_entries(self, reader: BinaryReader) -> None:
        """Parse the entry table."""
        if not self._header or self._header.entry_count == 0:
            return

        for i in range(self._header.entry_count):
            entry_id = reader.read_u32()
            key_offset = reader.read_u32()
            value_offset = reader.read_u32()

            # Save position
            current_pos = reader.tell()

            # Read key string
            key = ""
            if key_offset > 0 and key_offset < len(self._data):
                reader.seek(key_offset)
                key = reader.read_cstring()

            # Read value string
            value = ""
            if value_offset > 0 and value_offset < len(self._data):
                reader.seek(value_offset)
                value = self._read_localized_string(reader)

            # Restore position
            reader.seek(current_pos)

            self._entries.append(LOCEntry(id=entry_id, key=key, value=value))

    def _parse_string_table(self, reader: BinaryReader) -> None:
        """Parse a simple string table format (null-separated strings)."""
        strings = []
        current_string = []

        while reader.remaining() > 0:
            byte = reader.read_bytes(1)
            if byte == b"\x00":
                if current_string:
                    strings.append(b"".join(current_string).decode("utf-8", errors="replace"))
                    current_string = []
            else:
                current_string.append(byte)

        # Handle last string without null terminator
        if current_string:
            strings.append(b"".join(current_string).decode("utf-8", errors="replace"))

        # Create entries from strings
        for i, s in enumerate(strings):
            if s:  # Skip empty strings
                self._entries.append(LOCEntry(id=i, key=f"string_{i}", value=s))

    def _read_localized_string(self, reader: BinaryReader) -> str:
        """Read a localized string, handling UTF-16 if needed."""
        # Check for UTF-16 BOM
        start_pos = reader.tell()
        first_bytes = reader.peek(2)

        if first_bytes == b"\xff\xfe" or first_bytes == b"\xfe\xff":
            # UTF-16 encoded
            reader.skip(2)  # Skip BOM
            chars = []
            while reader.remaining() >= 2:
                char_bytes = reader.read_bytes(2)
                if char_bytes == b"\x00\x00":
                    break
                chars.append(char_bytes)

            encoding = "utf-16-le" if first_bytes == b"\xff\xfe" else "utf-16-be"
            return b"".join(chars).decode(encoding, errors="replace")
        else:
            # UTF-8 / ASCII
            return reader.read_cstring()

    @property
    def header(self) -> Optional[LOCHeader]:
        return self._header

    @property
    def entries(self) -> List[LOCEntry]:
        return self._entries

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def get_by_key(self, key: str) -> Optional[str]:
        """Get a localized string by its key."""
        for entry in self._entries:
            if entry.key == key:
                return entry.value
        return None

    def get_by_id(self, entry_id: int) -> Optional[str]:
        """Get a localized string by its ID."""
        for entry in self._entries:
            if entry.id == entry_id:
                return entry.value
        return None

    def items(self) -> Iterator[Tuple[str, str]]:
        """Iterate over (key, value) pairs."""
        for entry in self._entries:
            yield entry.key, entry.value

    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary mapping keys to values."""
        return {entry.key: entry.value for entry in self._entries}

    def to_json(self, indent: int = 2) -> str:
        """Export to JSON format."""
        data = {
            "entries": [
                {"id": e.id, "key": e.key, "value": e.value}
                for e in self._entries
            ]
        }
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def to_csv(self) -> str:
        """Export to CSV format."""
        lines = ["id,key,value"]
        for entry in self._entries:
            # Escape quotes and commas in value
            value = entry.value.replace('"', '""')
            lines.append(f'{entry.id},"{entry.key}","{value}"')
        return "\n".join(lines)

    def to_text(self) -> str:
        """Export to plain text format."""
        lines = []
        for entry in self._entries:
            lines.append(f"[{entry.key}]")
            lines.append(entry.value)
            lines.append("")
        return "\n".join(lines)

    def save_json(self, path: Path) -> None:
        """Save to JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    def save_csv(self, path: Path) -> None:
        """Save to CSV file."""
        Path(path).write_text(self.to_csv(), encoding="utf-8")

    def save_text(self, path: Path) -> None:
        """Save to text file."""
        Path(path).write_text(self.to_text(), encoding="utf-8")

    @classmethod
    def from_file(cls, path: Path) -> "LOCFile":
        """Load a LOC file from disk."""
        return cls(path)

    @classmethod
    def from_bytes(cls, data: bytes) -> "LOCFile":
        """Load a LOC file from bytes."""
        return cls(data)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[LOCEntry]:
        return iter(self._entries)

    def __repr__(self) -> str:
        return f"LOCFile(entries={len(self._entries)})"
