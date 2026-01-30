"""VRAM file format parser.

VRAM files contain packed texture data that pairs with NGP model files.
The NGP file's texture table defines offsets/sizes into the VRAM data.

Key characteristics:
- No header - raw texture data container
- Paired with NGP file by filename (model.ngp + model.vram)
- Contains multiple RTT textures packed sequentially
- Texture boundaries defined by NGP texture table
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

from .rtt import RTTTexture


@dataclass
class VRAMTextureInfo:
    """Information about a texture in the VRAM file."""

    index: int
    offset: int
    size: int
    texture: Optional[RTTTexture] = None


class VRAMFile:
    """Parser for VRAM texture container files."""

    def __init__(self, data: Union[bytes, Path]):
        if isinstance(data, Path):
            data = data.read_bytes()
        self._data = data
        self._textures: List[VRAMTextureInfo] = []

    @property
    def data(self) -> bytes:
        return self._data

    @property
    def size(self) -> int:
        return len(self._data)

    @property
    def textures(self) -> List[VRAMTextureInfo]:
        return self._textures

    def set_texture_layout(self, offsets_and_sizes: List[Tuple[int, int]]) -> None:
        """Define texture boundaries from NGP texture table.

        Args:
            offsets_and_sizes: List of (offset, size) tuples for each texture
        """
        self._textures = []

        for i, (offset, size) in enumerate(offsets_and_sizes):
            if offset + size > len(self._data):
                # Skip invalid entries
                continue

            texture_data = self._data[offset : offset + size]

            # Try to parse as RTT texture
            texture = None
            try:
                texture = RTTTexture(texture_data)
                if not texture.is_valid():
                    texture = None
            except (ValueError, EOFError):
                pass

            self._textures.append(
                VRAMTextureInfo(
                    index=i,
                    offset=offset,
                    size=size,
                    texture=texture,
                )
            )

    def get_texture_data(self, index: int) -> Optional[bytes]:
        """Get raw texture data by index."""
        if index < 0 or index >= len(self._textures):
            return None
        info = self._textures[index]
        return self._data[info.offset : info.offset + info.size]

    def get_texture(self, index: int) -> Optional[RTTTexture]:
        """Get parsed RTT texture by index."""
        if index < 0 or index >= len(self._textures):
            return None
        return self._textures[index].texture

    def extract_all(self, output_dir: Path, prefix: str = "texture") -> Iterator[Path]:
        """Extract all textures to files.

        Yields the path of each extracted texture file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for info in self._textures:
            filename = f"{prefix}_{info.index:03d}.rtt"
            output_path = output_dir / filename

            texture_data = self._data[info.offset : info.offset + info.size]
            output_path.write_bytes(texture_data)

            yield output_path

    def scan_for_textures(self) -> List[int]:
        """Scan VRAM data for RTT texture magic bytes (0x80).

        Returns offsets where textures might start.
        Useful when NGP texture table is missing/corrupt.
        """
        offsets = []
        i = 0

        while i < len(self._data) - 128:  # RTT header is 128 bytes
            if self._data[i] == 0x80:
                # Check if this looks like a valid RTT header
                try:
                    texture = RTTTexture(self._data[i:])
                    if texture.is_valid() and texture.width > 0 and texture.height > 0:
                        offsets.append(i)
                        # Skip past this texture's data
                        i += 128 + texture.get_total_data_size()
                        continue
                except (ValueError, EOFError):
                    pass
            i += 1

        return offsets

    def auto_detect_textures(self) -> int:
        """Automatically detect and parse textures in VRAM data.

        Returns the number of textures found.
        """
        offsets = self.scan_for_textures()
        self._textures = []

        for i, offset in enumerate(offsets):
            try:
                # Calculate size (to next texture or end of file)
                if i + 1 < len(offsets):
                    size = offsets[i + 1] - offset
                else:
                    size = len(self._data) - offset

                texture_data = self._data[offset : offset + size]
                texture = RTTTexture(texture_data)

                self._textures.append(
                    VRAMTextureInfo(
                        index=i,
                        offset=offset,
                        size=size,
                        texture=texture,
                    )
                )
            except (ValueError, EOFError):
                continue

        return len(self._textures)

    @classmethod
    def from_file(cls, path: Path) -> "VRAMFile":
        """Load a VRAM file from disk."""
        return cls(path)

    @classmethod
    def from_bytes(cls, data: bytes) -> "VRAMFile":
        """Load a VRAM file from bytes."""
        return cls(data)

    @classmethod
    def find_paired_file(cls, ngp_path: Path) -> Optional[Path]:
        """Find the VRAM file paired with an NGP file."""
        vram_path = ngp_path.with_suffix(".vram")
        if vram_path.exists():
            return vram_path

        # Try lowercase extension
        vram_path = ngp_path.with_suffix(".VRAM")
        if vram_path.exists():
            return vram_path

        return None

    def __len__(self) -> int:
        return len(self._textures)

    def __repr__(self) -> str:
        return f"VRAMFile(size={self.size}, textures={len(self._textures)})"
