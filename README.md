# Warhawk Toolkit

Extract and convert Warhawk PS3 game files.

## Overview

A 3-stage extraction pipeline for Warhawk game assets:

```
Stage 1: PSARC Extraction
    .psarc archive → Raw files (NGP, VRAM, RTT, LOC, etc.)

Stage 2: Warhawk Container Extraction
    NGP + VRAM files → Embedded textures (RTT)

Stage 3: Format Conversion
    RTT → DDS (standard texture format)
```

## Installation

```bash
pip install .
```

Or for development:

```bash
pip install -e ".[dev]"
```

## Usage

### Extract everything from a PSARC archive

```bash
# Extract and auto-convert Warhawk formats (extracts NGP textures, converts RTT→DDS)
warhawk extract game.psarc -o output_dir/

# List files without extracting
warhawk extract game.psarc --list-only

# Extract without converting
warhawk extract game.psarc --no-convert
```

### Extract textures from NGP files

```bash
# Extract textures with auto-detected VRAM file
warhawk ngp model.ngp -o output_dir/

# Specify VRAM file explicitly
warhawk ngp model.ngp --vram model.vram -o output_dir/

# Extract without DDS conversion
warhawk ngp model.ngp --no-dds
```

### Convert RTT textures to DDS

```bash
warhawk rtt2dds texture.rtt
warhawk rtt2dds texture.rtt -o custom_name.dds
```

### Extract 3D models from NGP files

```bash
# Extract all models with textures
warhawk models model.ngp -o output_dir/

# Specify VRAM file explicitly
warhawk models model.ngp --vram model.vram -o output_dir/

# Skip texture export
warhawk models model.ngp --no-textures

# Use UV2 (chroma) coordinates instead of UV1
warhawk models model.ngp --uv2
```

The toolkit extracts both **Type 1 (static)** and **Type 2 (rigged)** models:
- Type 1: Static meshes (buildings, props, terrain)
- Type 2: Rigged meshes (aircraft wings, animated parts)

Each model is exported as:
- `.obj` - 3D geometry with vertices, faces, normals, and UVs
- `.mtl` - Material file referencing the texture
- `.dds` - Texture in standard DDS format

### Extract localization strings

```bash
# Export to JSON (default)
warhawk loc strings.loc

# Export to CSV
warhawk loc strings.loc --format csv

# Export to plain text
warhawk loc strings.loc --format text
```

## Comparison with UnPSARC

| Tool | Base Files | Textures Extracted |
|------|------------|-------------------|
| UnPSARC | 854 | 498 (from PSARC only) |
| Warhawk Toolkit | 854 | **5,336** (from PSARC + NGP/VRAM) |

The toolkit extracts 10x more textures by processing NGP model files and extracting
embedded textures from paired VRAM containers.

## Supported Formats

| Format | Description | Output |
|--------|-------------|--------|
| PSARC | PlayStation Archive | All contained files |
| NGP | 3D model container | OBJ + RTT textures |
| VRAM | Texture container | (paired with NGP) |
| RTT | Texture format | → DDS |
| LOC | Localization | → JSON/CSV/TXT |

## Model Types

Warhawk uses two distinct model formats within NGP files:

| Type | Magic | Description | Vertex Format |
|------|-------|-------------|---------------|
| Type 1 | `0x00000001` | Static Mesh | 16-bit signed integers, normalized [0,2] |
| Type 2 | `0x00000002` | Rigged Mesh | 32-bit floats, world-space |

**Type 1 (Static Mesh)**: Used for buildings, terrain, and non-animated objects. Vertices use packed 16-bit coordinates normalized to a [0, 2] range, requiring scaling for real-world dimensions.

**Type 2 (Rigged Mesh)**: Used for animated parts like aircraft wings that can fold or take damage. Vertices use 32-bit floats in world-space coordinates (~739× larger scale than Type 1).

### UV Coordinates

Models may have two UV channels:
- **UV1**: Primary texture mapping (diffuse/albedo)
- **UV2**: Secondary mapping for chroma/skin textures

Use the `--uv2` flag to export with UV2 coordinates when working with chroma textures.

## Technical Details

- **Byte order**: Big-endian throughout (PS3 architecture)
- **NGP texture table**: Relative pointers at offset 0x10
- **RTT magic**: `0x80` at byte 0
- **NGP+VRAM pairing**: Same base filename, different extensions

### RTT Compression Types

| Value | Format | Description |
|-------|--------|-------------|
| 0x01/0x05 | RGBA | Uncompressed 32-bit |
| 0x06 | DXT1 | BC1, 4bpp, 1-bit alpha |
| 0x07 | DXT3 | BC2, 8bpp, explicit alpha |
| 0x08 | DXT5 | BC3, 8bpp, interpolated alpha |

## Python API

```python
from warhawk_toolkit.psarc import PSARCReader
from warhawk_toolkit.formats import RTTTexture, NGPFile, LOCFile
from warhawk_toolkit.converters import convert_rtt_to_dds

# Extract PSARC archive
with PSARCReader("game.psarc") as reader:
    for filename, path in reader.extract_all("output/"):
        print(f"Extracted: {filename}")

# Extract textures from NGP
ngp = NGPFile.from_file("model.ngp")  # Auto-detects model.vram
for i, rtt_path in ngp.extract_textures("output/", "model"):
    print(f"Extracted: {rtt_path}")

# Convert RTT to DDS
rtt = RTTTexture.from_file("texture.rtt")
convert_rtt_to_dds(rtt, "texture.dds")

# Parse localization
loc = LOCFile.from_file("strings.loc")
for key, value in loc.items():
    print(f"{key}: {value}")
```

## License

MIT
