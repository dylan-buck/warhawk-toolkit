# Warhawk Toolkit

Extract 3D models and textures from Warhawk PS3 game files.

## Quick Start

```bash
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
pip3 install .
warhawk full game.psarc -o output/
```

That's it. This extracts all files, textures (as DDS), and 3D models (as OBJ) in one command.

## What You Get

```
output/
├── *.dds          # Textures (viewable in most image editors)
├── *.obj          # 3D models (import into Blender, Maya, etc.)
├── *.mtl          # Material files (loaded automatically with OBJ)
└── ...            # Other game files
```

## Installation

**Requirements:** Python 3.9+ and Git

### 1. Install Python (if needed)

**macOS (Homebrew):**
```bash
brew install python
```

**macOS/Windows:** Or download from [python.org](https://www.python.org/downloads/)

**Linux (Debian/Ubuntu):**
```bash
sudo apt install python3 python3-pip python3-venv
```

### 2. Install Warhawk Toolkit

```bash
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
pip3 install .
```

**With a virtual environment (recommended):**
```bash
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install .
```

**For development:**
```bash
pip install -e ".[dev]"
```

## Commands

| Command | Purpose |
|---------|---------|
| `warhawk full` | Extract everything (recommended) |
| `warhawk extract` | Extract with options (--no-convert, --list-only) |
| `warhawk models` | Extract 3D models from a single NGP file |
| `warhawk ngp` | Extract textures from a single NGP file |
| `warhawk rtt2dds` | Convert a single RTT texture to DDS |
| `warhawk loc` | Extract localization strings |
| `warhawk uv-compare` | Diagnose UV coordinate issues |

### Preview Archive Contents

```bash
warhawk extract game.psarc --list-only
```

### Extract Without Converting

```bash
warhawk extract game.psarc --no-convert
```

---

## Advanced Usage

### Extract Models from a Single NGP File

```bash
warhawk models model.ngp -o output/
```

### UV Coordinates for Chroma Skins

Warhawk models have two UV sets:
- **UV1**: Body/diffuse textures (default)
- **UV2**: Chroma/skin textures (team colors, decals)

If your custom chroma skin looks wrong in-game, you probably designed it using UV1. Export with UV2 instead:

```bash
warhawk models --uv2 nemesis.ngp -o output/
```

### Diagnose UV Issues

```bash
warhawk uv-compare nemesis.ngp
```

This shows whether UV1 and UV2 differ and by how much.

### Extract Localization

```bash
warhawk loc strings.loc                    # JSON (default)
warhawk loc strings.loc --format csv       # CSV
warhawk loc strings.loc --format text      # Plain text
```

---

## Technical Reference

### Extraction Pipeline

```
Stage 1: PSARC Extraction
    .psarc archive → Raw files (NGP, VRAM, RTT, LOC, etc.)

Stage 2: Warhawk Container Extraction
    NGP + VRAM files → Embedded textures (RTT)

Stage 3: Format Conversion
    RTT → DDS (standard texture format)
    NGP → OBJ (3D model format)
```

### Supported Formats

| Format | Description | Output |
|--------|-------------|--------|
| PSARC | PlayStation Archive | All contained files |
| NGP | 3D model container | OBJ + RTT textures |
| VRAM | Texture container | (paired with NGP) |
| RTT | Texture format | → DDS |
| LOC | Localization | → JSON/CSV/TXT |

### Model Types

| Type | Magic | Description |
|------|-------|-------------|
| Type 1 | `0x00000001` | Static mesh (buildings, terrain, props) |
| Type 2 | `0x00000002` | Rigged mesh (aircraft wings, animated parts) |

### RTT Compression Types

| Value | Format | Description |
|-------|--------|-------------|
| 0x01/0x05 | RGBA | Uncompressed 32-bit |
| 0x06 | DXT1 | BC1, 4bpp, 1-bit alpha |
| 0x07 | DXT3 | BC2, 8bpp, explicit alpha |
| 0x08 | DXT5 | BC3, 8bpp, interpolated alpha |

### Texture Extraction Comparison

| Tool | Base Files | Textures Extracted |
|------|------------|-------------------|
| UnPSARC | 854 | 498 (from PSARC only) |
| Warhawk Toolkit | 854 | **5,336** (from PSARC + NGP/VRAM) |

---

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
