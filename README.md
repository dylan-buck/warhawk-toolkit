# Warhawk Toolkit

Extract 3D models and textures from Warhawk PS3 game files.

## Quick Start

```bash
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install .

warhawk full game.psarc
```

This extracts everything to `game_extracted/`: textures (DDS), 3D models (OBJ), and all other files.

## What You Get

```
game_extracted/
├── *.dds          # Textures (viewable in most image editors)
├── *.obj          # 3D models (import into Blender, Maya, etc.)
├── *.mtl          # Material files (loaded automatically with OBJ)
└── ...            # Other game files
```

## Installation

### macOS / Linux

**1. Install Python 3.9+ if you don't have it:**

```bash
# macOS (Homebrew)
brew install python

# Linux (Debian/Ubuntu)
sudo apt install python3 python3-pip python3-venv
```

**2. Clone and install:**

```bash
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

> **Note:** The virtual environment (venv) is required on modern macOS. It keeps the toolkit isolated from your system Python.

**3. Run:**

```bash
warhawk full game.psarc
```

> **Each time you open a new terminal**, activate the environment first:
> ```bash
> cd warhawk-toolkit
> source .venv/bin/activate
> ```

### Windows

**1. Install Python 3.9+:**

Download and run the installer from [python.org](https://www.python.org/downloads/).

During installation, check **"Add Python to PATH"**.

**2. Install Git:**

Download and run the installer from [git-scm.com](https://git-scm.com/download/win).

**3. Open Command Prompt and run:**

```cmd
git clone https://github.com/dylan-buck/warhawk-toolkit.git
cd warhawk-toolkit
python -m venv .venv
.venv\Scripts\activate
pip install .
```

> **What does `pip install .` mean?** The `.` means "install from the current folder". It reads `pyproject.toml` and installs the toolkit as a command you can run.

**4. Run:**

```cmd
warhawk full game.psarc
```

> **Each time you open a new Command Prompt**, activate the environment first:
> ```cmd
> cd warhawk-toolkit
> .venv\Scripts\activate
> ```

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
warhawk models model.ngp
```

### UV Coordinates for Chroma Skins

Warhawk models have two UV sets:
- **UV1**: Body/diffuse textures (default)
- **UV2**: Chroma/skin textures (team colors, decals)

If your custom chroma skin looks wrong in-game, you probably designed it using UV1. Export with UV2 instead:

```bash
warhawk models --uv2 nemesis.ngp
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
