"""NGP to OBJ 3D model converter.

Extracts 3D model geometry from NGP files and exports to OBJ format
with optional MTL materials and DDS textures.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .rtt_to_dds import convert_rtt_to_dds


@dataclass
class NGPModel:
    """Extracted model data."""

    header_offset: int
    vertices: List[Tuple[float, float, float]]
    faces: List[Tuple[int, int, int]]
    uvs: List[Tuple[float, float]]
    normals: List[Tuple[float, float, float]]
    uvs2: List[Tuple[float, float]]  # UV2 for chroma/skin textures
    texture_header_offset: int = -1
    model_type: int = 1  # 1 = Static Mesh, 2 = Rigged Mesh


# Linker type constants
LINKER_NORMALS = 0x00020004
LINKER_UV1 = 0x00080003
LINKER_UV_BUMP = 0x00090003
LINKER_UV2 = 0x000A0003


def dereference_relative_pointer(data: bytes, pointer_offset: int) -> int:
    """Dereference a relative pointer."""
    relative_offset = struct.unpack_from(">i", data, pointer_offset)[0]
    return pointer_offset + relative_offset


def find_models(ngp_data: bytes) -> Iterator[Tuple[int, int, int]]:
    """Find all model headers in NGP data.

    Yields (offset, header_length, model_type) for each model found.
    model_type: 1 = Static Mesh, 2 = Rigged Mesh
    """
    i = 0
    data_len = len(ngp_data)
    # Only need 4 bytes to read magic, bounds check each type separately
    while i < data_len - 4:
        magic = ngp_data[i:i+4]

        # Type 1: Static Mesh - magic 0x00000001, secondary check at 0x14
        if magic == b'\x00\x00\x00\x01':
            # Bounds check for Type 1 header fields
            if i + 0x2C <= data_len and ngp_data[i+0x14:i+0x18] == b'\x3C\x00\x00\x00':
                num_linkers = ngp_data[i + 0x26]
                header_length = 0x2C + (num_linkers * 0x0C)
                # Verify full header fits in data
                if i + header_length <= data_len:
                    yield i, header_length, 1
                    i += header_length
                    continue

        # Type 2: Rigged Mesh - magic 0x00000002
        # These have different structure, vertex pointer at 0x24 is relative
        if magic == b'\x00\x00\x00\x02':
            # Validate this looks like a Type 2 header by checking reasonable values
            # Type 2 headers have face offset at 0x44 which should be non-zero
            if i + 0x48 <= data_len:
                face_offset = struct.unpack_from(">I", ngp_data, i + 0x44)[0]
                if 0 < face_offset < data_len:
                    # Estimate header length - Type 2 has fixed 0x50 base + linkers
                    # For now, use a reasonable fixed size since linker count location differs
                    header_length = 0x50
                    if i + header_length <= data_len:
                        yield i, header_length, 2
                        i += header_length
                        continue

        i += 4


def extract_vertices(ngp_data: bytes, offset: int, count: int) -> List[Tuple[float, float, float]]:
    """Extract vertex positions from NGP data.

    Vertices are stored as 3 signed 16-bit values, scaled to [0, 2] range.
    Used for Type 1 (Static Mesh) models.
    """
    vertices = []
    for i in range(count):
        pos = offset + (i * 6)
        x = struct.unpack_from(">h", ngp_data, pos)[0]
        y = struct.unpack_from(">h", ngp_data, pos + 2)[0]
        z = struct.unpack_from(">h", ngp_data, pos + 4)[0]

        # Scale from signed 16-bit to [0, 2] range
        vertices.append((
            (x + 0x8000) / 0x8000,
            (y + 0x8000) / 0x8000,
            (z + 0x8000) / 0x8000,
        ))

    return vertices


def extract_vertices_float32(ngp_data: bytes, offset: int, max_count: int = 10000) -> List[Tuple[float, float, float]]:
    """Extract vertex positions stored as float32 triplets.

    Used for Type 2 (Rigged Mesh) models which store vertices as 3 big-endian
    float32 values per vertex (12 bytes total).

    Vertices are read until an invalid value is encountered (NaN or out of range).
    """
    vertices = []
    data_len = len(ngp_data)

    for i in range(max_count):
        pos = offset + (i * 12)
        if pos + 12 > data_len:
            break

        x = struct.unpack_from(">f", ngp_data, pos)[0]
        y = struct.unpack_from(">f", ngp_data, pos + 4)[0]
        z = struct.unpack_from(">f", ngp_data, pos + 8)[0]

        # Check for valid float values (not NaN, within reasonable range)
        if x != x or y != y or z != z:  # NaN check
            break
        if abs(x) > 100 or abs(y) > 100 or abs(z) > 100:
            break

        vertices.append((x, y, z))

    return vertices


def extract_faces(ngp_data: bytes, offset: int, index_count: int) -> List[Tuple[int, int, int]]:
    """Extract face indices from NGP data.

    Faces are stored as 3 unsigned 16-bit indices per triangle.
    Returns 1-indexed faces for OBJ format.
    """
    faces = []
    face_count = index_count // 3
    data_len = len(ngp_data)

    for i in range(face_count):
        pos = offset + (i * 6)
        # Bounds check to prevent buffer overrun
        if pos + 6 > data_len:
            break
        v1 = struct.unpack_from(">H", ngp_data, pos)[0] + 1  # 1-indexed
        v2 = struct.unpack_from(">H", ngp_data, pos + 2)[0] + 1
        v3 = struct.unpack_from(">H", ngp_data, pos + 4)[0] + 1
        faces.append((v1, v2, v3))

    return faces


def translate_uv(val: int) -> float:
    """Convert packed UV value to float in 0-1 range."""
    return ((val / 0x3800) ** 10) / 2


def extract_uvs_by_linker(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header: bytes,
    vertex_count: int,
    linker_type: int,
) -> List[Tuple[float, float]]:
    """Extract UV coordinates for a specific linker type.

    Args:
        linker_type: LINKER_UV1 (0x00080003) or LINKER_UV2 (0x000A0003)
    """
    uvs = []
    linker_start = 0x38

    for i in range(linker_start, len(header), 0x0C):
        if i + 0x0C > len(header):
            break

        linker = header[i:i+0x0C]
        current_type = struct.unpack_from(">I", linker, 0)[0]

        if current_type != linker_type:
            continue

        repeat_length = linker[0x04]
        is_in_ngp = linker[0x06] == 0x01
        offset = struct.unpack_from(">I", linker, 0x08)[0]

        # Select data source
        if is_in_ngp:
            data = ngp_data
        elif vram_data is not None:
            data = vram_data
        else:
            break

        # Extract UVs
        for j in range(vertex_count):
            pos = offset + (j * repeat_length)
            if pos + 4 > len(data):
                break

            u = struct.unpack_from(">H", data, pos)[0]
            v = struct.unpack_from(">H", data, pos + 2)[0]

            uvs.append((
                translate_uv(u),
                1 - translate_uv(v),  # Flip Y
            ))

        break  # Found the linker, stop searching

    return uvs


def extract_uvs(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header: bytes,
    vertex_count: int
) -> List[Tuple[float, float]]:
    """Extract UV1 coordinates from NGP/VRAM data."""
    return extract_uvs_by_linker(ngp_data, vram_data, header, vertex_count, LINKER_UV1)


def extract_uvs2(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header: bytes,
    vertex_count: int
) -> List[Tuple[float, float]]:
    """Extract UV2 (chroma) coordinates from NGP/VRAM data."""
    return extract_uvs_by_linker(ngp_data, vram_data, header, vertex_count, LINKER_UV2)


def extract_normals(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header: bytes,
    vertex_count: int
) -> List[Tuple[float, float, float]]:
    """Extract vertex normals from VRAM via linker 0x00020004.

    Normals are stored as 3x float32 per vertex.
    """
    normals = []
    linker_start = 0x38

    for i in range(linker_start, len(header), 0x0C):
        if i + 0x0C > len(header):
            break

        linker = header[i:i+0x0C]
        linker_type = struct.unpack_from(">I", linker, 0)[0]

        if linker_type != LINKER_NORMALS:
            continue

        repeat_length = linker[0x04]
        is_in_ngp = linker[0x06] == 0x01
        offset = struct.unpack_from(">I", linker, 0x08)[0]

        # Select data source
        if is_in_ngp:
            data = ngp_data
        elif vram_data is not None:
            data = vram_data
        else:
            break

        # Extract normals - they are stored as 3x float32
        for j in range(vertex_count):
            pos = offset + (j * repeat_length)
            if pos + 12 > len(data):
                break

            # Read 3 big-endian floats
            nx = struct.unpack_from(">f", data, pos)[0]
            ny = struct.unpack_from(">f", data, pos + 4)[0]
            nz = struct.unpack_from(">f", data, pos + 8)[0]

            normals.append((nx, ny, nz))

        break  # Found the linker, stop searching

    return normals


def find_texture_header(ngp_data: bytes, model_offset: int) -> int:
    """Find the texture header offset for a model.

    Returns -1 if no texture found.
    """
    try:
        ptr = dereference_relative_pointer(ngp_data, model_offset + 0x04)
        data_ptr = dereference_relative_pointer(ngp_data, ptr + 0x10)

        # Search for texture reference magic: 0x00111122
        for i in range(0, ptr - data_ptr, 4):
            pos = data_ptr + i
            if pos + 8 > len(ngp_data):
                break

            if ngp_data[pos:pos+4] == b'\x00\x11\x11\x22':
                # Check that next field is non-zero
                if ngp_data[pos+4:pos+8] != b'\x00\x00\x00\x00':
                    return dereference_relative_pointer(ngp_data, pos + 4)
    except (struct.error, IndexError):
        pass

    return -1


def extract_model(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
    model_type: int = 1,
) -> Optional[NGPModel]:
    """Extract a single model from NGP data.

    Args:
        model_type: 1 = Static Mesh, 2 = Rigged Mesh
    """
    if model_type == 1:
        return extract_model_type1(ngp_data, vram_data, header_offset)
    elif model_type == 2:
        return extract_model_type2(ngp_data, vram_data, header_offset)
    return None


def extract_model_type1(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
) -> Optional[NGPModel]:
    """Extract a Type 1 (Static Mesh) model from NGP data."""

    # Get header size from linker count
    num_linkers = ngp_data[header_offset + 0x26]
    header_size = 0x2C + (num_linkers * 0x0C)

    if header_offset + header_size > len(ngp_data):
        return None

    header = ngp_data[header_offset:header_offset + header_size]

    # Parse header fields for Type 1
    vertex_count = struct.unpack_from(">H", header, 0x24)[0]
    face_index_count = struct.unpack_from(">I", header, 0x1C)[0]
    faces_offset = struct.unpack_from(">I", header, 0x28)[0]
    vertex_offset = struct.unpack_from(">I", header, 0x34)[0]

    # Extract geometry
    vertices = extract_vertices(ngp_data, vertex_offset, vertex_count)
    faces = extract_faces(ngp_data, faces_offset, face_index_count)
    uvs = extract_uvs(ngp_data, vram_data, header, vertex_count)
    normals = extract_normals(ngp_data, vram_data, header, vertex_count)
    uvs2 = extract_uvs2(ngp_data, vram_data, header, vertex_count)

    # Find associated texture
    texture_offset = find_texture_header(ngp_data, header_offset)

    return NGPModel(
        header_offset=header_offset,
        vertices=vertices,
        faces=faces,
        uvs=uvs,
        normals=normals,
        uvs2=uvs2,
        texture_header_offset=texture_offset,
        model_type=1,
    )


def extract_model_type2(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
) -> Optional[NGPModel]:
    """Extract a Type 2 (Rigged Mesh) model from NGP data.

    Type 2 models use a different format than Type 1:
    - Vertex pointer at 0x24 (relative pointer)
    - Vertices stored as float32 triplets (12 bytes each) instead of int16
    - Face index count at 0x2C
    - Face offset at 0x44
    """
    # Type 2 has a minimum header size
    if header_offset + 0x50 > len(ngp_data):
        return None

    # Dereference the relative vertex pointer at 0x24
    vertex_offset = dereference_relative_pointer(ngp_data, header_offset + 0x24)
    faces_offset = struct.unpack_from(">I", ngp_data, header_offset + 0x44)[0]

    # Face index count is at +0x2C for Type 2 models
    face_index_count = struct.unpack_from(">I", ngp_data, header_offset + 0x2C)[0]

    # Validate offsets
    if vertex_offset < 0 or vertex_offset >= len(ngp_data):
        return None
    if faces_offset < 0 or faces_offset >= len(ngp_data):
        return None

    # Type 2 models use float32 vertices - extract by scanning for valid floats
    vertices = extract_vertices_float32(ngp_data, vertex_offset)

    if not vertices:
        return None

    # Extract faces
    faces = extract_faces(ngp_data, faces_offset, face_index_count)

    # Filter out faces that reference out-of-bounds vertices
    vertex_count = len(vertices)
    valid_faces = []
    for f in faces:
        if f[0] <= vertex_count and f[1] <= vertex_count and f[2] <= vertex_count:
            valid_faces.append(f)

    if not valid_faces:
        return None

    # Type 2 may have different linker structure - for now return without UVs/normals
    # The linkers may be at a different offset in Type 2 headers

    # Find associated texture
    texture_offset = find_texture_header(ngp_data, header_offset)

    return NGPModel(
        header_offset=header_offset,
        vertices=vertices,
        faces=valid_faces,
        uvs=[],
        normals=[],
        uvs2=[],
        texture_header_offset=texture_offset,
        model_type=2,
    )


def write_obj(
    model: NGPModel,
    output_path: Path,
    mtl_name: Optional[str] = None,
    use_uv2: bool = False,
) -> None:
    """Write model to OBJ file.

    Args:
        use_uv2: If True and UV2 is available, use UV2 instead of UV1
    """
    lines = []

    # Header comment
    model_type_str = "Static Mesh" if model.model_type == 1 else "Rigged Mesh"
    lines.append(f"# Warhawk {model_type_str} model")
    lines.append(f"# Vertices: {len(model.vertices)}, Faces: {len(model.faces)}")
    if model.normals:
        lines.append(f"# Normals: {len(model.normals)}")
    if model.uvs2:
        lines.append(f"# UV2 (Chroma) available: {len(model.uvs2)} coords")

    # Reference MTL if provided
    if mtl_name:
        lines.append(f"mtllib {mtl_name}")
        lines.append("usemtl Textured")

    lines.append(f"o model_{model.header_offset:x}")

    # Vertices
    for v in model.vertices:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    # Normals
    for n in model.normals:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")

    # UVs - use UV2 if requested and available, otherwise UV1
    uvs_to_use = model.uvs2 if (use_uv2 and model.uvs2) else model.uvs
    for uv in uvs_to_use:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")

    # Faces with proper format based on available data
    has_uvs = bool(uvs_to_use)
    has_normals = bool(model.normals)

    for f in model.faces:
        if has_uvs and has_normals:
            # f v/vt/vn format
            lines.append(f"f {f[0]}/{f[0]}/{f[0]} {f[1]}/{f[1]}/{f[1]} {f[2]}/{f[2]}/{f[2]}")
        elif has_uvs:
            # f v/vt format
            lines.append(f"f {f[0]}/{f[0]} {f[1]}/{f[1]} {f[2]}/{f[2]}")
        elif has_normals:
            # f v//vn format
            lines.append(f"f {f[0]}//{f[0]} {f[1]}//{f[1]} {f[2]}//{f[2]}")
        else:
            # f v format
            lines.append(f"f {f[0]} {f[1]} {f[2]}")

    output_path.write_text("\n".join(lines))


def write_mtl(output_path: Path, texture_filename: str) -> None:
    """Write MTL material file."""
    content = f"""newmtl Textured
Kd 1.0 1.0 1.0
map_Kd {texture_filename}
"""
    output_path.write_text(content)


def build_rtt_from_header(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
) -> Optional[bytes]:
    """Build RTT data from texture header in NGP."""
    if header_offset < 0 or header_offset + 0x10 > len(ngp_data):
        return None

    header = ngp_data[header_offset:header_offset + 0x10]

    compression = header[0]
    width = struct.unpack_from(">H", header, 0x04)[0]
    height = struct.unpack_from(">H", header, 0x06)[0]
    is_in_ngp = header[0x08] == 0x01
    num_mipmaps = header[0x0A]
    data_offset = struct.unpack_from(">I", header, 0x0C)[0]

    # Calculate texture data size
    if compression == 0x06:  # DXT1
        bytes_per_block = 8
    elif compression in (0x07, 0x08):  # DXT3/DXT5
        bytes_per_block = 16
    else:
        bytes_per_block = 4  # RGBA

    total_size = 0
    w, h = width, height
    for _ in range(max(1, num_mipmaps)):
        if compression in (0x06, 0x07, 0x08):
            blocks_w = max(1, (w + 3) // 4)
            blocks_h = max(1, (h + 3) // 4)
            total_size += blocks_w * blocks_h * bytes_per_block
        else:
            total_size += w * h * bytes_per_block
        w = max(1, w // 2)
        h = max(1, h // 2)

    # Get texture data
    if is_in_ngp:
        source = ngp_data
    elif vram_data:
        source = vram_data
    else:
        return None

    if data_offset + total_size > len(source):
        return None

    texture_data = source[data_offset:data_offset + total_size]

    # Build RTT header
    rtt_size = 0x80 + len(texture_data)
    rtt_header = bytearray(0x80)
    rtt_header[0] = 0x80

    size_minus_4 = rtt_size - 4
    rtt_header[1] = (size_minus_4 >> 16) & 0xFF
    rtt_header[2] = (size_minus_4 >> 8) & 0xFF
    rtt_header[3] = size_minus_4 & 0xFF

    # Copy texture header (first 12 bytes, clear in_ngp flag)
    header_copy = bytearray(header[:0x0C])
    header_copy[0x08] = 0x00
    rtt_header[4:16] = header_copy

    return bytes(rtt_header) + texture_data


def extract_models_from_ngp(
    ngp_path: Path,
    output_dir: Optional[Path] = None,
    vram_path: Optional[Path] = None,
    export_textures: bool = True,
    use_uv2: bool = False,
) -> Iterator[Tuple[Path, Optional[Path], Optional[Path]]]:
    """Extract all models from an NGP file.

    Args:
        use_uv2: If True and UV2 is available, use UV2 (chroma) instead of UV1

    Yields (obj_path, mtl_path, dds_path) for each model.
    """
    ngp_path = Path(ngp_path)

    if output_dir is None:
        output_dir = ngp_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    ngp_data = ngp_path.read_bytes()

    vram_data = None
    if vram_path and Path(vram_path).exists():
        vram_data = Path(vram_path).read_bytes()
    else:
        auto_vram = ngp_path.with_suffix(".vram")
        if auto_vram.exists():
            vram_data = auto_vram.read_bytes()

    base_name = ngp_path.stem

    # Find and extract all models (Type 1 and Type 2)
    for header_offset, header_length, model_type in find_models(ngp_data):
        model = extract_model(ngp_data, vram_data, header_offset, model_type)
        if model is None or not model.vertices or not model.faces:
            continue

        # Include model type in name for clarity
        type_suffix = "static" if model_type == 1 else "rigged"
        model_name = f"{base_name}_0x{header_offset:x}_{type_suffix}"
        obj_path = output_dir / f"{model_name}.obj"
        mtl_path = None
        dds_path = None

        # Export texture if available
        if export_textures and model.texture_header_offset >= 0:
            rtt_data = build_rtt_from_header(
                ngp_data, vram_data, model.texture_header_offset
            )
            if rtt_data:
                dds_path = output_dir / f"{model_name}.dds"
                try:
                    convert_rtt_to_dds(rtt_data, dds_path)
                    mtl_path = output_dir / f"{model_name}.mtl"
                    write_mtl(mtl_path, dds_path.name)
                except Exception:
                    dds_path = None
                    mtl_path = None

        # Write OBJ with normals and optionally UV2
        write_obj(model, obj_path, mtl_path.name if mtl_path else None, use_uv2)

        yield obj_path, mtl_path, dds_path
