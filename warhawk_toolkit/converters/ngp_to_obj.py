"""NGP to OBJ 3D model converter.

Extracts 3D model geometry from NGP files and exports to OBJ format
with optional MTL materials and DDS textures.
"""

import math
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .rtt_to_dds import convert_rtt_to_dds


@dataclass
class NGPExtractionStats:
    """Statistics from NGP model extraction."""

    type1_found: int = 0
    type2_found: int = 0
    type1_extracted: int = 0
    type2_extracted: int = 0
    textures_found: int = 0
    textures_extracted: int = 0
    models_with_uvs: int = 0
    models_with_normals: int = 0
    models_with_uv2: int = 0
    extraction_errors: List[str] = field(default_factory=list)

    @property
    def total_found(self) -> int:
        return self.type1_found + self.type2_found

    @property
    def total_extracted(self) -> int:
        return self.type1_extracted + self.type2_extracted

    def to_dict(self) -> dict:
        return {
            "type1_found": self.type1_found,
            "type2_found": self.type2_found,
            "type1_extracted": self.type1_extracted,
            "type2_extracted": self.type2_extracted,
            "total_found": self.total_found,
            "total_extracted": self.total_extracted,
            "textures_found": self.textures_found,
            "textures_extracted": self.textures_extracted,
            "models_with_uvs": self.models_with_uvs,
            "models_with_normals": self.models_with_normals,
            "models_with_uv2": self.models_with_uv2,
            "extraction_errors": self.extraction_errors,
        }


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


def extract_vertices_float32(ngp_data: bytes, offset: int, max_count: int = 20000) -> List[Tuple[float, float, float]]:
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


# ============================================================================
# Flexible Type 2 vertex extraction (variable stride support)
# ============================================================================

def decode_raw_indices(ngp_data: bytes, face_offset: int, index_count: int) -> Optional[List[int]]:
    """Decode raw u16 indices without converting to triangles."""
    need = face_offset + index_count * 2
    if face_offset < 0 or need > len(ngp_data) or index_count <= 0:
        return None
    return [struct.unpack_from(">H", ngp_data, face_offset + i * 2)[0] for i in range(index_count)]


def build_tris_trilist(indices: List[int]) -> List[Tuple[int, int, int]]:
    """Build triangles from a triangle list (every 3 indices = 1 triangle)."""
    tris = []
    n = len(indices) - (len(indices) % 3)
    for i in range(0, n, 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        if a == b or b == c or a == c:
            continue
        tris.append((a, b, c))
    return tris


def build_tris_strip(indices: List[int], restart: int = 0xFFFF) -> List[Tuple[int, int, int]]:
    """Build triangles from a triangle strip with restart index support."""
    tris: List[Tuple[int, int, int]] = []
    strip: List[int] = []

    def flush():
        nonlocal strip
        if len(strip) < 3:
            strip = []
            return
        flip = False
        for i in range(len(strip) - 2):
            a, b, c = strip[i], strip[i + 1], strip[i + 2]
            if a == b or b == c or a == c:
                flip = not flip
                continue
            tris.append((a, b, c) if not flip else (b, a, c))
            flip = not flip
        strip = []

    for v in indices:
        if v == restart:
            flush()
        else:
            strip.append(v)
    flush()
    return tris


def decode_vertices_strided(
    ngp_data: bytes,
    base_offset: int,
    vertex_count: int,
    stride: int,
    pos_offset: int,
    abs_limit: float = 1000.0,
) -> Optional[List[Tuple[float, float, float]]]:
    """Extract vertices with variable stride and position offset.

    Args:
        base_offset: Start of vertex data
        vertex_count: Number of vertices to extract
        stride: Bytes per vertex (12, 16, 20, 24, etc.)
        pos_offset: Offset within each vertex where XYZ starts
        abs_limit: Maximum allowed coordinate value
    """
    need = base_offset + pos_offset + (vertex_count - 1) * stride + 12
    if base_offset < 0 or need > len(ngp_data) or stride <= 0 or vertex_count <= 0:
        return None

    vertices = []
    for i in range(vertex_count):
        pos = base_offset + i * stride + pos_offset
        x = struct.unpack_from(">f", ngp_data, pos)[0]
        y = struct.unpack_from(">f", ngp_data, pos + 4)[0]
        z = struct.unpack_from(">f", ngp_data, pos + 8)[0]

        # Validate
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            return None
        if abs(x) > abs_limit or abs(y) > abs_limit or abs(z) > abs_limit:
            return None

        vertices.append((x, y, z))

    return vertices


def edge_length_squared(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    """Calculate squared edge length between two vertices."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return dx * dx + dy * dy + dz * dz


def bbox_span(vertices: List[Tuple[float, float, float]]) -> float:
    """Calculate bounding box span (sum of dimensions)."""
    if not vertices:
        return 0.0
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    zs = [v[2] for v in vertices]
    return (max(xs) - min(xs)) + (max(ys) - min(ys)) + (max(zs) - min(zs))


def score_mesh(
    vertices: List[Tuple[float, float, float]],
    triangles: List[Tuple[int, int, int]],
    sample_limit: int = 2500,
) -> float:
    """Score mesh quality based on edge length consistency.

    Lower score = better quality mesh.
    Returns infinity for invalid meshes.
    """
    if not vertices or not triangles:
        return float("inf")

    edges: List[float] = []
    for a, b, c in triangles[:min(len(triangles), sample_limit)]:
        if a >= len(vertices) or b >= len(vertices) or c >= len(vertices):
            return float("inf")
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        edges.extend([
            edge_length_squared(va, vb),
            edge_length_squared(vb, vc),
            edge_length_squared(vc, va),
        ])

    edges = [e for e in edges if e > 0 and math.isfinite(e)]
    if len(edges) < 60:
        return float("inf")

    edges.sort()
    median = edges[len(edges) // 2]
    if median <= 1e-12:
        return float("inf")

    p95 = edges[int(0.95 * (len(edges) - 1))]
    p99 = edges[int(0.99 * (len(edges) - 1))]

    # Penalize abnormal bounding box sizes
    span = bbox_span(vertices[:min(len(vertices), 4000)])
    span_penalty = 0.0
    if span < 0.05:
        span_penalty += 10.0
    if span > 8000.0:
        span_penalty += 10.0

    return (p99 / median) * 0.75 + (p95 / median) * 0.25 + span_penalty


def autodetect_type2_layout(
    ngp_data: bytes,
    vertex_offset: int,
    vertex_count: int,
    raw_indices: List[int],
    abs_limit: float = 1000.0,
    min_triangles: int = 1,
) -> Tuple[Optional[List[Tuple[float, float, float]]], List[Tuple[int, int, int]], Dict]:
    """Auto-detect the best vertex layout for Type 2 models.

    Tries multiple stride values, position offsets, and primitive types
    to find the layout that produces the best quality mesh.

    Returns (vertices, triangles, layout_metadata).
    """
    stride_candidates = [12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 64]
    pos_candidates = [0, 4, 8, 12, 16, 20]
    prim_candidates = ["trilist", "strip"]

    best_vertices: Optional[List[Tuple[float, float, float]]] = None
    best_triangles: List[Tuple[int, int, int]] = []
    best_meta: Dict = {}
    best_score = float("inf")

    for stride in stride_candidates:
        for pos_off in pos_candidates:
            vertices = decode_vertices_strided(
                ngp_data, vertex_offset, vertex_count, stride, pos_off, abs_limit
            )
            if not vertices:
                continue

            for prim in prim_candidates:
                if prim == "trilist":
                    triangles = build_tris_trilist(raw_indices)
                else:
                    triangles = build_tris_strip(raw_indices, 0xFFFF)

                if len(triangles) < min_triangles:
                    continue

                score = score_mesh(vertices, triangles)
                if score < best_score:
                    best_score = score
                    best_vertices = vertices
                    best_triangles = triangles
                    best_meta = {
                        "stride": stride,
                        "pos_offset": pos_off,
                        "primitive": prim,
                        "score": score,
                    }

    return best_vertices, best_triangles, best_meta


def translate_uv(val: int) -> float:
    """Convert packed UV value to float in 0-1 range.

    Note: This uses a non-linear formula (power of 10) which maps:
    - Input 0x3800 → Output 0.5
    - Input 0x3400 → Output ~0.28
    - Input 0x2800 → Output ~0.02

    The non-linearity is significant and small input differences
    compound dramatically, especially toward texture edges.
    """
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


def find_type2_linker_boundary(ngp_data: bytes, header_offset: int) -> int:
    """Find the boundary of the Type 2 linker search region.

    Scans forward from the header until hitting:
    - Next model magic (0x00000001 or 0x00000002)
    - Maximum search distance (0x1000 = 4KB)
    - End of data

    Returns the offset where the search should stop.
    """
    max_search = 0x1000  # 4KB max search window
    search_start = header_offset + 0x50  # After base Type 2 header
    search_end = min(header_offset + max_search, len(ngp_data) - 4)

    for i in range(search_start, search_end, 4):
        # Check for next model header magic
        magic = ngp_data[i:i + 4]
        if magic == b'\x00\x00\x00\x01' or magic == b'\x00\x00\x00\x02':
            # Verify this looks like a real header (not just coincidental bytes)
            # Type 1: check for 0x3C000000 at +0x14
            if magic == b'\x00\x00\x00\x01' and i + 0x18 <= len(ngp_data):
                if ngp_data[i + 0x14:i + 0x18] == b'\x3C\x00\x00\x00':
                    return i
            # Type 2: check for valid face offset at +0x44
            elif magic == b'\x00\x00\x00\x02' and i + 0x48 <= len(ngp_data):
                face_offset = struct.unpack_from(">I", ngp_data, i + 0x44)[0]
                if 0 < face_offset < len(ngp_data):
                    return i

    return search_end


def extract_uvs_type2(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
    vertex_count: int,
    linker_type: int = LINKER_UV1,
) -> List[Tuple[float, float]]:
    """Extract UV coordinates from Type 2 (Rigged Mesh) models.

    Type 2 models store linkers in a different region than Type 1.
    We search for UV linker patterns (0x00080003 for UV1, 0x000A0003 for UV2)
    in the extended header area, using dynamic boundary detection.

    Args:
        linker_type: LINKER_UV1 or LINKER_UV2
        vertex_count: Number of vertices to extract UVs for
    """
    uvs = []

    # Type 2 linkers are found in the region after the base header
    # Use dynamic boundary detection instead of fixed 0x200 limit
    search_start = header_offset + 0x40
    search_end = find_type2_linker_boundary(ngp_data, header_offset)

    for i in range(search_start, search_end, 4):
        if i + 12 > len(ngp_data):
            break

        current_type = struct.unpack_from(">I", ngp_data, i)[0]
        if current_type != linker_type:
            continue

        # Found a matching linker - parse it
        linker = ngp_data[i:i + 12]
        repeat_length = linker[0x04]
        is_in_ngp = linker[0x06] == 0x01
        data_offset = struct.unpack_from(">I", linker, 0x08)[0]

        # Validate
        if repeat_length == 0 or data_offset == 0:
            continue

        # Select data source
        if is_in_ngp:
            source = ngp_data
        elif vram_data is not None:
            source = vram_data
        else:
            continue

        # Validate data offset
        if data_offset >= len(source):
            continue

        # Extract UVs
        for j in range(vertex_count):
            pos = data_offset + (j * repeat_length)
            if pos + 4 > len(source):
                break

            u = struct.unpack_from(">H", source, pos)[0]
            v = struct.unpack_from(">H", source, pos + 2)[0]

            # Skip if we hit zeros (end of data)
            if u == 0 and v == 0 and j > 0:
                break

            uvs.append((
                translate_uv(u),
                1 - translate_uv(v),  # Flip Y
            ))

        break  # Found and processed a linker

    return uvs


def extract_normals_type2(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
    vertex_count: int,
) -> List[Tuple[float, float, float]]:
    """Extract vertex normals from Type 2 (Rigged Mesh) models.

    Searches the extended header region for normal linker (0x00020004)
    and parses float32 triplets from the data source.
    """
    normals = []

    # Use dynamic boundary detection
    search_start = header_offset + 0x40
    search_end = find_type2_linker_boundary(ngp_data, header_offset)

    for i in range(search_start, search_end, 4):
        if i + 12 > len(ngp_data):
            break

        current_type = struct.unpack_from(">I", ngp_data, i)[0]
        if current_type != LINKER_NORMALS:
            continue

        # Found a matching linker - parse it
        linker = ngp_data[i:i + 12]
        repeat_length = linker[0x04]
        is_in_ngp = linker[0x06] == 0x01
        data_offset = struct.unpack_from(">I", linker, 0x08)[0]

        # Validate
        if repeat_length == 0 or data_offset == 0:
            continue

        # Select data source
        if is_in_ngp:
            source = ngp_data
        elif vram_data is not None:
            source = vram_data
        else:
            continue

        # Validate data offset
        if data_offset >= len(source):
            continue

        # Extract normals as float32 triplets
        for j in range(vertex_count):
            pos = data_offset + (j * repeat_length)
            if pos + 12 > len(source):
                break

            nx = struct.unpack_from(">f", source, pos)[0]
            ny = struct.unpack_from(">f", source, pos + 4)[0]
            nz = struct.unpack_from(">f", source, pos + 8)[0]

            # Validate normals (should be unit vectors, allow some tolerance)
            if nx != nx or ny != ny or nz != nz:  # NaN check
                break
            if abs(nx) > 2 or abs(ny) > 2 or abs(nz) > 2:
                break

            normals.append((nx, ny, nz))

        break  # Found and processed a linker

    return normals


def detect_type2_vertex_count(
    ngp_data: bytes,
    vertex_offset: int,
    max_face_index: int,
    max_count: int = 20000,
) -> int:
    """Detect the actual vertex count for Type 2 models.

    Uses multiple validation methods:
    1. Max face index (from faces already parsed)
    2. Float32 validity scanning (detect NaN/out-of-range)

    Returns the most reliable vertex count estimate.
    """
    # Method 1: Use max face index as baseline
    baseline = max_face_index

    # Method 2: Scan for valid float32 vertices
    data_len = len(ngp_data)
    scanned_count = 0

    for i in range(max_count):
        pos = vertex_offset + (i * 12)
        if pos + 12 > data_len:
            break

        x = struct.unpack_from(">f", ngp_data, pos)[0]
        y = struct.unpack_from(">f", ngp_data, pos + 4)[0]
        z = struct.unpack_from(">f", ngp_data, pos + 8)[0]

        # Check for valid float values
        if x != x or y != y or z != z:  # NaN check
            break
        if abs(x) > 100 or abs(y) > 100 or abs(z) > 100:
            break

        scanned_count += 1

    # Cross-validate: use the larger of the two if they're close,
    # otherwise prefer max face index (more reliable)
    if scanned_count >= baseline:
        return scanned_count
    elif scanned_count >= baseline * 0.9:
        # Within 10% - use face index as it's authoritative
        return baseline
    else:
        # Significant mismatch - use face index but cap at scanned
        return min(baseline, scanned_count) if scanned_count > 0 else baseline


def extract_model_type2(
    ngp_data: bytes,
    vram_data: Optional[bytes],
    header_offset: int,
) -> Optional[NGPModel]:
    """Extract a Type 2 (Rigged Mesh) model from NGP data.

    Type 2 models use a different format than Type 1:
    - Vertex pointer at 0x24 (relative pointer)
    - Vertices stored as float32 in interleaved format (variable stride)
    - Face index count at 0x2C
    - Face offset at 0x44
    - UV/normal linkers located in extended header region (+0x40 to +0x1000)

    Uses flexible stride detection to handle different vertex layouts
    (12-64 byte strides with position at various offsets).
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

    # Decode raw indices first (needed for both trilist and strip detection)
    raw_indices = decode_raw_indices(ngp_data, faces_offset, face_index_count)
    if not raw_indices:
        return None

    # Estimate vertex count from max index in raw data
    max_index = max(idx for idx in raw_indices if idx != 0xFFFF)  # Exclude restart marker
    vertex_count = max_index + 1

    # Use flexible layout detection to find best stride/primitive combination
    vertices, triangles, layout_meta = autodetect_type2_layout(
        ngp_data, vertex_offset, vertex_count, raw_indices,
        abs_limit=1000.0, min_triangles=1
    )

    # If autodetect failed, fall back to simple fixed-stride approach
    if not vertices or not triangles:
        # Try the old fixed 12-byte stride method as fallback
        vertices = extract_vertices_float32(ngp_data, vertex_offset, max_count=vertex_count + 100)
        if vertices:
            # Use trilist as fallback
            triangles = build_tris_trilist(raw_indices)

    if not vertices or not triangles:
        return None

    # Convert to 1-indexed faces for OBJ format
    valid_faces = []
    vertex_count = len(vertices)
    for a, b, c in triangles:
        # Triangles from autodetect are 0-indexed, convert to 1-indexed
        if a < vertex_count and b < vertex_count and c < vertex_count:
            valid_faces.append((a + 1, b + 1, c + 1))

    if not valid_faces:
        return None

    # Extract UVs from Type 2 linker structure
    uvs = extract_uvs_type2(ngp_data, vram_data, header_offset, vertex_count, LINKER_UV1)

    # Also try to extract UV2 if present
    uvs2 = extract_uvs_type2(ngp_data, vram_data, header_offset, vertex_count, LINKER_UV2)

    # Extract normals using the Type 2 normals function
    normals = extract_normals_type2(ngp_data, vram_data, header_offset, vertex_count)

    # Find associated texture
    texture_offset = find_texture_header(ngp_data, header_offset)

    return NGPModel(
        header_offset=header_offset,
        vertices=vertices,
        faces=valid_faces,
        uvs=uvs,
        normals=normals,
        uvs2=uvs2,
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


def analyze_uv_differences(model: NGPModel) -> dict:
    """Analyze differences between UV1 and UV2 coordinate sets.

    Returns a dictionary with analysis results useful for diagnosing
    chroma skin misalignment issues.
    """
    result = {
        "uv1_count": len(model.uvs),
        "uv2_count": len(model.uvs2),
        "uv1_present": bool(model.uvs),
        "uv2_present": bool(model.uvs2),
        "identical": False,
        "avg_difference": 0.0,
        "max_difference": 0.0,
        "different_coords_count": 0,
        "overlap_percentage": 0.0,
    }

    if not model.uvs or not model.uvs2:
        return result

    if len(model.uvs) != len(model.uvs2):
        result["different_coords_count"] = abs(len(model.uvs) - len(model.uvs2))
        return result

    # Compare UV coordinates
    total_diff = 0.0
    max_diff = 0.0
    different_count = 0
    threshold = 0.001  # Coordinates within this are considered "same"

    for (u1, v1), (u2, v2) in zip(model.uvs, model.uvs2):
        diff = ((u1 - u2) ** 2 + (v1 - v2) ** 2) ** 0.5
        total_diff += diff
        max_diff = max(max_diff, diff)
        if diff > threshold:
            different_count += 1

    count = len(model.uvs)
    result["avg_difference"] = total_diff / count if count > 0 else 0.0
    result["max_difference"] = max_diff
    result["different_coords_count"] = different_count
    result["overlap_percentage"] = ((count - different_count) / count * 100) if count > 0 else 0.0
    result["identical"] = different_count == 0

    return result


def count_models_in_ngp(ngp_data: bytes) -> Tuple[int, int]:
    """Count Type 1 and Type 2 models in NGP data without full extraction.

    Returns (type1_count, type2_count).
    """
    type1_count = 0
    type2_count = 0

    for _, _, model_type in find_models(ngp_data):
        if model_type == 1:
            type1_count += 1
        else:
            type2_count += 1

    return type1_count, type2_count


def get_ngp_extraction_stats(
    ngp_path: Path,
    vram_path: Optional[Path] = None,
) -> NGPExtractionStats:
    """Get extraction statistics for an NGP file without writing output files.

    Analyzes the NGP file and returns statistics about what can be extracted.
    """
    stats = NGPExtractionStats()

    ngp_path = Path(ngp_path)
    ngp_data = ngp_path.read_bytes()

    vram_data = None
    if vram_path and Path(vram_path).exists():
        vram_data = Path(vram_path).read_bytes()
    else:
        auto_vram = ngp_path.with_suffix(".vram")
        if auto_vram.exists():
            vram_data = auto_vram.read_bytes()

    # Count textures from NGP format parser
    try:
        from ..formats.ngp import NGPFile
        ngp_file = NGPFile.from_bytes(ngp_data, vram_data)
        stats.textures_found = ngp_file.texture_count
        # Count extractable textures
        for i in range(ngp_file.texture_count):
            if ngp_file.get_texture_data(i) is not None:
                stats.textures_extracted += 1
    except Exception as e:
        stats.extraction_errors.append(f"Texture parsing error: {e}")

    # Find and analyze all models
    for header_offset, header_length, model_type in find_models(ngp_data):
        if model_type == 1:
            stats.type1_found += 1
        else:
            stats.type2_found += 1

        try:
            model = extract_model(ngp_data, vram_data, header_offset, model_type)
            if model is not None and model.vertices and model.faces:
                if model_type == 1:
                    stats.type1_extracted += 1
                else:
                    stats.type2_extracted += 1

                if model.uvs:
                    stats.models_with_uvs += 1
                if model.normals:
                    stats.models_with_normals += 1
                if model.uvs2:
                    stats.models_with_uv2 += 1
        except Exception as e:
            stats.extraction_errors.append(f"Model 0x{header_offset:x}: {e}")

    return stats


def extract_models_from_ngp(
    ngp_path: Path,
    output_dir: Optional[Path] = None,
    vram_path: Optional[Path] = None,
    export_textures: bool = True,
    use_uv2: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Iterator[Tuple[Path, Optional[Path], Optional[Path]]]:
    """Extract all models from an NGP file.

    Args:
        use_uv2: If True and UV2 is available, use UV2 (chroma) instead of UV1
        progress_callback: Optional callback(current, total) for progress reporting

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

    # Pre-count models for progress reporting
    model_headers = list(find_models(ngp_data))
    total_models = len(model_headers)

    # Find and extract all models (Type 1 and Type 2)
    for idx, (header_offset, header_length, model_type) in enumerate(model_headers):
        # Report progress
        if progress_callback is not None:
            progress_callback(idx + 1, total_models)

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
