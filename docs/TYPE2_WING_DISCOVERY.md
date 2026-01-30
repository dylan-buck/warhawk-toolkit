# Type 2 Rigged Model Wing Discovery

This document describes the discovery and resolution of missing wing geometry in Warhawk's rigged 3D models.

## The Mystery

When extracting 3D models from Warhawk NGP files, the main Warhawk aircraft body was missing its wings. The fuselage, cockpit, and tail surfaces extracted correctly, but the distinctive swept wings were absent.

Interestingly, the Nemesis aircraft (an enemy variant) extracted with complete geometry including wings. This discrepancy provided the key clue for the investigation.

## Investigation Process

### Step 1: Comparing Vertex Counts

Analysis of the extracted models revealed:
- **Warhawk**: 10,778 vertices (missing wings)
- **Nemesis**: 14,969 vertices (complete with wings)

The Nemesis model had roughly 4,000 more vertices, suggesting wing geometry was being extracted for Nemesis but not for Warhawk.

### Step 2: Understanding the Difference

The key insight was that Nemesis has wings integrated into its main body mesh as a single Type 1 (static) model, while Warhawk has modular construction:
- Main body: Type 1 static mesh
- Wings: Type 2 rigged mesh (separate, animatable)

Warhawk's wings are stored as Type 2 models to allow for animations (folding for carrier storage, damage states, etc.).

### Step 3: Analyzing Type 2 Extraction

Examining the Type 2 model data showed:
- Very few vertices being extracted (tens instead of thousands)
- X coordinates had impossibly wide values (spanning 70+ units)
- The extracted geometry was clearly wrong

### Step 4: Identifying the Format Difference

Binary analysis revealed the root cause:

| Attribute | Type 1 (Static) | Type 2 (Rigged) |
|-----------|-----------------|-----------------|
| Vertex format | signed int16 | float32 |
| Bytes per vertex | 6 (2 bytes × 3) | 12 (4 bytes × 3) |
| Coordinate space | Normalized [0, 2] | World-space units |
| Scale factor | ~739x smaller | Actual game units |

The original extraction code was reading Type 2 vertices as 16-bit integers, but they're actually stored as 32-bit floats.

## Root Cause

Type 2 vertex extraction was using the wrong data format:
- **Expected**: 3× signed int16 values (6 bytes total)
- **Actual**: 3× float32 values (12 bytes total)

This caused:
1. Incorrect vertex positions (interpreting float bits as integers)
2. Wrong vertex count (reading 12-byte stride as 6-byte)
3. Missing most of the wing geometry

## The Fix

### New Function: `extract_vertices_float32()`

```python
def extract_vertices_float32(ngp_data: bytes, offset: int, max_count: int = 10000) -> List[Tuple[float, float, float]]:
    """Extract vertex positions stored as float32 triplets.

    Used for Type 2 (Rigged Mesh) models which store vertices as 3 big-endian
    float32 values per vertex (12 bytes total).
    """
    vertices = []
    data_len = len(ngp_data)

    for i in range(max_count):
        pos = offset + (i * 12)  # 12 bytes per vertex
        if pos + 12 > data_len:
            break

        x = struct.unpack_from(">f", ngp_data, pos)[0]
        y = struct.unpack_from(">f", ngp_data, pos + 4)[0]
        z = struct.unpack_from(">f", ngp_data, pos + 8)[0]

        # Validate float values
        if x != x or y != y or z != z:  # NaN check
            break
        if abs(x) > 100 or abs(y) > 100 or abs(z) > 100:
            break

        vertices.append((x, y, z))

    return vertices
```

### Updated Type 2 Extraction

The `extract_model_type2()` function was updated to:
1. Use `extract_vertices_float32()` instead of `extract_vertices()`
2. Read face count from offset +0x2C (not +0x40 as originally assumed)
3. Properly dereference the relative vertex pointer at +0x24

## Results

After applying the fix:

### Wing Geometry Extraction
- **Vertex data offset**: 0x006C70
- **Vertices extracted**: 2,757
- **Faces extracted**: 600 triangles
- **Wingspan**: ~46 game units (tip-to-tip)

### Coordinate Comparison
| Measurement | Type 1 Body | Type 2 Wings |
|-------------|-------------|--------------|
| X range | 0.0 to 2.0 | -23.1 to +23.1 |
| Scale | Normalized | World-space |
| Scale ratio | 1× | ~739× larger |

The 739× scale difference explains why Type 2 vertices looked "impossibly wide" when interpreted with the wrong format.

## Technical Details

### Type 1 Header Structure (Static Mesh)
```
+0x00: Magic (0x00000001)
+0x14: Secondary magic (0x3C000000)
+0x1C: Face index count (uint32)
+0x24: Vertex count (uint16)
+0x26: Linker count (uint8)
+0x28: Face offset (uint32)
+0x34: Vertex offset (uint32)
+0x38+: Linker table (0x0C bytes each)
```

### Type 2 Header Structure (Rigged Mesh)
```
+0x00: Magic (0x00000002)
+0x24: Vertex pointer (relative, int32)
+0x2C: Face index count (uint32)
+0x44: Face offset (uint32)
```

### Vertex Format Comparison

**Type 1 (6 bytes per vertex)**:
```
[int16 X][int16 Y][int16 Z]
Range: -32768 to +32767, scaled to [0, 2]
```

**Type 2 (12 bytes per vertex)**:
```
[float32 X][float32 Y][float32 Z]
Range: Actual world-space coordinates
```

## Lessons Learned

1. **Format assumptions are dangerous**: Just because one model type uses int16 vertices doesn't mean all types do.
2. **Comparison debugging works**: Finding a working case (Nemesis) to compare against a broken case (Warhawk) quickly isolated the problem.
3. **Scale mismatches are a red flag**: When coordinates seem impossibly large or small, the data format is likely wrong.
4. **Rigged models often differ**: Games frequently use different formats for static vs. animatable geometry due to different processing requirements.

## Files Modified

- `warhawk_toolkit/converters/ngp_to_obj.py`:
  - Added `extract_vertices_float32()` function
  - Updated `extract_model_type2()` to use float32 format
  - Fixed face count offset for Type 2 models
