"""Tests for NGP to OBJ model converter."""

import pytest

from warhawk_toolkit.converters.ngp_to_obj import (
    NGPModel,
    analyze_uv_differences,
    translate_uv,
)


class TestTranslateUV:
    """Tests for UV coordinate translation."""

    def test_translate_uv_midpoint(self):
        """Input 0x3800 should produce 0.5."""
        result = translate_uv(0x3800)
        assert abs(result - 0.5) < 0.001

    def test_translate_uv_zero(self):
        """Input 0 should produce 0."""
        result = translate_uv(0)
        assert result == 0.0

    def test_translate_uv_nonlinear(self):
        """Verify non-linear behavior - values below 0x3800 compress heavily."""
        # Due to power-of-10, values below midpoint compress significantly
        result = translate_uv(0x2800)
        assert result < 0.1  # Much smaller than linear would suggest


class TestAnalyzeUVDifferences:
    """Tests for UV difference analysis."""

    def test_empty_uv_sets(self):
        """Model with no UVs."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0)],
            faces=[(1, 1, 1)],
            uvs=[],
            normals=[],
            uvs2=[],
        )
        result = analyze_uv_differences(model)
        assert result["uv1_present"] is False
        assert result["uv2_present"] is False
        assert result["identical"] is False

    def test_only_uv1_present(self):
        """Model with only UV1."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0)],
            faces=[(1, 1, 1)],
            uvs=[(0.5, 0.5)],
            normals=[],
            uvs2=[],
        )
        result = analyze_uv_differences(model)
        assert result["uv1_present"] is True
        assert result["uv2_present"] is False

    def test_only_uv2_present(self):
        """Model with only UV2."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0)],
            faces=[(1, 1, 1)],
            uvs=[],
            normals=[],
            uvs2=[(0.5, 0.5)],
        )
        result = analyze_uv_differences(model)
        assert result["uv1_present"] is False
        assert result["uv2_present"] is True

    def test_identical_uv_sets(self):
        """UV1 and UV2 with identical coordinates."""
        uvs = [(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)]
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],
            faces=[(1, 2, 3)],
            uvs=uvs.copy(),
            normals=[],
            uvs2=uvs.copy(),
        )
        result = analyze_uv_differences(model)
        assert result["identical"] is True
        assert result["overlap_percentage"] == 100.0
        assert result["different_coords_count"] == 0

    def test_different_uv_sets(self):
        """UV1 and UV2 with different coordinates."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],
            faces=[(1, 2, 3)],
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
            normals=[],
            uvs2=[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)],  # Different layout
        )
        result = analyze_uv_differences(model)
        assert result["identical"] is False
        assert result["different_coords_count"] == 3
        assert result["avg_difference"] > 0
        assert result["max_difference"] > 0

    def test_partial_overlap(self):
        """UV sets with some coordinates matching."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],
            faces=[(1, 2, 3)],
            uvs=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
            normals=[],
            uvs2=[(0.0, 0.0), (1.0, 0.0), (0.6, 0.9)],  # First two match
        )
        result = analyze_uv_differences(model)
        assert result["identical"] is False
        assert result["different_coords_count"] == 1
        assert 60 < result["overlap_percentage"] < 70  # ~66.7%

    def test_mismatched_uv_counts(self):
        """UV sets with different counts."""
        model = NGPModel(
            header_offset=0,
            vertices=[(0, 0, 0), (1, 0, 0), (0.5, 1, 0)],
            faces=[(1, 2, 3)],
            uvs=[(0.0, 0.0), (1.0, 0.0)],  # 2 coords
            normals=[],
            uvs2=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],  # 3 coords
        )
        result = analyze_uv_differences(model)
        assert result["uv1_count"] == 2
        assert result["uv2_count"] == 3
        assert result["different_coords_count"] == 1  # Count difference
