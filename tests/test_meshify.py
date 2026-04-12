"""Tests for meshify module components."""

import numpy as np
import pytest
import trimesh


class TestStagedReductions:
    def test_staged_reductions_sum(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        r1, r2 = staged_reductions(0.99, 0.5, 0.5)
        # After applying both reductions, overall keep should be 1 - 0.99 = 0.01
        keep_total = (1 - r1) * (1 - r2)
        np.testing.assert_almost_equal(keep_total, 0.01, decimal=6)

    def test_staged_reductions_asymmetric(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        r1, r2 = staged_reductions(0.90, 0.25, 0.75)
        keep_total = (1 - r1) * (1 - r2)
        np.testing.assert_almost_equal(keep_total, 0.10, decimal=6)

    def test_staged_reductions_invalid_fractions(self):
        from mesh_n_bone.meshify.meshify import staged_reductions

        with pytest.raises(AssertionError):
            staged_reductions(0.99, 0.3, 0.3)


class TestRepairMeshPymeshlab:
    def test_repair_simple_mesh(self, tiny_cube_mesh):
        from mesh_n_bone.meshify.meshify import Meshify

        repaired = Meshify.repair_mesh_pymeshlab(
            tiny_cube_mesh.vertices,
            tiny_cube_mesh.faces,
            remove_smallest_components=False,
        )
        assert len(repaired.vertices) > 0
        assert len(repaired.faces) > 0

    def test_is_mesh_valid(self, tiny_sphere_mesh):
        from mesh_n_bone.meshify.meshify import Meshify

        # Sphere should be valid
        assert Meshify.is_mesh_valid(tiny_sphere_mesh)


class TestDownsample:
    def test_flat_mode(self):
        from mesh_n_bone.meshify.downsample import flat_mode

        data = np.array([[[1, 1, 2], [1, 2, 2], [1, 1, 1]]])
        result = flat_mode(data)
        assert result == 1

    def test_flat_mode_except_zero(self):
        from mesh_n_bone.meshify.downsample import flat_mode_except_zero

        data = np.array([[[0, 0, 5], [5, 0, 5], [0, 0, 0]]])
        result = flat_mode_except_zero(data)
        assert result == 5

    def test_flat_binary_mode(self):
        from mesh_n_bone.meshify.downsample import flat_binary_mode

        # Majority nonzero
        data = np.array([[[1, 1, 1], [1, 0, 1], [1, 1, 1]]])
        assert flat_binary_mode(data) == 1

        # Majority zero
        data = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])
        assert flat_binary_mode(data) == 0

    def test_downsample_box(self):
        from mesh_n_bone.meshify.downsample import downsample_box

        box = np.array([[0, 0, 0], [10, 10, 10]])
        block_shape = np.array([2, 2, 2])
        result = downsample_box(box, block_shape)
        np.testing.assert_array_equal(result[0], [0, 0, 0])
        np.testing.assert_array_equal(result[1], [5, 5, 5])


class TestDefaultBlockShape:
    """Test _default_block_shape_pixels computes sensible multiples."""

    @staticmethod
    def _make_mock_array(chunk_shape, dtype):
        """Minimal object with chunk_shape and dtype."""
        from types import SimpleNamespace
        from funlib.geometry import Coordinate
        return SimpleNamespace(
            chunk_shape=Coordinate(chunk_shape),
            dtype=np.dtype(dtype),
        )

    def test_stays_within_budget(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((112, 112, 112), "uint64")
        block = meshify._default_block_shape_pixels(target_mb=128)
        actual_mb = int(np.prod(block)) * 8 / 1e6
        assert actual_mb <= 128, f"Block {block} uses {actual_mb:.0f} MB, exceeds 128 MB"

    def test_at_least_one_chunk(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((256, 256, 256), "uint64")
        block = meshify._default_block_shape_pixels(target_mb=10)
        # Even if budget is tiny, should be at least 1x chunk
        np.testing.assert_array_equal(block, [256, 256, 256])

    def test_larger_budget_gives_larger_block(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        meshify.segmentation_array = self._make_mock_array((64, 64, 64), "uint64")
        small = meshify._default_block_shape_pixels(target_mb=10)
        large = meshify._default_block_shape_pixels(target_mb=500)
        assert np.all(large >= small)

    def test_is_chunk_aligned(self):
        from mesh_n_bone.meshify.meshify import Meshify

        meshify = object.__new__(Meshify)
        chunk = (96, 96, 96)
        meshify.segmentation_array = self._make_mock_array(chunk, "uint32")
        block = meshify._default_block_shape_pixels(target_mb=128)
        # Must be an exact multiple of chunk shape
        assert np.all(block % np.array(chunk) == 0)


class TestClipPlaneDuplicateMerge:
    """Test that duplicate vertices at fixed-edge clip planes are merged."""

    def test_merge_clip_plane_duplicates(self):
        """Simulates two adjacent block meshes with shared clip-plane vertices.

        Fixed-edge simplification clips block meshes at half-voxel inward
        from chunk boundaries, producing exact-duplicate vertices that
        ``deduplicate_chunk_boundaries`` misses (mod != 0).  The merge
        step should collapse these duplicates.
        """
        # Create two open half-spheres meeting at z=0 (simulating clip plane)
        sphere = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        sphere.vertices += 20  # offset into positive quadrant

        # Split into two blocks at z=20 (the center)
        split_z = 20.0
        block_a = trimesh.intersections.slice_mesh_plane(
            sphere, [0, 0, -1], [0, 0, split_z], cap=False
        )
        block_b = trimesh.intersections.slice_mesh_plane(
            sphere, [0, 0, 1], [0, 0, split_z], cap=False
        )

        # Concatenate (simulating assembly without dedup)
        combined = trimesh.util.concatenate([block_a, block_b])
        n_before = len(combined.vertices)

        # The duplicate vertices at the split plane should exist
        at_plane = np.abs(combined.vertices[:, 2] - split_z) < 0.01
        # Each plane vertex exists twice (once from each block)
        unique_at_plane = np.unique(
            np.round(combined.vertices[at_plane], 4), axis=0
        )
        n_dups_at_plane = at_plane.sum() - len(unique_at_plane)
        assert n_dups_at_plane > 0, "Expected duplicate vertices at split plane"

        # Merge them (same as the fix in _assemble_mesh)
        combined.merge_vertices(merge_tex=False, merge_norm=False)
        n_after = len(combined.vertices)

        assert n_after < n_before, "merge_vertices should reduce vertex count"
        # All duplicates at the split plane should now be gone
        at_plane_after = np.abs(combined.vertices[:, 2] - split_z) < 0.01
        unique_after = np.unique(
            np.round(combined.vertices[at_plane_after], 4), axis=0
        )
        assert at_plane_after.sum() == len(unique_after), (
            "All duplicate vertices at the split plane should be merged"
        )
