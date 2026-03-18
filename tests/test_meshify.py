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
