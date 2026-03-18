"""Integration tests for watertightness after simplification and repair."""

import numpy as np
import os
import pytest
import trimesh

from mesh_n_bone.meshify.meshify import Meshify


class TestWatertightnessAfterRepair:
    """Test that meshes remain watertight after repair operations."""

    def test_sphere_stays_watertight_after_repair(self):
        """A watertight sphere should remain watertight after repair."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=20.0)
        assert mesh.is_watertight

        repaired = Meshify.repair_mesh_pymeshlab(
            mesh.vertices, mesh.faces, remove_smallest_components=True
        )

        assert repaired.is_watertight
        assert repaired.is_winding_consistent
        assert repaired.volume > 0

    def test_cube_stays_watertight_after_repair(self):
        """A watertight cube should remain watertight after repair."""
        mesh = trimesh.creation.box(extents=[20, 20, 20])
        assert mesh.is_watertight

        repaired = Meshify.repair_mesh_pymeshlab(
            mesh.vertices, mesh.faces, remove_smallest_components=False
        )

        assert repaired.is_watertight
        assert repaired.volume > 0

    def test_repair_fixes_non_manifold_mesh(self):
        """Repair should fix a mesh with duplicate faces."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        # Add duplicate faces to create non-manifold geometry
        bad_faces = np.vstack([mesh.faces, mesh.faces[:5]])
        bad_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=bad_faces)

        repaired = Meshify.repair_mesh_pymeshlab(
            bad_mesh.vertices, bad_mesh.faces, remove_smallest_components=True
        )

        assert len(repaired.faces) > 0
        assert repaired.is_winding_consistent


class TestWatertightnessAfterSimplification:
    """Test watertightness after mesh simplification and smoothing."""

    def test_simplification_preserves_watertightness(self, watertight_sphere_mesh):
        """Simplification + repair should produce a watertight mesh."""
        mesh = watertight_sphere_mesh
        assert Meshify.is_mesh_valid(mesh)

        result = Meshify.simplify_and_smooth_mesh(
            mesh,
            target_reduction=0.9,
            n_smoothing_iter=5,
            remove_smallest_components=True,
            aggressiveness=0.3,
            do_simplification=True,
            check_mesh_validity=True,
        )

        assert Meshify.is_mesh_valid(result)
        assert len(result.faces) < len(mesh.faces)
        assert result.volume > 0

    def test_high_reduction_still_watertight(self, watertight_sphere_mesh):
        """Even with 99% reduction, mesh should remain valid if possible."""
        mesh = watertight_sphere_mesh
        assert Meshify.is_mesh_valid(mesh)

        result = Meshify.simplify_and_smooth_mesh(
            mesh,
            target_reduction=0.99,
            n_smoothing_iter=5,
            remove_smallest_components=True,
            aggressiveness=0.3,
            do_simplification=True,
            check_mesh_validity=True,
        )

        assert Meshify.is_mesh_valid(result)
        assert len(result.faces) > 0

    def test_no_simplification_still_valid(self, watertight_sphere_mesh):
        """With do_simplification=False, mesh should still be repaired and valid."""
        mesh = watertight_sphere_mesh
        result = Meshify.simplify_and_smooth_mesh(
            mesh,
            target_reduction=0.99,
            n_smoothing_iter=5,
            remove_smallest_components=True,
            do_simplification=False,
            check_mesh_validity=True,
        )

        assert Meshify.is_mesh_valid(result)
        # Without simplification, face count should be similar
        assert len(result.faces) >= len(mesh.faces) * 0.9

    def test_smoothing_preserves_approximate_volume(self, watertight_sphere_mesh):
        """Smoothing should not drastically change the volume."""
        mesh = watertight_sphere_mesh
        original_volume = mesh.volume

        result = Meshify.simplify_and_smooth_mesh(
            mesh,
            target_reduction=0.5,
            n_smoothing_iter=10,
            remove_smallest_components=True,
            do_simplification=True,
            check_mesh_validity=True,
        )

        # Volume should be within 30% of original
        np.testing.assert_allclose(result.volume, original_volume, rtol=0.3)


class TestMeshValidity:
    """Test the is_mesh_valid checker."""

    def test_valid_sphere(self):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        assert Meshify.is_mesh_valid(mesh)

    def test_valid_box(self):
        mesh = trimesh.creation.box(extents=[5, 5, 5])
        assert Meshify.is_mesh_valid(mesh)

    def test_open_mesh_is_invalid(self):
        """A mesh with an open face should not be watertight."""
        # Create a single triangle (not a closed surface)
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        assert not Meshify.is_mesh_valid(mesh)
