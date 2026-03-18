"""Integration tests for the mesh analysis pipeline."""

import numpy as np
import os
import pytest
import trimesh

from mesh_n_bone.analyze.analyze import AnalyzeMeshes


class TestAnalyzeSingleMesh:
    """Test analysis metrics on known geometric shapes."""

    def test_sphere_volume_and_area(self, tmp_output_dir):
        """Sphere volume and surface area should match analytic values."""
        radius = 10.0
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=radius)
        mesh_path = os.path.join(tmp_output_dir, "sphere.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)

        expected_volume = (4 / 3) * np.pi * radius**3
        expected_area = 4 * np.pi * radius**2

        # Icosphere is approximate — allow 5% tolerance
        np.testing.assert_allclose(
            metrics["volume (nm^3)"], expected_volume, rtol=0.05
        )
        np.testing.assert_allclose(
            metrics["surface_area (nm^2)"], expected_area, rtol=0.05
        )

    def test_cube_volume(self, tmp_output_dir):
        """Cube volume should match expected value."""
        mesh = trimesh.creation.box(extents=[20, 20, 20])
        mesh_path = os.path.join(tmp_output_dir, "cube.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)
        np.testing.assert_almost_equal(metrics["volume (nm^3)"], 8000.0, decimal=0)
        np.testing.assert_almost_equal(metrics["surface_area (nm^2)"], 2400.0, decimal=0)

    def test_curvature_metrics_exist_and_are_finite(self, tmp_output_dir):
        """All curvature metrics should be computed and finite."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        mesh_path = os.path.join(tmp_output_dir, "curv_sphere.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)

        for curv_type in ["mean", "gaussian", "rms", "abs"]:
            for stat in ["mean", "median", "std"]:
                key = f"{curv_type}_curvature_{stat}"
                assert key in metrics, f"Missing metric: {key}"
                assert np.isfinite(metrics[key]), f"Non-finite metric: {key}"

    def test_thickness_metrics(self, tmp_output_dir):
        """Thickness metrics should be positive and finite."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        mesh_path = os.path.join(tmp_output_dir, "thick_sphere.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)

        assert metrics["thickness_mean"] > 0
        assert np.isfinite(metrics["thickness_mean"])
        assert np.isfinite(metrics["thickness_median"])
        assert np.isfinite(metrics["thickness_std"])

    def test_principal_inertia_and_oriented_bounds(self, tmp_output_dir):
        """PIC and oriented bounds should be computed for all 3 axes."""
        mesh = trimesh.creation.box(extents=[10, 20, 30])
        mesh_path = os.path.join(tmp_output_dir, "rect.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)

        for axis in range(3):
            assert f"pic_{axis}" in metrics
            assert f"pic_normalized_{axis}" in metrics
            assert f"ob_{axis}" in metrics
            assert f"ob_normalized_{axis}" in metrics
            assert metrics[f"pic_{axis}"] > 0
            assert metrics[f"ob_{axis}"] > 0

    def test_elongated_mesh_oriented_bounds(self, tmp_output_dir):
        """An elongated box should have one large and two small oriented bounds."""
        mesh = trimesh.creation.box(extents=[100, 10, 10])
        mesh_path = os.path.join(tmp_output_dir, "long.ply")
        mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)

        obs = sorted([metrics[f"ob_{i}"] for i in range(3)])
        # Largest dimension should be ~10x the smallest
        assert obs[2] / obs[0] > 5
