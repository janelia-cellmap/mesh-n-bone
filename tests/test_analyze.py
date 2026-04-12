"""Tests for mesh analysis module."""

import numpy as np
import os
import pytest
import trimesh


class TestAnalyzeMesh:
    def test_analyze_single_mesh(self, tiny_sphere_mesh, tmp_output_dir):
        from mesh_n_bone.analyze.analyze import AnalyzeMeshes

        mesh_path = os.path.join(tmp_output_dir, "sphere.ply")
        tiny_sphere_mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)
        assert metrics["id"] == "sphere"
        assert metrics["volume (nm^3)"] > 0
        assert metrics["surface_area (nm^2)"] > 0
        # Check curvature metrics exist
        assert "mean_curvature_mean" in metrics
        assert "gaussian_curvature_mean" in metrics
        assert "thickness_mean" in metrics

    def test_analyze_cube(self, tiny_cube_mesh, tmp_output_dir):
        from mesh_n_bone.analyze.analyze import AnalyzeMeshes

        mesh_path = os.path.join(tmp_output_dir, "42.ply")
        tiny_cube_mesh.export(mesh_path)

        metrics = AnalyzeMeshes.analyze_mesh(mesh_path)
        assert metrics["id"] == "42"
        # Cube volume = 10*10*10 = 1000
        np.testing.assert_almost_equal(metrics["volume (nm^3)"], 1000.0, decimal=0)
