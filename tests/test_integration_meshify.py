"""Integration tests for the meshify pipeline: zarr → marching cubes → chunk assembly → mesh output.

Tests the full blockwise workflow: reading segmentation from zarr, generating
per-chunk meshes, assembling across chunk boundaries, simplification, and
neuroglancer format output.
"""

import numpy as np
import os
import pytest
import tempfile
import trimesh

from funlib.persistence import prepare_ds
from funlib.geometry import Coordinate

from mesh_n_bone.meshify.meshify import Meshify


def _create_zarr_volume(tmpdir, vol, voxel_size=(8, 8, 8), chunk_shape=(16, 16, 16)):
    """Helper: write a labeled volume to zarr with funlib metadata."""
    zarr_path = os.path.join(tmpdir, "test.zarr")
    vs = Coordinate(*voxel_size)
    cs = Coordinate(*chunk_shape)
    ds = prepare_ds(
        f"{zarr_path}/labels/s0",
        shape=Coordinate(vol.shape),
        offset=Coordinate(0, 0, 0),
        voxel_size=vs,
        dtype=vol.dtype,
        chunk_shape=cs,
    )
    ds[ds.roi] = vol
    return f"{zarr_path}/labels/s0"


class TestMeshifyFromZarr:
    """End-to-end tests: zarr segmentation → PLY meshes."""

    def test_two_separate_objects(self, tmp_output_dir):
        """Two non-overlapping objects should produce two separate mesh files."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:14, 2:14, 2:14] = 1
        vol[18:30, 18:30, 18:30] = 2

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        meshes = sorted(os.listdir(mesh_dir))
        assert len(meshes) == 2
        assert "1.ply" in meshes
        assert "2.ply" in meshes

        for mesh_file in meshes:
            mesh = trimesh.load(os.path.join(mesh_dir, mesh_file))
            assert len(mesh.faces) > 0
            assert mesh.volume > 0

    def test_cross_chunk_object_is_watertight(self, tmp_output_dir):
        """An object spanning multiple chunks should assemble into a watertight mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        # Object spans all 8 chunks (chunk_shape=16)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert mesh.is_watertight
        # Volume should be close to expected (28^3 voxels * 8^3 nm^3/voxel)
        expected_vol = (28**3) * (8**3)
        np.testing.assert_allclose(mesh.volume, expected_vol, rtol=0.05)

    def test_cross_chunk_volume_accuracy(self, tmp_output_dir):
        """Chunk assembly should not lose or duplicate geometry at boundaries."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[6:26, 6:26, 6:26] = 1  # 20^3 voxel cube crossing chunk boundary at 16

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        expected_vol = (20**3) * (8**3)
        # Should be within 5% — marching cubes is approximate at boundaries
        np.testing.assert_allclose(mesh.volume, expected_vol, rtol=0.05)


class TestMeshifyWithSimplification:
    """Test meshify with simplification and repair enabled."""

    def test_simplification_reduces_faces(self, tmp_output_dir):
        """Simplification should produce fewer faces than the raw mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)

        # First: no simplification
        output_raw = os.path.join(tmp_output_dir, "output_raw")
        m_raw = Meshify(
            input_path=input_path,
            output_directory=output_raw,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m_raw.get_meshes()
        raw_mesh = trimesh.load(os.path.join(output_raw, "meshes", "1.ply"))

        # Second: with simplification
        output_simp = os.path.join(tmp_output_dir, "output_simp")
        m_simp = Meshify(
            input_path=input_path,
            output_directory=output_simp,
            num_workers=1,
            target_reduction=0.9,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=True,
            n_smoothing_iter=5,
            remove_smallest_components=False,
        )
        m_simp.get_meshes()
        simp_mesh = trimesh.load(os.path.join(output_simp, "meshes", "1.ply"))

        assert len(simp_mesh.faces) < len(raw_mesh.faces)
        assert simp_mesh.volume > 0

    def test_simplification_with_validity_check(self, tmp_output_dir):
        """With check_mesh_validity=True, output should be watertight."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            target_reduction=0.9,
            do_analysis=False,
            check_mesh_validity=True,
            do_simplification=True,
            n_smoothing_iter=5,
            remove_smallest_components=True,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert Meshify.is_mesh_valid(mesh)


class TestMeshifyWithDownsampling:
    """Test meshify with on-the-fly downsampling."""

    def test_downsample_factor(self, tmp_output_dir):
        """Downsampling should produce a coarser but valid mesh."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[2:30, 2:30, 2:30] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            downsample_factor=2,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        assert os.path.exists(os.path.join(mesh_dir, "1.ply"))
        mesh = trimesh.load(os.path.join(mesh_dir, "1.ply"))
        assert len(mesh.faces) > 0
        assert mesh.volume > 0


class TestMeshifyNeuroglancerOutput:
    """Test neuroglancer format output from meshify."""

    def test_legacy_neuroglancer_format(self, tmp_output_dir):
        """do_legacy_neuroglancer should write ngmesh files + metadata."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            do_legacy_neuroglancer=True,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        # Should have mesh data file, fragment file, info, and segment_properties
        assert os.path.exists(os.path.join(mesh_dir, "1"))
        assert os.path.exists(os.path.join(mesh_dir, "1:0"))
        assert os.path.exists(os.path.join(mesh_dir, "info"))
        assert os.path.exists(os.path.join(mesh_dir, "segment_properties", "info"))

        import json
        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_legacy_mesh"

    def test_singleres_multires_format(self, tmp_output_dir):
        """do_singleres_multires_neuroglancer should write Draco files + index."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
            do_singleres_multires_neuroglancer=True,
        )
        m.get_meshes()

        mesh_dir = os.path.join(output_dir, "meshes")
        # Should have Draco mesh + .index file
        assert os.path.exists(os.path.join(mesh_dir, "1"))
        assert os.path.exists(os.path.join(mesh_dir, "1.index"))
        assert os.path.exists(os.path.join(mesh_dir, "info"))

        import json
        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"


class TestMeshifyWithAnalysis:
    """Test meshify with built-in analysis."""

    def test_analysis_produces_csv(self, tmp_output_dir):
        """With do_analysis=True, a metrics CSV should be generated."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        input_path = _create_zarr_volume(tmp_output_dir, vol)
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=True,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        csv_path = os.path.join(output_dir, "metrics", "mesh_metrics.csv")
        assert os.path.exists(csv_path)

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 1
        assert "volume (nm^3)" in df.columns
        assert df["volume (nm^3)"].iloc[0] > 0


class TestMeshifyAnisotropicVoxels:
    """Test meshify with anisotropic voxel sizes."""

    def test_anisotropic_voxel_size(self, tmp_output_dir):
        """Anisotropic voxels should produce correctly scaled meshes."""
        vol = np.zeros((32, 32, 32), dtype=np.uint64)
        vol[4:28, 4:28, 4:28] = 1

        # Anisotropic: z is 2x coarser
        input_path = _create_zarr_volume(
            tmp_output_dir, vol, voxel_size=(8, 8, 16)
        )
        output_dir = os.path.join(tmp_output_dir, "output")

        m = Meshify(
            input_path=input_path,
            output_directory=output_dir,
            num_workers=1,
            do_analysis=False,
            check_mesh_validity=False,
            do_simplification=False,
            n_smoothing_iter=0,
            remove_smallest_components=False,
        )
        m.get_meshes()

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        assert len(mesh.faces) > 0

        # Mesh extents should reflect anisotropic voxel size
        extents = mesh.bounds[1] - mesh.bounds[0]
        # z extent should be ~2x the x/y extents
        z_ratio = extents[0] / extents[2]  # vertices are xyz reversed from zyx
        # Should be roughly 0.5 (z is 2x larger voxels, so same voxel count = 2x physical)
        assert 0.3 < z_ratio < 0.7 or 1.5 < z_ratio < 2.5
