"""Integration tests verifying mesh coordinate correctness across scales.

Tests that meshes from the meshify pipeline have correct world coordinates,
proper ZYX→XYZ handling, correct scaling by voxel size, and correct offsets.
Also tests that downsampled meshes at different LODs align spatially and
that sphere volumes are approximately correct.
"""

import os
import numpy as np
import pytest
import trimesh


def _run_meshify(input_path, output_dir, **kwargs):
    """Helper: run Meshify with sensible defaults for coordinate testing."""
    from mesh_n_bone.meshify.meshify import Meshify

    defaults = dict(
        num_workers=1,
        do_simplification=False,
        check_mesh_validity=False,
        do_analysis=False,
        n_smoothing_iter=0,
    )
    defaults.update(kwargs)
    meshify = Meshify(input_path=input_path, output_directory=output_dir, **defaults)
    meshify.get_meshes()
    return meshify


class TestCubeCoordinates:
    """Test that a cube mesh has correct world coordinates with offsets and voxel sizes."""

    def test_cube_bounds_at_lod0(self, zarr_cube_with_offset, tmp_output_dir):
        """LOD 0 mesh bounds should match expected world coordinates."""
        zarr_path, expected_min, expected_max, expected_center = zarr_cube_with_offset
        output_dir = os.path.join(tmp_output_dir, "cube_lod0")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))

        # Marching cubes boundaries should be within 1 voxel of expected
        # voxel_size=4 in all axes
        voxel_size = 4.0
        np.testing.assert_allclose(mesh.bounds[0], expected_min, atol=voxel_size,
                                   err_msg="LOD 0 mesh min bounds wrong")
        np.testing.assert_allclose(mesh.bounds[1], expected_max, atol=voxel_size,
                                   err_msg="LOD 0 mesh max bounds wrong")
        np.testing.assert_allclose(mesh.centroid, expected_center, atol=voxel_size,
                                   err_msg="LOD 0 mesh centroid wrong")

    def test_cube_bounds_zyx_not_flipped(self, zarr_cube_with_offset, tmp_output_dir):
        """Mesh should be in XYZ space with X corresponding to zarr's last axis.

        With offset=[100,200,300] (ZYX), the X coordinates should be near 300+,
        Y near 200+, Z near 100+.
        """
        zarr_path, _, _, _ = zarr_cube_with_offset
        output_dir = os.path.join(tmp_output_dir, "cube_zyx_check")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        centroid = mesh.centroid

        # X axis (zarr dim 2, offset 300) should be largest
        # Z axis (zarr dim 0, offset 100) should be smallest
        assert centroid[0] > centroid[2], (
            f"X centroid ({centroid[0]:.1f}) should be > Z centroid ({centroid[2]:.1f}) "
            f"since zarr X-offset=300 > Z-offset=100. Axes may be flipped."
        )
        assert centroid[1] > centroid[2], (
            f"Y centroid ({centroid[1]:.1f}) should be > Z centroid ({centroid[2]:.1f}) "
            f"since zarr Y-offset=200 > Z-offset=100. Axes may be flipped."
        )

    def test_multiscale_cubes_world_bounds(self, zarr_cube_with_offset, tmp_output_dir):
        """Each LOD mesh should have bounds matching expected world coordinates.

        The zarr cube at voxels [8:48]³ with voxel_size=[4,4,4] and
        offset=[100,200,300] (ZYX) spans multiple zarr chunks (16³),
        so blockwise assembly is exercised at each LOD.
        Coarser LODs shift by up to half a voxel at that resolution.
        """
        zarr_path, expected_min, expected_max, expected_center = zarr_cube_with_offset
        output_dir = os.path.join(tmp_output_dir, "cube_multiscale_bounds")
        _run_meshify(zarr_path, output_dir, do_multires=True, num_lods=3,
                     multires_strategy="downsample", delete_decimated_meshes=False)

        expected_extent = expected_max - expected_min  # [64, 64, 64]
        for lod in range(3):
            ply_path = os.path.join(output_dir, "mesh_lods", f"s{lod}", "1.ply")
            assert os.path.exists(ply_path), f"s{lod}/1.ply should exist"
            mesh = trimesh.load(ply_path)

            # At LOD N, effective voxel = base_voxel * 2^N.
            # Marching cubes isosurface can shift by up to half that voxel.
            lod_voxel = 4.0 * (2 ** lod)
            tol = lod_voxel

            np.testing.assert_allclose(
                mesh.bounds[0], expected_min, atol=tol,
                err_msg=f"LOD {lod} mesh min bounds wrong"
            )
            np.testing.assert_allclose(
                mesh.bounds[1], expected_max, atol=tol,
                err_msg=f"LOD {lod} mesh max bounds wrong"
            )
            np.testing.assert_allclose(
                mesh.centroid, expected_center, atol=tol,
                err_msg=f"LOD {lod} mesh centroid wrong"
            )

            extent = mesh.bounds[1] - mesh.bounds[0]
            np.testing.assert_allclose(
                extent, expected_extent, atol=2 * lod_voxel,
                err_msg=f"LOD {lod} extent differs from expected {expected_extent}"
            )


class TestOmeNgffCoordinates:
    """Test that OME-NGFF multiscales metadata is correctly picked up."""

    def test_ome_ngff_cube_bounds(self, zarr_cube_ome_ngff, tmp_output_dir):
        """Mesh from a zarr with OME-NGFF metadata should have correct world coords."""
        zarr_path, expected_min, expected_max, expected_center = zarr_cube_ome_ngff
        output_dir = os.path.join(tmp_output_dir, "ome_cube")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))

        voxel_size = 8.0
        np.testing.assert_allclose(mesh.bounds[0], expected_min, atol=voxel_size,
                                   err_msg="OME-NGFF cube min bounds wrong")
        np.testing.assert_allclose(mesh.bounds[1], expected_max, atol=voxel_size,
                                   err_msg="OME-NGFF cube max bounds wrong")
        np.testing.assert_allclose(mesh.centroid, expected_center, atol=voxel_size,
                                   err_msg="OME-NGFF cube centroid wrong")

    def test_ome_ngff_not_in_voxel_coords(self, zarr_cube_ome_ngff, tmp_output_dir):
        """Mesh vertices should NOT be in voxel coordinates (0-64 range)."""
        zarr_path, _, _, _ = zarr_cube_ome_ngff
        output_dir = os.path.join(tmp_output_dir, "ome_not_voxel")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        # With voxel_size=8, offset=100, the mesh should be at ~160-480 range
        # If voxel_size was missed, it would be at ~8-48 range
        assert mesh.bounds[1].max() > 100, (
            f"Mesh max coord ({mesh.bounds[1].max():.1f}) is too small — "
            f"voxel_size may not have been applied"
        )


class TestSphereGeometry:
    """Test that a sphere mesh has approximately correct volume and shape."""

    def test_sphere_centroid(self, zarr_sphere, tmp_output_dir):
        """Sphere mesh centroid should be at the expected world position."""
        zarr_path, expected_center, _, _ = zarr_sphere
        output_dir = os.path.join(tmp_output_dir, "sphere_centroid")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        # voxel_size=2, so tolerance of ~2
        np.testing.assert_allclose(mesh.centroid, expected_center, atol=2.0,
                                   err_msg="Sphere centroid wrong")

    def test_sphere_volume(self, zarr_sphere, tmp_output_dir):
        """Sphere mesh volume should be approximately (4/3)πr³."""
        zarr_path, _, _, expected_volume = zarr_sphere
        output_dir = os.path.join(tmp_output_dir, "sphere_volume")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        # Voxelized sphere volume differs from analytic — allow 15% tolerance
        # (voxelization of a sphere with r=20 voxels is quite good)
        assert mesh.is_watertight, "Sphere mesh should be watertight"
        np.testing.assert_allclose(
            abs(mesh.volume), expected_volume, rtol=0.15,
            err_msg="Sphere mesh volume differs too much from analytic"
        )

    def test_sphere_radius_from_bounds(self, zarr_sphere, tmp_output_dir):
        """Sphere bounding box should give approximately the right radius."""
        zarr_path, expected_center, expected_radius, _ = zarr_sphere
        output_dir = os.path.join(tmp_output_dir, "sphere_radius")
        _run_meshify(zarr_path, output_dir)

        mesh = trimesh.load(os.path.join(output_dir, "meshes", "1.ply"))
        extents = mesh.bounds[1] - mesh.bounds[0]
        # Diameter ≈ 2 * radius for each axis
        for i, axis in enumerate(["X", "Y", "Z"]):
            measured_radius = extents[i] / 2
            np.testing.assert_allclose(
                measured_radius, expected_radius, atol=4.0,
                err_msg=f"Sphere {axis}-axis radius wrong"
            )

    def test_sphere_multiscale_volume_consistent(self, zarr_sphere, tmp_output_dir):
        """Sphere volume should be approximately consistent across LOD levels."""
        zarr_path, _, _, expected_volume = zarr_sphere
        output_dir = os.path.join(tmp_output_dir, "sphere_multiscale_vol")
        _run_meshify(zarr_path, output_dir, do_multires=True, num_lods=3,
                     multires_strategy="downsample", delete_decimated_meshes=False)

        volumes = []
        for lod in range(3):
            ply_path = os.path.join(output_dir, "mesh_lods", f"s{lod}", "1.ply")
            if os.path.exists(ply_path):
                mesh = trimesh.load(ply_path)
                if mesh.is_watertight:
                    volumes.append(abs(mesh.volume))

        assert len(volumes) >= 2, "Need at least 2 LODs with watertight meshes"
        # Coarser LODs have less accurate volume — voxelization error grows with voxel size
        # LOD 0: ~5% error, LOD 1: ~15%, LOD 2: ~35% (sphere radius ~5 voxels)
        tolerances = [0.15, 0.25, 0.40]
        for i, vol in enumerate(volumes):
            rtol = tolerances[i] if i < len(tolerances) else 0.50
            np.testing.assert_allclose(
                vol, expected_volume, rtol=rtol,
                err_msg=f"LOD {i} sphere volume too far from analytic"
            )

    def test_sphere_multiscale_centroids_align(self, zarr_sphere, tmp_output_dir):
        """Sphere centroids at all LODs should be at the same world position."""
        zarr_path, expected_center, _, _ = zarr_sphere
        output_dir = os.path.join(tmp_output_dir, "sphere_multiscale_ctr")
        _run_meshify(zarr_path, output_dir, do_multires=True, num_lods=3,
                     multires_strategy="downsample", delete_decimated_meshes=False)

        for lod in range(3):
            ply_path = os.path.join(output_dir, "mesh_lods", f"s{lod}", "1.ply")
            if os.path.exists(ply_path):
                mesh = trimesh.load(ply_path)
                # At LOD 2, voxel=8, so tolerance ~8
                lod_voxel = 2.0 * (2 ** lod)
                np.testing.assert_allclose(
                    mesh.centroid, expected_center, atol=lod_voxel,
                    err_msg=f"LOD {lod} sphere centroid wrong"
                )
