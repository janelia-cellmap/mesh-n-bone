"""Integration tests for multi-scale mesh generation from a single zarr volume.

Tests both strategies:
- "downsample": downsample volume at each LOD, re-mesh
- "decimate": mesh at s0, then decimate for higher LODs
"""

import os
import numpy as np
import pytest
import struct


# Common kwargs for all tests
_BASE_KWARGS = dict(
    num_workers=1,
    do_simplification=False,
    check_mesh_validity=False,
    do_analysis=False,
    n_smoothing_iter=0,
)


class TestDownsampleStrategy:
    """Test the 'downsample' multires strategy."""

    def test_generates_lod_directories(self, zarr_segmentation, tmp_output_dir):
        """Should create mesh_lods/s0/, s1/ with PLY files."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "ds_lods")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            delete_decimated_meshes=False, **_BASE_KWARGS,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        assert os.path.isdir(mesh_lods_dir)
        for lod in range(2):
            scale_dir = os.path.join(mesh_lods_dir, f"s{lod}")
            assert os.path.isdir(scale_dir), f"s{lod} directory should exist"
            ply_files = [f for f in os.listdir(scale_dir) if f.endswith(".ply")]
            assert len(ply_files) > 0, f"s{lod} should contain PLY files"

    def test_lod1_has_fewer_faces(self, zarr_segmentation, tmp_output_dir):
        """Downsampled LOD 1 should have fewer faces than LOD 0."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        output_dir = os.path.join(tmp_output_dir, "ds_faces")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            delete_decimated_meshes=False, **_BASE_KWARGS,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s0_files = {f for f in os.listdir(f"{mesh_lods_dir}/s0") if f.endswith(".ply")}
        s1_files = {f for f in os.listdir(f"{mesh_lods_dir}/s1") if f.endswith(".ply")}
        common = s0_files & s1_files
        assert len(common) > 0

        for mesh_file in common:
            m0 = trimesh.load(os.path.join(mesh_lods_dir, "s0", mesh_file))
            m1 = trimesh.load(os.path.join(mesh_lods_dir, "s1", mesh_file))
            assert len(m1.faces) < len(m0.faces), (
                f"LOD 1 should have fewer faces ({len(m1.faces)} >= {len(m0.faces)})"
            )

    @pytest.mark.parametrize("method", ["mode_suppress_zero", "mode", "binary", "nearest"])
    def test_downsample_methods(self, zarr_segmentation, tmp_output_dir, method):
        """Each downsample method should produce valid meshes at both LODs."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, f"ds_{method}")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            delete_decimated_meshes=False, downsample_method=method, **_BASE_KWARGS,
        )
        meshify.get_meshes()

        for lod in range(2):
            scale_dir = os.path.join(output_dir, "mesh_lods", f"s{lod}")
            assert os.path.isdir(scale_dir)
            ply_files = [f for f in os.listdir(scale_dir) if f.endswith(".ply")]
            assert len(ply_files) > 0

    def test_downsample_with_simplification_face_count_monotonic(
        self, zarr_segmentation, tmp_output_dir
    ):
        """With simplification enabled, each LOD should have fewer faces than the previous."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        output_dir = os.path.join(tmp_output_dir, "ds_simplify_mono")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=3, multires_strategy="downsample",
            delete_decimated_meshes=False,
            num_workers=1, do_simplification=True,
            use_fixed_edge_simplification=True, target_reduction=0.99,
            check_mesh_validity=False, do_analysis=False, n_smoothing_iter=0,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s0_files = {f for f in os.listdir(f"{mesh_lods_dir}/s0") if f.endswith(".ply")}
        s1_files = {f for f in os.listdir(f"{mesh_lods_dir}/s1") if f.endswith(".ply")}
        s2_files = {f for f in os.listdir(f"{mesh_lods_dir}/s2") if f.endswith(".ply")}
        common = s0_files & s1_files & s2_files
        assert len(common) > 0

        for mesh_file in common:
            m0 = trimesh.load(os.path.join(mesh_lods_dir, "s0", mesh_file))
            m1 = trimesh.load(os.path.join(mesh_lods_dir, "s1", mesh_file))
            m2 = trimesh.load(os.path.join(mesh_lods_dir, "s2", mesh_file))
            # For very small meshes (<20 faces), simplification/remeshing
            # can add a face during repair, so only enforce monotonicity
            # on meshes large enough for it to be meaningful.
            if len(m0.faces) >= 20:
                assert len(m0.faces) >= len(m1.faces), (
                    f"{mesh_file}: s0 ({len(m0.faces)}) should have >= faces than s1 ({len(m1.faces)})"
                )
            if len(m1.faces) >= 20:
                assert len(m1.faces) >= len(m2.faces), (
                    f"{mesh_file}: s1 ({len(m1.faces)}) should have >= faces than s2 ({len(m2.faces)})"
                )

    def test_downsample_with_simplification_preserves_spatial_extent(
        self, zarr_segmentation, tmp_output_dir
    ):
        """Simplification should not collapse the spatial extent of downsampled LODs."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        output_dir = os.path.join(tmp_output_dir, "ds_simplify_extent")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            delete_decimated_meshes=False,
            num_workers=1, do_simplification=True,
            use_fixed_edge_simplification=True, target_reduction=0.99,
            check_mesh_validity=False, do_analysis=False, n_smoothing_iter=0,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s0_files = {f for f in os.listdir(f"{mesh_lods_dir}/s0") if f.endswith(".ply")}
        s1_files = {f for f in os.listdir(f"{mesh_lods_dir}/s1") if f.endswith(".ply")}
        common = s0_files & s1_files
        assert len(common) > 0

        for mesh_file in common:
            m0 = trimesh.load(os.path.join(mesh_lods_dir, "s0", mesh_file))
            m1 = trimesh.load(os.path.join(mesh_lods_dir, "s1", mesh_file))
            extent_s0 = m0.bounds[1] - m0.bounds[0]
            extent_s1 = m1.bounds[1] - m1.bounds[0]
            for axis in range(3):
                if extent_s0[axis] > 0:
                    ratio = extent_s1[axis] / extent_s0[axis]
                    assert ratio > 0.5, (
                        f"{mesh_file} axis {axis}: s1 extent ({extent_s1[axis]:.1f}) "
                        f"is less than 50% of s0 ({extent_s0[axis]:.1f})"
                    )

    def test_downsample_block_boundary_clipping(
        self, zarr_cube_with_offset, tmp_output_dir
    ):
        """Boundary clipping should work correctly for downsampled multi-block meshes."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        zarr_path, expected_min, expected_max, _ = zarr_cube_with_offset
        output_dir = os.path.join(tmp_output_dir, "ds_boundary")
        meshify = Meshify(
            input_path=zarr_path, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            delete_decimated_meshes=False,
            num_workers=1, do_simplification=True,
            use_fixed_edge_simplification=True,
            check_mesh_validity=False, do_analysis=False, n_smoothing_iter=0,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s1_dir = os.path.join(mesh_lods_dir, "s1")
        assert os.path.isdir(s1_dir)
        ply_files = [f for f in os.listdir(s1_dir) if f.endswith(".ply")]
        assert len(ply_files) > 0

        for ply_file in ply_files:
            m = trimesh.load(os.path.join(s1_dir, ply_file))
            mesh_min = m.vertices.min(axis=0)
            mesh_max = m.vertices.max(axis=0)
            # Allow tolerance of 2x the downsampled voxel size (4*2=8)
            tolerance = 16
            assert np.all(mesh_min >= expected_min - tolerance), (
                f"{ply_file}: vertices below expected min {expected_min} (got {mesh_min})"
            )
            assert np.all(mesh_max <= expected_max + tolerance), (
                f"{ply_file}: vertices above expected max {expected_max} (got {mesh_max})"
            )

    def test_invalid_downsample_method_raises(self, zarr_segmentation, tmp_output_dir):
        """Invalid downsample method should raise ValueError."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "ds_invalid")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="downsample",
            downsample_method="nonexistent", **_BASE_KWARGS,
        )
        with pytest.raises(ValueError, match="Unknown downsample_method"):
            meshify.get_meshes()


class TestDecimateStrategy:
    """Test the 'decimate' multires strategy (default)."""

    def test_generates_neuroglancer_output(self, zarr_segmentation, tmp_output_dir):
        """Decimate strategy should produce neuroglancer multires output."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "dec_ng")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=3, multires_strategy="decimate",
            **_BASE_KWARGS,
        )
        meshify.get_meshes()

        multires_dir = os.path.join(output_dir, "multires")
        assert os.path.isdir(multires_dir)
        assert os.path.isfile(os.path.join(multires_dir, "info"))
        assert os.path.isdir(os.path.join(multires_dir, "segment_properties"))
        index_files = [f for f in os.listdir(multires_dir) if f.endswith(".index")]
        assert len(index_files) > 0

    def test_decimate_lod1_has_fewer_or_equal_faces(self, zarr_segmentation, tmp_output_dir):
        """Decimated LOD 1 should have <= faces than LOD 0 (small meshes may plateau at minimum)."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        output_dir = os.path.join(tmp_output_dir, "dec_faces")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="decimate",
            delete_decimated_meshes=False, **_BASE_KWARGS,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s0_files = {f for f in os.listdir(f"{mesh_lods_dir}/s0") if f.endswith(".ply")}
        s1_files = {f for f in os.listdir(f"{mesh_lods_dir}/s1") if f.endswith(".ply")}
        common = s0_files & s1_files
        assert len(common) > 0

        for mesh_file in common:
            m0 = trimesh.load(os.path.join(mesh_lods_dir, "s0", mesh_file))
            m1 = trimesh.load(os.path.join(mesh_lods_dir, "s1", mesh_file))
            assert len(m1.faces) <= len(m0.faces)


class TestNeuroglancerOutput:
    """Test neuroglancer output format for both strategies."""

    @pytest.mark.parametrize("strategy", ["downsample", "decimate"])
    def test_neuroglancer_index_has_at_least_one_lod(self, zarr_segmentation, tmp_output_dir, strategy):
        """Neuroglancer index should contain at least 1 LOD (small meshes may be truncated)."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, f"ng_{strategy}")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy=strategy,
            **_BASE_KWARGS,
        )
        meshify.get_meshes()

        multires_dir = os.path.join(output_dir, "multires")
        index_files = [f for f in os.listdir(multires_dir) if f.endswith(".index")]
        assert len(index_files) > 0

        for index_file in index_files:
            with open(os.path.join(multires_dir, index_file), "rb") as f:
                content = f.read()

            offset = 24  # skip chunk_shape (12) + grid_origin (12)
            num_lods = struct.unpack_from("I", content, offset)[0]
            assert num_lods >= 1, f"Should have at least 1 LOD, got {num_lods}"

    def test_invalid_strategy_raises(self, zarr_segmentation, tmp_output_dir):
        """Invalid multires_strategy should raise ValueError."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "bad_strategy")
        meshify = Meshify(
            input_path=zarr_segmentation, output_directory=output_dir,
            do_multires=True, num_lods=2, multires_strategy="nonexistent",
            **_BASE_KWARGS,
        )
        with pytest.raises(ValueError, match="Unknown multires_strategy"):
            meshify.get_meshes()
