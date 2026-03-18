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

    @pytest.mark.parametrize("method", ["mode_suppress_zero", "mode", "binary"])
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

    def test_decimate_lod1_has_fewer_faces(self, zarr_segmentation, tmp_output_dir):
        """Decimated LOD 1 should have fewer faces than LOD 0."""
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
            assert len(m1.faces) < len(m0.faces)


class TestNeuroglancerOutput:
    """Test neuroglancer output format for both strategies."""

    @pytest.mark.parametrize("strategy", ["downsample", "decimate"])
    def test_neuroglancer_index_has_correct_num_lods(self, zarr_segmentation, tmp_output_dir, strategy):
        """Neuroglancer index should contain the correct number of LODs."""
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
            assert num_lods == 2, f"Should have 2 LODs, got {num_lods}"

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
