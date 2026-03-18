"""Integration tests for multi-scale mesh generation from a single zarr volume.

Tests the full pipeline: zarr segmentation → meshify at multiple downsampled
scales → neuroglancer multiresolution output.
"""

import os
import numpy as np
import pytest
import struct


class TestMultiscaleMeshify:
    """Test that Meshify with do_multires=True generates meshes at multiple
    scales from a single zarr volume and produces neuroglancer output."""

    def test_multiscale_generates_lod_directories(self, zarr_segmentation, tmp_output_dir):
        """Meshify with do_multires=True should create mesh_lods/s0/, s1/, etc."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "multiscale_output")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
        )
        meshify.get_meshes()

        # Check mesh_lods directories exist with PLY files
        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        assert os.path.isdir(mesh_lods_dir), "mesh_lods directory should exist"
        for lod in range(2):
            scale_dir = os.path.join(mesh_lods_dir, f"s{lod}")
            assert os.path.isdir(scale_dir), f"s{lod} directory should exist"
            ply_files = [f for f in os.listdir(scale_dir) if f.endswith(".ply")]
            assert len(ply_files) > 0, f"s{lod} should contain PLY mesh files"

    def test_multiscale_generates_neuroglancer_output(self, zarr_segmentation, tmp_output_dir):
        """The multires directory should contain neuroglancer index+data files."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "multiscale_ng_output")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
        )
        meshify.get_meshes()

        multires_dir = os.path.join(output_dir, "multires")
        assert os.path.isdir(multires_dir), "multires directory should exist"

        # Should have info file
        info_path = os.path.join(multires_dir, "info")
        assert os.path.isfile(info_path), "info file should exist"

        # Should have segment_properties
        seg_props_dir = os.path.join(multires_dir, "segment_properties")
        assert os.path.isdir(seg_props_dir), "segment_properties dir should exist"

        # Should have index files for at least one mesh ID
        index_files = [f for f in os.listdir(multires_dir) if f.endswith(".index")]
        assert len(index_files) > 0, "Should have .index files for mesh segments"

    def test_multiscale_lod1_has_fewer_faces(self, zarr_segmentation, tmp_output_dir):
        """Meshes at LOD 1 (downsampled volume) should have fewer faces than LOD 0."""
        from mesh_n_bone.meshify.meshify import Meshify
        import trimesh

        output_dir = os.path.join(tmp_output_dir, "multiscale_faces_output")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
        )
        meshify.get_meshes()

        mesh_lods_dir = os.path.join(output_dir, "mesh_lods")
        s0_dir = os.path.join(mesh_lods_dir, "s0")
        s1_dir = os.path.join(mesh_lods_dir, "s1")

        # Get PLY files that exist in both s0 and s1
        s0_files = {f for f in os.listdir(s0_dir) if f.endswith(".ply")}
        s1_files = {f for f in os.listdir(s1_dir) if f.endswith(".ply")}
        common = s0_files & s1_files

        assert len(common) > 0, "At least one mesh should exist at both LOD levels"

        for mesh_file in common:
            mesh_s0 = trimesh.load(os.path.join(s0_dir, mesh_file))
            mesh_s1 = trimesh.load(os.path.join(s1_dir, mesh_file))
            assert len(mesh_s1.faces) < len(mesh_s0.faces), (
                f"LOD 1 mesh {mesh_file} should have fewer faces than LOD 0 "
                f"({len(mesh_s1.faces)} >= {len(mesh_s0.faces)})"
            )

    def test_multiscale_neuroglancer_index_format(self, zarr_segmentation, tmp_output_dir):
        """Neuroglancer index files should have valid multilod_draco format."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "multiscale_index_output")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
        )
        meshify.get_meshes()

        multires_dir = os.path.join(output_dir, "multires")
        index_files = [f for f in os.listdir(multires_dir) if f.endswith(".index")]
        assert len(index_files) > 0

        for index_file in index_files:
            index_path = os.path.join(multires_dir, index_file)
            with open(index_path, "rb") as f:
                content = f.read()

            # Parse neuroglancer multilod index format:
            # 3 floats: chunk_shape
            # 3 floats: grid_origin
            # 1 uint32: num_lods
            offset = 0
            chunk_shape = struct.unpack_from("3f", content, offset)
            offset += 12
            grid_origin = struct.unpack_from("3f", content, offset)
            offset += 12
            num_lods = struct.unpack_from("I", content, offset)[0]
            offset += 4

            assert num_lods >= 1, f"Should have at least 1 LOD, got {num_lods}"
            assert all(c > 0 for c in chunk_shape), f"Chunk shape should be positive: {chunk_shape}"


class TestMultiscaleDownsampleMethods:
    """Test that different downsample methods produce valid meshes."""

    @pytest.mark.parametrize("method", ["mode_suppress_zero", "mode", "binary"])
    def test_downsample_method(self, zarr_segmentation, tmp_output_dir, method):
        """Each downsample method should produce valid meshes at both LOD levels."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, f"ds_{method}")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
            downsample_method=method,
        )
        meshify.get_meshes()

        # Both LOD dirs should have mesh files
        for lod in range(2):
            scale_dir = os.path.join(output_dir, "mesh_lods", f"s{lod}")
            assert os.path.isdir(scale_dir)
            ply_files = [f for f in os.listdir(scale_dir) if f.endswith(".ply")]
            assert len(ply_files) > 0, f"s{lod} with method={method} should have meshes"

    def test_invalid_downsample_method_raises(self, zarr_segmentation, tmp_output_dir):
        """An invalid downsample method should raise ValueError."""
        from mesh_n_bone.meshify.meshify import Meshify

        output_dir = os.path.join(tmp_output_dir, "ds_invalid")
        meshify = Meshify(
            input_path=zarr_segmentation,
            output_directory=output_dir,
            num_workers=1,
            do_multires=True,
            num_lods=2,
            do_simplification=False,
            check_mesh_validity=False,
            do_analysis=False,
            n_smoothing_iter=0,
            downsample_method="nonexistent",
        )
        with pytest.raises(ValueError, match="Unknown downsample_method"):
            meshify.get_meshes()
