"""Integration tests for the multires pipeline.

Tests the full workflow: mesh → decomposition → Draco compression → neuroglancer format.
"""

import json
import numpy as np
import os
import pytest
import struct
import trimesh

from mesh_n_bone.multires.decomposition import (
    generate_mesh_decomposition,
    my_slice_faces_plane,
)
from mesh_n_bone.multires.decimation import pyfqmr_decimate
from mesh_n_bone.multires.multires import generate_neuroglancer_multires_mesh
from mesh_n_bone.util.mesh_io import mesh_loader, write_mesh_files
from mesh_n_bone.util.neuroglancer import (
    write_info_file,
    write_segment_properties_file,
)


class TestDecompositionPipeline:
    """Test mesh decomposition into spatial fragments with Draco compression."""

    def test_single_lod_decomposition(self, tmp_output_dir):
        """A mesh should decompose into fragments that tile the bounding box."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        mesh.vertices += 100
        mesh_path = os.path.join(tmp_output_dir, "test_mesh.ply")
        mesh.export(mesh_path)

        lod_0_box_size = np.array([40.0, 40.0, 40.0])
        grid_origin = np.floor(mesh.vertices.min(axis=0) - 1)

        fragments = generate_mesh_decomposition(
            mesh_path=mesh_path,
            lod_0_box_size=lod_0_box_size,
            grid_origin=grid_origin,
            start_fragment=np.array([0, 0, 0]),
            end_fragment=np.array([5, 5, 5]),
            current_lod=0,
            num_chunks=np.array([1, 1, 1]),
        )

        assert fragments is not None
        assert len(fragments) > 0

        # Each fragment should have valid Draco bytes
        for frag in fragments:
            assert frag.draco_bytes is not None
            assert len(frag.draco_bytes) > 12
            assert frag.offset == len(frag.draco_bytes)
            assert len(frag.position) == 3

    def test_per_axis_box_size(self, tmp_output_dir):
        """Per-axis box sizes should produce valid fragments (no degenerate triangles)."""
        # Elongated mesh
        mesh = trimesh.creation.box(extents=[200, 40, 40])
        mesh.vertices += 150
        mesh_path = os.path.join(tmp_output_dir, "elongated.ply")
        mesh.export(mesh_path)

        # Per-axis box size matching the elongation
        lod_0_box_size = np.array([100.0, 40.0, 40.0])
        grid_origin = np.floor(mesh.vertices.min(axis=0) - 1)

        fragments = generate_mesh_decomposition(
            mesh_path=mesh_path,
            lod_0_box_size=lod_0_box_size,
            grid_origin=grid_origin,
            start_fragment=np.array([0, 0, 0]),
            end_fragment=np.array([4, 10, 10]),
            current_lod=0,
            num_chunks=np.array([1, 1, 1]),
        )

        assert fragments is not None
        assert len(fragments) > 0

    def test_higher_lod_decomposition(self, tmp_output_dir):
        """LOD > 0 should produce fragments with 2x larger effective box sizes."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=50.0)
        mesh.vertices += 100
        mesh_path = os.path.join(tmp_output_dir, "lod1_mesh.ply")
        mesh.export(mesh_path)

        lod_0_box_size = np.array([30.0, 30.0, 30.0])
        grid_origin = np.floor(mesh.vertices.min(axis=0) - 1)

        fragments_lod0 = generate_mesh_decomposition(
            mesh_path=mesh_path,
            lod_0_box_size=lod_0_box_size,
            grid_origin=grid_origin,
            start_fragment=np.array([0, 0, 0]),
            end_fragment=np.array([8, 8, 8]),
            current_lod=0,
            num_chunks=np.array([1, 1, 1]),
        )

        fragments_lod1 = generate_mesh_decomposition(
            mesh_path=mesh_path,
            lod_0_box_size=lod_0_box_size,
            grid_origin=grid_origin,
            start_fragment=np.array([0, 0, 0]),
            end_fragment=np.array([4, 4, 4]),
            current_lod=1,
            num_chunks=np.array([1, 1, 1]),
        )

        assert fragments_lod0 is not None
        assert fragments_lod1 is not None
        # Higher LOD should have fewer or equal fragments (larger effective box)
        assert len(fragments_lod1) <= len(fragments_lod0)

    def test_empty_region_returns_none(self, tmp_output_dir):
        """Decomposing a region with no mesh should return None."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        mesh.vertices += 50
        mesh_path = os.path.join(tmp_output_dir, "small.ply")
        mesh.export(mesh_path)

        grid_origin = np.array([0.0, 0.0, 0.0])
        lod_0_box_size = np.array([5.0, 5.0, 5.0])

        # Request a region far from the mesh
        result = generate_mesh_decomposition(
            mesh_path=mesh_path,
            lod_0_box_size=lod_0_box_size,
            grid_origin=grid_origin,
            start_fragment=np.array([100, 100, 100]),
            end_fragment=np.array([110, 110, 110]),
            current_lod=0,
            num_chunks=np.array([2, 2, 2]),
        )
        assert result is None


class TestDecimation:
    """Test mesh decimation with pyfqmr."""

    def test_decimation_reduces_faces(self, tmp_output_dir):
        """Decimation should produce a mesh with fewer faces."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        mesh.vertices += 100

        input_dir = os.path.join(tmp_output_dir, "input")
        output_dir = os.path.join(tmp_output_dir, "output")
        os.makedirs(input_dir)
        os.makedirs(os.path.join(output_dir, "s1"))
        mesh.export(os.path.join(input_dir, "1.ply"))

        original_faces = len(mesh.faces)

        pyfqmr_decimate(
            id=1,
            lod=1,
            input_path=input_dir,
            output_path=output_dir,
            ext=".ply",
            decimation_factor=4,
            aggressiveness=7,
        )

        decimated_path = os.path.join(output_dir, "s1", "1.ply")
        assert os.path.exists(decimated_path)

        dec_mesh = trimesh.load(decimated_path)
        assert len(dec_mesh.faces) < original_faces
        assert len(dec_mesh.faces) > 0

    def test_decimation_multiple_lods(self, tmp_output_dir):
        """Higher LODs should have progressively fewer faces."""
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        mesh.vertices += 100

        input_dir = os.path.join(tmp_output_dir, "input")
        output_dir = os.path.join(tmp_output_dir, "output")
        os.makedirs(input_dir)
        mesh.export(os.path.join(input_dir, "1.ply"))

        face_counts = [len(mesh.faces)]
        for lod in [1, 2, 3]:
            os.makedirs(os.path.join(output_dir, f"s{lod}"), exist_ok=True)
            pyfqmr_decimate(
                id=1,
                lod=lod,
                input_path=input_dir,
                output_path=output_dir,
                ext=".ply",
                decimation_factor=4,
                aggressiveness=7,
            )
            dec_mesh = trimesh.load(os.path.join(output_dir, f"s{lod}", "1.ply"))
            face_counts.append(len(dec_mesh.faces))

        # Each LOD should have fewer faces than the previous
        for i in range(1, len(face_counts)):
            assert face_counts[i] < face_counts[i - 1]


class TestFullMultiresPipeline:
    """Test the complete multires pipeline end-to-end."""

    def test_single_mesh_multires(self, multires_mesh_dir):
        """Full pipeline: LOD meshes → decomposition → Draco → neuroglancer files."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=None,
        )

        multires_dir = os.path.join(output_path, "multires")
        assert os.path.exists(multires_dir)

        # Mesh data file should exist
        mesh_file = os.path.join(multires_dir, "1")
        assert os.path.exists(mesh_file)
        assert os.path.getsize(mesh_file) > 0

        # Index file should exist and be parseable
        index_file = os.path.join(multires_dir, "1.index")
        assert os.path.exists(index_file)

        with open(index_file, "rb") as f:
            data = f.read()
        # Index starts with chunk_shape (3 floats), grid_origin (3 floats), num_lods (1 uint)
        assert len(data) >= 28  # minimum: 12 + 12 + 4 bytes
        chunk_shape = struct.unpack("<3f", data[0:12])
        grid_origin = struct.unpack("<3f", data[12:24])
        num_lods = struct.unpack("<I", data[24:28])[0]
        assert num_lods >= 1
        assert all(cs > 0 for cs in chunk_shape)

    def test_multires_with_explicit_box_size(self, multires_mesh_dir):
        """Pipeline with explicit per-axis box size."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=np.array([30.0, 30.0, 30.0]),
        )

        multires_dir = os.path.join(output_path, "multires")
        assert os.path.exists(os.path.join(multires_dir, "1"))
        assert os.path.exists(os.path.join(multires_dir, "1.index"))

    def test_neuroglancer_metadata_files(self, multires_mesh_dir):
        """Info and segment_properties files should be valid JSON."""
        output_path = multires_mesh_dir
        multires_dir = os.path.join(output_path, "multires")

        # Generate multires mesh first
        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
        )

        # Write metadata
        write_info_file(multires_dir)
        write_segment_properties_file(multires_dir)

        # Verify info file
        with open(os.path.join(multires_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["vertex_quantization_bits"] == 10

        # Verify segment properties
        sp_path = os.path.join(multires_dir, "segment_properties", "info")
        assert os.path.exists(sp_path)
        with open(sp_path) as f:
            sp = json.load(f)
        assert sp["@type"] == "neuroglancer_segment_properties"
        assert "1" in sp["inline"]["ids"]
