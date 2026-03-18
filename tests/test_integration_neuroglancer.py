"""Integration tests for neuroglancer format output files."""

import json
import numpy as np
import os
import pytest
import struct
import trimesh

from mesh_n_bone.util.neuroglancer import (
    write_info_file,
    write_segment_properties_file,
    write_ngmesh,
    write_ngmesh_metadata,
    write_singleres_multires_files,
    write_singleres_multires_metadata,
    write_singleres_index_file,
    write_precomputed_annotations,
)
from mesh_n_bone.util.mesh_io import mesh_loader


class TestNgmeshFormat:
    """Test legacy neuroglancer mesh format."""

    def test_write_and_read_ngmesh(self, tmp_output_dir):
        """Write an ngmesh file and verify it can be read back."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        mesh.vertices += 50

        path = os.path.join(tmp_output_dir, "test_mesh")
        write_ngmesh(mesh.vertices, mesh.faces, path)

        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Read it back using mesh_loader (which handles ngmesh format)
        vertices, faces = mesh_loader(path)
        assert vertices is not None
        assert faces is not None
        assert vertices.shape[0] == len(mesh.vertices)
        assert faces.shape[0] == len(mesh.faces)

    def test_ngmesh_bytes_output(self):
        """write_ngmesh with no file arg should return bytes."""
        mesh = trimesh.creation.box(extents=[5, 5, 5])
        result = write_ngmesh(mesh.vertices, mesh.faces)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_ngmesh_metadata(self, tmp_output_dir):
        """Metadata files should be valid JSON with correct format."""
        mesh_dir = os.path.join(tmp_output_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        # Create a mock mesh with fragment file
        mesh = trimesh.creation.box(extents=[5, 5, 5])
        write_ngmesh(mesh.vertices, mesh.faces, os.path.join(mesh_dir, "42"))
        with open(os.path.join(mesh_dir, "42:0"), "w") as f:
            json.dump({"fragments": ["./42"]}, f)

        write_ngmesh_metadata(mesh_dir)

        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_legacy_mesh"

        sp_path = os.path.join(mesh_dir, "segment_properties", "info")
        assert os.path.exists(sp_path)
        with open(sp_path) as f:
            sp = json.load(f)
        assert "42" in sp["inline"]["ids"]


class TestSingleresMultiresFormat:
    """Test single-resolution multires (Draco) format."""

    def test_write_singleres_multires(self, tmp_output_dir):
        """Write a mesh in singleres multires format and verify output."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        mesh.vertices += 50

        path = os.path.join(tmp_output_dir, "1")
        res, quantized_verts = write_singleres_multires_files(
            mesh.vertices, mesh.faces, path
        )

        # Mesh data file
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

        # Index file
        index_path = path + ".index"
        assert os.path.exists(index_path)

        # Parse index file
        with open(index_path, "rb") as f:
            data = f.read()

        chunk_shape = struct.unpack("<3f", data[0:12])
        grid_origin = struct.unpack("<3f", data[12:24])
        num_lods = struct.unpack("<I", data[24:28])[0]

        assert num_lods == 1
        assert all(cs > 0 for cs in chunk_shape)

    def test_singleres_metadata(self, tmp_output_dir):
        """Metadata for singleres multires format."""
        mesh_dir = os.path.join(tmp_output_dir, "meshes")
        os.makedirs(mesh_dir, exist_ok=True)

        mesh = trimesh.creation.box(extents=[5, 5, 5])
        mesh.vertices += 10
        write_singleres_multires_files(
            mesh.vertices, mesh.faces, os.path.join(mesh_dir, "1")
        )

        write_singleres_multires_metadata(mesh_dir)

        with open(os.path.join(mesh_dir, "info")) as f:
            info = json.load(f)
        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["vertex_quantization_bits"] == 10


class TestMultilodDracoFormat:
    """Test multilod draco info and segment properties."""

    def test_info_file(self, tmp_output_dir):
        write_info_file(tmp_output_dir)

        with open(os.path.join(tmp_output_dir, "info")) as f:
            info = json.load(f)

        assert info["@type"] == "neuroglancer_multilod_draco"
        assert info["vertex_quantization_bits"] == 10
        assert info["lod_scale_multiplier"] == 1
        assert len(info["transform"]) == 12

    def test_segment_properties_file(self, tmp_output_dir):
        """segment_properties should list all mesh IDs from .index files."""
        # Create mock .index files
        for mesh_id in [1, 5, 42]:
            with open(os.path.join(tmp_output_dir, f"{mesh_id}.index"), "w") as f:
                f.write("")  # content doesn't matter for this test

        write_segment_properties_file(tmp_output_dir)

        sp_path = os.path.join(tmp_output_dir, "segment_properties", "info")
        with open(sp_path) as f:
            sp = json.load(f)

        assert sp["@type"] == "neuroglancer_segment_properties"
        ids = sp["inline"]["ids"]
        assert "1" in ids
        assert "5" in ids
        assert "42" in ids
        # IDs should be sorted numerically
        assert ids == sorted(ids, key=int)


class TestPrecomputedAnnotations:
    """Test neuroglancer annotation output."""

    def test_point_annotations(self, tmp_output_dir):
        """Write point annotations and verify info file."""
        output_dir = os.path.join(tmp_output_dir, "annotations")
        coords = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.float32)
        ids = np.array([1, 2], dtype=np.uint64)
        properties = {"value": np.array([1.5, 2.5], dtype=np.float32)}

        write_precomputed_annotations(
            output_dir,
            annotation_type="point",
            ids=ids,
            coords=coords,
            properties_dict=properties,
        )

        # Verify info file
        with open(os.path.join(output_dir, "info")) as f:
            info = json.load(f)
        assert info["annotation_type"] == "point"
        assert info["@type"] == "neuroglancer_annotations_v1"

        # Verify spatial data file exists
        assert os.path.exists(os.path.join(output_dir, "spatial0", "0_0_0"))

    def test_line_annotations(self, tmp_output_dir):
        """Write line annotations with 6 coordinates per entry."""
        output_dir = os.path.join(tmp_output_dir, "line_annotations")
        # Line annotations have start + end = 6 coords per entry
        coords = np.array(
            [[10, 20, 30, 40, 50, 60], [1, 2, 3, 4, 5, 6]], dtype=np.float32
        )
        ids = np.array([1, 2], dtype=np.uint64)
        properties = {"length": np.array([50.0, 5.0], dtype=np.float32)}

        write_precomputed_annotations(
            output_dir,
            annotation_type="line",
            ids=ids,
            coords=coords,
            properties_dict=properties,
        )

        with open(os.path.join(output_dir, "info")) as f:
            info = json.load(f)
        assert info["annotation_type"] == "line"


class TestIndexFileParsing:
    """Test writing and reading back index files."""

    def test_singleres_index_roundtrip(self, tmp_output_dir):
        """Write an index file and verify its binary structure."""
        path = os.path.join(tmp_output_dir, "test.index")
        grid_origin = np.array([10.0, 20.0, 30.0])
        chunk_shape = np.array([50.0, 50.0, 50.0])

        write_singleres_index_file(
            path=path,
            grid_origin=grid_origin,
            fragment_positions=[[0, 0, 0], [1, 0, 0]],
            fragment_offsets=[100, 200],
            current_lod=0,
            lods=[0],
            chunk_shape=chunk_shape,
        )

        with open(path, "rb") as f:
            data = f.read()

        cs = struct.unpack("<3f", data[0:12])
        go = struct.unpack("<3f", data[12:24])
        num_lods = struct.unpack("<I", data[24:28])[0]

        np.testing.assert_array_almost_equal(cs, [50, 50, 50])
        np.testing.assert_array_almost_equal(go, [10, 20, 30])
        assert num_lods == 1
