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

        lod_0_box_size = 40.0
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

    def test_higher_lod_decomposition(self, tmp_output_dir):
        """LOD > 0 should produce fragments with 2x larger effective box sizes."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=50.0)
        mesh.vertices += 100
        mesh_path = os.path.join(tmp_output_dir, "lod1_mesh.ply")
        mesh.export(mesh_path)

        lod_0_box_size = 30.0
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
        lod_0_box_size = 5.0

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
        """Higher LODs should have progressively fewer faces until the minimum."""
        mesh = trimesh.creation.icosphere(subdivisions=5, radius=50.0)
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

        # Each LOD should have <= faces than the previous
        for i in range(1, len(face_counts)):
            assert face_counts[i] <= face_counts[i - 1]
        # At least the first decimation should reduce faces
        assert face_counts[1] < face_counts[0]


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

    def test_multires_with_explicit_scalar_box_size(self, multires_mesh_dir):
        """Pipeline with explicit scalar box size (broadcast to per-axis)."""
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

    def test_multires_with_per_axis_box_size(self, multires_mesh_dir):
        """Pipeline with per-axis box size for elongated meshes."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=np.array([20.0, 30.0, 40.0]),
        )

        multires_dir = os.path.join(output_path, "multires")
        assert os.path.exists(os.path.join(multires_dir, "1"))
        assert os.path.exists(os.path.join(multires_dir, "1.index"))

        # Verify index file has per-axis chunk_shape
        with open(os.path.join(multires_dir, "1.index"), "rb") as f:
            data = f.read()
        chunk_shape = struct.unpack("<3f", data[0:12])
        np.testing.assert_allclose(chunk_shape, [20.0, 30.0, 40.0])

    def test_small_mesh_grid_center_matches_mesh_center(self, multires_mesh_dir):
        """Grid bounding-box center should approximately match the mesh center.

        The grid_origin is set so that Neuroglancer's navigation (which
        uses the grid bounding box) lands near the actual mesh.
        """
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=None,  # auto-compute from mesh
        )

        # Read the mesh to get the actual center
        from mesh_n_bone.util.mesh_io import mesh_loader
        vertices, _ = mesh_loader(
            os.path.join(output_path, "mesh_lods", "s0", "1.ply")
        )
        actual_center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2

        # Read the index file to get the grid center
        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()

        chunk_shape = np.frombuffer(data, "<f", 3, 0).copy()
        grid_origin = np.frombuffer(data, "<f", 3, 12).copy()
        num_lods = struct.unpack_from("<I", data, 24)[0]
        off = 28 + 4 * num_lods + 12 * num_lods
        num_frags_per_lod = np.frombuffer(data, "<I", num_lods, off).copy()
        off += 4 * num_lods
        nf = num_frags_per_lod[0]
        positions = np.frombuffer(data, "<I", nf * 3, off).reshape(3, nf).T.copy()

        grid_min = grid_origin + positions.min(axis=0) * chunk_shape
        grid_max = grid_origin + (positions.max(axis=0) + 1) * chunk_shape
        grid_center = (grid_min + grid_max) / 2

        offset = np.abs(grid_center - actual_center)
        assert np.all(offset < chunk_shape * 0.6), (
            f"Grid center {grid_center} too far from mesh center "
            f"{actual_center} (offset {offset}, chunk_shape {chunk_shape})"
        )

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
        assert info["vertex_quantization_bits"] == 16

        # Verify segment properties
        sp_path = os.path.join(multires_dir, "segment_properties", "info")
        assert os.path.exists(sp_path)
        with open(sp_path) as f:
            sp = json.load(f)
        assert sp["@type"] == "neuroglancer_segment_properties"
        assert "1" in sp["inline"]["ids"]


class TestLodTruncation:
    """Test that LOD truncation correctly includes all valid LODs.

    The old multiresolution-mesh-creator had a bug: `lods = lods[:idx]`
    without `else: idx += 1`, which always dropped the last valid LOD.
    With lods=[0,1], only LOD 0 was processed. The new code fixes this.
    """

    def test_all_valid_lods_included(self, multires_mesh_dir):
        """When all LODs have decreasing face counts, all should be included."""
        output_path = multires_mesh_dir

        # Use a small box_size to force multi-chunk so the single-LOD
        # truncation for single-chunk meshes does not apply.
        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=np.array([20.0, 30.0, 40.0]),
        )

        # Parse the index file to check num_lods
        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()
        num_lods = struct.unpack("<I", data[24:28])[0]
        # Both LODs should be present (the old code would produce only 1)
        assert num_lods == 2, (
            f"Expected 2 LODs in index file, got {num_lods}. "
            "The last valid LOD may be getting dropped."
        )

    def test_lod_truncation_on_non_decreasing_faces(self, tmp_output_dir):
        """If a higher LOD has >= faces than the previous, truncate there."""
        output_path = os.path.join(tmp_output_dir, "truncation_test")
        mesh_lods = os.path.join(output_path, "mesh_lods")

        # LOD 0: 642 faces (icosphere subdiv=3)
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        mesh.vertices += 100
        s0_dir = os.path.join(mesh_lods, "s0")
        os.makedirs(s0_dir)
        mesh.export(os.path.join(s0_dir, "1.ply"))

        # LOD 1: SAME mesh (not decimated) — should trigger truncation
        s1_dir = os.path.join(mesh_lods, "s1")
        os.makedirs(s1_dir)
        mesh.export(os.path.join(s1_dir, "1.ply"))

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
            lod_0_box_size=None,
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()
        num_lods = struct.unpack("<I", data[24:28])[0]
        # LOD 1 has same face count as LOD 0, so it should be truncated
        assert num_lods == 1

    def test_three_lods_all_valid(self, tmp_output_dir):
        """Three LODs with progressively fewer faces should all be included."""
        output_path = os.path.join(tmp_output_dir, "three_lods")
        mesh_lods = os.path.join(output_path, "mesh_lods")

        mesh = trimesh.creation.icosphere(subdivisions=4, radius=50.0)
        mesh.vertices += 100

        import pyfqmr

        for lod in range(3):
            lod_dir = os.path.join(mesh_lods, f"s{lod}")
            os.makedirs(lod_dir)
            if lod == 0:
                mesh.export(os.path.join(lod_dir, "1.ply"))
            else:
                simplifier = pyfqmr.Simplify()
                simplifier.setMesh(mesh.vertices, mesh.faces)
                target = max(len(mesh.faces) // (4 ** lod), 4)
                simplifier.simplify_mesh(
                    target_count=target, aggressiveness=7,
                    preserve_border=False, verbose=False,
                )
                v, f, _ = simplifier.getMesh()
                trimesh.Trimesh(v, f).export(os.path.join(lod_dir, "1.ply"))

        # Use a small box_size to force multi-chunk so the single-LOD
        # truncation for single-chunk meshes does not apply.
        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1, 2],
            original_ext=".ply",
            lod_0_box_size=np.array([20.0, 20.0, 20.0]),
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()
        num_lods = struct.unpack("<I", data[24:28])[0]
        assert num_lods == 3, f"Expected 3 LODs, got {num_lods}"


class TestIndexFileFormat:
    """Test the neuroglancer multilod_draco index file format."""

    def test_index_lod_scales_are_powers_of_two(self, multires_mesh_dir):
        """lod_scales should be [1, 2, 4, ...] (powers of 2)."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()

        num_lods = struct.unpack("<I", data[24:28])[0]
        lod_scales = struct.unpack(f"<{num_lods}f", data[28:28 + 4 * num_lods])

        for i, scale in enumerate(lod_scales):
            assert scale == pytest.approx(2 ** i), (
                f"LOD {i} scale should be {2**i}, got {scale}"
            )

    def test_index_vertex_offsets_are_zero(self, multires_mesh_dir):
        """vertex_offsets should be all zeros (no per-LOD vertex offset)."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()

        num_lods = struct.unpack("<I", data[24:28])[0]
        vo_start = 28 + 4 * num_lods
        vo_end = vo_start + 4 * num_lods * 3
        vertex_offsets = struct.unpack(f"<{num_lods * 3}f", data[vo_start:vo_end])
        assert all(v == 0.0 for v in vertex_offsets)

    def test_index_has_fragments_at_each_lod(self, multires_mesh_dir):
        """Each LOD should have at least one fragment in the index."""
        output_path = multires_mesh_dir

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0, 1],
            original_ext=".ply",
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()

        num_lods = struct.unpack("<I", data[24:28])[0]
        vo_end = 28 + 4 * num_lods + 4 * num_lods * 3
        nf_end = vo_end + 4 * num_lods
        num_frags_per_lod = struct.unpack(f"<{num_lods}I", data[vo_end:nf_end])

        for lod, nf in enumerate(num_frags_per_lod):
            assert nf >= 1, f"LOD {lod} should have at least 1 fragment, got {nf}"

    def test_per_axis_heuristic_produces_3d_box_size(self, tmp_output_dir):
        """When lod_0_box_size is None, the heuristic should produce per-axis values."""
        output_path = os.path.join(tmp_output_dir, "heuristic_test")
        mesh_lods = os.path.join(output_path, "mesh_lods")

        # Create an elongated mesh (different extents per axis)
        mesh = trimesh.creation.box(extents=[200, 50, 100])
        mesh.vertices += 200
        s0_dir = os.path.join(mesh_lods, "s0")
        os.makedirs(s0_dir)
        mesh.export(os.path.join(s0_dir, "1.ply"))

        generate_neuroglancer_multires_mesh(
            id=1,
            num_subtask_workers=1,
            output_path=output_path,
            lods=[0],
            original_ext=".ply",
            lod_0_box_size=None,
        )

        index_file = os.path.join(output_path, "multires", "1.index")
        with open(index_file, "rb") as f:
            data = f.read()
        chunk_shape = np.array(struct.unpack("<3f", data[0:12]))
        # Chunk shape should be per-axis (not necessarily all equal)
        assert chunk_shape.shape == (3,)
        assert all(cs > 0 for cs in chunk_shape)
