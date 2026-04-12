"""Tests for mesh_io utilities."""

import numpy as np
import os
import pytest
import tempfile
import trimesh

from mesh_n_bone.util.mesh_io import (
    Fragment,
    CompressedFragment,
    mesh_loader,
    zorder_fragments,
)


class TestMeshLoader:
    def test_load_ply(self, tiny_cube_mesh, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "test.ply")
        tiny_cube_mesh.export(path)
        vertices, faces = mesh_loader(path)
        assert vertices is not None
        assert faces is not None
        assert vertices.shape[1] == 3
        assert faces.shape[1] == 3

    def test_load_obj(self, tiny_cube_mesh, tmp_output_dir):
        path = os.path.join(tmp_output_dir, "test.obj")
        tiny_cube_mesh.export(path)
        vertices, faces = mesh_loader(path)
        assert vertices is not None
        assert faces is not None

    def test_load_nonexistent_returns_none(self):
        vertices, faces = mesh_loader("/nonexistent/path/mesh.ply")
        assert vertices is None
        assert faces is None


class TestFragment:
    def test_fragment_creation(self, sample_vertices_and_faces):
        vertices, faces = sample_vertices_and_faces
        frag = Fragment(vertices, faces, [(0, 0, 0)])
        assert np.array_equal(frag.vertices, vertices)
        assert np.array_equal(frag.faces, faces)
        assert frag.lod_0_fragment_pos == [(0, 0, 0)]

    def test_fragment_update(self, sample_vertices_and_faces):
        vertices, faces = sample_vertices_and_faces
        frag = Fragment(vertices, faces, [(0, 0, 0)])
        frag.update(vertices + 10, faces, (1, 1, 1))
        assert len(frag.vertices) == 8  # 4 + 4
        assert len(frag.faces) == 8  # 4 + 4
        assert (1, 1, 1) in frag.lod_0_fragment_pos


class TestZorderFragments:
    def test_sorting(self):
        frag_a = CompressedFragment(
            draco_bytes=b"a",
            position=np.array([0, 0, 1]),
            offset=1,
            lod_0_positions=np.array([[0, 0, 1]]),
        )
        frag_b = CompressedFragment(
            draco_bytes=b"b",
            position=np.array([0, 0, 0]),
            offset=1,
            lod_0_positions=np.array([[0, 0, 0]]),
        )
        sorted_frags = zorder_fragments([frag_a, frag_b])
        # (0,0,0) should come before (0,0,1) in z-order
        assert np.array_equal(sorted_frags[0].position, np.array([0, 0, 0]))
