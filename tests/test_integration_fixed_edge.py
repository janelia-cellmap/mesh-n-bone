"""Integration tests for fixed-edge simplification."""

import numpy as np
import pytest
import trimesh

from mesh_n_bone.meshify.fixed_edge import (
    simplify_mesh,
    remove_boundary_vertices,
    detect_seam_vertices,
    denoise_seams_inplace,
    weld_vertices,
    pymeshlab_simplify,
    fqmr_simplify,
)


class TestFixedEdgeSimplification:
    """Test boundary-preserving simplification."""

    def test_simplify_reduces_faces(self):
        """Simplification should reduce face count."""
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        original_faces = len(mesh.faces)

        result = simplify_mesh(
            mesh,
            target_reduction=0.9,
            voxel_size=(1, 1, 1),
            block_size=None,
            fix_edges=False,
        )

        assert len(result.faces) < original_faces
        assert len(result.faces) > 0

    def test_simplify_with_boundary_preservation(self):
        """Fix_edges=True should preserve mesh boundaries."""
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=20.0)
        original_faces = len(mesh.faces)

        result = simplify_mesh(
            mesh,
            target_reduction=0.8,
            voxel_size=(1, 1, 1),
            block_size=None,
            fix_edges=True,
        )

        assert len(result.faces) < original_faces
        assert len(result.faces) > 0

    def test_simplify_with_block_clipping(self):
        """Block-based clipping should remove vertices outside the block."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        # Mesh is centered at origin, extends ~[-10, 10]
        # Set block_size to clip it
        block_size = np.array([15.0, 15.0, 15.0])

        result = simplify_mesh(
            mesh,
            target_reduction=0.5,
            voxel_size=(1, 1, 1),
            block_size=block_size,
            fix_edges=False,
        )

        # Some vertices should have been clipped
        if len(result.faces) > 0:
            assert np.all(result.vertices >= -0.5)
            assert np.all(result.vertices <= block_size.max())

    def test_simplify_empty_mesh(self):
        """Simplifying an empty mesh should return an empty mesh."""
        mesh = trimesh.Trimesh(
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
        )
        result = simplify_mesh(
            mesh, target_reduction=0.9, voxel_size=(1, 1, 1)
        )
        assert len(result.faces) == 0


class TestRemoveBoundaryVertices:
    """Test boundary vertex removal."""

    def test_no_block_size_returns_original(self):
        """Without block_size, mesh should be returned as-is."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=5.0)
        result = remove_boundary_vertices(mesh, voxel_size=(1, 1, 1))
        assert len(result.faces) == len(mesh.faces)

    def test_clipping_removes_vertices(self):
        """Clipping with a small block should remove vertices."""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=10.0)
        # Mesh is centered at origin, extends ~[-10, 10]
        # Small block should clip most of it
        block_size = np.array([8.0, 8.0, 8.0])
        result = remove_boundary_vertices(
            mesh, voxel_size=(1, 1, 1), block_size=block_size
        )
        assert len(result.faces) < len(mesh.faces)


class TestWeldVertices:
    """Test vertex welding."""

    def test_welding_merges_close_vertices(self):
        """Vertices within epsilon should be merged."""
        # Two triangles sharing an edge, but with slightly offset vertices
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, 1, 0],
            [1.0001, 0, 0],  # very close to vertex 1
            [2, 0, 0],
            [1.5, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        result = weld_vertices(mesh, epsilon=0.001)
        assert len(result.vertices) < len(vertices)

    def test_welding_no_merge_when_far(self):
        """Vertices farther than epsilon should not be merged."""
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 1, 0],
            [5, 0, 0], [6, 0, 0], [5.5, 1, 0],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

        result = weld_vertices(mesh, epsilon=0.001)
        # No vertices should be merged
        assert len(result.vertices) == len(vertices)


class TestSeamDetectionAndDenoising:
    """Test seam detection and Taubin denoising."""

    def test_detect_seams_on_sharp_edges(self):
        """A cube has sharp 90-degree edges that should be detected as seams."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        seam_verts = detect_seam_vertices(mesh, angle_degrees=45)
        # Cube has many sharp edges, so many vertices should be seam vertices
        assert len(seam_verts) > 0

    def test_detect_seams_on_smooth_sphere(self):
        """A smooth sphere should have few or no seam vertices at high angle threshold."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        seam_verts = detect_seam_vertices(mesh, angle_degrees=80)
        # Smooth sphere should have very few sharp edges
        assert len(seam_verts) < len(mesh.vertices) * 0.1

    def test_denoise_seams_runs_without_error(self):
        """Seam denoising should run without errors on a cube."""
        mesh = trimesh.creation.box(extents=[10, 10, 10])
        # Should not raise
        denoise_seams_inplace(
            mesh,
            seam_angle_deg=45,
            k_ring=1,
            taubin_iters=3,
        )
        assert len(mesh.faces) > 0


class TestBoundaryAwareReduction:
    """Test that fix_edges caps reduction to avoid high-valence artifacts."""

    @staticmethod
    def _make_partial_block_mesh(block_size, voxel_size=1):
        """Create a mesh that only partially fills a block.

        The mesh is an open hemisphere occupying one corner of the block,
        resulting in a large open boundary after clipping — the scenario
        that triggers high-valence artifacts without the cap.
        """
        import pymeshlab
        # Create a sphere that extends beyond the block on three sides
        sphere = trimesh.creation.icosphere(subdivisions=4, radius=block_size * 0.8)
        # Shift so most of it is inside [0, block_size]
        sphere.vertices += block_size * 0.4
        return sphere

    def test_fix_edges_caps_target_faces(self):
        """With fix_edges=True, target_faces should be capped based on boundary count."""
        block_size = np.array([30.0, 30.0, 30.0])
        mesh = self._make_partial_block_mesh(30.0)

        result = simplify_mesh(
            mesh,
            target_reduction=0.95,
            voxel_size=(1, 1, 1),
            block_size=block_size,
            fix_edges=True,
        )

        # With the cap, max valence should stay reasonable (< 30)
        if len(result.vertices) > 0:
            valences = np.array(
                [len(result.vertex_neighbors[i]) for i in range(len(result.vertices))]
            )
            assert valences.max() < 30, (
                f"Max valence {valences.max()} exceeds limit; "
                "boundary-aware cap may not be working"
            )

    def test_fix_edges_aggressive_reduction_no_spikes(self):
        """Aggressive reduction with fix_edges should not create spike vertices."""
        block_size = np.array([30.0, 30.0, 30.0])
        mesh = self._make_partial_block_mesh(30.0)

        result = simplify_mesh(
            mesh,
            target_reduction=0.9,
            voxel_size=(1, 1, 1),
            block_size=block_size,
            fix_edges=True,
        )

        if len(result.vertices) > 3 and len(result.faces) > 1:
            # Check for spike vertices (face normals pointing opposite directions)
            vertex_faces = result.vertex_faces
            spike_count = 0
            for vi in range(len(result.vertices)):
                faces = vertex_faces[vi]
                faces = faces[faces >= 0]
                if len(faces) < 3:
                    continue
                fnormals = result.face_normals[faces]
                mean_n = fnormals.mean(axis=0)
                norm = np.linalg.norm(mean_n)
                if norm < 1e-10:
                    continue
                mean_n /= norm
                dots = np.dot(fnormals, mean_n)
                if dots.min() < -0.5:
                    spike_count += 1
            assert spike_count == 0, f"Found {spike_count} spike vertices"

    def test_no_cap_without_fix_edges(self):
        """Without fix_edges, reduction should not be capped."""
        mesh = trimesh.creation.icosphere(subdivisions=4, radius=10.0)
        original_faces = len(mesh.faces)

        result = simplify_mesh(
            mesh,
            target_reduction=0.95,
            voxel_size=(1, 1, 1),
            block_size=None,
            fix_edges=False,
        )

        # Should achieve close to the requested reduction
        assert len(result.faces) < 0.15 * original_faces


class TestSimplificationBackends:
    """Test both pymeshlab and fqmr simplification backends."""

    def test_pymeshlab_simplify(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        target_faces = len(mesh.faces) // 4

        v_out, f_out = pymeshlab_simplify(
            mesh.vertices, mesh.faces, target_faces=target_faces
        )

        assert len(f_out) > 0
        assert len(f_out) <= target_faces * 1.5  # allow some tolerance

    def test_fqmr_simplify(self):
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=10.0)
        target_faces = len(mesh.faces) // 4

        v_out, f_out = fqmr_simplify(
            mesh.vertices, mesh.faces,
            target_faces=target_faces,
            preserve_border=False,
        )

        assert len(f_out) > 0
        assert len(f_out) <= target_faces * 1.5
