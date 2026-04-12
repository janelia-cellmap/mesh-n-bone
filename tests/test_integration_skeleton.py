"""Integration tests for the skeletonization pipeline."""

import numpy as np
import os
import pytest
import trimesh

from mesh_n_bone.skeletonize.skeleton import CustomSkeleton
from mesh_n_bone.skeletonize.skeletonize import Skeletonize


# Path to the compiled CGAL skeletonizer binary
_CGAL_BINARY = os.path.join(
    os.path.dirname(__file__), os.pardir,
    "cgal_skeletonize_mesh", "skeletonize_mesh",
)
_HAS_CGAL = os.path.isfile(_CGAL_BINARY) and os.access(_CGAL_BINARY, os.X_OK)


class TestCGALSkeletonization:
    """End-to-end tests that run the compiled CGAL skeletonizer binary."""

    @pytest.mark.skipif(not _HAS_CGAL, reason="CGAL binary not built")
    def test_sphere_produces_skeleton(self, tmp_output_dir):
        """Skeletonizing a sphere should produce a skeleton with vertices."""
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=50.0)
        mesh.vertices += 100
        ply_path = os.path.join(tmp_output_dir, "sphere.ply")
        skel_path = os.path.join(tmp_output_dir, "sphere_skel.txt")
        mesh.export(ply_path)

        Skeletonize.cgal_skeletonize_mesh(
            ply_path, skel_path, base_loop_subdivision_iterations=0,
        )

        assert os.path.exists(skel_path)
        skel = Skeletonize.read_skeleton_from_custom_file(skel_path)
        assert len(skel.vertices) > 0
        assert len(skel.edges) > 0

    @pytest.mark.skipif(not _HAS_CGAL, reason="CGAL binary not built")
    def test_tube_skeleton_is_linear(self, tmp_output_dir):
        """Skeletonizing a tube should produce a roughly linear skeleton."""
        mesh = trimesh.creation.cylinder(radius=5.0, height=100.0, sections=16)
        mesh.vertices += [50, 50, 50]
        ply_path = os.path.join(tmp_output_dir, "tube.ply")
        skel_path = os.path.join(tmp_output_dir, "tube_skel.txt")
        mesh.export(ply_path)

        Skeletonize.cgal_skeletonize_mesh(
            ply_path, skel_path, base_loop_subdivision_iterations=0,
        )

        skel = Skeletonize.read_skeleton_from_custom_file(skel_path)
        assert len(skel.vertices) >= 2
        # The skeleton should span roughly the tube's height
        verts = np.array(skel.vertices)
        extent = verts.max(axis=0) - verts.min(axis=0)
        assert extent.max() > 50  # at least half the tube height

    @pytest.mark.skipif(not _HAS_CGAL, reason="CGAL binary not built")
    def test_full_pipeline_with_cgal(self, tmp_output_dir):
        """Full pipeline: CGAL skeletonize → prune → simplify → metrics."""
        mesh = trimesh.creation.cylinder(radius=5.0, height=80.0, sections=16)
        mesh.vertices += 100
        ply_path = os.path.join(tmp_output_dir, "pipe.ply")
        skel_path = os.path.join(tmp_output_dir, "pipe_skel.txt")
        mesh.export(ply_path)

        Skeletonize.cgal_skeletonize_mesh(
            ply_path, skel_path, base_loop_subdivision_iterations=0,
        )

        output_dir = os.path.join(tmp_output_dir, "pipe_output")
        metrics = Skeletonize.process_custom_skeleton(
            skeleton_path=skel_path,
            output_directory=output_dir,
            min_branch_length_nm=5.0,
            simplification_tolerance_nm=1.0,
        )

        assert metrics["lsp (nm)"] > 30
        assert metrics["radius mean (nm)"] > 0
        assert metrics["num branches"] >= 1


class TestSkeletonProcessingPipeline:
    """Test skeleton creation → pruning → simplification → metrics."""

    def _make_t_skeleton(self):
        """Create a T-shaped skeleton for testing.
        Main branch: (0,0,0)→(50,0,0)→(100,0,0) with many intermediate points.
        Side branch: (50,0,0)→(50,2,0) — short, should be pruned.
        """
        # Main branch with many intermediate points (for simplification testing)
        main_pts = [(i, 0, 0) for i in range(0, 101, 5)]
        # Short side branch
        side_pts = [(50, 1, 0), (50, 2, 0)]
        vertices = main_pts + side_pts
        n_main = len(main_pts)

        edges = [(i, i + 1) for i in range(n_main - 1)]
        # Connect junction (vertex at x=50, index=10) to side branch
        junction_idx = 10  # (50,0,0)
        edges.append((junction_idx, n_main))
        edges.append((n_main, n_main + 1))

        radii = [1.0] * len(vertices)
        return CustomSkeleton(vertices, edges, radii)

    def _make_y_skeleton(self):
        """Y-shaped skeleton: two long branches meeting at a junction.
        Useful for testing longest shortest path."""
        # Stem: 0→50 along x
        stem = [(i, 0, 0) for i in range(0, 51, 10)]
        # Branch A: (50,0,0)→(100,50,0)
        branch_a = [(50 + i, i, 0) for i in range(10, 51, 10)]
        # Branch B: (50,0,0)→(100,-50,0)
        branch_b = [(50 + i, -i, 0) for i in range(10, 51, 10)]

        vertices = stem + branch_a + branch_b
        n_stem = len(stem)
        n_a = len(branch_a)

        edges = [(i, i + 1) for i in range(n_stem - 1)]
        junction_idx = n_stem - 1
        # Connect junction to branch A
        edges.append((junction_idx, n_stem))
        edges.extend((n_stem + i, n_stem + i + 1) for i in range(n_a - 1))
        # Connect junction to branch B
        edges.append((junction_idx, n_stem + n_a))
        edges.extend(
            (n_stem + n_a + i, n_stem + n_a + i + 1) for i in range(len(branch_b) - 1)
        )

        radii = [2.0] * len(vertices)
        return CustomSkeleton(vertices, edges, radii)

    def test_pruning_removes_short_branches(self):
        """Pruning should remove branches shorter than the threshold."""
        skel = self._make_t_skeleton()
        pruned = skel.prune(min_branch_length_nm=5.0)

        # The short branch (length 2) should be removed
        assert len(pruned.vertices) < len(skel.vertices)
        # Main branch endpoints should still exist
        vertex_xs = [v[0] for v in pruned.vertices]
        assert 0 in vertex_xs
        assert 100 in vertex_xs

    def test_pruning_keeps_long_branches(self):
        """Pruning with a low threshold should keep all branches."""
        skel = self._make_t_skeleton()
        pruned = skel.prune(min_branch_length_nm=0.5)

        # Side branch (length 2) should be kept with threshold 0.5
        assert len(pruned.vertices) == len(skel.vertices)

    def test_simplification_reduces_points(self):
        """RDP simplification should reduce collinear points."""
        skel = self._make_t_skeleton()
        pruned = skel.prune(min_branch_length_nm=5.0)
        simplified = pruned.simplify(tolerance_nm=1.0)

        # Main branch is a straight line — should simplify to ~2 points
        assert len(simplified.vertices) < len(pruned.vertices)
        # Endpoints should be preserved
        vertex_arr = np.array(simplified.vertices)
        assert np.any(np.all(np.isclose(vertex_arr, [0, 0, 0]), axis=1))
        assert np.any(np.all(np.isclose(vertex_arr, [100, 0, 0]), axis=1))

    def test_longest_shortest_path(self):
        """LSP of a Y-skeleton should be the path between the two branch tips."""
        skel = self._make_y_skeleton()
        lsp = Skeletonize.get_longest_shortest_path_distance(skel)

        # The longest shortest path goes from branch A tip to branch B tip,
        # passing through the stem. Should be > 100 (two branches of ~70 each)
        assert lsp > 100

    def test_neuroglancer_roundtrip(self, tmp_output_dir):
        """Write and read back a skeleton in neuroglancer format."""
        skel = self._make_y_skeleton()
        path = os.path.join(tmp_output_dir, "skel", "test_y")
        skel.write_neuroglancer_skeleton(path)

        verts, edges = CustomSkeleton.read_neuroglancer_skeleton(path)
        assert verts.shape[0] == len(skel.vertices)
        assert verts.shape[1] == 3
        assert edges.shape[1] == 2

    def test_full_skeleton_processing_pipeline(self, tmp_output_dir):
        """Full pipeline: skeleton → prune → simplify → metrics → write."""
        skel = self._make_y_skeleton()

        # Write skeleton to file in custom format
        skel_path = os.path.join(tmp_output_dir, "skeleton.txt")
        with open(skel_path, "w") as f:
            for v, r in zip(skel.vertices, skel.radii):
                f.write(f"v {v[0]} {v[1]} {v[2]} {r}\n")
            for e in skel.edges:
                f.write(f"e {e[0]} {e[1]}\n")

        # Read back
        read_skel = Skeletonize.read_skeleton_from_custom_file(skel_path)
        assert len(read_skel.vertices) == len(skel.vertices)
        assert len(read_skel.edges) == len(skel.edges)

        # Process through full pipeline
        output_dir = os.path.join(tmp_output_dir, "output")
        metrics = Skeletonize.process_custom_skeleton(
            skeleton_path=skel_path,
            output_directory=output_dir,
            min_branch_length_nm=5.0,
            simplification_tolerance_nm=1.0,
        )

        assert "lsp (nm)" in metrics
        assert metrics["lsp (nm)"] > 0
        assert "radius mean (nm)" in metrics
        assert metrics["radius mean (nm)"] > 0
        assert "num branches" in metrics
        assert metrics["num branches"] >= 1

        # Verify output files exist
        skel_id = os.path.basename(skel_path).split(".")[0]
        assert os.path.exists(os.path.join(output_dir, "skeleton", "full", skel_id))
        assert os.path.exists(
            os.path.join(output_dir, "skeleton", "simplified", skel_id)
        )


class TestSkeletonMetrics:
    """Test skeleton metric calculations."""

    def test_straight_line_lsp(self):
        """LSP of a straight line should equal its length."""
        vertices = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        edges = [(0, 1), (1, 2)]
        skel = CustomSkeleton(vertices, edges)

        lsp = Skeletonize.get_longest_shortest_path_distance(skel)
        np.testing.assert_almost_equal(lsp, 20.0)

    def test_radius_statistics(self):
        """Radius statistics should reflect the input radii."""
        vertices = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        edges = [(0, 1), (1, 2)]
        radii = [2.0, 4.0, 6.0]
        skel = CustomSkeleton(vertices, edges, radii)

        np.testing.assert_almost_equal(np.mean(skel.radii), 4.0)
        np.testing.assert_almost_equal(np.std(skel.radii), np.std([2, 4, 6]))
