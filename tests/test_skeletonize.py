"""Tests for skeletonization module."""

import numpy as np
import pytest

from mesh_n_bone.skeletonize.skeleton import CustomSkeleton, Source


class TestCustomSkeleton:
    def test_create_simple_skeleton(self):
        vertices = [(0, 0, 0), (1, 0, 0), (2, 0, 0)]
        edges = [(0, 1), (1, 2)]
        skel = CustomSkeleton(vertices, edges)
        assert len(skel.vertices) == 3
        assert len(skel.edges) == 2
        assert len(skel.polylines) > 0

    def test_skeleton_to_graph(self):
        vertices = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        edges = [(0, 1), (1, 2)]
        skel = CustomSkeleton(vertices, edges)
        g = skel.skeleton_to_graph()
        assert g.number_of_nodes() == 3
        assert g.number_of_edges() == 2

    def test_prune_short_branch(self):
        # Create a T-shaped skeleton
        # Main branch: 0 -- 1 -- 2 (length 20)
        # Short side branch: 1 -- 3 (length 1)
        vertices = [(0, 0, 0), (10, 0, 0), (20, 0, 0), (10, 1, 0)]
        edges = [(0, 1), (1, 2), (1, 3)]
        radii = [1.0, 1.0, 1.0, 1.0]
        skel = CustomSkeleton(vertices, edges, radii)

        pruned = skel.prune(min_branch_length_nm=5)
        # The short branch (length 1) should be pruned
        assert len(pruned.vertices) == 3

    def test_simplify(self):
        # Straight line with many points
        vertices = [(i, 0, 0) for i in range(10)]
        edges = [(i, i + 1) for i in range(9)]
        radii = [1.0] * 10
        polylines = [np.array(vertices)]
        skel = CustomSkeleton(vertices, edges, radii, polylines)

        simplified = skel.simplify(tolerance_nm=0.5)
        # Straight line should simplify to just endpoints
        assert len(simplified.vertices) <= len(skel.vertices)

    def test_lineseg_dists(self):
        p = np.array([[0, 1, 0]])
        a = np.array([0, 0, 0])
        b = np.array([1, 0, 0])
        dists = CustomSkeleton.lineseg_dists(p, a, b)
        np.testing.assert_almost_equal(dists[0], 1.0)

    def test_find_branchpoints_and_endpoints(self):
        import networkx as nx

        g = nx.Graph()
        # T-shape: 0-1-2, 1-3
        g.add_edges_from([(0, 1), (1, 2), (1, 3)])
        branchpoints, endpoints = CustomSkeleton.find_branchpoints_and_endpoints(g)
        assert 1 in branchpoints
        assert set(endpoints) == {0, 2, 3}


class TestNeuroglancerSkeletonIO:
    def test_write_and_read(self, tmp_output_dir):
        import os

        vertices = [(0, 0, 0), (10, 0, 0), (20, 0, 0)]
        edges = [(0, 1), (1, 2)]
        skel = CustomSkeleton(vertices, edges)

        path = os.path.join(tmp_output_dir, "skel", "test_skel")
        skel.write_neuroglancer_skeleton(path)

        verts_read, edges_read = CustomSkeleton.read_neuroglancer_skeleton(path)
        assert verts_read.shape == (3, 3)
        assert edges_read.shape == (2, 2)


class TestSource:
    def test_source_default(self):
        s = Source()
        assert s.vertex_attributes == []
