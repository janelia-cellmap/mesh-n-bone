"""Unit + small-integration tests for lossless_simplify.

Covers the contract that adjacent blocks running ``lossless_chunk_simplify``
independently produce *identical* vertex sets on their shared chunk-boundary
plane — the property that makes seamless assembly possible without
cross-block communication.
"""

import numpy as np
import pytest
import trimesh

from mesh_n_bone.meshify.lossless_simplify import (
    collapse_planar_vertices,
    simplify_boundary_polylines,
    lossless_chunk_simplify,
    block_boundary_planes,
)


def _flat_grid(n=20, plane="z", value=0.0):
    """An NxN flat grid in the chosen plane, triangulated as 2 tris per quad."""
    xs, ys = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    flat = np.full_like(xs, value, dtype=float)
    if plane == "z":
        verts = np.column_stack([xs.ravel(), ys.ravel(), flat.ravel()])
    elif plane == "y":
        verts = np.column_stack([xs.ravel(), flat.ravel(), ys.ravel()])
    else:
        verts = np.column_stack([flat.ravel(), xs.ravel(), ys.ravel()])
    faces = []
    for i in range(n):
        for j in range(n):
            a = i * (n + 1) + j
            b = a + 1
            c = a + (n + 1)
            d = c + 1
            faces.append([a, b, d])
            faces.append([a, d, c])
    return np.asarray(verts, dtype=np.float64), np.asarray(faces, dtype=np.int64)


class TestCoplanarCollapse:
    def test_flat_grid_collapses_to_single_polygon(self):
        v, f = _flat_grid(n=10)
        # 10x10 grid = 200 triangles. Lossless collapse via 1-ring planarity
        # removes all interior vertices and triangulates the boundary polygon.
        nv, nf = collapse_planar_vertices(v, f)
        # The remaining vertices should be exactly the boundary loop.
        assert len(nv) == 40  # 4 * 10 boundary verts of an 11x11 grid
        # Triangulating a 40-vertex polygon = 38 triangles
        assert len(nf) == 38
        # Surface still planar at z = 0
        assert np.all(np.abs(nv[:, 2]) < 1e-9)

    def test_lock_vertex_mask_preserves_locked(self):
        v, f = _flat_grid(n=10)
        # Lock the central vertex
        center_xy = (5, 5)
        center_idx = center_xy[0] * 11 + center_xy[1]
        lock = np.zeros(len(v), dtype=bool)
        lock[center_idx] = True
        nv, nf = collapse_planar_vertices(v, f, lock_vertex_mask=lock)
        # Center vertex must survive
        center_pos = v[center_idx]
        assert any(np.allclose(p, center_pos) for p in nv)

    def test_cube_unchanged(self):
        box = trimesh.creation.box(extents=[1, 1, 1])
        v = np.asarray(box.vertices, dtype=np.float64)
        f = np.asarray(box.faces, dtype=np.int64)
        nv, nf = collapse_planar_vertices(v, f)
        # 6 faces × 2 triangles = 12. Faces are mutually perpendicular, no
        # interior vertex shares a planar 1-ring → nothing collapses.
        assert len(nf) == 12

    def test_mc_cube_collapses_to_near_minimum_with_feature_edges(self):
        """With the (opt-in) feature-edge extension, an MC cube collapses
        to near the 8-vert/12-face theoretical minimum. Off by default
        because it damages NG per-vertex normal averaging on curved
        surfaces; on for cube-like data."""
        from zmesh import Mesher
        vol = np.zeros((40, 40, 40), dtype=np.uint32)
        vol[5:35, 5:35, 5:35] = 1
        mesher = Mesher((1.0, 1.0, 1.0))
        mesher.mesh(vol, close=False)
        m = mesher.get(1, normals=False)
        v = np.asarray(m.vertices, dtype=np.float64)[:, ::-1]
        f = np.asarray(m.faces, dtype=np.int64)
        nv, nf = collapse_planar_vertices(v, f, collapse_feature_edges=True)
        assert len(nf) < 100, (
            f"MC box should collapse to near-minimum; got {len(nf)} faces"
        )

    def test_mc_sphere_pure_lossless_is_bit_exact(self):
        """Pure Pass A on a curved MC surface invents no new vertex
        positions: every surviving vertex is one of the originals.
        The geometry is bit-exact even though NG's *unweighted* face-
        normal averaging means per-vertex shading may shift slightly
        (it depends on triangle counts, and retriangulation changes
        them — a fundamental property of the NG renderer, not the
        simplifier)."""
        from zmesh import Mesher
        vol = np.zeros((40, 40, 40), dtype=np.uint32)
        zz, yy, xx = np.indices(vol.shape) - 20
        vol[(xx*xx + yy*yy + zz*zz) < 12*12] = 1
        mesher = Mesher((1.0, 1.0, 1.0))
        mesher.mesh(vol, close=False)
        m = mesher.get(1, normals=False)
        v = np.asarray(m.vertices, dtype=np.float64)[:, ::-1]
        f = np.asarray(m.faces, dtype=np.int64)
        nv, nf = collapse_planar_vertices(v, f)
        in_orig = set(map(tuple, np.round(v, 6).tolist()))
        in_out = set(map(tuple, np.round(nv, 6).tolist()))
        assert in_out.issubset(in_orig)


class TestDouglasPeucker:
    """The DP implementation must be a pure function of its 2D polyline
    input — the same polyline produces the same output regardless of how
    you got there (i.e., regardless of which block built it)."""

    def test_collinear_polyline_collapses_to_endpoints(self):
        from mesh_n_bone.meshify.lossless_simplify import _douglas_peucker_2d
        pts = np.array([[float(i), 0.0] for i in range(11)])
        keep = _douglas_peucker_2d(pts, eps=0.5)
        assert keep.sum() == 2  # only the two endpoints survive
        assert keep[0] and keep[-1]

    def test_deterministic_independent_of_block(self):
        """Same input → same output, byte-identical. This is the property
        that makes adjacent blocks agree without communication."""
        from mesh_n_bone.meshify.lossless_simplify import _douglas_peucker_2d
        pts = np.array([
            [0.0, 0.0], [1.0, 0.1], [2.0, 0.0], [3.0, 5.0],
            [4.0, 0.0], [5.0, 0.0], [6.0, 0.0],
        ])
        k1 = _douglas_peucker_2d(pts, eps=0.5)
        k2 = _douglas_peucker_2d(pts.copy(), eps=0.5)
        np.testing.assert_array_equal(k1, k2)
        # The spike at (3, 5) must survive
        assert k1[3]


class TestEndToEndChunkSimplify:
    def test_solid_box_two_blocks_stitch(self):
        """Two adjacent blocks of a solid box: lossless_chunk_simplify on
        each block produces identical boundary vertices."""
        from zmesh import Mesher

        vol = np.zeros((20, 20, 20), dtype=np.uint32)
        vol[2:18, 2:18, 2:18] = 1
        # Split at x = 10
        pad = 1
        vol_a = vol[:, :, : 10 + pad]
        vol_b = vol[:, :, 10 - pad :]

        def mc(v, x_origin):
            mesher = Mesher((1, 1, 1))
            mesher.mesh(v.astype(np.uint32), close=False)
            m = mesher.get(1, normals=False)
            verts = np.asarray(m.vertices, dtype=np.float64)[:, ::-1]
            verts[:, 0] += x_origin
            return verts, np.asarray(m.faces, dtype=np.int64)

        v_a, f_a = mc(vol_a, 0.0)
        v_b, f_b = mc(vol_b, 10.0 - pad)
        # Clip each at the shared plane x=10
        ma = trimesh.intersections.slice_mesh_plane(
            trimesh.Trimesh(v_a, f_a, process=False),
            plane_normal=[-1, 0, 0], plane_origin=[10, 0, 0], cap=False,
        )
        mb = trimesh.intersections.slice_mesh_plane(
            trimesh.Trimesh(v_b, f_b, process=False),
            plane_normal=[1, 0, 0], plane_origin=[10, 0, 0], cap=False,
        )
        v_a, f_a = np.asarray(ma.vertices), np.asarray(ma.faces, dtype=np.int64)
        v_b, f_b = np.asarray(mb.vertices), np.asarray(mb.faces, dtype=np.int64)

        # Run lossless on each block, locking the shared plane
        plane = [(0, 10.0)]
        v_a_s, f_a_s = lossless_chunk_simplify(v_a, f_a, boundary_planes=plane)
        v_b_s, f_b_s = lossless_chunk_simplify(v_b, f_b, boundary_planes=plane)

        # Boundary vertex sets must match
        def on_plane_set(v, axis, value, tol=1e-3):
            on = np.abs(v[:, axis] - value) < tol
            return set(map(tuple, np.round(v[on] / tol).astype(np.int64).tolist()))

        p_a = on_plane_set(v_a_s, 0, 10.0)
        p_b = on_plane_set(v_b_s, 0, 10.0)
        assert p_a == p_b, (
            f"boundary mismatch after lossless: "
            f"only-in-0={p_a - p_b}, only-in-1={p_b - p_a}"
        )

        # Both blocks should have significantly fewer faces than raw MC
        assert len(f_a_s) < 0.5 * len(f_a)
        assert len(f_b_s) < 0.5 * len(f_b)


def test_block_boundary_planes_helper():
    origin = np.array([10.0, 20.0, 30.0])
    size = np.array([5.0, 6.0, 7.0])
    bp = block_boundary_planes(origin, size)
    assert (0, 10.0) in bp and (0, 15.0) in bp
    assert (1, 20.0) in bp and (1, 26.0) in bp
    assert (2, 30.0) in bp and (2, 37.0) in bp
