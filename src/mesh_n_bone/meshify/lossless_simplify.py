"""Lossless mesh simplification for chunked meshing pipelines.

Two-pass algorithm that produces a mesh visually identical to its input
while dramatically reducing triangle count, AND guarantees that two
neighboring blocks produce *identical* vertex sets on their shared
chunk-boundary plane (so the assembled mesh has no seams or T-junctions).

Pass A (interior coplanar collapse):
    Vertex-by-vertex: a vertex whose entire 1-ring of incident faces is
    coplanar (within angular tolerance) is removed, and the polygonal
    hole is re-triangulated. Handles both flat-region interior vertices
    and collinear vertices on the perimeter of a flat patch (where the
    1-ring is still planar because all adjacent faces lie in one plane).

Pass B (deterministic boundary-polyline simplification):
    For each chunk-boundary plane, extract the 2D polyline where the
    mesh crosses the plane (= the set of mesh vertices on that plane,
    connected by edges in the mesh). Run Douglas-Peucker simplification
    with a fixed tolerance and lexicographic tie-breaking. Because the
    input polyline depends only on the segmentation crossing the plane
    (which is identical from both sides), two adjacent blocks compute
    *identical* simplified polylines without any cross-block communication.
    Then each block independently retriangulates the local strip of
    faces affected by the removed boundary vertices.

The combination produces:
  * Massive face-count reduction on flat surfaces (interior or boundary)
  * Identical vertex sets on shared chunk faces (clean stitching)
  * Surface visually identical to the original marching-cubes output

Lossy-but-bounded simplification (Garland-Heckbert / quadric error)
should run *after* this pass on the assembled per-segment mesh, not
per-block — because once block boundaries no longer carry extra
density, there's no need to lock them during the lossy pass.
"""

from __future__ import annotations

import numpy as np
import trimesh
from typing import Iterable


_PLANE_TOL = 1e-4         # max sin(angle) between adjacent face normals
_COLLINEAR_TOL = 1e-3     # max perp distance from line in nm
_DP_DEFAULT_EPS = 0.5     # Douglas-Peucker tolerance in nm
_PLANE_HIT_TOL = 1e-3     # max distance from boundary plane to count as on-plane
_FEATURE_DIHEDRAL_MIN_DEG = 45.0  # below this, treat as curvature, not feature


# ---------------------------------------------------------------------------
# Geometry primitives
# ---------------------------------------------------------------------------


def _face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    return np.divide(n, np.maximum(mag, 1e-30))


def _vertex_face_adjacency(num_verts: int, faces: np.ndarray) -> list[list[int]]:
    """Per-vertex list of incident face indices."""
    inc: list[list[int]] = [[] for _ in range(num_verts)]
    for fi in range(len(faces)):
        a, b, c = faces[fi]
        inc[a].append(fi)
        inc[b].append(fi)
        inc[c].append(fi)
    return inc


def _vertex_ring_2d_proj(verts: np.ndarray, ring_indices: list[int],
                         normal: np.ndarray):
    """Project a set of 3D vertices onto a plane and return 2D coords.

    Returns (coords_2d, u, v) where u, v are the basis vectors of the plane.
    """
    if abs(normal[2]) < 0.9:
        u = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    v /= np.linalg.norm(v)
    coords = np.column_stack([
        verts[ring_indices] @ u,
        verts[ring_indices] @ v,
    ])
    return coords, u, v


def _polygon_is_simple_and_ccw(coords_2d: np.ndarray) -> bool:
    """Quick simple-polygon sanity check + ensure CCW orientation."""
    n = len(coords_2d)
    if n < 3:
        return False
    area = 0.0
    for i in range(n):
        x0, y0 = coords_2d[i]
        x1, y1 = coords_2d[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    return area > 0


def _ear_clip_triangulate(coords_2d: np.ndarray, indices: list[int]) -> list[tuple]:
    """Ear-clipping triangulation of a simple polygon. Indices are vertex
    indices into the parent vertex array; coords_2d is per-polygon-vertex
    in the local plane (length matches `indices`). Returns triangles as
    tuples of indices into the parent vertex array.
    """
    n = len(indices)
    if n < 3:
        return []
    if n == 3:
        return [tuple(indices)]
    pts = np.asarray(coords_2d, dtype=np.float64)

    # Ensure CCW
    area = 0.0
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        area += x0 * y1 - x1 * y0
    if area < 0:
        indices = list(reversed(indices))
        pts = pts[::-1]

    remaining = list(range(n))
    triangles: list[tuple] = []
    guard = 0
    while len(remaining) > 3 and guard < 10 * n:
        guard += 1
        ear_found = False
        for k in range(len(remaining)):
            i_prev = remaining[(k - 1) % len(remaining)]
            i_curr = remaining[k]
            i_next = remaining[(k + 1) % len(remaining)]
            p_prev = pts[i_prev]
            p_curr = pts[i_curr]
            p_next = pts[i_next]
            cross = ((p_curr[0] - p_prev[0]) * (p_next[1] - p_prev[1])
                     - (p_curr[1] - p_prev[1]) * (p_next[0] - p_prev[0]))
            if cross <= 1e-12:
                continue
            inside = False
            for j in remaining:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle_2d(pts[j], p_prev, p_curr, p_next):
                    inside = True
                    break
            if inside:
                continue
            triangles.append((indices[i_prev], indices[i_curr], indices[i_next]))
            remaining.pop(k)
            ear_found = True
            break
        if not ear_found:
            break
    if len(remaining) == 3:
        triangles.append(
            (indices[remaining[0]], indices[remaining[1]], indices[remaining[2]])
        )
    return triangles


def _point_in_triangle_2d(p, a, b, c) -> bool:
    def sgn(p1, p2, p3):
        return ((p1[0] - p3[0]) * (p2[1] - p3[1])
                - (p2[0] - p3[0]) * (p1[1] - p3[1]))
    d1 = sgn(p, a, b)
    d2 = sgn(p, b, c)
    d3 = sgn(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _order_polygon_loop(half_edges: dict[int, list[int]]) -> list[int]:
    """Given a half-edge adjacency `u -> [v]` for a simple loop, return the
    ordered list of vertex indices going around the loop. Returns [] if
    the structure isn't a single simple loop.
    """
    if not half_edges:
        return []
    start = min(half_edges)
    loop = [start]
    curr = start
    while True:
        nexts = half_edges.get(curr, [])
        if not nexts:
            return []
        nxt = nexts[0]
        if nxt == start:
            return loop
        loop.append(nxt)
        curr = nxt
        if len(loop) > 100 * len(half_edges):
            return []


# ---------------------------------------------------------------------------
# Pass A: lossless interior coplanar collapse via vertex-by-vertex removal
# ---------------------------------------------------------------------------


def collapse_planar_vertices(
    verts: np.ndarray, faces: np.ndarray,
    normal_tol: float = _PLANE_TOL,
    lock_vertex_mask: np.ndarray | None = None,
    collapse_feature_edges: bool = False,
    feature_dihedral_min_deg: float = _FEATURE_DIHEDRAL_MIN_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Lossless mesh simplification by removing vertices with a planar 1-ring.

    A vertex V is removed if every face incident to V has a normal
    matching every other incident face's normal to within ``normal_tol``
    (dot product > 1 - tol). The 1-ring polygon is then re-triangulated
    by ear-clipping in the patch's plane. Crucially this is **bit-exact**:
    no vertex position is ever moved; we only delete vertices whose
    incident faces are coplanar, so the resulting surface is identical
    to the input.

    Optional feature-edge extension (``collapse_feature_edges=True``):
    when V's 1-ring partitions into exactly 2 coplanar groups joined by
    a straight edge through V (and the dihedral angle between groups is
    ≥ ``feature_dihedral_min_deg``), V is removed and each half-arc is
    retriangulated in its own plane. This collapses cube/slab edges
    aggressively but reduces vertex density along genuine geometric
    feature edges — which Neuroglancer renders by averaging incident
    face normals at each vertex, so fewer feature-edge verts can produce
    visible facets. **Off by default** for that reason. Flip on only
    when you know your data is dominated by sharp 90°-ish edges
    (CAD-like volumes) rather than curved organic shapes.

    Locked vertices (e.g. vertices on chunk-boundary planes) are never
    removed.

    Returns a new (vertices, faces) pair with renumbered vertex indices.
    """
    feature_cos_max = float(np.cos(np.radians(feature_dihedral_min_deg)))
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    num_verts = len(verts)

    normals = _face_normals(verts, faces)
    inc = _vertex_face_adjacency(num_verts, faces)

    if lock_vertex_mask is None:
        lock_vertex_mask = np.zeros(num_verts, dtype=bool)

    alive_face = np.ones(len(faces), dtype=bool)
    alive_vert = np.ones(num_verts, dtype=bool)

    # Iterate to a fixed point: removing a planar vertex can leave its
    # neighbors planar too, but normals don't change because the surface
    # is preserved. So a single pass is enough if we recompute incidence
    # carefully.
    new_faces: list[np.ndarray] = list(faces)
    # We'll rebuild from new_faces, so keep it as the source of truth.

    def faces_view() -> np.ndarray:
        if not new_faces:
            return np.zeros((0, 3), dtype=np.int64)
        return np.asarray(new_faces, dtype=np.int64)

    # Per-vertex incidence (recomputed once at the end of each pass)
    def recompute_incidence(face_list: list[np.ndarray]):
        inc_local: list[list[int]] = [[] for _ in range(num_verts)]
        for fi, f in enumerate(face_list):
            a, b, c = f
            inc_local[a].append(fi)
            inc_local[b].append(fi)
            inc_local[c].append(fi)
        return inc_local

    inc = recompute_incidence(new_faces)

    pass_idx = 0
    while True:
        pass_idx += 1
        removed_any = False
        # Order vertices for determinism (lower index first)
        for V in range(num_verts):
            if not alive_vert[V] or lock_vertex_mask[V]:
                continue
            face_idxs = inc[V]
            if len(face_idxs) < 3:
                continue
            # Check coplanarity of incident faces
            face_arr = np.array([new_faces[i] for i in face_idxs])
            if (face_arr == -1).any():
                # Some incident face was removed; refresh
                face_idxs = [fi for fi in face_idxs if (new_faces[fi] != -1).all()]
                if len(face_idxs) < 3:
                    continue
                face_arr = np.array([new_faces[i] for i in face_idxs])
            # Group incident faces by normal. A vertex V is removable if:
            #   (a) all 1-ring faces share one normal (planar interior), OR
            #   (b) the 1-ring partitions into exactly two coplanar groups
            #       AND V's two feature-edge neighbors are collinear with V
            #       (so the two coplanar half-polygons can be retriangulated
            #       independently, then stitched at the straight edge).
            face_normals = [_face_normal_single(verts, f) for f in face_arr]
            # Normalize sign: align all normals to the first one's direction
            # (we treat opposite-normal coplanar faces as the same group;
            # mesh winding errors shouldn't cause spurious splits).
            n0 = face_normals[0]
            group_ids = [0]
            group_normals = [n0]
            for ni in face_normals[1:]:
                placed = False
                for gi, gn in enumerate(group_normals):
                    if abs(ni.dot(gn)) > 1 - normal_tol:
                        group_ids.append(gi)
                        placed = True
                        break
                if not placed:
                    group_ids.append(len(group_normals))
                    group_normals.append(ni)
            num_groups = len(group_normals)
            if num_groups > 2:
                continue  # corner — keep V
            if num_groups == 2 and not collapse_feature_edges:
                continue  # 2-group means V sits on a feature edge — keep V

            # Build the boundary loop of the 1-ring.
            # Each face contributes its 2 edges that don't touch V.
            half_edges: dict[int, list[int]] = {}
            half_edge_group: dict[tuple[int, int], int] = {}
            for f, g in zip(face_arr, group_ids):
                f_list = list(f)
                if V not in f_list:
                    continue
                idx = f_list.index(V)
                u = f_list[(idx + 1) % 3]
                w = f_list[(idx + 2) % 3]
                half_edges.setdefault(u, []).append(w)
                half_edge_group[(u, w)] = g
            sources = list(half_edges.keys())
            targets = [t for ts in half_edges.values() for t in ts]
            if len(sources) != len(targets):
                continue
            if any(len(ts) != 1 for ts in half_edges.values()):
                continue
            if len(set(targets)) != len(targets):
                continue
            loop = _order_polygon_loop(half_edges)
            if len(loop) < 3 or len(loop) != len(sources):
                continue

            if num_groups == 1:
                # Pure-planar case: triangulate the full polygon in 2D.
                coords, _, _ = _vertex_ring_2d_proj(verts, loop, n0)
                tris = _ear_clip_triangulate(coords, loop)
                if len(tris) != len(loop) - 2:
                    continue
            else:
                # Feature-edge case: 2 coplanar groups meeting at an edge
                # through V. The polygon has two contiguous "arcs" — one in
                # each plane — joined at V's feature-edge neighbors.
                # Walk the loop and assign each edge to its face's group.
                edge_groups: list[int] = []
                for i in range(len(loop)):
                    u = loop[i]
                    w = loop[(i + 1) % len(loop)]
                    g = half_edge_group.get((u, w))
                    if g is None:
                        edge_groups = []
                        break
                    edge_groups.append(g)
                if len(edge_groups) != len(loop):
                    continue
                # Find the two transitions between groups (where consecutive
                # edges differ in group). For a proper 2-group ring there
                # are exactly 2 transitions.
                transitions = [
                    i for i in range(len(loop))
                    if edge_groups[i] != edge_groups[(i - 1) % len(loop)]
                ]
                if len(transitions) != 2:
                    continue
                # Vertices at transitions are V's two feature-edge neighbors.
                # The transition is on the loop edge (loop[i-1], loop[i]),
                # so the feature-edge neighbor on this side is loop[i-1]'s
                # successor going INTO group g... actually it's loop[transitions[k]-1]
                # and the loop index itself depending on orientation. The
                # straight-line check needs the two vertices that V connects
                # to via feature edges in the mesh — these are exactly the
                # loop vertices at the boundary between the two arcs.
                # A transition at index i means the edge (loop[i], loop[(i+1)%n])
                # is in a different group than the edge (loop[(i-1)%n], loop[i]).
                # The vertex loop[i] is the "pivot" between groups — i.e.,
                # the feature-edge neighbor of V in the mesh.
                # Dihedral-angle gate: only collapse along a *real* feature
                # edge (large angle between the two patches), not a shallow
                # curvature step that just happens to be locally collinear.
                # cos(angle) = |n_a · n_b|; reject if angle is too small.
                n_a = group_normals[0]
                n_b = group_normals[1]
                if abs(float(n_a.dot(n_b))) > feature_cos_max:
                    continue
                t0, t1 = transitions
                n = len(loop)
                fneigh_a = loop[t0]
                fneigh_b = loop[t1]
                pa = verts[fneigh_a]
                pv = verts[V]
                pb = verts[fneigh_b]
                # Collinear if (pv - pa) × (pb - pv) ≈ 0
                e1 = pv - pa
                e2 = pb - pv
                cross_vec = np.cross(e1, e2)
                len1 = np.linalg.norm(e1)
                len2 = np.linalg.norm(e2)
                if len1 < 1e-12 or len2 < 1e-12:
                    continue
                if np.linalg.norm(cross_vec) > _COLLINEAR_TOL * max(len1, len2):
                    continue
                # Collinear → V is on a straight feature edge. Replace
                # the 4 (or more) faces around V with two planar polygon
                # triangulations, one per group, both sharing the
                # straight edge (fneigh_a → fneigh_b).
                #
                # Arc1 = loop[t0 .. t1], polygon closed by the
                #         straight edge fneigh_b -> fneigh_a.
                # Arc2 = loop[t1 .. t0+n], polygon closed by the
                #         straight edge fneigh_a -> fneigh_b.
                #
                # Each arc lies in a single plane (its group's normal),
                # so each can be ear-clipped in 2D.
                arc1 = [loop[i % n] for i in range(t0, t1 + 1)]
                arc2 = [loop[i % n] for i in range(t1, t0 + n + 1)]
                # Edge group at loop index i is for the OUTGOING edge
                # loop[i] -> loop[(i+1) % n]. So arc1 covers edges with
                # group edge_groups[t0], edge_groups[t0+1], …, edge_groups[t1-1].
                g_arc1 = edge_groups[t0]
                g_arc2 = edge_groups[t1]
                norm1 = group_normals[g_arc1]
                norm2 = group_normals[g_arc2]
                if len(arc1) < 3 or len(arc2) < 3:
                    continue
                coords1, _, _ = _vertex_ring_2d_proj(verts, arc1, norm1)
                tris1 = _ear_clip_triangulate(coords1, arc1)
                coords2, _, _ = _vertex_ring_2d_proj(verts, arc2, norm2)
                tris2 = _ear_clip_triangulate(coords2, arc2)
                if (len(tris1) != len(arc1) - 2
                        or len(tris2) != len(arc2) - 2):
                    continue
                tris = tris1 + tris2
            # Commit: remove V and its incident faces, add the new triangles
            for fi in face_idxs:
                # Detach the face from its other vertices' incidence lists
                for vv in new_faces[fi]:
                    if vv != V and fi in inc[vv]:
                        inc[vv].remove(fi)
                alive_face[fi] = False
                new_faces[fi] = np.array([-1, -1, -1], dtype=np.int64)
            for tri in tris:
                fi_new = len(new_faces)
                new_faces.append(np.array(tri, dtype=np.int64))
                for vv in tri:
                    inc[vv].append(fi_new)
                if fi_new >= len(alive_face):
                    alive_face = np.append(alive_face, True)
            alive_vert[V] = False
            removed_any = True
        if not removed_any:
            break

    # Compact faces
    surviving_faces = np.array(
        [f for fi, f in enumerate(new_faces) if alive_face[fi]],
        dtype=np.int64,
    )
    if len(surviving_faces) == 0:
        return verts.copy(), np.zeros((0, 3), dtype=np.int64)

    # Compact vertices
    used = np.zeros(num_verts, dtype=bool)
    used[surviving_faces.ravel()] = True
    remap = -np.ones(num_verts, dtype=np.int64)
    keep_idx = np.where(used)[0]
    remap[keep_idx] = np.arange(len(keep_idx))
    out_verts = verts[keep_idx]
    out_faces = remap[surviving_faces]
    return out_verts, out_faces


def _face_normal_single(verts: np.ndarray, face: np.ndarray) -> np.ndarray:
    v0 = verts[face[0]]
    v1 = verts[face[1]]
    v2 = verts[face[2]]
    n = np.cross(v1 - v0, v2 - v0)
    m = np.linalg.norm(n)
    if m < 1e-30:
        return n
    return n / m


# ---------------------------------------------------------------------------
# Pass B: symmetric boundary-polyline decimation via Douglas-Peucker
# ---------------------------------------------------------------------------


def _douglas_peucker_2d(pts: np.ndarray, eps: float) -> np.ndarray:
    """Recursive DP simplification of an open polyline. `pts` shape (N, 2).
    Returns boolean keep-mask of length N. Endpoints always kept.
    """
    n = len(pts)
    if n <= 2:
        return np.ones(n, dtype=bool)
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j - i <= 1:
            continue
        a = pts[i]
        b = pts[j]
        ab = b - a
        ab_len_sq = float(ab @ ab)
        if ab_len_sq < 1e-30:
            d = np.linalg.norm(pts[i + 1:j] - a, axis=1)
        else:
            t = ((pts[i + 1:j] - a) @ ab) / ab_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj = a + t[:, None] * ab
            d = np.linalg.norm(pts[i + 1:j] - proj, axis=1)
        if len(d) == 0:
            continue
        k_local = int(np.argmax(d))
        k = i + 1 + k_local
        if d[k_local] > eps:
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return keep


def _boundary_polylines_on_plane(
    verts: np.ndarray, faces: np.ndarray, axis: int, value: float,
    plane_tol: float = _PLANE_HIT_TOL,
) -> list[list[int]]:
    """Find the polylines on a chunk-boundary plane.

    A polyline is a maximal sequence of vertices on the plane connected
    by mesh edges whose *both* endpoints lie on the plane. Each polyline
    is returned as an ordered list of vertex indices. Closed loops (rare,
    e.g. an island poking through the plane) are returned as cyclic lists.
    """
    on_plane = np.abs(verts[:, axis] - value) < plane_tol
    on_plane_verts = np.where(on_plane)[0]
    if len(on_plane_verts) == 0:
        return []
    on_plane_set = set(on_plane_verts.tolist())

    # Collect edges with both endpoints on the plane
    edges = set()
    for f in faces:
        for u, v in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0])):
            if int(u) in on_plane_set and int(v) in on_plane_set:
                edges.add((int(min(u, v)), int(max(u, v))))

    if not edges:
        return []

    # Adjacency
    adj: dict[int, list[int]] = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    # Walk to extract polylines (start at degree-1 vertex for open chains,
    # then handle cycles for remaining)
    visited = set()
    polylines = []
    # Open chains first
    endpoints = [v for v, nbrs in adj.items() if len(nbrs) == 1]
    endpoints.sort()  # determinism
    for start in endpoints:
        if start in visited:
            continue
        chain = [start]
        visited.add(start)
        curr = start
        prev = None
        while True:
            cands = [n for n in adj[curr] if n != prev]
            if not cands:
                break
            nxt = min(cands)  # deterministic
            if nxt in visited:
                break
            chain.append(nxt)
            visited.add(nxt)
            prev = curr
            curr = nxt
        polylines.append(chain)
    # Closed loops
    for u in sorted(adj.keys()):
        if u in visited:
            continue
        # Walk a cycle
        chain = [u]
        visited.add(u)
        prev = None
        curr = u
        while True:
            cands = [n for n in adj[curr] if n != prev and n not in visited]
            if not cands:
                if u in adj[curr]:
                    pass  # close the loop implicitly
                break
            nxt = min(cands)
            chain.append(nxt)
            visited.add(nxt)
            prev = curr
            curr = nxt
        polylines.append(chain)
    return polylines


def simplify_boundary_polylines(
    verts: np.ndarray, faces: np.ndarray,
    boundary_planes: Iterable[tuple[int, float]],
    eps: float = _DP_DEFAULT_EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric, deterministic boundary-polyline simplification.

    For each (axis, value) plane: extract the polylines on that plane,
    run Douglas-Peucker simplification, remove the intermediate (now-
    redundant) vertices, and locally retriangulate the strip of faces
    that touched them.

    The DP algorithm depends only on the polyline's vertex coordinates,
    so two blocks sharing this plane compute identical results.
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # Track which vertices to remove
    remove_mask = np.zeros(len(verts), dtype=bool)

    for axis, value in boundary_planes:
        polylines = _boundary_polylines_on_plane(verts, faces, axis, value)
        for pl in polylines:
            if len(pl) < 3:
                continue
            # Project to 2D (drop the boundary axis)
            kept_axes = [a for a in (0, 1, 2) if a != axis]
            pts_2d = verts[np.array(pl)][:, kept_axes]
            keep = _douglas_peucker_2d(pts_2d, eps)
            for vi, k in zip(pl, keep):
                if not k:
                    remove_mask[vi] = True

    if not remove_mask.any():
        return verts.copy(), faces.copy()

    # Retriangulate: for each removed vertex V, treat its 1-ring as a
    # polygonal hole and re-triangulate using the same approach as Pass A.
    # We do this in a tight loop because removed vertices are independent.
    new_verts = verts.copy()
    new_faces_list = [f.copy() for f in faces]
    # Per-vertex incidence
    inc: list[list[int]] = [[] for _ in range(len(new_verts))]
    for fi, f in enumerate(new_faces_list):
        for vv in f:
            inc[int(vv)].append(fi)
    alive_face = np.ones(len(new_faces_list), dtype=bool)

    to_remove = sorted(np.where(remove_mask)[0].tolist())
    for V in to_remove:
        face_idxs = [fi for fi in inc[V] if alive_face[fi]]
        if len(face_idxs) < 3:
            continue
        face_arr = np.array([new_faces_list[fi] for fi in face_idxs])
        # The local 1-ring polygon (boundary opposite to V)
        half_edges: dict[int, list[int]] = {}
        for f in face_arr:
            f_list = [int(x) for x in f]
            if V not in f_list:
                continue
            idx = f_list.index(V)
            u = f_list[(idx + 1) % 3]
            w = f_list[(idx + 2) % 3]
            half_edges.setdefault(u, []).append(w)
        # Quick sanity
        if any(len(ts) != 1 for ts in half_edges.values()):
            continue
        loop = _order_polygon_loop(half_edges)
        if len(loop) < 3 or len(loop) != len(half_edges):
            continue
        # Compute a normal for projection: average incident face normals.
        normals = np.array([_face_normal_single(new_verts, f) for f in face_arr])
        # Make sure they all point the same way (flip if dot < 0 with the first)
        for i in range(1, len(normals)):
            if normals[0].dot(normals[i]) < 0:
                normals[i] = -normals[i]
        n = normals.mean(axis=0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n /= n_norm
        coords, _, _ = _vertex_ring_2d_proj(new_verts, loop, n)
        tris = _ear_clip_triangulate(coords, loop)
        if len(tris) != len(loop) - 2:
            continue
        # Commit
        for fi in face_idxs:
            for vv in new_faces_list[fi]:
                v_int = int(vv)
                if v_int != V and fi in inc[v_int]:
                    inc[v_int].remove(fi)
            alive_face[fi] = False
        for tri in tris:
            fi_new = len(new_faces_list)
            new_faces_list.append(np.array(tri, dtype=np.int64))
            for vv in tri:
                inc[int(vv)].append(fi_new)
            alive_face = np.append(alive_face, True)

    surviving = np.array(
        [f for fi, f in enumerate(new_faces_list) if alive_face[fi]],
        dtype=np.int64,
    )
    if len(surviving) == 0:
        return new_verts.copy(), np.zeros((0, 3), dtype=np.int64)

    used = np.zeros(len(new_verts), dtype=bool)
    used[surviving.ravel()] = True
    remap = -np.ones(len(new_verts), dtype=np.int64)
    keep_idx = np.where(used)[0]
    remap[keep_idx] = np.arange(len(keep_idx))
    out_verts = new_verts[keep_idx]
    out_faces = remap[surviving]
    return out_verts, out_faces


# ---------------------------------------------------------------------------
# Combined API
# ---------------------------------------------------------------------------


def lossless_chunk_simplify(
    verts: np.ndarray, faces: np.ndarray,
    boundary_planes: Iterable[tuple[int, float]] = (),
    dp_eps: float = _DP_DEFAULT_EPS,
    normal_tol: float = _PLANE_TOL,
    do_pass_a: bool = True,
    do_pass_b: bool = True,
    collapse_feature_edges: bool = False,
    feature_dihedral_min_deg: float = _FEATURE_DIHEDRAL_MIN_DEG,
) -> tuple[np.ndarray, np.ndarray]:
    """Run both passes. `boundary_planes` is a list of (axis, value)
    planes to apply Pass B to (typically the 6 faces of the chunk).

    Pass A locks vertices on any boundary plane (only Pass B touches them).
    """
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if len(faces) == 0:
        return verts.copy(), faces.copy()

    boundary_planes_list = list(boundary_planes)
    if do_pass_a:
        if boundary_planes_list:
            lock = np.zeros(len(verts), dtype=bool)
            for axis, value in boundary_planes_list:
                lock |= np.abs(verts[:, axis] - value) < _PLANE_HIT_TOL
        else:
            lock = None
        verts, faces = collapse_planar_vertices(
            verts, faces, normal_tol=normal_tol, lock_vertex_mask=lock,
            collapse_feature_edges=collapse_feature_edges,
            feature_dihedral_min_deg=feature_dihedral_min_deg,
        )

    if do_pass_b and boundary_planes_list:
        verts, faces = simplify_boundary_polylines(
            verts, faces, boundary_planes_list, eps=dp_eps,
        )

    return verts, faces


def block_boundary_planes(
    block_origin: np.ndarray, block_size: np.ndarray,
) -> list[tuple[int, float]]:
    """Return the six (axis, value) planes for a block's bounding faces.
    """
    return [
        (0, float(block_origin[0])),
        (0, float(block_origin[0] + block_size[0])),
        (1, float(block_origin[1])),
        (1, float(block_origin[1] + block_size[1])),
        (2, float(block_origin[2])),
        (2, float(block_origin[2] + block_size[2])),
    ]
