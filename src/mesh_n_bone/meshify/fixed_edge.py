"""
Blockwise -> Merge -> Seam Denoise -> Global Simplify

Dependencies: numpy, trimesh, pyfqmr, pymeshlab
"""

import numpy as np
import trimesh
import pymeshlab

try:
    from pyfqmr import Simplify
except Exception as e:
    raise RuntimeError("pyfqmr is required. Install with `pip install pyfqmr`.") from e


def pymeshlab_simplify(
    verts, faces, target_faces, aggressiveness=0.3,
    preserve_border=True, preserve_topology=True, verbose=False,
):
    """Simplify a mesh using PyMeshLab quadric edge-collapse decimation.

    Performs decimation followed by removal of duplicate faces, duplicate
    vertices, and unreferenced vertices.

    Parameters
    ----------
    verts : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle indices with shape ``(M, 3)``.
    target_faces : int
        Desired number of faces after simplification.
    aggressiveness : float, optional
        Quality threshold passed to the decimation filter. Lower values
        produce more conservative simplification. Default is ``0.3``.
    preserve_border : bool, optional
        If ``True``, boundary edges are preserved during decimation.
        Default is ``True``.
    preserve_topology : bool, optional
        If ``True``, the mesh topology is preserved. Default is ``True``.
    verbose : bool, optional
        Unused; kept for API compatibility. Default is ``False``.

    Returns
    -------
    verts_out : numpy.ndarray
        Simplified vertex positions with dtype ``float64``.
    faces_out : numpy.ndarray
        Simplified triangle indices with dtype ``int32``.
    """
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(verts, faces))
    ms.meshing_decimation_quadric_edge_collapse(
        targetfacenum=target_faces,
        preserveboundary=preserve_border,
        preservetopology=preserve_topology,
        preservenormal=True,
        boundaryweight=np.inf,
        planarquadric=True,
        qualitythr=aggressiveness,
    )
    ms.meshing_remove_duplicate_faces()
    ms.meshing_remove_duplicate_vertices()
    ms.meshing_remove_unreferenced_vertices()
    m = ms.current_mesh()
    return m.vertex_matrix().astype(np.float64), m.face_matrix().astype(np.int32)


def fqmr_simplify(
    verts, faces, target_faces, preserve_border, aggressiveness=7, verbose=False,
):
    """Simplify a mesh using the pyfqmr (Fast Quadric Mesh Reduction) library.

    Handles multiple pyfqmr API versions (``setMesh``/``set_mesh``) and
    return formats (2- or 3-tuple) transparently.

    Parameters
    ----------
    verts : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle indices with shape ``(M, 3)``.
    target_faces : int
        Desired number of faces after simplification. Clamped to a minimum
        of ``4``.
    preserve_border : bool
        If ``True``, boundary edges are preserved during decimation.
    aggressiveness : float, optional
        Controls how aggressively the algorithm simplifies. Higher values
        allow more aggressive reduction. Default is ``7``.
    verbose : bool, optional
        If ``True``, pyfqmr prints progress information. Default is ``False``.

    Returns
    -------
    verts_out : numpy.ndarray
        Simplified vertex positions with dtype ``float64``.
    faces_out : numpy.ndarray
        Simplified triangle indices with dtype ``int32``.

    Raises
    ------
    RuntimeError
        If the installed pyfqmr version has an incompatible API or returns
        an unexpected number of values.
    """
    simp = Simplify()
    if hasattr(simp, "setMesh"):
        simp.setMesh(verts.astype(np.float64), faces.astype(np.int32))
        simp.simplify_mesh(
            target_count=int(max(12, target_faces)),
            preserve_border=preserve_border,
            aggressiveness=aggressiveness,
            verbose=verbose,
        )
        result = simp.getMesh()
    elif hasattr(simp, "set_mesh"):
        simp.set_mesh(verts.astype(np.float64), faces.astype(np.int32))
        simp.simplify_mesh(
            target_count=int(max(12, target_faces)),
            preserve_border=preserve_border,
            aggressiveness=aggressiveness,
            verbose=verbose,
        )
        result = simp.get_mesh()
    else:
        raise RuntimeError("Incompatible pyfqmr version.")

    if len(result) == 2:
        v_out, f_out = result
    elif len(result) == 3:
        v_out, f_out, _ = result
    else:
        raise RuntimeError(f"Unexpected pyfqmr return format: got {len(result)} values")

    return v_out.astype(np.float64), f_out.astype(np.int32)


def repair_cleanup(mesh, rezero=False):
    """Clean up a trimesh by removing degenerate and duplicate geometry.

    Sequentially removes degenerate faces, duplicate faces, and
    unreferenced vertices. Optionally re-zeros the mesh origin and
    fixes face normals.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh to repair in place.
    rezero : bool, optional
        If ``True`` and the mesh has faces, translate the mesh so its
        bounding box starts at the origin. Default is ``False``.

    Returns
    -------
    trimesh.Trimesh
        The repaired mesh (same object, modified in place).
    """
    # Remove degenerate faces
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    else:
        mask = mesh.nondegenerate_faces()
        if not mask.all():
            mesh.update_faces(mask)
    # Remove duplicate faces
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    else:
        mesh.update_faces(mesh.unique_faces())
    # Remove unreferenced vertices
    mesh.remove_unreferenced_vertices()
    if rezero and len(mesh.faces) > 0 and mesh.bounds is not None:
        mesh.rezero()
    if len(mesh.faces) > 0:
        mesh.fix_normals()
    return mesh


def weld_vertices(mesh, epsilon, block_size=None, roi_offset=None, verbose=False):
    """Weld vertices that are within an epsilon distance of each other.

    Uses a grid-based spatial hashing approach to efficiently find and
    merge nearby vertices via a union-find structure. When ``block_size``
    and ``roi_offset`` are provided, only vertices near block boundaries
    are considered for welding.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh. A copy is made internally.
    epsilon : float
        Maximum distance between vertices to be merged.
    block_size : numpy.ndarray or None, optional
        Shape ``(3,)`` array giving the block dimensions. When provided
        together with ``roi_offset``, restricts welding to vertices near
        block boundaries. Default is ``None``.
    roi_offset : numpy.ndarray or None, optional
        Shape ``(3,)`` array giving the ROI origin offset in world
        coordinates. Used with ``block_size`` to compute boundary
        proximity. Default is ``None``.
    verbose : bool, optional
        Unused; kept for API compatibility. Default is ``False``.

    Returns
    -------
    trimesh.Trimesh
        A new mesh with nearby vertices merged and geometry cleaned up
        via ``repair_cleanup``.
    """
    from collections import defaultdict

    mesh = mesh.copy()
    verts_before = len(mesh.vertices)
    mesh.merge_vertices(merge_tex=False, merge_norm=False)

    if len(mesh.vertices) > 0 and epsilon > 0:
        vertices = mesh.vertices

        if block_size is not None and roi_offset is not None:
            rel_coords = vertices - roi_offset
            boundary_dist = np.minimum(
                np.abs(rel_coords % block_size),
                np.abs(block_size - (rel_coords % block_size)),
            )
            near_boundary = np.any(boundary_dist <= epsilon * 2, axis=1)
            if near_boundary.sum() == 0:
                return repair_cleanup(mesh)
            boundary_indices = np.where(near_boundary)[0]
            vertices_to_process = vertices[boundary_indices]
        else:
            boundary_indices = np.arange(len(vertices))
            vertices_to_process = vertices

            vmin = vertices_to_process.min(axis=0)
            grid_coords = np.floor((vertices_to_process - vmin) / epsilon).astype(np.int32)
            grid = defaultdict(list)
            for local_idx, coord in enumerate(grid_coords):
                grid[tuple(coord)].append(local_idx)

            vertex_map = np.arange(len(vertices), dtype=np.int32)

            for cell, local_indices in grid.items():
                for i in range(len(local_indices)):
                    for j in range(i + 1, len(local_indices)):
                        local_i, local_j = local_indices[i], local_indices[j]
                        global_i = boundary_indices[local_i]
                        global_j = boundary_indices[local_j]
                        dist = np.linalg.norm(
                            vertices_to_process[local_i] - vertices_to_process[local_j]
                        )
                        if dist <= epsilon:
                            root_i, root_j = global_i, global_j
                            while vertex_map[root_i] != root_i:
                                root_i = vertex_map[root_i]
                            while vertex_map[root_j] != root_j:
                                root_j = vertex_map[root_j]
                            if root_i < root_j:
                                vertex_map[root_j] = root_i
                            elif root_j < root_i:
                                vertex_map[root_i] = root_j

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0, 1]:
                            if dx == dy == dz == 0:
                                continue
                            neighbor_cell = (cell[0] + dx, cell[1] + dy, cell[2] + dz)
                            if neighbor_cell in grid:
                                for local_i in local_indices:
                                    for local_j in grid[neighbor_cell]:
                                        global_i = boundary_indices[local_i]
                                        global_j = boundary_indices[local_j]
                                        dist = np.linalg.norm(
                                            vertices_to_process[local_i]
                                            - vertices_to_process[local_j]
                                        )
                                        if dist <= epsilon:
                                            root_i, root_j = global_i, global_j
                                            while vertex_map[root_i] != root_i:
                                                root_i = vertex_map[root_i]
                                            while vertex_map[root_j] != root_j:
                                                root_j = vertex_map[root_j]
                                            if root_i < root_j:
                                                vertex_map[root_j] = root_i
                                            elif root_j < root_i:
                                                vertex_map[root_i] = root_j

            # Path compression
            for i in range(len(vertex_map)):
                root = i
                while vertex_map[root] != root:
                    root = vertex_map[root]
                vertex_map[i] = root

            unique_roots = np.unique(vertex_map)
            if len(unique_roots) < len(vertices):
                inverse_map = np.zeros(len(vertices), dtype=np.int32)
                for new_idx, root in enumerate(unique_roots):
                    inverse_map[vertex_map == root] = new_idx
                new_vertices = vertices[unique_roots]
                new_faces = inverse_map[mesh.faces.ravel()].reshape(-1, 3)
                mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)

    return repair_cleanup(mesh)


def remove_boundary_vertices(mesh, voxel_size, block_size=None, verbose=False):
    """Clip a mesh at block boundaries and remove exterior vertices.

    For each axis, slices the mesh at planes located half a voxel inward
    from the block origin and half a voxel inward from the block extent.
    Vertices that fall outside the valid interior range are then removed
    along with any faces that reference them.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to clip.
    voxel_size : array_like
        Shape ``(3,)`` voxel dimensions used to compute a half-voxel
        tolerance for clipping planes.
    block_size : array_like or None, optional
        Shape ``(3,)`` block dimensions in world coordinates. If ``None``,
        the mesh is returned unchanged. Default is ``None``.
    verbose : bool, optional
        Unused; kept for API compatibility. Default is ``False``.

    Returns
    -------
    trimesh.Trimesh
        The clipped and cleaned mesh. Returns an empty mesh if all
        geometry is removed.
    """
    if len(mesh.vertices) == 0:
        return mesh
    if block_size is None:
        return mesh

    tolerance = 0.5 * np.array(voxel_size)

    for axis in range(3):
        plane_normal = np.zeros(3)
        plane_normal[axis] = 1.0
        plane_origin = np.zeros(3)
        plane_origin[axis] = tolerance[axis]
        mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=False,
        )
        if mesh is None or len(mesh.faces) == 0:
            return trimesh.Trimesh(
                vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
            )

        plane_normal = np.zeros(3)
        plane_normal[axis] = -1.0
        plane_origin = np.zeros(3)
        plane_origin[axis] = block_size[axis] - tolerance[axis]
        mesh = trimesh.intersections.slice_mesh_plane(
            mesh, plane_normal=plane_normal, plane_origin=plane_origin, cap=False,
        )
        if mesh is None or len(mesh.faces) == 0:
            return trimesh.Trimesh(
                vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
            )

    vertices = mesh.vertices
    below_min = vertices < tolerance
    beyond_max = vertices > (block_size - tolerance)
    outside_valid_range = np.any(below_min | beyond_max, axis=1)

    if not np.any(outside_valid_range):
        return repair_cleanup(mesh)

    keep_mask = ~outside_valid_range
    keep_indices = np.where(keep_mask)[0]
    vertex_map = np.full(len(vertices), -1, dtype=np.int32)
    vertex_map[keep_indices] = np.arange(len(keep_indices))
    new_vertices = vertices[keep_indices]
    face_mask = np.all(vertex_map[mesh.faces] >= 0, axis=1)
    new_faces = vertex_map[mesh.faces[face_mask]]

    if len(new_faces) == 0:
        return trimesh.Trimesh(
            vertices=np.empty((0, 3)), faces=np.empty((0, 3), dtype=np.int32)
        )

    new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False)
    return repair_cleanup(new_mesh)


def simplify_mesh(
    mesh, target_reduction, voxel_size, block_size=None,
    aggressiveness=0.3, verbose=False, use_pymeshlab=True, fix_edges=False,
):
    """Simplify a mesh with optional boundary preservation.

    First removes vertices outside block boundaries via
    ``remove_boundary_vertices``, then decimates the mesh to the
    requested face count using either PyMeshLab or pyfqmr.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to simplify.
    target_reduction : float
        Fraction of faces to remove, in the range ``[0, 1)``. For
        example, ``0.5`` removes roughly half the faces.
    voxel_size : array_like
        Shape ``(3,)`` voxel dimensions passed to
        ``remove_boundary_vertices``.
    block_size : array_like or None, optional
        Shape ``(3,)`` block dimensions for boundary clipping. If
        ``None``, no boundary clipping is performed. Default is ``None``.
    aggressiveness : float, optional
        Aggressiveness parameter forwarded to the simplification backend.
        Default is ``0.3``.
    verbose : bool, optional
        If ``True``, print progress information. Default is ``False``.
    use_pymeshlab : bool, optional
        If ``True``, use PyMeshLab for decimation; otherwise use pyfqmr.
        Default is ``True``.
    fix_edges : bool, optional
        If ``True``, boundary edges are preserved during decimation.
        Default is ``False``.

    Returns
    -------
    trimesh.Trimesh
        The simplified and cleaned mesh.
    """
    mesh = remove_boundary_vertices(mesh, voxel_size, block_size, verbose=verbose)
    F = mesh.faces
    if len(F) == 0:
        return mesh

    target_faces = int(max(12, (1 - target_reduction) * F.shape[0]))
    if fix_edges:
        # When preserving borders, boundary vertices cannot be collapsed.
        # If target_faces is too low relative to the number of boundary
        # vertices, the simplifier is forced to collapse all interior
        # geometry into single high-valence fan vertices.  Cap the target
        # so that at least 2 faces per boundary vertex remain, which keeps
        # vertex valence reasonable.
        ms_tmp = pymeshlab.MeshSet()
        ms_tmp.add_mesh(pymeshlab.Mesh(mesh.vertices, F))
        ms_tmp.compute_selection_from_mesh_border()
        n_boundary = int(ms_tmp.current_mesh().vertex_selection_array().sum())
        min_faces = 2 * n_boundary
        if target_faces < min_faces:
            target_faces = min_faces
    if use_pymeshlab:
        v_out, f_out = pymeshlab_simplify(
            mesh.vertices, F,
            target_faces=target_faces,
            aggressiveness=0.3,
            preserve_border=fix_edges,
            preserve_topology=True,
            verbose=verbose,
        )
    else:
        v_out, f_out = fqmr_simplify(
            mesh.vertices, F,
            target_faces=target_faces,
            preserve_border=fix_edges,
            aggressiveness=aggressiveness,
            verbose=False,
        )
    m2 = trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    return repair_cleanup(m2)


def detect_seam_vertices(mesh, angle_degrees, verbose=False):
    """Detect vertices along sharp seam edges in a mesh.

    Identifies edges where the dihedral angle between adjacent faces
    exceeds a threshold, then returns the unique vertex indices that
    belong to those sharp edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh to analyze.
    angle_degrees : float
        Dihedral angle threshold in degrees. Edges with an angle at or
        above this value are considered sharp seams.
    verbose : bool, optional
        Unused; kept for API compatibility. Default is ``False``.

    Returns
    -------
    numpy.ndarray
        Sorted unique vertex indices (dtype ``int32``) on sharp seam
        edges. Returns an empty array if no seams are found.
    """
    adj = mesh.face_adjacency
    ang = mesh.face_adjacency_angles
    if ang is None or len(ang) == 0:
        return np.array([], dtype=np.int32)
    thresh = np.deg2rad(angle_degrees)
    sharp_mask = ang >= thresh
    if not np.any(sharp_mask):
        return np.array([], dtype=np.int32)
    # Get edges shared between adjacent sharp face pairs
    sharp_pair_indices = np.where(sharp_mask)[0]
    if hasattr(mesh, 'face_adjacency_edges'):
        shared = mesh.face_adjacency_edges[sharp_pair_indices]
    else:
        sharp_pairs = adj[sharp_mask]
        shared = trimesh.graph.shared_edges(mesh.faces, sharp_pairs)
    return np.unique(shared.reshape(-1)).astype(np.int32)


def vertex_adjacency_list(mesh):
    """Build a per-vertex adjacency list from the mesh edge graph.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input mesh whose ``edges_unique`` attribute is used.

    Returns
    -------
    list of numpy.ndarray
        A list of length ``len(mesh.vertices)`` where each element is a
        1-D ``int32`` array of neighboring vertex indices.
    """
    nbrs = [[] for _ in range(len(mesh.vertices))]
    edges = mesh.edges_unique
    for u, v in edges:
        nbrs[u].append(v)
        nbrs[v].append(u)
    return [np.array(n, dtype=np.int32) for n in nbrs]


def expand_k_ring(seed_vertices, nbrs, k):
    """Expand a set of seed vertices by ``k`` neighborhood rings.

    Performs a breadth-first expansion from ``seed_vertices`` using the
    adjacency list ``nbrs``, adding all vertices reachable within ``k``
    edge hops.

    Parameters
    ----------
    seed_vertices : array_like
        Initial vertex indices to expand from.
    nbrs : list of array_like
        Per-vertex adjacency list as returned by
        ``vertex_adjacency_list``.
    k : int
        Number of rings (hops) to expand.

    Returns
    -------
    numpy.ndarray
        All vertex indices within ``k`` rings of the seeds, including
        the seeds themselves (dtype ``int32``).
    """
    ring = set(int(i) for i in seed_vertices)
    frontier = set(ring)
    for _ in range(k):
        new_frontier = set()
        for v in frontier:
            for w in nbrs[v]:
                if w not in ring:
                    ring.add(w)
                    new_frontier.add(w)
        frontier = new_frontier
        if not frontier:
            break
    return np.fromiter(ring, dtype=np.int32)


def taubin_constrained(mesh, subset_idx, lamb=0.5, mu=-0.53, iterations=10, verbose=False):
    """Apply non-shrinking Taubin smoothing to a vertex subset.

    Alternates positive (``lamb``) and negative (``mu``) Laplacian
    smoothing steps to smooth the surface without net shrinkage. Only
    the vertices in ``subset_idx`` are updated; all other vertices
    remain fixed.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh whose vertices are modified in place.
    subset_idx : array_like
        Indices of the vertices to smooth.
    lamb : float, optional
        Positive smoothing factor for the shrinking step.
        Default is ``0.5``.
    mu : float, optional
        Negative smoothing factor for the inflation step. Should be
        negative with ``|mu| > lamb`` to prevent shrinkage.
        Default is ``-0.53``.
    iterations : int, optional
        Number of shrink/inflate iteration pairs. Default is ``10``.
    verbose : bool, optional
        Unused; kept for API compatibility. Default is ``False``.
    """
    if len(subset_idx) == 0:
        return

    V = mesh.vertices.view(np.ndarray).copy()
    nbrs = vertex_adjacency_list(mesh)
    subset = np.asarray(subset_idx, dtype=np.int32)

    def smooth_step(Vcur, step):
        Vnew = Vcur.copy()
        for v in subset:
            n = nbrs[v]
            if n.size == 0:
                continue
            mean_nb = Vcur[n].mean(axis=0)
            Vnew[v] = Vcur[v] + step * (mean_nb - Vcur[v])
        return Vnew

    for _ in range(iterations):
        V = smooth_step(V, lamb)
        V = smooth_step(V, mu)

    mesh.vertices = V
    repair_cleanup(mesh)


def denoise_seams_inplace(
    mesh, seam_angle_deg, k_ring, taubin_iters,
    lamb=0.5, mu=-0.53, verbose=False,
):
    """Detect and smooth seam artifacts on a mesh in place.

    Combines ``detect_seam_vertices``, ``expand_k_ring``, and
    ``taubin_constrained`` to find sharp seam edges, expand the affected
    region by ``k_ring`` hops, and apply Taubin smoothing to that band.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Mesh to denoise in place.
    seam_angle_deg : float
        Dihedral angle threshold in degrees for detecting seam edges.
    k_ring : int
        Number of neighborhood rings to expand around detected seam
        vertices before smoothing.
    taubin_iters : int
        Number of Taubin smoothing iteration pairs.
    lamb : float, optional
        Positive smoothing factor. Default is ``0.5``.
    mu : float, optional
        Negative inflation factor. Default is ``-0.53``.
    verbose : bool, optional
        If ``True``, print diagnostic information. Default is ``False``.
    """
    seam_verts = detect_seam_vertices(mesh, angle_degrees=seam_angle_deg, verbose=verbose)
    if seam_verts.size == 0:
        return

    band = expand_k_ring(seam_verts, vertex_adjacency_list(mesh), k=k_ring)
    taubin_constrained(mesh, band, lamb=lamb, mu=mu, iterations=taubin_iters, verbose=verbose)
