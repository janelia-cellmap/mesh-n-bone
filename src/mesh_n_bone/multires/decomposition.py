from trimesh.intersections import slice_faces_plane
import numpy as np
import DracoPy

from mesh_n_bone.util import mesh_io
from mesh_n_bone.util.logging import print_with_datetime, capture_draco_output
import logging

logger = logging.getLogger(__name__)


def my_slice_faces_plane(vertices, faces, plane_normal, plane_origin):
    """Slice a mesh along a plane, returning only the portion behind the plane.

    Thin wrapper around ``trimesh.intersections.slice_faces_plane`` that
    gracefully handles the case where the entire mesh lies on one side of
    the cutting plane (which would otherwise raise a ``ValueError``).

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle face indices with shape ``(M, 3)``.
    plane_normal : numpy.ndarray
        Unit normal of the slicing plane (shape ``(3,)``).  Vertices on the
        side the normal points toward are kept.
    plane_origin : numpy.ndarray
        A point on the slicing plane (shape ``(3,)``).

    Returns
    -------
    vertices : numpy.ndarray
        Vertex positions of the sliced mesh.
    faces : numpy.ndarray
        Face indices of the sliced mesh.
    """
    if len(vertices) > 0 and len(faces) > 0:
        try:
            vertices, faces, _ = slice_faces_plane(
                vertices, faces, plane_normal, plane_origin
            )
        except ValueError as e:
            if str(e) != "input must be 1D integers!":
                raise
    return vertices, faces


def update_fragment_dict(dictionary, fragment_pos, vertices, faces, lod_0_fragment_pos):
    """Insert or update a fragment in the position-keyed dictionary.

    If ``fragment_pos`` already exists in ``dictionary``, the fragment's
    vertices and faces are extended via ``Fragment.update``.  Otherwise a
    new ``mesh_io.Fragment`` is created and inserted.

    Parameters
    ----------
    dictionary : dict
        Mutable mapping from fragment position tuples to
        ``mesh_io.Fragment`` objects.  Modified in place.
    fragment_pos : tuple of int
        ``(x, y, z)`` grid position of the fragment at the current LOD.
    vertices : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle face indices with shape ``(M, 3)``.
    lod_0_fragment_pos : list of int
        ``[x, y, z]`` grid position of the sub-fragment at LOD 0.
    """
    if fragment_pos in dictionary:
        fragment = dictionary[fragment_pos]
        fragment.update(vertices, faces, lod_0_fragment_pos)
        dictionary[fragment_pos] = fragment
    else:
        dictionary[fragment_pos] = mesh_io.Fragment(
            vertices, faces, [lod_0_fragment_pos]
        )


def generate_mesh_decomposition(
    mesh_path, lod_0_box_size, grid_origin, start_fragment, end_fragment,
    current_lod, num_chunks, vertex_quantization_bits=16,
):
    """Decompose a mesh into spatially-gridded, Draco-compressed fragments.

    The mesh is loaded from ``mesh_path``, translated so that
    ``grid_origin`` becomes the coordinate origin, and then sliced into
    axis-aligned boxes of size ``lod_0_box_size * 2**current_lod``.  For
    LOD >= 1 each box is further subdivided into a 2x2x2 grid of
    sub-fragments at the LOD 0 scale.  Every resulting fragment is
    quantized, Draco-encoded, and returned as a
    ``mesh_io.CompressedFragment``.

    Parameters
    ----------
    mesh_path : str
        Path to the mesh file to decompose.
    lod_0_box_size : numpy.ndarray
        Per-axis size of a single fragment at LOD 0 (shape ``(3,)``).
    grid_origin : numpy.ndarray
        World-space origin of the fragment grid (shape ``(3,)``).  Vertices
        are shifted by ``-grid_origin`` before slicing.
    start_fragment : numpy.ndarray
        Inclusive ``(x, y, z)`` start indices of the fragment range to
        process (at the current LOD scale).
    end_fragment : numpy.ndarray
        Exclusive ``(x, y, z)`` end indices of the fragment range to
        process (at the current LOD scale).
    current_lod : int
        Level of detail being generated.  ``0`` means full resolution;
        higher values correspond to coarser decimation levels.
    num_chunks : numpy.ndarray
        Total number of Dask chunks along each axis (shape ``(3,)``).
        Used to decide whether to pre-clip the mesh to the task's slab.

    Returns
    -------
    list of mesh_io.CompressedFragment or None
        Draco-compressed fragments that contain geometry, or ``None`` if
        the mesh has no vertices within the requested region.
    """

    vertices, faces = mesh_io.mesh_loader(mesh_path)

    combined_fragments_dictionary = {}
    fragments = []

    nyz, nxz, nxy = np.eye(3)

    if current_lod != 0:
        start_fragment *= 2
        end_fragment *= 2
        sub_box_size = lod_0_box_size * 2 ** (current_lod - 1)
    else:
        sub_box_size = lod_0_box_size

    vertices -= grid_origin

    # Set up slab for current dask task
    n = np.eye(3)
    for dimension in range(3):
        if num_chunks[dimension] > 1:
            n_d = n[dimension, :]
            plane_origin = n_d * end_fragment[dimension] * sub_box_size
            vertices, faces = my_slice_faces_plane(vertices, faces, -n_d, plane_origin)
            if len(vertices) == 0:
                return None
            plane_origin = n_d * start_fragment[dimension] * sub_box_size
            vertices, faces = my_slice_faces_plane(vertices, faces, n_d, plane_origin)

    if len(vertices) == 0:
        return None

    for x in range(start_fragment[0], end_fragment[0]):
        plane_origin_yz = nyz * (x + 1) * sub_box_size
        vertices_yz, faces_yz = my_slice_faces_plane(
            vertices, faces, -nyz, plane_origin_yz
        )

        for y in range(start_fragment[1], end_fragment[1]):
            plane_origin_xz = nxz * (y + 1) * sub_box_size
            vertices_xz, faces_xz = my_slice_faces_plane(
                vertices_yz, faces_yz, -nxz, plane_origin_xz
            )

            for z in range(start_fragment[2], end_fragment[2]):
                plane_origin_xy = nxy * (z + 1) * sub_box_size
                vertices_xy, faces_xy = my_slice_faces_plane(
                    vertices_xz, faces_xz, -nxy, plane_origin_xy
                )

                lod_0_fragment_position = tuple(np.array([x, y, z]))
                if current_lod != 0:
                    fragment_position = tuple(np.array([x, y, z]) // 2)
                else:
                    fragment_position = lod_0_fragment_position

                update_fragment_dict(
                    combined_fragments_dictionary,
                    fragment_position,
                    vertices_xy,
                    faces_xy,
                    list(lod_0_fragment_position),
                )

                vertices_xz, faces_xz = my_slice_faces_plane(
                    vertices_xz, faces_xz, nxy, plane_origin_xy
                )

            vertices_yz, faces_yz = my_slice_faces_plane(
                vertices_yz, faces_yz, nxz, plane_origin_xz
            )

        vertices, faces = my_slice_faces_plane(vertices, faces, nyz, plane_origin_yz)

    # Compress fragments
    for fragment_pos, fragment in combined_fragments_dictionary.items():
        try:
            if len(fragment.vertices) > 0:
                current_box_size = lod_0_box_size * 2**current_lod
                quantization_origin = np.asarray(fragment_pos) * current_box_size
                quantization_bits = vertex_quantization_bits
                # The spec requires Draco-encoded INTEGER positions in
                # [0, 2^bits).  Compute per-axis integer positions:
                #   q[j] = round(local[j] / current_box_size[j] * max_q)
                # then encode with range=max_q so DracoPy stores them
                # as-is.  Neuroglancer decodes via:
                #   world[j] = grid_origin[j] + chunk_shape[j] * 2^lod
                #              * (frag_pos[j] + q[j] / max_q)
                max_q = float((1 << quantization_bits) - 1)
                local_vertices = fragment.vertices.astype(np.float64) - quantization_origin
                local_vertices = np.clip(local_vertices, 0.0, current_box_size)

                # Snap vertices that fall within half a quantization
                # step of a chunk boundary to the boundary itself. The
                # slicer may emit boundary vertices a hair off the
                # plane in float64; without this snap the same
                # conceptual vertex in two adjacent chunks can round
                # to two different lattice positions, decoding to two
                # slightly-different world coords and creating a
                # visible sub-pixel seam at every chunk boundary.
                half_step = current_box_size / max_q / 2.0
                local_vertices = np.where(
                    local_vertices < half_step, 0.0, local_vertices,
                )
                local_vertices = np.where(
                    local_vertices > current_box_size - half_step,
                    current_box_size,
                    local_vertices,
                )

                # Compute integer quantized positions per axis
                int_positions = np.round(
                    local_vertices * (max_q / current_box_size)
                ).astype(np.float64)
                int_positions = np.clip(int_positions, 0.0, max_q)

                draco_bytes, _ = capture_draco_output(
                    2,
                    DracoPy.encode,
                    points=int_positions,
                    faces=fragment.faces,
                    quantization_bits=quantization_bits,
                    quantization_range=max_q,
                    quantization_origin=[0.0, 0.0, 0.0],
                )

                if len(draco_bytes) > 12:
                    fragment = mesh_io.CompressedFragment(
                        draco_bytes,
                        np.asarray(fragment_pos),
                        len(draco_bytes),
                        np.asarray(fragment.lod_0_fragment_pos),
                    )
                    fragments.append(fragment)

        except Exception as e:
            if "All triangles are degenerate" in str(e):
                print_with_datetime(
                    f"Skipping degenerate fragment {mesh_path}, {fragment_pos}",
                    logger,
                )
            else:
                raise Exception(
                    f"Error processing fragment {mesh_path},{lod_0_box_size},{grid_origin},"
                    f"{start_fragment},{end_fragment},{current_lod},{num_chunks}: {e}"
                )
    return fragments
