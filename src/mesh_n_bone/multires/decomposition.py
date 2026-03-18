from trimesh.intersections import slice_faces_plane
import numpy as np
import DracoPy

from mesh_n_bone.util import mesh_io
from mesh_n_bone.util.logging import print_with_datetime, capture_draco_output
import logging

logger = logging.getLogger(__name__)


def my_slice_faces_plane(vertices, faces, plane_normal, plane_origin):
    """Wrapper for trimesh slice_faces_plane to catch error that happens if the
    whole mesh is to one side of the plane."""
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
    """Update dictionary mapping fragment positions to Fragment objects."""
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
    current_lod, num_chunks,
):
    """Decompose a mesh into fragments of size lod_0_box_size * 2**current_lod.
    Each fragment is also subdivided by 2x2x2."""

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
                quantization_bits = 10

                # Snap to quantization lattice
                max_q = float((1 << quantization_bits) - 1)
                local_vertices = fragment.vertices.astype(np.float64) - quantization_origin
                local_vertices = np.clip(local_vertices, 0.0, current_box_size)
                local_vertices = (
                    np.round(local_vertices * (max_q / current_box_size))
                    * (current_box_size / max_q)
                )

                # Pre-scale for per-axis chunk_shape
                max_cbs = float(np.max(current_box_size))
                scale_factors = max_cbs / current_box_size
                scaled_local = local_vertices * scale_factors
                scaled_origin = np.asarray(fragment_pos, dtype=float) * max_cbs
                quantized_vertices = scaled_origin + scaled_local

                draco_bytes, _ = capture_draco_output(
                    2,
                    DracoPy.encode,
                    points=quantized_vertices,
                    faces=fragment.faces,
                    quantization_bits=quantization_bits,
                    quantization_range=max_cbs,
                    quantization_origin=scaled_origin,
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
