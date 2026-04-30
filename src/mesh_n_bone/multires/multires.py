from contextlib import ExitStack
import numpy as np
import os
import time
import dask
from dask.distributed import worker_client
import logging

from mesh_n_bone.util import mesh_io, dask_util
from mesh_n_bone.util.logging import Timing_Messager, print_with_datetime
from mesh_n_bone.util import neuroglancer
from mesh_n_bone.multires.decomposition import generate_mesh_decomposition
from mesh_n_bone.multires.decimation import (
    generate_decimated_meshes,
    delete_decimated_mesh_files,
)
from mesh_n_bone.config import read_multires_config

logger = logging.getLogger(__name__)


DEFAULT_TARGET_FACES_PER_LOD0_CHUNK = 25_000


def generate_neuroglancer_multires_mesh(
    id, num_subtask_workers, output_path, lods, original_ext, lod_0_box_size=None,
    vertex_quantization_bits=16,
    target_faces_per_lod0_chunk=DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
):
    """Create a complete multiresolution mesh for a single segment.

    Reads the mesh at each LOD, computes an octree decomposition, and
    writes Neuroglancer precomputed fragment and manifest files.

    Parameters
    ----------
    id : int
        Segment ID.
    num_subtask_workers : int
        Number of Dask sub-workers to use for decomposition.
    output_path : str
        Root output directory (multires files go under
        ``output_path/multires/``).
    lods : list of int
        LOD levels to process, e.g. ``[0, 1, 2]``.
    original_ext : str
        File extension of LOD 0 meshes (e.g. ``".ply"``).
    lod_0_box_size : ndarray of float or None
        Chunk box size for LOD 0. If ``None``, computed from mesh
        bounding box targeting ``target_faces_per_lod0_chunk`` faces
        per chunk.
    target_faces_per_lod0_chunk : int
        Target face count per LOD-0 fragment used by the
        auto-sizing heuristic. Lower values produce more, smaller
        chunks (and force multi-LOD output for smaller meshes);
        higher values keep small meshes in a single fragment so
        they collapse to 1 LOD. Ignored when *lod_0_box_size* is
        passed explicitly.
    """
    with ExitStack() as stack:
        if num_subtask_workers > 1:
            client = stack.enter_context(worker_client())

        os.makedirs(f"{output_path}/multires", exist_ok=True)
        os.system(
            f"rm -rf {output_path}/multires/{id} {output_path}/multires/{id}.index"
        )

        vertex_min = None
        vertex_max = None
        # Union of vertex bboxes across ALL valid LODs. The grid below
        # is sized from LOD 0's bbox (the canonical extent), but
        # decimated LODs can drift slightly outside it; the empty-
        # placeholder logic needs to cover the full union so NG always
        # has a finer-LOD fragment under any region a coarser LOD has
        # geometry in (otherwise zoom-in transitions show two scales
        # at once on the regions LOD 0 doesn't cover but LOD 1+ does).
        union_vertex_min = None
        union_vertex_max = None
        previous_num_faces = np.inf
        for idx, current_lod in enumerate(lods):
            if current_lod == 0:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
            else:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}.ply"

            vertices, faces = mesh_io.mesh_loader(mesh_path)
            if faces is None:
                break

            num_faces = len(faces)
            if num_faces >= previous_num_faces:
                break
            # LOD-0 bounds drive the chunk grid; union bounds drive
            # the empty-placeholder envelope (which must cover every
            # LOD's actual face footprint, not just LOD 0's).
            if current_lod == 0 and vertices is not None:
                vertex_min = vertices.min(axis=0)
                vertex_max = vertices.max(axis=0)
            if vertices is not None and len(vertices) > 0:
                lod_vmin = vertices.min(axis=0)
                lod_vmax = vertices.max(axis=0)
                if union_vertex_min is None:
                    union_vertex_min = lod_vmin.copy()
                    union_vertex_max = lod_vmax.copy()
                else:
                    union_vertex_min = np.minimum(union_vertex_min, lod_vmin)
                    union_vertex_max = np.maximum(union_vertex_max, lod_vmax)

            if lod_0_box_size is None and current_lod == 0:
                distances_per_axis = np.ceil(
                    vertices.max(axis=0) - vertices.min(axis=0)
                )
                # Surface-area scaling: triangles distribute across the
                # mesh's 2-D surface, so total chunks ∝ N²-on-axis. Use
                # sqrt for the per-axis count.
                heuristic_num_chunks = np.ceil(num_faces / target_faces_per_lod0_chunk)
                num_chunks_per_axis = max(
                    1, int(np.ceil(np.sqrt(heuristic_num_chunks)))
                )
                # Cap so each top-LOD chunk fits within the mesh extent
                # along every axis. Otherwise NG's per-chunk LOD-select
                # gate doesn't fire for the oversized root chunk and
                # the coarsest LOD stays painted persistently. The
                # constraint is num_chunks_per_axis >= octree_unit.
                octree_unit = 2 ** (len(lods) - 1)
                num_chunks_per_axis = max(num_chunks_per_axis, octree_unit)
                lod_0_box_size = (
                    np.ceil(distances_per_axis / num_chunks_per_axis) + 1
                )

            previous_num_faces = num_faces
        else:
            # Loop completed without break — all LODs are valid
            idx += 1

        lods = lods[:idx]

        # Compute the LOD 0 chunk grid from the s0 mesh extent.
        mesh_extent = vertex_max - vertex_min
        num_chunks_per_axis = np.maximum(
            np.ceil(mesh_extent / lod_0_box_size).astype(int), 1
        )

        # No LOD-count truncation here. The union-bbox empty-placeholder
        # logic in ``rewrite_index_with_empty_fragments`` keeps the
        # listed-fragment grid tight regardless of LOD count, and a
        # single-LOD-0-chunk mesh still benefits from extra decimation
        # passes when the user is zoomed far out.

        # Center the mesh within the full octree grid so that
        # Neuroglancer's bounding-box center matches the actual mesh
        # center.
        octree_unit = 2 ** (len(lods) - 1)
        total_chunks_per_axis = (
            np.ceil(num_chunks_per_axis / octree_unit).astype(int)
            * octree_unit
        )
        full_grid_extent = total_chunks_per_axis * lod_0_box_size
        bbox_center = (vertex_min + vertex_max) / 2
        grid_origin = np.floor(bbox_center - full_grid_extent / 2)
        grid_origin = np.clip(
            grid_origin,
            np.ceil(vertex_max - full_grid_extent),
            np.floor(vertex_min),
        )

        # Convert union-of-LOD-vertex bbox to LOD-0 chunk units. This is
        # the face-correct footprint we list at every LOD: a chunk with
        # any face in any LOD must have a fragment (possibly empty) at
        # every other LOD whose chunk contains it, so cross-LOD zoom
        # transitions don't show two scales side by side.
        union_lod0_min = np.maximum(
            np.floor((union_vertex_min - grid_origin) / lod_0_box_size).astype(int),
            0,
        )
        union_lod0_max = np.maximum(
            np.ceil((union_vertex_max - grid_origin) / lod_0_box_size).astype(int),
            union_lod0_min + 1,
        )

        results = []
        for idx, current_lod in enumerate(lods):
            if current_lod == 0:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
            else:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}.ply"

            vertices, _ = mesh_io.mesh_loader(mesh_path)

            if vertices is not None:
                vertices -= grid_origin

                current_box_size = lod_0_box_size * 2**current_lod
                start_fragment = np.maximum(
                    vertices.min(axis=0) // current_box_size, np.array([0, 0, 0])
                ).astype(int)
                end_fragment = (vertices.max(axis=0) // current_box_size + 1).astype(int)

                del vertices

                max_number_of_chunks = end_fragment - start_fragment
                dimensions_sorted = np.argsort(-max_number_of_chunks)
                num_chunks = np.array([1, 1, 1])

                for _ in range(num_subtask_workers + 1):
                    for d in dimensions_sorted:
                        if num_chunks[d] < max_number_of_chunks[d]:
                            num_chunks[d] += 1
                            if np.prod(num_chunks) > num_subtask_workers:
                                num_chunks[d] -= 1
                            break

                stride = np.ceil(
                    1.0 * (end_fragment - start_fragment) / num_chunks
                ).astype(int)

                decomposition_results = []
                for x in range(start_fragment[0], end_fragment[0], stride[0]):
                    for y in range(start_fragment[1], end_fragment[1], stride[1]):
                        for z in range(start_fragment[2], end_fragment[2], stride[2]):
                            current_start_fragment = np.array([x, y, z])
                            current_end_fragment = current_start_fragment + stride
                            if num_subtask_workers == 1:
                                decomposition_results.append(
                                    generate_mesh_decomposition(
                                        mesh_path,
                                        lod_0_box_size,
                                        grid_origin,
                                        current_start_fragment,
                                        current_end_fragment,
                                        current_lod,
                                        num_chunks,
                                        vertex_quantization_bits,
                                    )
                                )
                            else:
                                results.append(
                                    dask.delayed(generate_mesh_decomposition)(
                                        mesh_path,
                                        lod_0_box_size,
                                        grid_origin,
                                        current_start_fragment,
                                        current_end_fragment,
                                        current_lod,
                                        num_chunks,
                                        vertex_quantization_bits,
                                    )
                                )

                if num_subtask_workers > 1:
                    client.rebalance()
                    decomposition_results = dask.compute(*results)

                results = []

                decomposition_results = [
                    fragments for fragments in decomposition_results if fragments
                ]

                fragments = [
                    fragment
                    for fragments in decomposition_results
                    for fragment in fragments
                ]

                del decomposition_results

                mesh_io.write_mesh_files(
                    f"{output_path}/multires",
                    f"{id}",
                    grid_origin,
                    fragments,
                    current_lod,
                    lods[: idx + 1],
                    np.asarray(lod_0_box_size, dtype=float),
                    union_lod0_min=union_lod0_min,
                    union_lod0_max=union_lod0_max,
                )

                del fragments


def _mesh_intersects_roi(mesh_path, roi_begin, roi_end):
    """Check if a mesh's bounding box intersects the given ROI.

    Parameters
    ----------
    mesh_path : str
        Path to a mesh file.
    roi_begin : ndarray, shape (3,)
        ROI lower bound in XYZ world coordinates.
    roi_end : ndarray, shape (3,)
        ROI upper bound in XYZ world coordinates.

    Returns
    -------
    bool
        ``True`` if the mesh's axis-aligned bounding box overlaps the ROI.
    """
    vertices, _ = mesh_io.mesh_loader(mesh_path)
    if vertices is None or len(vertices) == 0:
        return False
    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    # Check for overlap in all 3 dimensions
    return np.all(mesh_min <= roi_end) and np.all(mesh_max >= roi_begin)


def generate_all_neuroglancer_multires_meshes(
    output_path, num_workers, ids, lods, original_ext, file_sizes,
    lod_0_box_size=None, vertex_quantization_bits=16,
    target_faces_per_lod0_chunk=DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
):
    """Generate Neuroglancer multiresolution meshes for all segments.

    Distributes work across Dask workers, allocating sub-workers to each
    segment proportional to its mesh file size.

    Parameters
    ----------
    output_path : str
        Root output directory.
    num_workers : int
        Total number of Dask workers.
    ids : list of int
        Segment IDs to process.
    lods : list of int
        LOD levels, e.g. ``[0, 1, 2]``.
    original_ext : str
        File extension of LOD 0 meshes.
    file_sizes : ndarray of float
        File sizes of LOD 0 meshes (used for work balancing).
    lod_0_box_size : ndarray of float or None
        Chunk box size for LOD 0. ``None`` for auto-computation.
    target_faces_per_lod0_chunk : int
        Forwarded to :func:`generate_neuroglancer_multires_mesh`.
    """

    def get_number_of_subtask_workers(file_sizes, num_workers):
        total_file_size = np.sum(file_sizes)
        num_workers_per_byte = num_workers / total_file_size
        num_subtask_workers = np.ceil(file_sizes * num_workers_per_byte).astype(int)
        return num_subtask_workers

    num_subtask_workers = get_number_of_subtask_workers(file_sizes, num_workers)
    variable_args_list = []
    fixed_args_list = [
        output_path, lods, original_ext, lod_0_box_size,
        vertex_quantization_bits, target_faces_per_lod0_chunk,
    ]
    for idx, id in enumerate(ids):
        variable_args_list.append((id, num_subtask_workers[idx]))
    dask_util.compute_bag(
        generate_neuroglancer_multires_mesh,
        f"{output_path}/variable_args_to_multires.npy",
        variable_args_list,
        fixed_args_list,
        num_workers,
    )


def run_multires(config_path, num_workers, roi=None):
    """Main entry point for the multiresolution pipeline.

    Reads a config, generates decimated LODs, builds Neuroglancer
    precomputed multiresolution meshes, and writes metadata files.

    Parameters
    ----------
    config_path : str
        Directory containing ``run-config.yaml``.
    num_workers : int
        Number of Dask workers.
    roi : dict or None
        Optional ROI with ``begin``/``end`` or ``offset``/``shape``
        keys in XYZ world coordinates. Only meshes intersecting this
        region will be processed.
    """
    submission_directory = os.getcwd()
    required_settings, optional_decimation_settings, optional_properties_settings = (
        read_multires_config(config_path)
    )

    input_path = required_settings["input_path"]
    output_path = required_settings["output_path"]
    num_lods = required_settings["num_lods"]

    lod_0_box_size = optional_decimation_settings["box_size"]
    skip_decimation = optional_decimation_settings["skip_decimation"]
    decimation_factor = optional_decimation_settings["decimation_factor"]
    aggressiveness = optional_decimation_settings["aggressiveness"]
    delete_decimated_meshes_flag = optional_decimation_settings["delete_decimated_meshes"]
    target_faces_per_lod0_chunk = optional_decimation_settings["target_faces_per_lod0_chunk"]

    segment_properties_csv = optional_properties_settings["segment_properties_csv"]
    segment_properties_columns = optional_properties_settings["segment_properties_columns"]
    segment_properties_id_column = optional_properties_settings["segment_properties_id_column"]

    # Merge ROI from config if not provided via CLI
    if roi is None:
        roi = optional_decimation_settings.get("roi")

    # Parse ROI into begin/end arrays (XYZ world coordinates)
    roi_begin = None
    roi_end = None
    if roi is not None:
        if "begin" in roi and "end" in roi:
            roi_begin = np.asarray(roi["begin"], dtype=float)
            roi_end = np.asarray(roi["end"], dtype=float)
        elif "offset" in roi and "shape" in roi:
            roi_begin = np.asarray(roi["offset"], dtype=float)
            roi_end = roi_begin + np.asarray(roi["shape"], dtype=float)
        else:
            raise ValueError(
                "roi must have 'begin'+'end' or 'offset'+'shape' keys"
            )

    execution_directory = dask_util.setup_execution_directory(config_path, logger)
    logpath = f"{execution_directory}/output.log"

    from mesh_n_bone.util.logging import tee_streams

    with tee_streams(logpath):
        try:
            os.chdir(execution_directory)

            lods = list(range(num_lods))

            mesh_ids = []
            mesh_ext = None
            file_sizes = []
            with os.scandir(input_path) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    root, ext = os.path.splitext(name)
                    if mesh_ext is None:
                        mesh_ext = ext
                    file_sizes.append(entry.stat(follow_symlinks=False).st_size)
                    mesh_ids.append(int(root))
            t0 = time.time()

            # Filter meshes by ROI if specified
            if roi_begin is not None:
                kept_ids = []
                kept_sizes = []
                for idx, mesh_id in enumerate(mesh_ids):
                    mesh_path = os.path.join(input_path, f"{mesh_id}{mesh_ext}")
                    if _mesh_intersects_roi(mesh_path, roi_begin, roi_end):
                        kept_ids.append(mesh_id)
                        kept_sizes.append(file_sizes[idx])
                logger.info(
                    f"ROI filter: {len(kept_ids)}/{len(mesh_ids)} meshes "
                    f"intersect the specified region"
                )
                mesh_ids = kept_ids
                file_sizes = kept_sizes

            if not skip_decimation:
                with dask_util.start_dask(num_workers, "decimation", logger):
                    with Timing_Messager("Generating decimated meshes", logger):
                        generate_decimated_meshes(
                            input_path, output_path, lods, mesh_ids, mesh_ext,
                            decimation_factor, aggressiveness, num_workers,
                        )

            with dask_util.start_dask(num_workers, "multires creation", logger):
                with Timing_Messager("Generating multires meshes", logger):
                    generate_all_neuroglancer_multires_meshes(
                        output_path, num_workers, mesh_ids, lods, mesh_ext,
                        np.array(file_sizes), lod_0_box_size,
                        vertex_quantization_bits=16,
                        target_faces_per_lod0_chunk=target_faces_per_lod0_chunk,
                    )

            with Timing_Messager("Writing info and segment properties files", logger):
                multires_output_path = f"{output_path}/multires"
                neuroglancer.write_segment_properties_file(
                    multires_output_path,
                    csv_path=segment_properties_csv,
                    csv_columns=segment_properties_columns,
                    csv_id_column=segment_properties_id_column,
                )
                neuroglancer.write_info_file(
                    multires_output_path, vertex_quantization_bits=16,
                )

            if not skip_decimation and delete_decimated_meshes_flag:
                with dask_util.start_dask(
                    num_workers, "delete decimated meshes", logger
                ):
                    with Timing_Messager("Deleting decimated meshes", logger):
                        delete_decimated_mesh_files(
                            output_path, lods, mesh_ids, num_workers,
                        )
                        os.system(f"rm -rf {output_path}/mesh_lods")

            print_with_datetime(
                f"Complete! Elapsed time: {time.time() - t0}", logger
            )
        finally:
            os.chdir(submission_directory)
