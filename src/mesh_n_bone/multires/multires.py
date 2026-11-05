from contextlib import ExitStack
import numpy as np
import os
import shutil
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
# Hemibrain uses 1024 voxels per LOD-0 chunk (= 16 μm at their 16-nm
# meshing scale). Same number used here as a sensible dataset-level
# default: 1024 × voxel_size for a single isotropic chunk size that
# behaves consistently across mesh sizes.
DEFAULT_BOX_SIZE_VOXELS = 1024
# A LOD whose decimated face count would fall below this is dropped from
# per-segment output — further decimation produces degenerate meshes.
MIN_USEFUL_FACES_PER_LOD = 50


def generate_neuroglancer_multires_mesh(
    id, num_subtask_workers, output_path, lods, original_ext, lod_0_box_size=None,
    vertex_quantization_bits=16,
    target_faces_per_lod0_chunk=DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
    voxel_size_nm=1.0,
    min_useful_faces=MIN_USEFUL_FACES_PER_LOD,
    center_octree=True,
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
        higher values keep small meshes in fewer fragments and may
        reduce the effective LOD count to avoid an over-padded
        octree. Ignored when *lod_0_box_size* is passed explicitly.
    """
    with ExitStack() as stack:
        if num_subtask_workers > 1:
            client = stack.enter_context(worker_client())

        os.makedirs(f"{output_path}/multires", exist_ok=True)
        # Clear a previous attempt's files. Directly, not via a shell: a
        # process spawn per segment costs hours over millions of segments.
        for stale in (f"{output_path}/multires/{id}", f"{output_path}/multires/{id}.index"):
            try:
                os.remove(stale)
            except FileNotFoundError:
                pass
            except IsADirectoryError:
                shutil.rmtree(stale)

        vertex_min = None
        vertex_max = None
        requested_lods = list(lods)
        requested_lod_count = len(requested_lods)
        lod_face_counts = []
        previous_num_faces = np.inf
        lod_truncation_reason = None
        for idx, current_lod in enumerate(lods):
            if current_lod == 0:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}{original_ext}"
            else:
                mesh_path = f"{output_path}/mesh_lods/s{current_lod}/{id}.ply"

            vertices, faces = mesh_io.mesh_loader(mesh_path)
            if faces is None:
                lod_truncation_reason = (
                    f"Segment {id} using {idx}/{requested_lod_count} requested "
                    f"LODs; dropping LOD {current_lod} and later because "
                    f"{mesh_path} could not be loaded or has no faces."
                )
                break

            num_faces = len(faces)
            if num_faces > previous_num_faces:
                previous_lod, previous_faces = lod_face_counts[-1]
                lod_truncation_reason = (
                    f"Segment {id} using {idx}/{requested_lod_count} requested "
                    f"LODs; dropping LOD {current_lod} and later because it "
                    f"has {num_faces} faces, more than LOD "
                    f"{previous_lod} ({previous_faces} faces)."
                )
                break
            if current_lod > 0 and num_faces < min_useful_faces:
                lod_truncation_reason = (
                    f"Segment {id} using {idx}/{requested_lod_count} requested "
                    f"LODs; dropping LOD {current_lod} and later because it "
                    f"has {num_faces} faces, below the useful-detail "
                    f"threshold ({min_useful_faces}) — further decimation "
                    f"produces degenerate meshes."
                )
                break
            lod_face_counts.append((current_lod, num_faces))
            # LOD-0 bounds drive the chunk grid.
            if current_lod == 0 and vertices is not None:
                vertex_min = vertices.min(axis=0)
                vertex_max = vertices.max(axis=0)

            # Capture LOD-0 face count for the chunk-shape heuristic
            # below. We compute lod_0_box_size AFTER the loop so it
            # reflects the effective num_lods (after any face-count
            # truncation), not the requested num_lods.
            if current_lod == 0:
                lod0_num_faces = num_faces
                lod0_distances_per_axis = np.ceil(
                    vertices.max(axis=0) - vertices.min(axis=0)
                )

            previous_num_faces = num_faces
        else:
            # Loop completed without break — all LODs are valid
            idx += 1

        # A trailing plateau adds nothing: a coarser level with the same
        # face count as the one before it draws the same triangles at far
        # zoom and only stores the mesh again. Equal counts stay in the
        # middle of the chain, where a cheaper level follows and the
        # format cannot skip a level.
        while len(lod_face_counts) > 1 and lod_face_counts[-1][1] == lod_face_counts[-2][1]:
            dropped_lod, dropped_faces = lod_face_counts.pop()
            idx -= 1
            lod_truncation_reason = (
                f"Segment {id} using {idx}/{requested_lod_count} requested "
                f"LODs; dropping trailing LOD {dropped_lod} because it has "
                f"{dropped_faces} faces, the same as LOD {lod_face_counts[-1][0]}."
            )

        lods = lods[:idx]
        if lod_truncation_reason is not None:
            logger.info(lod_truncation_reason)
        if not lods:
            logger.warning(
                "Segment %s has no valid LOD meshes from requested LODs %s; "
                "skipping Neuroglancer multires output.",
                id,
                requested_lods,
            )
            return

        if lod_0_box_size is None:
            if target_faces_per_lod0_chunk != DEFAULT_TARGET_FACES_PER_LOD0_CHUNK:
                # User explicitly tuned target_faces_per_lod0_chunk →
                # use the adaptive surface-area heuristic.
                heuristic_num_chunks = np.ceil(
                    lod0_num_faces / target_faces_per_lod0_chunk
                )
                num_chunks_per_axis = max(
                    1, int(np.ceil(np.sqrt(heuristic_num_chunks)))
                )
                lod_0_box_size = (
                    np.ceil(lod0_distances_per_axis / num_chunks_per_axis) + 1
                )
            else:
                # Dataset-level default: fixed N voxels per chunk side
                # (matches hemibrain's setup at 1024 voxels). Gives
                # consistent LOD-transition zoom thresholds across mesh
                # sizes — a 1 μm mito and a 100 μm neuron get the same
                # physical chunk size, just different chunk counts.
                lod_0_box_size = np.full(
                    3, DEFAULT_BOX_SIZE_VOXELS * voxel_size_nm, dtype=float,
                )

        # Compute the LOD 0 chunk grid from the s0 mesh extent.
        mesh_extent = vertex_max - vertex_min
        num_chunks_per_axis = np.maximum(
            np.ceil(mesh_extent / lod_0_box_size).astype(int), 1
        )

        # Position the octree relative to the mesh bbox.
        #
        # center_octree=True (default): the mesh sits in the middle of
        # the LOD-(top) chunk. NG's "double-click to focus" uses the
        # top-LOD chunk center as the segment's centroid — centering
        # makes that land on the mesh. The cost is that meshes which
        # would otherwise fit in a single LOD-0 chunk straddle the
        # octree's center boundary, producing 2×2×2 = 8 LOD-0 fragments
        # and visible chunk seams at high zoom.
        #
        # center_octree=False: align the octree to floor(bbox_min).
        # No internal-seam subdivision for sub-octree-sized meshes
        # (they fit in 1 LOD-0 chunk). Click-to-focus may land off the
        # mesh — toward the LOD-(top) chunk center, which is at the
        # mesh's near corner for sub-chunk meshes. Choose this when
        # seam-free rendering matters more than precise click-to-center
        # (e.g. dense-organelle visualization).
        octree_unit = 2 ** (len(lods) - 1)
        total_chunks_per_axis = (
            np.ceil(num_chunks_per_axis / octree_unit).astype(int)
            * octree_unit
        )
        full_grid_extent = total_chunks_per_axis * lod_0_box_size
        if center_octree:
            bbox_center = (vertex_min + vertex_max) / 2
            grid_origin = np.floor(bbox_center - full_grid_extent / 2)
            grid_origin = np.clip(
                grid_origin,
                np.ceil(vertex_max - full_grid_extent),
                np.floor(vertex_min),
            )
        else:
            grid_origin = np.floor(vertex_min)

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


# Pack files of small segments' multires chunks, written by
# generate_small_multires_batch and read by the shard packer.
SMALL_CHUNKS_SUBDIR = "_small_chunks"
_CHUNK_MANIFEST_DTYPE = np.dtype([
    ("id", "u8"), ("offset", "u8"), ("bytes", "u8"), ("manifest_size", "u8"),
])


def _local_scratch():
    """A directory on this machine's own disk: the dask worker's local
    directory on a cluster node, else the system temporary directory."""
    try:
        from distributed import get_worker
        return get_worker().local_directory
    except (ImportError, ValueError):
        import tempfile
        return tempfile.gettempdir()


def generate_small_multires_batch(
    entries, chunk_dir, name, lods, original_ext=".ply", lod_0_box_size=None,
    vertex_quantization_bits=16,
    target_faces_per_lod0_chunk=DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
    voxel_size_nm=1.0,
    min_useful_faces=MIN_USEFUL_FACES_PER_LOD,
    center_octree=True,
):
    """Multires chunks for a batch of small segments, into one pack file.

    *entries* is a list of ``(segment id, [(lod, pack path, offset,
    bytes), ...])``: where each of the segment's level meshes sits in the
    packs the meshing passes wrote. The meshes are copied to a scratch
    directory on this machine's own disk, laid out as
    :func:`generate_neuroglancer_multires_mesh` reads them, and that same
    function writes each segment's ``<id>`` and ``<id>.index`` there. The
    results go into ``<chunk_dir>/<name>.pack``, one shard-ready chunk per
    segment (fragment data then manifest, as the shard packer joins
    them), with a manifest of offsets written last.

    So a segment costs no file operation on the shared filesystem beyond
    its share of a few sequential reads and one write. A batch whose
    manifest exists was finished by an earlier attempt and is skipped.

    Returns the ids of the segments that got a chunk.
    """
    import tempfile
    from mesh_n_bone.util.sharded_mesh_util import _read_segment_chunk

    manifest_dir = os.path.join(chunk_dir, "_manifests")
    manifest_path = os.path.join(manifest_dir, f"{name}.npy")
    if os.path.exists(manifest_path):
        return [int(i) for i in np.load(manifest_path)["id"]]

    with tempfile.TemporaryDirectory(prefix="multires-", dir=_local_scratch()) as tmp:
        reads = sorted(
            (path, int(offset), int(nbytes), int(lod), int(segment_id))
            for segment_id, sources in entries
            for lod, path, offset, nbytes in sources
        )
        made = set()
        handle_path, handle = None, None
        try:
            for path, offset, nbytes, lod, segment_id in reads:
                if path != handle_path:
                    if handle is not None:
                        handle.close()
                    handle_path, handle = path, open(path, "rb")
                handle.seek(offset)
                data = handle.read(nbytes)
                if len(data) != nbytes:
                    raise IOError(
                        f"{path} is shorter than its manifest says "
                        f"({len(data)} of {nbytes} bytes at {offset})"
                    )
                lod_dir = f"{tmp}/mesh_lods/s{lod}"
                if lod_dir not in made:
                    os.makedirs(lod_dir, exist_ok=True)
                    made.add(lod_dir)
                ext = original_ext if lod == 0 else ".ply"
                with open(f"{lod_dir}/{segment_id}{ext}", "wb") as fh:
                    fh.write(data)
        finally:
            if handle is not None:
                handle.close()

        for segment_id, _ in entries:
            generate_neuroglancer_multires_mesh(
                int(segment_id), 1, tmp, lods, original_ext, lod_0_box_size,
                vertex_quantization_bits=vertex_quantization_bits,
                target_faces_per_lod0_chunk=target_faces_per_lod0_chunk,
                voxel_size_nm=voxel_size_nm,
                min_useful_faces=min_useful_faces,
                center_octree=center_octree,
            )

        os.makedirs(manifest_dir, exist_ok=True)
        pack_path = os.path.join(chunk_dir, f"{name}.pack")
        rows = []
        offset = 0
        with open(pack_path + ".partial", "wb") as fh:
            for segment_id, _ in entries:
                chunk = _read_segment_chunk(f"{tmp}/multires", int(segment_id))
                if chunk is None:
                    continue
                binary, manifest_size = chunk
                fh.write(binary)
                rows.append((int(segment_id), offset, len(binary), manifest_size))
                offset += len(binary)
        os.replace(pack_path + ".partial", pack_path)
        with open(manifest_path + ".partial", "wb") as fh:
            np.save(fh, np.array(rows, dtype=_CHUNK_MANIFEST_DTYPE))
        os.replace(manifest_path + ".partial", manifest_path)
    return [row[0] for row in rows]


def read_small_chunk_index(chunk_dir):
    """``{segment id: (pack path, offset, bytes, manifest_size)}`` for
    every chunk :func:`generate_small_multires_batch` wrote under
    *chunk_dir*."""
    manifest_dir = os.path.join(chunk_dir, "_manifests")
    index = {}
    if not os.path.isdir(manifest_dir):
        return index
    for name in sorted(os.listdir(manifest_dir)):
        if not name.endswith(".npy") or name.endswith(".partial.npy"):
            continue
        pack_path = os.path.join(chunk_dir, name[:-len(".npy")] + ".pack")
        for row in np.load(os.path.join(manifest_dir, name)):
            index[int(row["id"])] = (
                pack_path, int(row["offset"]), int(row["bytes"]), int(row["manifest_size"]),
            )
    return index


def generate_all_neuroglancer_multires_meshes(
    output_path, num_workers, ids, lods, original_ext, file_sizes,
    lod_0_box_size=None, vertex_quantization_bits=16,
    target_faces_per_lod0_chunk=DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
    voxel_size_nm=1.0,
    min_useful_faces=MIN_USEFUL_FACES_PER_LOD,
    center_octree=True,
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
        voxel_size_nm, min_useful_faces, center_octree,
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
    (
        required_settings,
        optional_decimation_settings,
        optional_properties_settings,
        sharding_settings,
    ) = read_multires_config(config_path)

    input_path = required_settings["input_path"]
    output_path = required_settings["output_path"]
    num_lods = required_settings["num_lods"]

    lod_0_box_size = optional_decimation_settings["box_size"]
    skip_decimation = optional_decimation_settings["skip_decimation"]
    decimation_factor = optional_decimation_settings["decimation_factor"]
    aggressiveness = optional_decimation_settings["aggressiveness"]
    delete_decimated_meshes_flag = optional_decimation_settings["delete_decimated_meshes"]
    target_faces_per_lod0_chunk = optional_decimation_settings["target_faces_per_lod0_chunk"]
    voxel_size_nm = float(optional_decimation_settings.get("voxel_size_nm", 1.0))
    center_octree = bool(optional_decimation_settings.get("center_octree", True))
    retry_on_oom = optional_decimation_settings.get("retry_on_oom", True)
    memory_retry_max = optional_decimation_settings.get("memory_retry_max", 3)

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

            effective_workers = dask_util.effective_num_workers(
                num_workers, len(mesh_ids), logger, "multires",
            )

            if not skip_decimation:
                def _run_decimation(workers, config):
                    with dask_util.start_dask(
                        workers, "decimation", logger, config=config,
                    ):
                        with Timing_Messager("Generating decimated meshes", logger):
                            generate_decimated_meshes(
                                input_path, output_path, lods, mesh_ids, mesh_ext,
                                decimation_factor, aggressiveness, workers,
                            )
                dask_util.run_with_oom_retry(
                    _run_decimation, effective_workers, "decimation", logger,
                    max_retries=memory_retry_max, retry_on_oom=retry_on_oom,
                )

            def _run_multires(workers, config):
                with dask_util.start_dask(
                    workers, "multires creation", logger, config=config,
                ):
                    with Timing_Messager("Generating multires meshes", logger):
                        generate_all_neuroglancer_multires_meshes(
                            output_path, workers, mesh_ids, lods, mesh_ext,
                            np.array(file_sizes), lod_0_box_size,
                            vertex_quantization_bits=16,
                            target_faces_per_lod0_chunk=target_faces_per_lod0_chunk,
                            voxel_size_nm=voxel_size_nm,
                            center_octree=center_octree,
                        )
            dask_util.run_with_oom_retry(
                _run_multires, effective_workers, "multires creation", logger,
                max_retries=memory_retry_max, retry_on_oom=retry_on_oom,
            )

            multires_output_path = f"{output_path}/multires"
            with Timing_Messager("Writing segment properties file", logger):
                # Write segment_properties before sharding so it can still scan
                # the per-segment `.index` files; they get removed once shards
                # are packed.
                neuroglancer.write_segment_properties_file(
                    multires_output_path,
                    csv_path=segment_properties_csv,
                    csv_columns=segment_properties_columns,
                    csv_id_column=segment_properties_id_column,
                )

            if sharding_settings["sharded"]:
                from mesh_n_bone.util import sharded_mesh_util

                params = sharded_mesh_util.choose_shard_params(len(mesh_ids))
                for key in ("preshift_bits", "minishard_bits", "shard_bits"):
                    override = sharding_settings.get(key)
                    if override is not None:
                        params[key] = int(override)
                spec = sharded_mesh_util.make_sharding_spec(**params)

                def _pack(workers, config):
                    with dask_util.start_dask(
                        workers, "shard packing", logger, config=config,
                    ):
                        with Timing_Messager(
                            f"Packing meshes into sharded format ({params})", logger
                        ):
                            sharded_mesh_util.pack_meshes_to_shards(
                                multires_output_path, mesh_ids, spec, workers,
                            )

                dask_util.run_with_oom_retry(
                    _pack, effective_workers, "shard packing", logger,
                    max_retries=memory_retry_max, retry_on_oom=retry_on_oom,
                )

                with Timing_Messager("Writing sharded info file", logger):
                    sharded_mesh_util.write_sharded_info_file(
                        multires_output_path, spec, vertex_quantization_bits=16,
                        lod_scale_multiplier=voxel_size_nm,
                    )

                if sharding_settings["delete_unsharded_files"]:
                    with Timing_Messager(
                        "Deleting unsharded per-segment files", logger
                    ):
                        sharded_mesh_util.delete_unsharded_segment_files(
                            multires_output_path, mesh_ids,
                            num_workers=max(1, int(num_workers)), logger=logger,
                        )
            else:
                with Timing_Messager("Writing info file", logger):
                    neuroglancer.write_info_file(
                        multires_output_path, vertex_quantization_bits=16,
                        lod_scale_multiplier=voxel_size_nm,
                    )

            if not skip_decimation and delete_decimated_meshes_flag:
                with dask_util.start_dask(
                    effective_workers, "delete decimated meshes", logger
                ):
                    with Timing_Messager("Deleting decimated meshes", logger):
                        delete_decimated_mesh_files(
                            output_path, lods, mesh_ids, effective_workers,
                        )
                        mesh_lods_dir = f"{output_path}/mesh_lods"
                        if os.path.isdir(mesh_lods_dir) and not os.path.islink(mesh_lods_dir):
                            dask_util.parallel_delete_tree(
                                mesh_lods_dir, num_workers=effective_workers, logger=logger,
                            )

            print_with_datetime(
                f"Complete! Elapsed time: {time.time() - t0}", logger
            )
        finally:
            os.chdir(submission_directory)
