"""Mesh generation from segmentation volumes using zmesh and dask."""

from dataclasses import asdict, dataclass
import copy
from funlib.geometry import Roi, Coordinate
import numpy as np
import os
import logging
from zmesh import Mesher
from zmesh import Mesh as Zmesh
import dask.bag as db
from cloudvolume.mesh import Mesh as CloudVolumeMesh
import shutil
import trimesh
import json
import pymeshlab
import fastremap
from mesh_n_bone.util import dask_util
from mesh_n_bone.util.logging import Timing_Messager
from mesh_n_bone.util.zarr_io import (
    open_dataset,
    split_dataset_path,
    read_raw_voxel_size,
    read_raw_offset,
    _read_attrs,
    _get_multiscales,
    _extract_ome_scale_translation,
    _first_multiscales_dataset_path,
    _path_basename,
    _path_dirname,
    _path_join,
)
from mesh_n_bone.util.image_data_interface import (
    _detect_zarr_driver,
    open_ds_tensorstore,
    to_ndarray_tensorstore,
)
from mesh_n_bone.meshify.downsample import (
    downsample_labels_3d_suppress_zero,
    downsample_labels_3d,
    downsample_binary_3d,
)

logger = logging.getLogger(__name__)


_OME_UNIT_TO_ABBREVIATION = {
    "angstrom": "Å",
    "nanometer": "nm",
    "micrometer": "um",
    "millimeter": "mm",
    "centimeter": "cm",
    "meter": "m",
}


def _read_ome_ngff_transform(input_path):
    """Extract voxel_size, offset, and coordinate unit from OME-NGFF metadata.

    Reads multiscales from the parent group of *input_path*. Robust to
    OME-Zarr v0.4 / v0.5 layouts, non-ZYX axes, root-level
    coordinateTransformations, and arbitrary dataset paths — all via
    :func:`mesh_n_bone.util.zarr_io._extract_ome_scale_translation`.

    Returns ``(voxel_size, offset, coordinate_units)`` in ZYX order
    (or ``(None, None, None)`` when no metadata is found). The voxel
    size and offset are returned as ``np.ndarray`` for consistency
    with existing callers.
    """
    zarr_root_path, dataset_path = split_dataset_path(input_path)
    if dataset_path:
        dataset_name = _path_basename(dataset_path)
        parent_path = _path_dirname(dataset_path)
        parent_dir = _path_join(zarr_root_path, parent_path) if parent_path else zarr_root_path
    else:
        dataset_name = _path_basename(input_path)
        parent_dir = _path_dirname(input_path)

    try:
        parent_attrs = _read_attrs(parent_dir)
        if not dataset_path:
            input_attrs = _read_attrs(input_path)
            selected_dataset_path = _first_multiscales_dataset_path(input_attrs)
            if selected_dataset_path:
                parent_attrs = input_attrs
                dataset_name = selected_dataset_path
        multiscales = _get_multiscales(parent_attrs)
        if not multiscales:
            return None, None, None

        scale, translation = _extract_ome_scale_translation(
            parent_attrs, dataset_name=dataset_name,
        )
        if scale is None and translation is None:
            return None, None, None

        # Pull the unit off the first spatial axis (units are per-axis but
        # mesh-n-bone treats voxel_size isotropically here).
        coordinate_units = None
        for ax in multiscales[0].get("axes", []) or []:
            if isinstance(ax, dict) and ax.get("type") == "space":
                unit = ax.get("unit")
                if unit is not None:
                    coordinate_units = _OME_UNIT_TO_ABBREVIATION.get(unit, unit)
                break

        voxel_size = np.array(scale, dtype=float) if scale is not None else None
        offset = np.array(translation, dtype=float) if translation is not None else None
        return voxel_size, offset, coordinate_units
    except Exception as e:
        logger.debug(f"Could not read OME-NGFF metadata: {e}")
        return None, None, None


try:
    from mesh_n_bone.meshify.fixed_edge import simplify_mesh

    FIXED_EDGE_AVAILABLE = True
except ImportError as e:
    FIXED_EDGE_AVAILABLE = False
    logger.warning(
        f"Fixed edge mesh utilities not available: {e}. "
        "Fixed edge simplification will not work."
    )


def staged_reductions(target_reduction_total, frac1, frac2):
    """Compute per-stage reductions for a two-stage simplification pipeline.

    Splits an overall target reduction into two successive stages so that
    applying both in sequence achieves the total.

    Parameters
    ----------
    target_reduction_total : float
        Overall target reduction, e.g. 0.99 removes 99% of faces.
    frac1 : float
        Fraction of the total simplification performed in stage 1.
    frac2 : float
        Fraction of the total simplification performed in stage 2.
        Must satisfy ``frac1 + frac2 == 1``.

    Returns
    -------
    tuple[float, float]
        ``(reduction_stage_1, reduction_stage_2)`` — per-stage reduction
        ratios such that applying them sequentially yields
        *target_reduction_total*.
    """
    assert abs(frac1 + frac2 - 1.0) < 1e-6, "fractions must sum to 1"
    keep_total = 1 - target_reduction_total
    r1 = 1 - keep_total**frac1
    r2 = 1 - keep_total**frac2
    return r1, r2


def _chunk_stage_1_reduction(config):
    """Return the chunk-level reduction for fixed-edge chunk processing.

    ``use_fixed_edge_simplification=False`` means skip chunk-level
    decimation, but still use the fixed-edge clipping path when global
    simplification is enabled.
    """
    if not config["use_fixed_edge_simplification"]:
        return 0.0
    stage_1_reduction, _ = staged_reductions(
        config["target_reduction"],
        config["stage_1_reduction_fraction"],
        1 - config["stage_1_reduction_fraction"],
    )
    return stage_1_reduction


# Thread-local tensorstore handle cache so each worker opens once
_thread_local_ts = {}


@dataclass(frozen=True)
class AssemblyMeshEstimate:
    """Estimated assembly cost for one final segment mesh."""

    mesh_id: str
    num_files: int
    ply_bytes: int
    vertex_count: int
    face_count: int
    raw_mesh_bytes: int
    estimated_peak_bytes: int


@dataclass(frozen=True)
class AssemblyWave:
    """One assembly pass using a single Dask process-per-job setting."""

    processes: int
    workers: int
    batches: list[list[str]]
    max_estimated_peak_bytes: int
    total_ply_bytes: int
    config: dict | None


def _read_ply_header_counts(path, max_header_bytes=64 * 1024):
    """Return ``(vertices, faces)`` from a PLY header without reading geometry."""
    header = bytearray()
    marker = b"end_header"
    with open(path, "rb") as f:
        while marker not in header and len(header) < max_header_bytes:
            chunk = f.read(1024)
            if not chunk:
                break
            header.extend(chunk)

    end = header.find(marker)
    if end < 0:
        raise ValueError(f"PLY header terminator not found in {path!r}")

    text = header[:end].decode("ascii", errors="ignore")
    vertices = None
    faces = None
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertices = int(parts[2])
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            faces = int(parts[2])

    if vertices is None or faces is None:
        raise ValueError(f"PLY vertex/face counts not found in {path!r}")
    return vertices, faces


def _assembly_memory_amplification(
    do_simplification=True,
    smooth_before_simplify=True,
    check_mesh_validity=True,
    has_custom_roi=False,
):
    """Choose a conservative peak-RSS multiplier for assembled mesh arrays."""
    if do_simplification:
        # A 658 MiB / 31M-face chunked PLY sample measured 16.6 GiB peak
        # RSS for decimate-then-smooth assembly without validity repair.
        # That is ~22x over raw float64/int32 mesh arrays after the fixed
        # baseline, so keep a little headroom above the observed case.
        amplification = 32 if smooth_before_simplify else 24
    else:
        # Even without simplification, concatenate/consolidate/boundary
        # dedup can briefly duplicate several full-size mesh arrays.
        amplification = 20
    if check_mesh_validity or has_custom_roi:
        amplification += 4
    return amplification


def _estimate_assembly_peak_bytes(
    ply_bytes,
    vertex_count,
    face_count,
    amplification,
    baseline_bytes=1 << 30,
):
    """Estimate peak RSS for one mesh assembly from chunk PLY totals."""
    raw_mesh_bytes = vertex_count * 3 * 8 + face_count * 3 * 4
    # Header parsing can fail for malformed or future PLY variants. In that
    # case, file size is still a useful lower bound on in-memory geometry.
    effective_mesh_bytes = max(raw_mesh_bytes, int(ply_bytes * 1.1))
    return raw_mesh_bytes, int(baseline_bytes + amplification * effective_mesh_bytes)


def _scan_assembly_mesh_estimates(dirname, amplification):
    """Scan chunked mesh PLYs and estimate assembly memory per segment id."""
    estimates = []
    for mesh_id in sorted(os.listdir(dirname)):
        mesh_dir = os.path.join(dirname, mesh_id)
        if not os.path.isdir(mesh_dir):
            continue

        num_files = 0
        ply_bytes = 0
        vertex_count = 0
        face_count = 0
        for name in os.listdir(mesh_dir):
            if not name.endswith(".ply"):
                continue
            num_files += 1
            path = os.path.join(mesh_dir, name)
            try:
                ply_bytes += os.path.getsize(path)
                vertices, faces = _read_ply_header_counts(path)
            except (OSError, ValueError) as e:
                logger.debug("Could not read PLY header for %s: %s", path, e)
                continue
            vertex_count += vertices
            face_count += faces

        if num_files == 0:
            continue
        raw_mesh_bytes, estimated_peak_bytes = _estimate_assembly_peak_bytes(
            ply_bytes, vertex_count, face_count, amplification,
        )
        estimates.append(
            AssemblyMeshEstimate(
                mesh_id=str(mesh_id),
                num_files=num_files,
                ply_bytes=int(ply_bytes),
                vertex_count=int(vertex_count),
                face_count=int(face_count),
                raw_mesh_bytes=int(raw_mesh_bytes),
                estimated_peak_bytes=int(estimated_peak_bytes),
            )
        )
    return estimates


def _jobqueue_settings(config):
    """Return ``(cluster_type, settings)`` for a dask-jobqueue config."""
    if not config:
        return None, None
    jobqueue = config.get("jobqueue", {}) or {}
    if not jobqueue:
        return None, None
    cluster_type, settings = next(iter(jobqueue.items()))
    return cluster_type, settings


def _job_memory_bytes(settings):
    """Parse job memory bytes from a dask-jobqueue settings dict."""
    if not settings or "memory" not in settings:
        return None
    from dask.utils import parse_bytes
    return int(parse_bytes(str(settings["memory"])))


def _recommended_assembly_processes(
    estimated_peak_bytes,
    job_memory_bytes,
    base_processes,
    memory_fraction=0.60,
):
    """Pick processes/job so one estimated assembly fits per process."""
    base_processes = max(1, int(base_processes))
    if not job_memory_bytes or estimated_peak_bytes <= 0:
        return base_processes
    usable_job_bytes = int(job_memory_bytes * memory_fraction)
    processes = usable_job_bytes // int(estimated_peak_bytes)
    return max(1, min(base_processes, int(processes)))


def _assembly_config_for_processes(config, cluster_type, processes):
    """Copy a Dask config and set processes/cores for one-thread workers."""
    if config is None or cluster_type is None:
        return None
    adjusted = copy.deepcopy(config)
    dask_util.set_jobqueue_processes(adjusted, cluster_type, processes)
    return adjusted


def _balanced_assembly_batches(estimates, max_batches):
    """Greedily balance mesh ids into batches by estimated processing weight."""
    import heapq

    if not estimates:
        return []
    max_batches = max(1, min(int(max_batches), len(estimates)))
    heap = [(0, i, []) for i in range(max_batches)]
    weights_by_id = {}
    for estimate in estimates:
        weights_by_id[estimate.mesh_id] = max(
            estimate.estimated_peak_bytes, estimate.ply_bytes, 1,
        )
    for estimate in sorted(
        estimates, key=lambda e: weights_by_id[e.mesh_id], reverse=True,
    ):
        weight, index, mesh_ids = heapq.heappop(heap)
        mesh_ids = mesh_ids + [estimate.mesh_id]
        heapq.heappush(
            heap, (weight + weights_by_id[estimate.mesh_id], index, mesh_ids)
        )

    batches = [mesh_ids for weight, index, mesh_ids in heap if mesh_ids]
    batches.sort(
        key=lambda ids: sum(weights_by_id[mesh_id] for mesh_id in ids),
        reverse=True,
    )
    return batches


def _plan_assembly_waves(
    estimates,
    requested_workers,
    config=None,
    batches_per_worker=4,
    memory_fraction=0.60,
):
    """Group mesh ids into assembly waves with memory-aware Dask configs."""
    if not estimates:
        return []

    cluster_type, settings = _jobqueue_settings(config)
    base_processes = int((settings or {}).get("processes", 1) or 1)
    job_memory = _job_memory_bytes(settings)

    estimates_by_processes = {}
    for estimate in estimates:
        processes = _recommended_assembly_processes(
            estimate.estimated_peak_bytes,
            job_memory,
            base_processes,
            memory_fraction=memory_fraction,
        )
        estimates_by_processes.setdefault(processes, []).append(estimate)

    waves = []
    requested_workers = max(1, int(requested_workers))
    for processes in sorted(estimates_by_processes):
        wave_estimates = estimates_by_processes[processes]
        nominal_workers = max(
            1, requested_workers * int(processes) // max(1, base_processes)
        )
        max_batches = max(1, nominal_workers * int(batches_per_worker))
        batches = _balanced_assembly_batches(wave_estimates, max_batches)
        # Keep enough workers to fill one job when processes > task count.
        workers = min(nominal_workers, max(len(batches), int(processes)))
        waves.append(
            AssemblyWave(
                processes=int(processes),
                workers=int(max(1, workers)),
                batches=batches,
                max_estimated_peak_bytes=max(
                    e.estimated_peak_bytes for e in wave_estimates
                ),
                total_ply_bytes=sum(e.ply_bytes for e in wave_estimates),
                config=_assembly_config_for_processes(
                    config, cluster_type, int(processes)
                ),
            )
        )
    return waves


def _write_assembly_memory_plan(output_directory, estimates, waves):
    """Write a small JSON record explaining the assembly scheduling plan."""
    plan = {
        "mesh_estimates": [asdict(e) for e in estimates],
        "waves": [
            {
                "processes": wave.processes,
                "workers": wave.workers,
                "num_batches": len(wave.batches),
                "num_meshes": sum(len(batch) for batch in wave.batches),
                "max_estimated_peak_bytes": wave.max_estimated_peak_bytes,
                "total_ply_bytes": wave.total_ply_bytes,
                "batches": wave.batches,
            }
            for wave in waves
        ],
    }
    plan_dir = os.path.join(output_directory, "assembly_metadata")
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(plan_dir, "assembly_memory_plan.json"), "w") as f:
        json.dump(plan, f, indent=2)


def _estimate_block_target_mb_from_dask_config(
    config_path="dask-config.yaml",
    fallback_mb=128,
    processing_amplification=6,
):
    """Pick a per-block memory budget by working backward from worker RAM.

    Reads ``dask-config.yaml`` and divides per-worker memory by an
    amplification factor approximating "peak working memory ÷ voxel-array
    memory" for a block.

    The default amplification (6) is set from an empirical RSS-peak
    measurement across uint64 block sizes 32 MB → 512 MB at sparse (10
    segments) and dense (1000 segments) densities. After subtracting the
    fixed ~114 MB Python + zmesh + pymeshlab + trimesh import baseline,
    the asymptotic amplification was ~0.1× on sparse blocks and
    ~1.1–1.3× on dense blocks — i.e., per-task peak working memory grows
    almost exactly with the voxel array on dense data and not at all on
    sparse data. ``6`` leaves a ~4.5× safety margin over the empirical
    worst case for unforeseen dense / degenerate cases.

    For the user's typical LSF config (180 GB / 12 processes = 15 GB per
    worker) this yields ~2.5 GB per block; for a 60 GB-per-worker box
    it'd give ~10 GB. No cap — the amplification factor already encodes
    "leave headroom" and a cap on top would hide the math.

    Returns *fallback_mb* if the dask config is missing/unparseable.

    Parameters
    ----------
    config_path : str
        Path to dask-config.yaml. Default reads from cwd.
    fallback_mb : int
        Returned when the dask config is missing or unparseable. Also
        the floor on the auto-tuned result so tiny workers don't get
        pathologically small blocks.
    processing_amplification : int
        "Block peak RAM ÷ block voxel array" multiplier. Empirical
        asymptote is ~1.3×; the 6× default is 4.5× safety margin.
    """
    try:
        from dask.utils import parse_bytes
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        jq = cfg.get("jobqueue", {}) or {}
        if not jq:
            return fallback_mb
        # Take the first (only) configured cluster type.
        _, settings = next(iter(jq.items()))
        mem_bytes = parse_bytes(str(settings["memory"]))
        processes = int(settings.get("processes") or 1)
        per_worker_mb = (mem_bytes / processes) / 1e6
        target = per_worker_mb / processing_amplification
        # Floor so tiny workers don't get pathologically small blocks.
        return float(max(fallback_mb, target))
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return fallback_mb


def _normalize_target_ids(value):
    """Coerce a target_ids spec to a frozenset[int] or None.

    Accepts:
      - None: process every label found in the volume (default)
      - int: a single segment id
      - list/tuple/set of ints: multiple ids
      - str: path to a CSV file containing one id per row, either
        headerless (uses first column) or with a column named
        ``id``, ``ID``, ``Object ID``, or ``object_id``.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return frozenset([int(value)])
    if isinstance(value, str):
        import pandas as pd
        df = pd.read_csv(value)
        col = next(
            (c for c in ("id", "ID", "Object ID", "object_id") if c in df.columns),
            df.columns[0],
        )
        return frozenset(int(v) for v in df[col].dropna().tolist())
    return frozenset(int(i) for i in value)


def _get_chunked_mesh_worker(block_index, tmpdirname, config):
    """Run marching cubes on a single block and write per-segment PLYs.

    This is a module-level function so only lightweight *config* dict
    (scalars, tuples, strings) is serialised to workers — no zarr arrays.
    Receives only the block's sequential index; the block's ROI is
    materialized inside the worker via
    :func:`mesh_n_bone.util.dask_util.block_from_index`, so the driver
    never has to hold a list of millions of block objects.

    Parameters
    ----------
    block_index : int
        Sequential block index produced by ``db.range(num_blocks)``.
    tmpdirname : str
        Temporary directory for writing per-segment block meshes.
    config : dict
        Worker config from ``Meshify._get_worker_config()``. Must
        include ``block_roi_begin``, ``block_roi_end``,
        ``block_size_world`` and ``block_padding``.
    """
    block = dask_util.block_from_index(
        block_index,
        config["block_roi_begin"],
        config["block_roi_end"],
        config["block_size_world"],
        padding=config.get("block_padding"),
    )
    dataset_path = config["dataset_path"]
    if dataset_path not in _thread_local_ts:
        _thread_local_ts[dataset_path] = open_ds_tensorstore(dataset_path)
    ts_dataset = _thread_local_ts[dataset_path]

    voxel_size = Coordinate(config["voxel_size"])
    roi_offset = Coordinate(config["roi_offset"])
    output_voxel_size = Coordinate(config["output_voxel_size"])

    mesher = Mesher(output_voxel_size[::-1])
    segmentation_block = to_ndarray_tensorstore(
        ts_dataset, block.roi, voxel_size, roi_offset,
        swap_axes=config["swap_axes"], fill_value=0, source_path=dataset_path,
    )
    if segmentation_block.dtype.byteorder == ">":
        swapped_dtype = segmentation_block.dtype.newbyteorder()
        segmentation_block = segmentation_block.view(swapped_dtype).byteswap()

    downsample_factor = config["downsample_factor"]
    if downsample_factor:
        dm = config["downsample_method"]
        if dm == "nearest":
            segmentation_block = segmentation_block[
                ::downsample_factor, ::downsample_factor, ::downsample_factor
            ].copy()
        else:
            methods = {
                "mode_suppress_zero": downsample_labels_3d_suppress_zero,
                "mode": downsample_labels_3d,
                "binary": downsample_binary_3d,
            }
            if dm not in methods:
                raise ValueError(
                    f"Unknown downsample_method '{dm}'. "
                    f"Choose from: {list(methods.keys()) + ['nearest']}"
                )
            ds_func = methods[dm]
            segmentation_block, _ = ds_func(segmentation_block, downsample_factor)

    # If the user supplied target_ids, zero out every voxel whose label
    # isn't in the keep list BEFORE running marching cubes. That way zmesh
    # only does work for the requested objects; blocks containing none of
    # them exit immediately. We optionally renumber the surviving labels
    # to small consecutive ids so zmesh's internal arrays use a smaller
    # dtype, then map the resulting mesh ids back to the originals.
    target_ids_tuple = config.get("target_ids")
    inv_remap = None
    if target_ids_tuple:
        keep = list(target_ids_tuple)
        segmentation_block = fastremap.mask_except(
            segmentation_block, keep, in_place=False
        )
        if not segmentation_block.any():
            return  # no target ids in this block; skip MC entirely
        if segmentation_block.dtype.itemsize > 2 and len(keep) < (1 << 16):
            remap = {old: new for new, old in enumerate(sorted(keep), start=1)}
            inv_remap = {new: old for old, new in remap.items()}
            segmentation_block = fastremap.remap(
                segmentation_block, remap, preserve_missing_labels=True,
            ).astype(np.uint16)

    block_offset = np.array(block.roi.get_begin())
    # Correct for the half-kernel shift introduced by downsampling:
    # a downsampled voxel at index 0 represents original voxels [0, ds),
    # centered at (ds-1)/2 original voxels from the origin.
    ds_shift = (downsample_factor - 1) / 2 * np.array(voxel_size) if downsample_factor else np.zeros(3)
    mesher.mesh(segmentation_block, close=False)
    for id in mesher.ids():
        original_id = inv_remap[int(id)] if inv_remap is not None else int(id)
        mesh = mesher.get_mesh(id)

        if config["do_simplification"]:
            mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            # Shift by half a voxel so clip planes in
            # remove_boundary_vertices land exactly on the MC crossing
            # vertices (midpoints between padding and unpadded voxels).
            # This makes both adjacent blocks clip at the same world
            # plane, producing matching boundary vertices.  Padding
            # parallel-edge vertices end up strictly outside [0,
            # block_size] and are removed by the strict > check.
            half_pad = 0.5 * np.array(output_voxel_size)[::-1]
            mesh_tri.vertices -= half_pad

            ds = config["downsample_factor"] or 1
            block_size_voxels = np.array(config["read_write_block_shape_pixels"]) // ds
            block_size_world = (block_size_voxels * output_voxel_size)[::-1]

            mesh_tri_simplified = simplify_mesh(
                mesh_tri,
                voxel_size=output_voxel_size,
                target_reduction=_chunk_stage_1_reduction(config),
                block_size=block_size_world,
                aggressiveness=config["default_aggressiveness"],
                verbose=False,
                fix_edges=True,
            )
            # Segments living entirely in this block's padding zone end up
            # empty after boundary clipping. Skip writing — they'll be
            # produced by the adjacent block whose core contains them.
            if len(mesh_tri_simplified.vertices) == 0:
                continue
            half_pad_offset = block_offset + 0.5 * np.array(output_voxel_size)
            mesh_tri_simplified.vertices += half_pad_offset[::-1] + ds_shift[::-1]

            mesh_simplified = CloudVolumeMesh(
                mesh_tri_simplified.vertices,
                mesh_tri_simplified.faces,
                normals=None,
            )

            os.makedirs(f"{tmpdirname}/{original_id}", exist_ok=True)
            with open(f"{tmpdirname}/{original_id}/block_{block.index}.ply", "wb") as fp:
                fp.write(mesh_simplified.to_ply())
        else:
            if len(mesh.vertices) == 0:
                continue
            mesh.vertices += block_offset[::-1] + ds_shift[::-1]
            os.makedirs(f"{tmpdirname}/{original_id}", exist_ok=True)
            with open(f"{tmpdirname}/{original_id}/block_{block.index}.ply", "wb") as fp:
                fp.write(mesh.to_ply())


class Meshify:
    """Generate triangle meshes from a segmentation volume.

    Uses `zmesh <https://github.com/seung-lab/zmesh>`_ for marching-cubes
    meshing and Dask for parallel processing.  The pipeline:

    1. Splits the volume into blocks and runs marching cubes per block.
    2. Assembles per-segment block meshes, deduplicates boundary vertices.
    3. Optionally simplifies, smooths, repairs, and validates each mesh.
    4. Writes output as PLY, legacy Neuroglancer, or multiresolution
       Neuroglancer precomputed format.

    Parameters
    ----------
    input_path : str
        Path to the input segmentation dataset (Zarr or N5).
    output_directory : str
        Directory where output meshes and metadata are written.
    roi : Roi or dict or None
        Region of interest to process. Accepts a ``funlib.geometry.Roi``,
        a dict with ``begin``/``end`` or ``offset``/``shape`` keys, or
        ``None`` for the full volume.
    max_num_voxels : float
        Maximum number of voxels in a segment before it is skipped.
    max_num_blocks : float
        Maximum number of blocks a segment may span before skipping.
    read_write_block_shape_pixels : list of int or None
        Block shape in voxels for chunked processing. Defaults to the
        dataset's chunk shape.
    downsample_factor : int or None
        Factor by which to downsample the volume before meshing.
    target_reduction : float
        Fraction of faces to remove during simplification (0–1).
    num_workers : int
        Number of Dask workers for parallel processing.
    remove_smallest_components : bool
        If ``True``, keep only the largest connected component.
    n_smoothing_iter : int
        Number of Taubin smoothing iterations.
    smooth_before_simplify : bool
        If ``True`` (default), apply Taubin smoothing BEFORE quadric
        decimation in the assembly stage. Smoothing on the dense mesh
        recovers the underlying continuous surface (the voxel staircase
        gets averaged out by local neighbors); decimation then collapses
        a clean surface. Empirically ~2× lower RMS deviation from truth
        at the same final face count vs the reverse order. Set ``False``
        to restore the legacy decimate-then-smooth ordering.
    target_ids : int, list of int, str, or None
        Only meshify the listed segment ids. ``None`` (default) processes
        every label found in the volume. Accepts a single int, a list of
        ints, or a path to a CSV file (uses the first column or a column
        named ``id``/``Object ID``/...). When set, the chunk worker zeros
        out every voxel whose label isn't in the keep list BEFORE running
        marching cubes (via ``fastremap.mask_except``), and blocks that
        contain none of the targets are skipped entirely. Labels are
        optionally renumbered to a smaller dtype for zmesh efficiency
        and mapped back to the originals when writing PLYs.
    default_aggressiveness : float
        Aggressiveness parameter for quadric-error simplification.
    check_mesh_validity : bool
        If ``True``, validate that meshes are watertight with
        consistent winding.
    do_simplification : bool
        If ``True``, simplify meshes to *target_reduction*.
    do_analysis : bool
        If ``True``, run geometric analysis after mesh generation.
    do_legacy_neuroglancer : bool
        Write single-resolution Neuroglancer precomputed format.
    do_singleres_multires_neuroglancer : bool
        Write single-resolution meshes wrapped in multires metadata.
    use_fixed_edge_simplification : bool
        When ``True``, simplification runs in two stages: a per-chunk
        pass with block-boundary vertices pinned (so they survive
        assembly), followed by a global pass on the assembled mesh. The
        split between the two stages is controlled by
        ``stage_1_reduction_fraction``. When ``False``, a single
        standard simplification pass runs after assembly.
    fixed_edge_merge_weld_epsilon : float
        Vertex-merge tolerance for fixed-edge simplification.
    fixed_edge_seam_angle_deg : float
        Dihedral angle threshold (degrees) for seam detection.
    fixed_edge_k_ring : int
        K-ring expansion around seam vertices for denoising.
    fixed_edge_taubin_iters : int
        Taubin smoothing iterations during seam denoising.
    fixed_edge_taubin_lambda : float
        Lambda parameter for Taubin smoothing.
    fixed_edge_taubin_mu : float
        Mu parameter for Taubin smoothing.
    stage_1_reduction_fraction : float
        Fraction of total reduction applied in stage 1 (per-block).
    do_multires : bool
        If ``True``, generate multiresolution meshes instead of
        single-resolution output.
    num_lods : int
        Number of levels of detail for multiresolution output.
    lod_0_box_size : array-like or None
        Chunk box size for LOD 0. ``None`` for auto-computation.
    downsample_method : str
        Downsampling method for in-worker volume downsampling:
        ``"mode"`` (majority-label voting, default), ``"mode_suppress_zero"``
        (mode that ignores background voxels — keeps thin segments
        visible at coarse LODs but inflates their apparent extent),
        ``"binary"``, or ``"nearest"`` (stride).
    multires_strategy : str
        Strategy for generating LODs (default ``"downsample"``):
          - ``"decimate"``: mesh s0 once, face-decimate that mesh for
            higher LODs by ``decimation_factor`` per LOD.
          - ``"downsample"``: mesh each LOD from a coarser volume,
            preferring pre-existing OME-NGFF multiscale levels when
            available (see ``use_existing_scales``); otherwise
            downsampling in-worker via ``downsample_method``. Each
            LOD's effective ``target_reduction`` is auto-computed so
            the per-LOD face count drops by ``decimation_factor``
            (4x raw-MC drop from voxel doubling, plus
            ``4 / decimation_factor`` extra decimation per LOD).
            Default — produces hemibrain-density face counts when
            paired with the matched ``target_reduction`` and
            ``decimation_factor`` defaults.
    use_existing_scales : bool
        When ``True`` (default) and ``multires_strategy="downsample"``,
        each LOD prefers reading the matching pre-existing OME-NGFF
        multiscale level (``s_k``) over re-downsampling the input
        in-worker. Set ``False`` to force every LOD to downsample the
        input via ``downsample_method`` instead — useful when the
        dataset's own downsampling is poor quality or you want
        consistent ``mode_suppress_zero`` behavior at every LOD that the
        source's pre-built scales weren't built with.
    decimation_factor : int
        Per-LOD face-count reduction factor (default 6, hemibrain-matched).
        In the decimate strategy it's the literal ratio between
        consecutive LODs. In the downsample strategy it's the target
        per-LOD ratio used to derive each LOD's ``target_reduction``.
    decimation_aggressiveness : int
        Aggressiveness for pyfqmr decimation across LODs.
    delete_decimated_meshes : bool
        If ``True``, remove intermediate LOD mesh files after the
        multiresolution pipeline completes.
    segment_properties_csv : str or None
        Path to a CSV with per-segment properties for Neuroglancer.
    segment_properties_columns : list of str or None
        Columns to include from the CSV (``None`` for all).
    segment_properties_id_column : str
        Column name in the CSV containing segment IDs.
    coordinate_units : str
        Spatial unit label written to metadata (e.g. ``"nm"``).
    voxel_size_nm : list of float or None
        Explicit voxel size override in the same units as
        *coordinate_units*. ``None`` to read from dataset metadata.
    """

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        roi: Roi = None,
        max_num_voxels=np.inf,
        max_num_blocks=np.inf,
        read_write_block_shape_pixels: list = None,
        downsample_factor: int | None = None,
        target_reduction: float = 0.933,
        num_workers: int = 10,
        remove_smallest_components: bool = False,
        n_smoothing_iter: int = 2,
        default_aggressiveness: int = 0.3,
        check_mesh_validity: bool = False,
        do_simplification: bool = True,
        do_analysis: bool = False,
        do_legacy_neuroglancer=False,
        do_singleres_multires_neuroglancer=False,
        use_fixed_edge_simplification: bool = False,
        fixed_edge_merge_weld_epsilon: float = 1e-4,
        fixed_edge_seam_angle_deg: float = 35.0,
        fixed_edge_k_ring: int = 2,
        fixed_edge_taubin_iters: int = 12,
        fixed_edge_taubin_lambda: float = 0.5,
        fixed_edge_taubin_mu: float = -0.53,
        stage_1_reduction_fraction: float = 0.5,
        do_multires: bool = True,
        num_lods: int = 4,
        lod_0_box_size=None,
        target_faces_per_lod0_chunk: int = 25_000,
        downsample_method: str = "mode",
        multires_strategy: str = "downsample",
        use_existing_scales: bool = True,
        decimation_factor: int = 6,
        decimation_aggressiveness: int = 7,
        delete_decimated_meshes: bool = True,
        segment_properties_csv: str = None,
        segment_properties_columns: list = None,
        segment_properties_id_column: str = "Object ID",
        coordinate_units: str = "nm",
        voxel_size_nm: list = None,
        retry_on_oom: bool = True,
        memory_retry_max: int = 3,
        sharded: bool = True,
        shard_bits: int | None = None,
        minishard_bits: int | None = None,
        preshift_bits: int | None = None,
        delete_unsharded_files: bool = True,
        smooth_before_simplify: bool = True,
        target_ids: int | list[int] | None = None,
    ):
        filename, dataset_name = split_dataset_path(input_path)
        self.segmentation_array = open_dataset(filename, dataset_name)
        self._dataset_path = (
            getattr(self.segmentation_array, "_dataset_path", None)
            or (_path_join(filename, dataset_name) if dataset_name else filename)
        )
        self.output_directory = output_directory
        self.input_path = input_path
        # Both N5 and neuroglancer precomputed store voxels in XYZ order
        # with no per-axis labels, so we need to swap to ZYX at read time.
        driver = _detect_zarr_driver(self._dataset_path)
        self._swap_axes = driver in ("n5", "neuroglancer_precomputed")

        # Get true (possibly non-integer) voxel size and offset from
        # the underlying data. self.true_offset is the float-precision
        # OME translation / funlib offset; self.roi.offset stays
        # integer for funlib Roi arithmetic, and the final mesh
        # vertices get corrected with true_offset before output.
        self.true_voxel_size = np.array(read_raw_voxel_size(self.segmentation_array))
        self.true_offset = np.array(read_raw_offset(self.segmentation_array))

        # Check if voxel_size is just defaults (1,1,1) and
        # try OME-NGFF multiscales metadata from the parent zarr group.
        # Skip for precomputed (no OME metadata).
        if driver == "neuroglancer_precomputed":
            ome_voxel_size, ome_offset, ome_units = None, None, None
        else:
            ome_voxel_size, ome_offset, ome_units = _read_ome_ngff_transform(input_path)

        if ome_units is not None and coordinate_units == "nm":
            coordinate_units = ome_units

        if voxel_size_nm is not None:
            # Explicit voxel size in nm — only affects mesh vertex scaling,
            # not the block/ROI coordinate system (so ROI stays in dataset units)
            voxel_size_nm = np.atleast_1d(np.asarray(voxel_size_nm, dtype=float))
            if voxel_size_nm.shape == (1,):
                voxel_size_nm = np.repeat(voxel_size_nm, 3)
            logger.info(f"Using user-specified voxel_size_nm {voxel_size_nm}")
            self.true_voxel_size = voxel_size_nm.copy()
        elif ome_voxel_size is not None:
            if all(v == 1 for v in self.segmentation_array.voxel_size):
                logger.info(
                    f"Using OME-NGFF voxel_size {ome_voxel_size} "
                    f"(attrs returned {self.segmentation_array.voxel_size})"
                )
                self.true_voxel_size = ome_voxel_size.copy()
                ome_voxel_size_coord = Coordinate(int(v) for v in ome_voxel_size)
                ome_offset_coord = (
                    Coordinate(int(v) for v in ome_offset)
                    if ome_offset is not None
                    else Coordinate(0, 0, 0)
                )
                self.segmentation_array.voxel_size = ome_voxel_size_coord
                array_shape = self.segmentation_array.data.shape[-3:]
                self.segmentation_array.roi = Roi(
                    ome_offset_coord, Coordinate(array_shape) * ome_voxel_size_coord
                )
                # Track the float-precision OME translation so the
                # final-mesh rescale can place vertices at sub-unit
                # positions even when ome_offset_coord rounds them.
                if ome_offset is not None:
                    self.true_offset = np.array(ome_offset, dtype=float)
                else:
                    self.true_offset = np.zeros(3, dtype=float)

        if roi is not None:
            if not isinstance(roi, Roi):
                # Accept dict with offset+shape or begin+end from YAML config
                if isinstance(roi, dict):
                    if "begin" in roi and "end" in roi:
                        begin = Coordinate(roi["begin"])
                        end = Coordinate(roi["end"])
                        roi = Roi(begin, end - begin)
                    elif "offset" in roi and "shape" in roi:
                        roi = Roi(
                            Coordinate(roi["offset"]),
                            Coordinate(roi["shape"]),
                        )
                    else:
                        raise ValueError(
                            "roi dict must have 'offset'+'shape' or 'begin'+'end' keys"
                        )
                else:
                    raise ValueError(
                        "roi must be a Roi object or a dict with "
                        "'offset'+'shape' or 'begin'+'end' keys"
                    )
            self.roi = roi
            self.has_custom_roi = True
        else:
            self.roi = self.segmentation_array.roi
            self.has_custom_roi = False
        self.num_workers = num_workers

        if read_write_block_shape_pixels:
            self.read_write_block_shape_pixels = np.array(read_write_block_shape_pixels)
        else:
            self.read_write_block_shape_pixels = (
                self._default_block_shape_pixels()
            )

        self.max_num_blocks = max_num_blocks
        self.base_voxel_size_funlib = self.segmentation_array.voxel_size

        self.output_voxel_size_funlib = max(
            self.base_voxel_size_funlib, Coordinate(1, 1, 1)
        )
        self.downsample_factor = downsample_factor
        if self.downsample_factor:
            self.output_voxel_size_funlib = Coordinate(
                np.array(self.output_voxel_size_funlib) * self.downsample_factor
            )
            self.true_voxel_size *= self.downsample_factor
        self.target_reduction = target_reduction

        self.check_mesh_validity = check_mesh_validity
        self.remove_smallest_components = remove_smallest_components
        self.n_smoothing_iter = n_smoothing_iter
        self.do_analysis = do_analysis
        self.do_legacy_neuroglancer = do_legacy_neuroglancer
        self.do_singleres_multires_neuroglancer = do_singleres_multires_neuroglancer
        self.do_simplification = do_simplification
        self.default_aggressiveness = default_aggressiveness

        self.use_fixed_edge_simplification = use_fixed_edge_simplification
        if self.use_fixed_edge_simplification and not FIXED_EDGE_AVAILABLE:
            raise RuntimeError(
                "Fixed edge simplification requested but dependencies not available. "
                "Ensure pyfqmr is installed (`pip install pyfqmr`)."
            )

        self.fixed_edge_merge_weld_epsilon = fixed_edge_merge_weld_epsilon
        self.fixed_edge_seam_angle_deg = fixed_edge_seam_angle_deg
        self.fixed_edge_k_ring = fixed_edge_k_ring
        self.fixed_edge_taubin_iters = fixed_edge_taubin_iters
        self.fixed_edge_taubin_lambda = fixed_edge_taubin_lambda
        self.fixed_edge_taubin_mu = fixed_edge_taubin_mu

        self.stage_1_reduction_fraction = stage_1_reduction_fraction
        self.stage_2_reduction_fraction = 1 - self.stage_1_reduction_fraction

        self.do_multires = do_multires
        self.num_lods = num_lods
        if lod_0_box_size is not None:
            self.lod_0_box_size = np.atleast_1d(np.asarray(lod_0_box_size, dtype=float))
            if self.lod_0_box_size.shape == (1,):
                self.lod_0_box_size = np.repeat(self.lod_0_box_size, 3)
        else:
            self.lod_0_box_size = None
        self.target_faces_per_lod0_chunk = target_faces_per_lod0_chunk
        self.downsample_method = downsample_method
        self.input_path = input_path
        self.multires_strategy = multires_strategy
        self.use_existing_scales = use_existing_scales
        self.decimation_factor = decimation_factor
        self.decimation_aggressiveness = decimation_aggressiveness
        self.delete_decimated_meshes = delete_decimated_meshes
        self.segment_properties_csv = segment_properties_csv
        self.segment_properties_columns = segment_properties_columns
        self.segment_properties_id_column = segment_properties_id_column
        self.sharded = sharded
        self.shard_bits = shard_bits
        self.minishard_bits = minishard_bits
        self.preshift_bits = preshift_bits
        self.delete_unsharded_files = delete_unsharded_files
        self.smooth_before_simplify = smooth_before_simplify
        self.target_ids = _normalize_target_ids(target_ids)
        self.coordinate_units = coordinate_units
        self.retry_on_oom = retry_on_oom
        self.memory_retry_max = memory_retry_max

    def _default_block_shape_pixels(self, target_mb=None):
        """Choose a default block shape as a chunk-aligned multiple.

        Picks the largest integer multiple of the dataset's chunk shape
        whose memory footprint stays at or below ``target_mb``. Larger
        blocks reduce the number of block boundaries (and therefore
        frozen boundary vertices during fixed-edge simplification),
        and shrink dask graph scheduling overhead.

        When ``target_mb`` is ``None`` (the default) we estimate a
        sensible value from the dask-config.yaml in the current
        directory: per-worker RAM divided by an empirical processing
        amplification factor (~8x, covering MC working memory + stage-1
        simplification scratch + headroom for Python/libs). Capped at
        1 GB so a single slow block doesn't tail the whole run. Falls
        back to 128 MB if no dask-config can be parsed.

        Parameters
        ----------
        target_mb : int, float, or None
            Target memory budget per block in megabytes. ``None``
            triggers auto-tuning from the dask-config.

        Returns
        -------
        numpy.ndarray
            Block shape in voxels, as a multiple of the chunk shape.
        """
        if target_mb is None:
            target_mb = _estimate_block_target_mb_from_dask_config()
        chunk = np.array(self.segmentation_array.chunk_shape)
        itemsize = self.segmentation_array.dtype.itemsize
        chunk_bytes = int(np.prod(chunk)) * itemsize
        target_bytes = target_mb * 1e6
        # Find the largest multiplier whose cube fits in the budget
        # Total bytes = chunk_bytes * multiplier^3
        max_mult = int((target_bytes / chunk_bytes) ** (1 / 3))
        # Don't exceed the ROI dimensions
        if hasattr(self, "roi") and self.roi is not None:
            roi_pixels = np.array(self.roi.shape) / np.array(
                self.segmentation_array.voxel_size
            )
            max_by_roi = int(np.min(roi_pixels / chunk))
            max_mult = min(max_mult, max_by_roi)
        multiplier = max(1, max_mult)
        return chunk * multiplier

    def _get_downsample_function(self):
        """Return the appropriate downsample function based on config."""
        methods = {
            "mode_suppress_zero": downsample_labels_3d_suppress_zero,
            "mode": downsample_labels_3d,
            "binary": downsample_binary_3d,
            "nearest": None,
        }
        if self.downsample_method not in methods:
            raise ValueError(
                f"Unknown downsample_method '{self.downsample_method}'. "
                f"Choose from: {list(methods.keys())}"
            )
        return methods[self.downsample_method]

    @staticmethod
    def my_cloudvolume_concatenate(*meshes):
        """Concatenate multiple meshes into a single CloudVolume ``Mesh``.

        Face indices are offset so that they reference the correct
        vertices in the combined vertex array.

        Parameters
        ----------
        *meshes : cloudvolume.mesh.Mesh
            Meshes to concatenate.

        Returns
        -------
        cloudvolume.mesh.Mesh
            Combined mesh with all vertices and re-indexed faces.
        """
        vertex_ct = np.zeros(len(meshes) + 1, np.uint32)
        vertex_ct[1:] = np.cumsum([len(mesh) for mesh in meshes])
        vertices = np.concatenate([mesh.vertices for mesh in meshes])
        faces = np.concatenate(
            [mesh.faces + vertex_ct[i] for i, mesh in enumerate(meshes)]
        )
        normals = None
        return CloudVolumeMesh(vertices, faces, normals)

    def _get_worker_config(self):
        """Return a lightweight, pickle-safe dict of parameters for workers."""
        return {
            "dataset_path": self._dataset_path,
            "swap_axes": self._swap_axes,
            "voxel_size": tuple(self.segmentation_array.voxel_size),
            "roi_offset": tuple(self.segmentation_array.roi.offset),
            "output_voxel_size": tuple(self.output_voxel_size_funlib),
            "downsample_factor": self.downsample_factor,
            "downsample_method": self.downsample_method,
            "use_fixed_edge_simplification": self.use_fixed_edge_simplification,
            "do_simplification": self.do_simplification,
            "target_reduction": self.target_reduction,
            "stage_1_reduction_fraction": self.stage_1_reduction_fraction,
            "stage_2_reduction_fraction": self.stage_2_reduction_fraction,
            "read_write_block_shape_pixels": self.read_write_block_shape_pixels.tolist(),
            "default_aggressiveness": self.default_aggressiveness,
            # Sorted tuple for pickling; worker materializes a numpy array
            # for np.isin. None means "process every label found in block."
            "target_ids": (
                tuple(sorted(self.target_ids)) if self.target_ids is not None else None
            ),
        }

    @staticmethod
    def is_mesh_valid(mesh):
        """Check whether a mesh has consistent winding, is watertight, and has positive volume.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            Mesh to validate.

        Returns
        -------
        bool
            ``True`` if the mesh passes all three checks.
        """
        return mesh.is_winding_consistent and mesh.is_watertight and mesh.volume > 0

    def get_chunked_meshes(self, dirname):
        """Generate per-block meshes for the entire ROI using Dask.

        Driver enumerates blocks lazily by index — only the count is
        computed up front (O(1)), and each worker materializes its own
        block ROI from the index. This avoids building a list of
        millions of block objects on the driver for large volumes, and
        shrinks per-task graph payload from a Roi-carrying object down
        to a single int.

        Parameters
        ----------
        dirname : str
            Directory where per-segment block mesh PLYs are written.
        """
        block_size_world = (
            self.read_write_block_shape_pixels * self.output_voxel_size_funlib
        )
        num_blocks = dask_util.count_blocks(self.roi, block_size_world)

        worker_config = self._get_worker_config()
        # Pass enough info to reconstruct each block's ROI in the worker.
        worker_config["block_roi_begin"] = tuple(self.roi.get_begin())
        worker_config["block_roi_end"] = tuple(self.roi.get_end())
        worker_config["block_size_world"] = tuple(int(v) for v in block_size_world)
        worker_config["block_padding"] = tuple(self.output_voxel_size_funlib)

        effective_workers = dask_util.effective_num_workers(
            self.num_workers, num_blocks, logger, "generate chunked meshes",
        )

        def _run(workers, config):
            # Use the rounded estimator here rather than workers * 10 directly:
            # dask.bag.range puts all remainder elements in the final partition.
            npartitions = dask_util.guesstimate_npartitions(num_blocks, workers)
            bag = db.range(num_blocks, npartitions=npartitions).map(
                _get_chunked_mesh_worker, dirname, worker_config
            )
            with dask_util.start_dask(
                workers, "generate chunked meshes", logger, config=config,
            ):
                with Timing_Messager(
                    f"Generating chunked meshes ({num_blocks} blocks)", logger,
                ):
                    bag.compute()

        dask_util.run_with_oom_retry(
            _run, effective_workers, "generate chunked meshes", logger,
            max_retries=self.memory_retry_max, retry_on_oom=self.retry_on_oom,
        )

    @staticmethod
    def simplify_and_smooth_mesh(
        mesh,
        target_reduction=0.933,
        n_smoothing_iter=2,
        remove_smallest_components=True,
        aggressiveness=0.3,
        do_simplification=True,
        check_mesh_validity=True,
        preserve_open_boundaries=False,
        smooth_before_simplify=True,
    ):
        """Simplify, smooth, and optionally repair a mesh.

        Applies quadric-error simplification and Taubin smoothing.  By
        default smoothing runs FIRST on the dense input, then quadric
        decimation collapses the smoothed surface to ``target_reduction``.
        That ordering preserves silhouette and underlying-surface fidelity
        better than the reverse for voxel-derived meshes (empirically
        ~2× lower RMS deviation from analytic / source-mesh truth at the
        same final face count). The order can be flipped via
        ``smooth_before_simplify=False`` for backward compatibility.

        If the result is invalid (non-watertight or inconsistent winding),
        retries with progressively lower aggressiveness until a valid mesh
        is obtained or simplification is skipped entirely.

        Parameters
        ----------
        mesh : trimesh.Trimesh or mesh-like
            Input mesh with ``.vertices`` and ``.faces`` attributes.
        target_reduction : float
            Fraction of faces to remove (0–1).
        n_smoothing_iter : int
            Number of Taubin smoothing iterations.
        remove_smallest_components : bool
            If ``True``, keep only the largest connected component before
            processing.
        aggressiveness : float
            Starting aggressiveness for simplification.
        do_simplification : bool
            If ``False``, skip simplification entirely.
        check_mesh_validity : bool
            If ``True``, validate the mesh after each attempt and retry
            on failure.
        preserve_open_boundaries : bool
            If ``True``, pin boundary vertices during simplification and
            restore them after smoothing.
        smooth_before_simplify : bool
            If ``True`` (default), smooth before decimating. If ``False``,
            decimate first then smooth (the older behavior, kept for
            backwards compatibility).

        Returns
        -------
        trimesh.Trimesh
            Processed mesh.
        """
        def get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_reduction, aggressiveness, do_simplification
        ):
            def _decimate(input_mesh):
                if not do_simplification:
                    return input_mesh
                return simplify_mesh(
                    input_mesh,
                    voxel_size=None,
                    target_reduction=target_reduction,
                    aggressiveness=aggressiveness,
                    verbose=False,
                    fix_edges=preserve_open_boundaries,
                )

            def _smooth(input_mesh):
                if n_smoothing_iter <= 0:
                    return input_mesh
                ms = pymeshlab.MeshSet()
                ms.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=input_mesh.vertices,
                        face_matrix=input_mesh.faces,
                    )
                )
                if preserve_open_boundaries:
                    # Identify boundary vertices and save their positions
                    ms.compute_selection_from_mesh_border()
                    border_mask = ms.current_mesh().vertex_selection_array()
                    border_positions = input_mesh.vertices[border_mask].copy()

                ms.apply_coord_taubin_smoothing(
                    lambda_=0.5,
                    mu=-0.53,
                    stepsmoothnum=n_smoothing_iter,
                )
                m = ms.current_mesh()
                verts = m.vertex_matrix()

                if preserve_open_boundaries:
                    # Restore boundary vertex positions
                    verts[border_mask] = border_positions

                return trimesh.Trimesh(vertices=verts, faces=m.face_matrix())

            if smooth_before_simplify:
                # Smooth the dense mesh first so Taubin has all the
                # original vertices to redistribute, then decimate the
                # already-smooth surface.
                simplified_mesh = _smooth(mesh)
                simplified_mesh = _decimate(simplified_mesh)
            else:
                # Legacy order: decimate first, then smooth the sparse result.
                simplified_mesh = _decimate(mesh)
                simplified_mesh = _smooth(simplified_mesh)
            del mesh

            if not check_mesh_validity:
                return simplified_mesh
            cleaned_mesh = Meshify.repair_mesh_pymeshlab(
                simplified_mesh.vertices,
                simplified_mesh.faces,
                remove_smallest_components=remove_smallest_components,
            )
            del simplified_mesh
            return cleaned_mesh

        if remove_smallest_components:
            if type(mesh) != trimesh.base.Trimesh:
                mesh = trimesh.Trimesh(mesh.vertices, mesh.faces)

            components = mesh.split(only_watertight=check_mesh_validity)
            if len(components) > 0:
                mesh = components[0]
                for m in components[1:]:
                    if len(m.faces) > len(mesh.faces):
                        mesh = m

        com = mesh.vertices.mean(axis=0)
        vertices = mesh.vertices - com
        faces = mesh.faces
        mesh = trimesh.Trimesh(vertices, faces)

        output_trimesh_mesh = mesh.copy()
        if check_mesh_validity and not Meshify.is_mesh_valid(output_trimesh_mesh):
            output_trimesh_mesh.export("failed_mesh.ply")
            raise Exception(
                f"Initial mesh is not valid, "
                f"{output_trimesh_mesh.is_winding_consistent=},"
                f"{output_trimesh_mesh.is_watertight=},"
                f"{output_trimesh_mesh.volume=}."
            )

        if do_simplification:
            target_faces = int(max(12, (1 - target_reduction) * output_trimesh_mesh.faces.shape[0]))
            do_simplification = output_trimesh_mesh.faces.shape[0] > target_faces
        trimesh_mesh = get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_reduction, aggressiveness, do_simplification
        )

        retry_simplification_for_validity = False
        if check_mesh_validity:
            if Meshify.is_mesh_valid(trimesh_mesh):
                output_trimesh_mesh = trimesh_mesh
            else:
                retry_simplification_for_validity = True
        else:
            output_trimesh_mesh = trimesh_mesh

        aggressiveness -= 0.05
        while (
            (
                len(output_trimesh_mesh.faces)
                < 0.5 * len(trimesh_mesh.faces) * (1 - target_reduction)
                or retry_simplification_for_validity
            )
            and aggressiveness >= -0.05
            and do_simplification
        ):
            logger.info(f"Retrying with aggressiveness: {aggressiveness}")
            trimesh_mesh = get_cleaned_simplified_and_smoothed_mesh(
                mesh,
                target_reduction,
                aggressiveness,
                do_simplification=aggressiveness >= 0,
            )
            aggressiveness -= 0.05
            if check_mesh_validity:
                if Meshify.is_mesh_valid(trimesh_mesh):
                    output_trimesh_mesh = trimesh_mesh
                    retry_simplification_for_validity = False
            else:
                output_trimesh_mesh = trimesh_mesh

        if do_simplification and aggressiveness < -0.05:
            logger.warning(
                f"Mesh with {len(output_trimesh_mesh.faces)} faces "
                "had to be processed unsimplified."
            )

        if len(output_trimesh_mesh.faces) == 0:
            raise Exception(
                f"Mesh with {len(output_trimesh_mesh.faces)} faces "
                "could not be smoothed and cleaned even without simplification."
            )

        output_trimesh_mesh.vertices += com
        output_trimesh_mesh.fix_normals()
        return output_trimesh_mesh

    @staticmethod
    def repair_mesh_pymeshlab(
        vertices,
        faces,
        remove_smallest_components=True,
        max_hole_size=30,
        verbose=False,
    ):
        """Repair a mesh using PyMeshLab.

        Removes duplicate faces/vertices, repairs non-manifold edges and
        vertices, closes small holes, re-orients faces, and optionally
        removes small connected components.

        Parameters
        ----------
        vertices : ndarray, shape (V, 3)
            Vertex positions.
        faces : ndarray, shape (F, 3)
            Triangle face indices.
        remove_smallest_components : bool
            If ``True``, remove all but the largest component.
        max_hole_size : int
            Maximum number of edges in a hole to close. Set to 0 to
            skip hole closing.
        verbose : bool
            Not currently used; reserved for future logging.

        Returns
        -------
        trimesh.Trimesh
            Repaired mesh.
        """
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))

        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()

        ms.meshing_repair_non_manifold_edges(method="Split Vertices")
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")

        if max_hole_size > 0:
            ms.meshing_close_holes(maxholesize=max_hole_size)
        ms.meshing_repair_non_manifold_vertices(vertdispratio=0)

        if remove_smallest_components:
            try:
                if hasattr(pymeshlab, "PercentageValue"):
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=pymeshlab.PercentageValue(0)
                    )
                elif hasattr(pymeshlab, "Percentage"):
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=pymeshlab.Percentage(0)
                    )
                else:
                    ms.meshing_remove_connected_component_by_diameter(
                        mincomponentdiag=0
                    )
            except Exception as e:
                logger.warning(
                    f"Could not remove small components: {e}. Skipping."
                )

        ms.meshing_re_orient_faces_coherently()

        m = ms.current_mesh()
        verts_out = m.vertex_matrix()
        faces_out = m.face_matrix()
        return trimesh.Trimesh(vertices=verts_out, faces=faces_out)

    def _assemble_mesh(self, mesh_id):
        """Assemble block meshes for a single segment into a final mesh.

        Concatenates per-block PLYs, deduplicates boundary vertices,
        simplifies, smooths, validates, and writes the output mesh.

        Parameters
        ----------
        mesh_id : str
            Segment ID whose block meshes will be assembled.
        """
        if not os.path.exists(f"{self.dirname}/{mesh_id}"):
            return
        # Echo the segment id to worker stderr so it lands in the LSF
        # worker .err file. If the worker is later killed (e.g. by OOM),
        # the last mesh_id logged by that worker identifies what it was
        # processing — grep job-logs/LSFCluster-*.err for "assemble
        # mesh_id=".
        import sys
        print(f"assemble mesh_id={mesh_id}", file=sys.stderr, flush=True)

        mesh_files = [
            f for f in os.listdir(f"{self.dirname}/{mesh_id}") if f.endswith(".ply")
        ]
        if len(mesh_files) >= self.max_num_blocks:
            logger.warning(
                f"Mesh {mesh_id} has too many blocks "
                f"{len(mesh_files)}>{self.max_num_blocks}. Skipping."
            )
            skipped_path = f"{self.output_directory}/too_big_skipped"
            os.makedirs(skipped_path, exist_ok=True)
            with open(f"{skipped_path}/{mesh_id}.txt", "a") as f:
                f.write(
                    f"Mesh {mesh_id} has too many blocks "
                    f"{len(mesh_files)}>{self.max_num_blocks}. Skipping.\n"
                )
                f.write(", ".join(mesh_files))
            shutil.rmtree(f"{self.dirname}/{mesh_id}")
            return

        block_meshes = []
        for mesh_file in mesh_files:
            with open(f"{self.dirname}/{mesh_id}/{mesh_file}", "rb") as f:
                mesh = Zmesh.from_ply(f.read())
                block_meshes.append(mesh)

        if len(block_meshes) > 1:
            try:
                mesh = Meshify.my_cloudvolume_concatenate(*block_meshes)
            except Exception as e:
                raise Exception(f"{mesh_id} failed, with error: {e}")
            del block_meshes
            mesh = mesh.consolidate()
            # A segment can end up empty here when fixed-edge clipping
            # removed every vertex of every per-block PLY. Bail out before
            # deduplicate_chunk_boundaries, which can't handle empty verts.
            if len(mesh.vertices) == 0:
                logger.warning(
                    "Mesh %s: skipping — concatenated %d per-block PLY(s) "
                    "produced zero vertices (likely a sliver clipped away by "
                    "fixed-edge simplification).",
                    mesh_id, len(mesh_files),
                )
                shutil.rmtree(f"{self.dirname}/{mesh_id}", ignore_errors=True)
                return
            chunk_size = (
                self.read_write_block_shape_pixels * self.base_voxel_size_funlib
            )
            mesh = mesh.deduplicate_chunk_boundaries(
                chunk_size=chunk_size[::-1],
                offset=self.roi.offset[::-1],
            )
            if self.do_simplification:
                # The half-voxel shift places clip planes on the MC
                # crossing vertices between padding and unpadded voxels.
                # Both adjacent blocks clip at the same world plane and
                # produce identical boundary vertices.  merge_vertices
                # merges them even though they aren't at exact chunk-
                # size multiples (where deduplicate looks).
                tri_tmp = trimesh.Trimesh(
                    vertices=mesh.vertices, faces=mesh.faces, process=False
                )
                tri_tmp.merge_vertices(merge_tex=False, merge_norm=False)
                mesh = CloudVolumeMesh(
                    tri_tmp.vertices, tri_tmp.faces, normals=None
                )

        # When using a custom ROI, meshes cut at the boundary are intentionally
        # open — skip hole-closing and watertight validity checks.
        check_validity = self.check_mesh_validity and not self.has_custom_roi
        hole_size = 0 if self.has_custom_roi else 30

        # A segment can end up with zero vertices when its entire surface
        # sat on a chunk boundary that fixed-edge clipping removed, or when
        # an ROI cut leaves nothing behind. pymeshlab refuses an empty
        # vertex matrix, so skip the segment entirely.
        if len(mesh.vertices) == 0:
            logger.warning(
                "Mesh %s: skipping — boundary dedup/merge_vertices left "
                "zero vertices.",
                mesh_id,
            )
            shutil.rmtree(f"{self.dirname}/{mesh_id}", ignore_errors=True)
            return

        if check_validity or self.has_custom_roi:
            try:
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                del mesh
                mesh = Meshify.repair_mesh_pymeshlab(
                    vertices,
                    faces,
                    remove_smallest_components=self.remove_smallest_components,
                    max_hole_size=hole_size,
                )
            except Exception as e:
                raise Exception(f"{mesh_id} failed, with error: {e}")

        try:
            if self.use_fixed_edge_simplification and self.do_simplification:
                _, stage_2_reduction = staged_reductions(
                    self.target_reduction,
                    self.stage_1_reduction_fraction,
                    self.stage_2_reduction_fraction,
                )
                mesh = Meshify.simplify_and_smooth_mesh(
                    mesh,
                    stage_2_reduction,
                    self.n_smoothing_iter,
                    self.remove_smallest_components,
                    self.default_aggressiveness,
                    self.do_simplification,
                    check_validity,
                    preserve_open_boundaries=self.has_custom_roi,
                    smooth_before_simplify=self.smooth_before_simplify,
                )
            else:
                mesh = Meshify.simplify_and_smooth_mesh(
                    mesh,
                    self.target_reduction,
                    self.n_smoothing_iter,
                    self.remove_smallest_components,
                    self.default_aggressiveness,
                    self.do_simplification,
                    check_validity,
                    preserve_open_boundaries=self.has_custom_roi,
                    smooth_before_simplify=self.smooth_before_simplify,
                )

            if len(mesh.faces) == 0:
                _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        # Correct for differences between funlib's integer voxel_size /
        # offset (used during marching cubes and Roi math) and the true
        # float-precision values from OME-NGFF translation / scale.
        # Rewriting from voxel-index space gives:
        #   true_pos = true_offset + true_voxel * (int_pos - int_offset) / int_voxel
        #            = true_offset + scale * (int_pos - int_offset)
        voxel_mismatch = (
            list(self.true_voxel_size) != list(self.output_voxel_size_funlib)
        )
        true_offset_attr = getattr(self, "true_offset", None)
        roi_attr = getattr(self, "roi", None)
        offset_mismatch = False
        if true_offset_attr is not None and roi_attr is not None:
            int_offset_xyz = np.array(roi_attr.offset[::-1], dtype=float)
            true_offset_xyz = np.array(true_offset_attr[::-1], dtype=float)
            offset_mismatch = not np.allclose(int_offset_xyz, true_offset_xyz)
        if voxel_mismatch or offset_mismatch:
            scale = np.array(self.true_voxel_size[::-1]) / np.array(
                self.output_voxel_size_funlib[::-1]
            )
            int_offset_xyz = np.array(self.roi.offset[::-1], dtype=float)
            true_offset_xyz = np.array(
                getattr(self, "true_offset", self.roi.offset)[::-1], dtype=float
            )
            mesh.vertices -= int_offset_xyz
            mesh.vertices *= scale
            mesh.vertices += true_offset_xyz

        from mesh_n_bone.util.neuroglancer import (
            write_ngmesh,
            write_ngmesh_metadata,
            write_singleres_multires_files,
        )

        if self.do_legacy_neuroglancer:
            write_ngmesh(
                mesh.vertices,
                mesh.faces,
                f"{self.output_directory}/meshes/{mesh_id}",
            )
            with open(f"{self.output_directory}/meshes/{mesh_id}:0", "w") as f:
                f.write(json.dumps({"fragments": [f"./{mesh_id}"]}))
        elif self.do_singleres_multires_neuroglancer:
            write_singleres_multires_files(
                mesh.vertices, mesh.faces, f"{self.output_directory}/meshes/{mesh_id}"
            )
        else:
            _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
        shutil.rmtree(f"{self.dirname}/{mesh_id}")

    def _assemble_mesh_batch(self, mesh_ids):
        """Assemble a memory-balanced batch of segment ids sequentially."""
        for mesh_id in mesh_ids:
            self._assemble_mesh(mesh_id)

    def assemble_meshes(self, dirname):
        """Assemble all per-segment block meshes and write final outputs.

        Parameters
        ----------
        dirname : str
            Directory containing per-segment subdirectories of block PLYs.
        """
        from mesh_n_bone.util.neuroglancer import (
            write_ngmesh_metadata,
            write_singleres_multires_metadata,
        )

        os.makedirs(f"{self.output_directory}/meshes/", exist_ok=True)
        self.dirname = dirname
        amplification = _assembly_memory_amplification(
            do_simplification=self.do_simplification,
            smooth_before_simplify=self.smooth_before_simplify,
            check_mesh_validity=self.check_mesh_validity,
            has_custom_roi=self.has_custom_roi,
        )
        estimates = _scan_assembly_mesh_estimates(dirname, amplification)
        if not estimates:
            logger.warning("No chunked mesh PLYs found in %s; skipping assembly.", dirname)
            shutil.rmtree(dirname, ignore_errors=True)
            return

        assembly_config = None
        if self.num_workers > 1:
            try:
                assembly_config = dask_util._load_dask_config()
            except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
                logger.warning(
                    "Could not load dask-config.yaml for assembly memory "
                    "planning (%s); using default worker count.",
                    e,
                )

        waves = _plan_assembly_waves(
            estimates, self.num_workers, config=assembly_config,
        )
        _write_assembly_memory_plan(self.output_directory, estimates, waves)

        largest = max(estimates, key=lambda e: e.estimated_peak_bytes)
        logger.info(
            "Assembly memory estimate: largest mesh_id=%s, chunked_ply=%.1f MiB, "
            "raw_mesh=%.1f MiB, estimated_peak=%.1f GiB, amplification=%sx.",
            largest.mesh_id,
            largest.ply_bytes / 2**20,
            largest.raw_mesh_bytes / 2**20,
            largest.estimated_peak_bytes / 2**30,
            amplification,
        )
        for i, wave in enumerate(waves, start=1):
            logger.info(
                "Assembly wave %d/%d: processes/job=%d, workers=%d, "
                "batches=%d, meshes=%d, max_estimated_peak=%.1f GiB.",
                i, len(waves), wave.processes, wave.workers, len(wave.batches),
                sum(len(batch) for batch in wave.batches),
                wave.max_estimated_peak_bytes / 2**30,
            )

        # Drop the zarr-backed array before dask serialises self — assembly
        # only reads PLY files, not the segmentation volume.
        saved_array = self.segmentation_array
        self.segmentation_array = None

        try:
            for i, wave in enumerate(waves, start=1):
                phase_name = f"assemble meshes wave {i}/{len(waves)}"

                def _run(workers, config, wave=wave, phase_name=phase_name):
                    bag = db.from_sequence(
                        wave.batches, npartitions=len(wave.batches),
                    ).map(self._assemble_mesh_batch)
                    with dask_util.start_dask(
                        workers, phase_name, logger, config=config,
                    ):
                        with Timing_Messager(
                            f"Assembling meshes ({phase_name})", logger,
                        ):
                            bag.compute()

                dask_util.run_with_oom_retry(
                    _run, wave.workers, phase_name, logger,
                    max_retries=self.memory_retry_max,
                    retry_on_oom=self.retry_on_oom,
                    config=wave.config,
                )
        finally:
            self.segmentation_array = saved_array
        if self.do_legacy_neuroglancer:
            write_ngmesh_metadata(f"{self.output_directory}/meshes")
        elif self.do_singleres_multires_neuroglancer:
            write_singleres_multires_metadata(f"{self.output_directory}/meshes")
        shutil.rmtree(dirname)

    def _generate_meshes_at_scale(self, output_mesh_dir, downsample_factor=None,
                                   target_reduction_override=None,
                                   input_dataset_path=None):
        """Generate meshes at a given downsampling level, writing PLYs to output_mesh_dir.

        This creates a temporary Meshify-like pipeline that:
        1. Reads the segmentation volume (with optional extra downsampling)
        2. Runs marching cubes per block
        3. Assembles block meshes into per-segment PLYs

        ``target_reduction_override`` lets the caller swap in a per-LOD
        reduction value for this run (used by the downsample multires
        strategy so each LOD lands at hemibrain-equivalent face count).

        ``input_dataset_path`` lets the caller point at a different
        OME-NGFF multiscale level for this LOD (e.g. read ``.../s2``
        directly instead of downsampling ``.../s0`` by 4 in-worker).
        When this is given, ``downsample_factor`` should typically be 1
        (or omitted) since the source is already at the desired
        resolution.
        """
        # Save/restore state so we can temporarily override output + downsample
        orig_output = self.output_directory
        orig_downsample = self.downsample_factor
        orig_voxel_size = self.output_voxel_size_funlib
        orig_true_voxel = self.true_voxel_size.copy()
        orig_do_legacy = self.do_legacy_neuroglancer
        orig_do_singleres = self.do_singleres_multires_neuroglancer
        orig_target_reduction = self.target_reduction
        orig_seg_array = self.segmentation_array
        orig_dataset_path = self._dataset_path
        orig_input_path = self.input_path
        orig_base_voxel = self.base_voxel_size_funlib
        orig_true_offset = self.true_offset.copy()
        orig_roi = self.roi

        try:
            # Override to write PLYs (not neuroglancer format) to the scale dir
            self.output_directory = output_mesh_dir
            self.do_legacy_neuroglancer = False
            self.do_singleres_multires_neuroglancer = False

            if input_dataset_path is not None:
                # Swap to a pre-existing multiscale level. Reads that
                # dataset directly — no in-worker downsampling.
                from mesh_n_bone.util.zarr_io import (
                    open_dataset, split_dataset_path,
                    read_raw_voxel_size, read_raw_offset,
                )
                f_name, d_name = split_dataset_path(input_dataset_path)
                new_arr = open_dataset(f_name, d_name)
                self.segmentation_array = new_arr
                self._dataset_path = (
                    getattr(new_arr, "_dataset_path", None) or input_dataset_path
                )
                self.input_path = input_dataset_path
                self.base_voxel_size_funlib = new_arr.voxel_size
                self.output_voxel_size_funlib = new_arr.voxel_size
                # Prefer OME-NGFF fractional precision when available
                ome_vs, ome_off, _ = _read_ome_ngff_transform(input_dataset_path)
                self.true_voxel_size = (
                    np.array(ome_vs, dtype=float) if ome_vs is not None
                    else np.array(read_raw_voxel_size(new_arr), dtype=float)
                )
                self.true_offset = (
                    np.array(ome_off, dtype=float) if ome_off is not None
                    else np.array(read_raw_offset(new_arr), dtype=float)
                )
                self.downsample_factor = None
                # Refresh ROI when we didn't have a custom one. For a
                # custom ROI we keep the user-specified world bbox.
                if not self.has_custom_roi:
                    self.roi = new_arr.roi
            elif downsample_factor is not None:
                self.downsample_factor = downsample_factor
                self.output_voxel_size_funlib = Coordinate(
                    np.array(self.base_voxel_size_funlib) * downsample_factor
                )
                self.true_voxel_size = orig_true_voxel / (orig_downsample or 1) * downsample_factor

            if target_reduction_override is not None:
                self.target_reduction = target_reduction_override

            os.makedirs(self.output_directory, exist_ok=True)
            tmp_chunked_dir = self.output_directory + "/tmp_chunked"
            os.makedirs(tmp_chunked_dir, exist_ok=True)
            self.get_chunked_meshes(tmp_chunked_dir)
            self.assemble_meshes(tmp_chunked_dir)
            shutil.rmtree(tmp_chunked_dir, ignore_errors=True)
        finally:
            self.output_directory = orig_output
            self.downsample_factor = orig_downsample
            self.output_voxel_size_funlib = orig_voxel_size
            self.true_voxel_size = orig_true_voxel
            self.do_legacy_neuroglancer = orig_do_legacy
            self.do_singleres_multires_neuroglancer = orig_do_singleres
            self.target_reduction = orig_target_reduction
            self.segmentation_array = orig_seg_array
            self._dataset_path = orig_dataset_path
            self.input_path = orig_input_path
            self.base_voxel_size_funlib = orig_base_voxel
            self.true_offset = orig_true_offset
            self.roi = orig_roi

    def _discover_existing_scales(self):
        """Return ``{relative_downsample_factor: dataset_path}`` for OME-NGFF
        multiscale levels available in the input zarr, relative to the
        input dataset's own scale.

        For example, if ``input_path`` is ``.../seg/s0`` (scale=[1,1,1])
        and the group exposes s0..s3, this returns ``{1: '...s0', 2:
        '...s1', 4: '...s2', 8: '...s3'}``. If the input is already s1,
        the same group gives ``{1: '...s1', 2: '...s2', 4: '...s3'}``
        (s0 is finer than input → not useful for a downsample pyramid).

        Cached after first call. Returns ``{}`` when the input has no
        recognisable multiscales metadata (precomputed sources, plain
        zarr arrays without OME, etc.) — caller should fall back to
        in-worker downsampling.
        """
        if hasattr(self, "_existing_scales_cache"):
            return self._existing_scales_cache

        result = {}
        try:
            from mesh_n_bone.util.zarr_io import (
                _get_multiscales,
                _extract_ome_scale_translation,
                _read_parent_attrs,
                _path_join,
                _path_dirname,
                _path_basename,
            )
            parent_attrs = _read_parent_attrs(self.segmentation_array)
            multiscales = _get_multiscales(parent_attrs)
            if multiscales and isinstance(multiscales, list) and multiscales:
                ms = multiscales[0]
                input_ds_name = _path_basename(self._dataset_path)
                input_scale, _ = _extract_ome_scale_translation(
                    parent_attrs, dataset_name=input_ds_name,
                )
                if input_scale is not None and all(s > 0 for s in input_scale):
                    parent_path = _path_dirname(self._dataset_path)
                    for ds in ms.get("datasets", []) or []:
                        ds_name = ds.get("path")
                        if not ds_name:
                            continue
                        ds_scale, _ = _extract_ome_scale_translation(
                            parent_attrs, dataset_name=ds_name,
                        )
                        if ds_scale is None:
                            continue
                        ratios = [d / i for d, i in zip(ds_scale, input_scale)]
                        if not all(abs(r - ratios[0]) < 1e-6 for r in ratios):
                            continue
                        ratio = ratios[0]
                        if ratio < 1.0 - 1e-6:
                            continue
                        ratio_int = int(round(ratio))
                        if abs(ratio - ratio_int) > 1e-6:
                            continue
                        if ratio_int < 1 or (ratio_int & (ratio_int - 1)) != 0:
                            continue
                        result[ratio_int] = _path_join(parent_path, ds_name)
        except Exception as e:
            logger.warning(
                "Could not discover existing multiscale levels (%s); "
                "falling back to in-worker downsampling for all LODs.", e,
            )

        self._existing_scales_cache = result
        return result

    def _per_lod_target_reduction(self, lod: int) -> float:
        """Per-LOD target_reduction for the downsample multires strategy.

        Raw MC face count drops by 4x per scale step (voxel doubling).
        We want each LOD to land at the previous LOD's face count divided
        by ``decimation_factor`` (default 6 for hemibrain). So the keep
        ratio at LOD k is::

            keep(k) = (1 - target_reduction) * (4 / decimation_factor)**k

        For ``target_reduction=0.933`` (hemibrain s1 LOD-0 anchor) and
        ``decimation_factor=6``, this gives 0.933, 0.955, 0.970, 0.980 at
        LODs 0..3 — matching hemibrain's measured face counts within ~1-2%.
        """
        keep_0 = max(0.0, 1.0 - float(self.target_reduction))
        ratio = 4.0 / float(self.decimation_factor)
        keep_k = keep_0 * (ratio ** lod)
        return max(0.0, min(1.0, 1.0 - keep_k))

    def _get_downsample_factor_for_lod(self, lod):
        """Get the total downsample factor for a given LOD level.

        If the base config already has a downsample_factor, multiply it.
        Otherwise, use 2^lod (lod=0 means no downsampling).
        """
        base = self.downsample_factor or 1
        return base * (2 ** lod)

    def get_multiscale_meshes(self):
        """Generate meshes and create neuroglancer multiresolution output.

        Two strategies are supported via self.multires_strategy:

        "decimate" (default):
            1. Generate meshes at scale 0 from the zarr volume
            2. Decimate each mesh with pyfqmr for LODs 1, 2, ...
            3. Feed all LODs into the neuroglancer multires pipeline
            Best for thin/elongated structures (e.g. mitochondria).

        "downsample":
            1. For each LOD, downsample the volume by 2^lod
            2. Run marching cubes at each downsampled resolution
            3. Feed all LODs into the neuroglancer multires pipeline
            Best for thick/compact structures.
        """
        from mesh_n_bone.multires.multires import generate_all_neuroglancer_multires_meshes
        from mesh_n_bone.multires.decimation import (
            generate_decimated_meshes,
            delete_decimated_mesh_files,
        )
        from mesh_n_bone.util import neuroglancer

        os.makedirs(self.output_directory, exist_ok=True)
        mesh_lods_dir = f"{self.output_directory}/mesh_lods"
        os.makedirs(mesh_lods_dir, exist_ok=True)

        lods = list(range(self.num_lods))

        if self.multires_strategy == "decimate":
            self._generate_multires_decimate(mesh_lods_dir, lods)
        elif self.multires_strategy == "downsample":
            self._generate_multires_downsample(mesh_lods_dir, lods)
        else:
            raise ValueError(
                f"Unknown multires_strategy '{self.multires_strategy}'. "
                f"Choose from: 'decimate', 'downsample'"
            )

        # Collect mesh IDs and file sizes from s0
        s0_dir = f"{mesh_lods_dir}/s0"
        mesh_ids = []
        file_sizes = []
        mesh_ext = None
        with os.scandir(s0_dir) as it:
            for entry in it:
                if not entry.is_file():
                    continue
                name = entry.name
                root, ext = os.path.splitext(name)
                try:
                    mesh_id = int(root)
                except ValueError:
                    continue
                if mesh_ext is None:
                    mesh_ext = ext
                mesh_ids.append(mesh_id)
                file_sizes.append(entry.stat(follow_symlinks=False).st_size)

        if not mesh_ids:
            logger.warning("No meshes found at s0, skipping multires generation")
            return

        logger.info(f"Generating neuroglancer multires for {len(mesh_ids)} meshes with {self.num_lods} LODs")

        effective_workers = dask_util.effective_num_workers(
            self.num_workers, len(mesh_ids), logger, "multires creation",
        )

        def _run(workers, config):
            with dask_util.start_dask(
                workers, "multires creation", logger, config=config,
            ):
                with Timing_Messager("Generating multires meshes", logger):
                    generate_all_neuroglancer_multires_meshes(
                        self.output_directory,
                        workers,
                        mesh_ids,
                        lods,
                        mesh_ext,
                        np.array(file_sizes, dtype=float),
                        self.lod_0_box_size,
                        target_faces_per_lod0_chunk=self.target_faces_per_lod0_chunk,
                    )

        dask_util.run_with_oom_retry(
            _run, effective_workers, "multires creation", logger,
            max_retries=self.memory_retry_max, retry_on_oom=self.retry_on_oom,
        )

        multires_path = f"{self.output_directory}/multires"
        with Timing_Messager("Writing segment properties file", logger):
            # Always write segment_properties before sharding (it scans
            # `.index` files, which are removed once shards are packed).
            neuroglancer.write_segment_properties_file(
                multires_path,
                csv_path=self.segment_properties_csv,
                csv_columns=self.segment_properties_columns,
                csv_id_column=self.segment_properties_id_column,
            )

        if self.sharded:
            from mesh_n_bone.util import sharded_mesh_util

            params = sharded_mesh_util.choose_shard_params(len(mesh_ids))
            for key, override in (
                ("preshift_bits", self.preshift_bits),
                ("minishard_bits", self.minishard_bits),
                ("shard_bits", self.shard_bits),
            ):
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
                            multires_path, mesh_ids, spec, workers,
                        )

            dask_util.run_with_oom_retry(
                _pack, effective_workers, "shard packing", logger,
                max_retries=self.memory_retry_max, retry_on_oom=self.retry_on_oom,
            )

            with Timing_Messager("Writing sharded info file", logger):
                sharded_mesh_util.write_sharded_info_file(
                    multires_path, spec, vertex_quantization_bits=16,
                )

            if self.delete_unsharded_files:
                with Timing_Messager(
                    "Deleting unsharded per-segment files", logger
                ):
                    sharded_mesh_util.delete_unsharded_segment_files(
                        multires_path, mesh_ids,
                    )
        else:
            with Timing_Messager("Writing info file", logger):
                neuroglancer.write_info_file(multires_path)

        if self.delete_decimated_meshes:
            with Timing_Messager("Cleaning up intermediate mesh files", logger):
                shutil.rmtree(mesh_lods_dir, ignore_errors=True)

        logger.info("Multires pipeline complete")

    def _generate_multires_decimate(self, mesh_lods_dir, lods):
        """Strategy: mesh at s0, then decimate for higher LODs."""
        from mesh_n_bone.multires.decimation import generate_decimated_meshes

        # Step 1: Generate s0 meshes from the zarr volume
        s0_dir = f"{mesh_lods_dir}/s0"
        logger.info("Generating meshes at LOD 0 from segmentation volume")
        self._generate_meshes_at_scale(s0_dir, self.downsample_factor)

        # Move meshes from s0/meshes/ up to s0/
        s0_mesh_subdir = f"{s0_dir}/meshes"
        if os.path.isdir(s0_mesh_subdir):
            for f in os.listdir(s0_mesh_subdir):
                shutil.move(f"{s0_mesh_subdir}/{f}", f"{s0_dir}/{f}")
            os.rmdir(s0_mesh_subdir)

        # Collect mesh IDs from s0
        mesh_ids = []
        mesh_ext = None
        for f in os.listdir(s0_dir):
            root, ext = os.path.splitext(f)
            if ext in (".ply", ".obj"):
                if mesh_ext is None:
                    mesh_ext = ext
                try:
                    mesh_ids.append(int(root))
                except ValueError:
                    continue

        if not mesh_ids:
            logger.warning("No meshes found at s0")
            return

        # Step 2: Decimate s0 meshes for LODs 1, 2, ...
        if len(lods) > 1:
            logger.info(f"Decimating meshes for LODs 1-{len(lods)-1} "
                        f"(factor={self.decimation_factor}, aggressiveness={self.decimation_aggressiveness})")
            effective_workers = dask_util.effective_num_workers(
                self.num_workers, len(mesh_ids), logger, "decimation",
            )

            def _run(workers, config):
                with dask_util.start_dask(
                    workers, "decimation", logger, config=config,
                ):
                    with Timing_Messager("Generating decimated meshes", logger):
                        generate_decimated_meshes(
                            s0_dir,
                            self.output_directory,
                            lods,
                            mesh_ids,
                            mesh_ext,
                            self.decimation_factor,
                            self.decimation_aggressiveness,
                            workers,
                        )

            dask_util.run_with_oom_retry(
                _run, effective_workers, "decimation", logger,
                max_retries=self.memory_retry_max,
                retry_on_oom=self.retry_on_oom,
            )

    def _generate_multires_downsample(self, mesh_lods_dir, lods):
        """Strategy: downsample volume at each LOD, re-mesh.

        Each LOD's ``target_reduction`` is computed via
        ``_per_lod_target_reduction`` so the per-LOD face count tracks a
        constant ``decimation_factor`` ratio between consecutive LODs
        (default 6 for hemibrain). LOD 0 uses the configured
        ``target_reduction`` unchanged.

        If the input zarr exposes OME-NGFF multiscales metadata, each
        LOD prefers reading from the matching pre-existing scale (e.g.
        ``s_k`` for LOD k) rather than downsampling the input in-worker.
        Falls back to in-worker downsampling per-LOD when no matching
        scale exists.
        """
        existing_scales = (
            self._discover_existing_scales() if self.use_existing_scales else {}
        )
        base_ds = self.downsample_factor or 1
        for lod in lods:
            scale_dir = f"{mesh_lods_dir}/s{lod}"
            target_factor = 2 ** lod
            tr_lod = self._per_lod_target_reduction(lod)

            existing_path = existing_scales.get(target_factor)
            if existing_path is not None and base_ds == 1:
                # Read directly from a pre-computed scale; no in-worker downsample
                logger.info(
                    f"Generating meshes at LOD {lod} "
                    f"(reading existing scale {existing_path}, "
                    f"target_reduction {tr_lod:.4f})"
                )
                self._generate_meshes_at_scale(
                    scale_dir,
                    target_reduction_override=tr_lod,
                    input_dataset_path=existing_path,
                )
            else:
                # Fall back to in-worker downsampling of the input volume
                ds_factor = (self.downsample_factor if lod == 0
                             else self._get_downsample_factor_for_lod(lod))
                logger.info(
                    f"Generating meshes at LOD {lod} "
                    f"(downsample factor {ds_factor}, "
                    f"target_reduction {tr_lod:.4f})"
                )
                self._generate_meshes_at_scale(
                    scale_dir, ds_factor, target_reduction_override=tr_lod,
                )

            # Move meshes from s{lod}/meshes/ up to s{lod}/
            scale_mesh_subdir = f"{scale_dir}/meshes"
            if os.path.isdir(scale_mesh_subdir):
                for f in os.listdir(scale_mesh_subdir):
                    shutil.move(f"{scale_mesh_subdir}/{f}", f"{scale_dir}/{f}")
                os.rmdir(scale_mesh_subdir)

    def get_meshes(self):
        """Generate meshes: chunk, assemble, and optionally analyze.

        If do_multires is True, generates meshes at multiple downsampled
        scales and creates neuroglancer multiresolution output.
        """
        if self.do_multires:
            self.get_multiscale_meshes()
            return

        os.makedirs(self.output_directory, exist_ok=True)
        tmp_chunked_dir = self.output_directory + "/tmp_chunked"
        os.makedirs(tmp_chunked_dir, exist_ok=True)
        self.get_chunked_meshes(tmp_chunked_dir)
        self.assemble_meshes(tmp_chunked_dir)
        shutil.rmtree(tmp_chunked_dir, ignore_errors=True)

        if self.do_analysis:
            from mesh_n_bone.analyze.analyze import AnalyzeMeshes

            analyze = AnalyzeMeshes(
                self.output_directory + "/meshes",
                self.output_directory + "/metrics",
                num_workers=self.num_workers,
            )
            analyze.analyze()
