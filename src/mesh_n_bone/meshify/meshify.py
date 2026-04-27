"""Mesh generation from segmentation volumes using zmesh and dask."""

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
from mesh_n_bone.util import dask_util
from mesh_n_bone.util.logging import Timing_Messager
from mesh_n_bone.util.zarr_io import (
    open_dataset,
    split_dataset_path,
    read_raw_voxel_size,
    _read_attrs,
    _get_multiscales,
    _extract_ome_scale_translation,
)
from mesh_n_bone.util.image_data_interface import open_ds_tensorstore, to_ndarray_tensorstore
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
    for ext in (".zarr", ".n5"):
        if ext in input_path:
            parts = input_path.split(ext + "/")
            zarr_root_path = parts[0] + ext
            dataset_path = parts[1] if len(parts) > 1 else ""
            break
    else:
        return None, None, None

    dataset_name = os.path.basename(dataset_path)
    parent_path = os.path.dirname(dataset_path)
    parent_dir = os.path.join(zarr_root_path, parent_path) if parent_path else zarr_root_path

    try:
        parent_attrs = _read_attrs(parent_dir)
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


# Thread-local tensorstore handle cache so each worker opens once
_thread_local_ts = {}


def _get_chunked_mesh_worker(block, tmpdirname, config):
    """Run marching cubes on a single block and write per-segment PLYs.

    This is a module-level function so only lightweight *config* dict
    (scalars, tuples, strings) is serialised to workers — no zarr arrays.

    Parameters
    ----------
    block : DaskBlock
        Block specification with ROI and index.
    tmpdirname : str
        Temporary directory for writing per-segment block meshes.
    config : dict
        Worker config from ``Meshify._get_worker_config()``.
    """
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
        swap_axes=config["swap_axes"], fill_value=0,
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

    block_offset = np.array(block.roi.get_begin())
    # Correct for the half-kernel shift introduced by downsampling:
    # a downsampled voxel at index 0 represents original voxels [0, ds),
    # centered at (ds-1)/2 original voxels from the origin.
    ds_shift = (downsample_factor - 1) / 2 * np.array(voxel_size) if downsample_factor else np.zeros(3)
    mesher.mesh(segmentation_block, close=False)
    for id in mesher.ids():
        mesh = mesher.get_mesh(id)
        os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)

        if config["use_fixed_edge_simplification"] and config["do_simplification"]:
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

            stage_1_reduction, _ = staged_reductions(
                config["target_reduction"],
                config["stage_1_reduction_fraction"],
                1 - config["stage_1_reduction_fraction"],
            )

            ds = config["downsample_factor"] or 1
            block_size_voxels = np.array(config["read_write_block_shape_pixels"]) // ds
            block_size_world = (block_size_voxels * output_voxel_size)[::-1]

            mesh_tri_simplified = simplify_mesh(
                mesh_tri,
                voxel_size=output_voxel_size,
                target_reduction=stage_1_reduction,
                block_size=block_size_world,
                aggressiveness=config["default_aggressiveness"],
                verbose=False,
                fix_edges=True,
            )
            half_pad_offset = block_offset + 0.5 * np.array(output_voxel_size)
            mesh_tri_simplified.vertices += half_pad_offset[::-1] + ds_shift[::-1]

            mesh_simplified = CloudVolumeMesh(
                mesh_tri_simplified.vertices,
                mesh_tri_simplified.faces,
                normals=None,
            )

            with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                fp.write(mesh_simplified.to_ply())
        else:
            mesh.vertices += block_offset[::-1] + ds_shift[::-1]
            with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
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
        Use boundary-preserving simplification that pins block-edge
        vertices during the per-chunk stage.
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
        Downsampling method: ``"mode_suppress_zero"``,
        ``"mode"``, or ``"binary"``.
    multires_strategy : str
        Strategy for generating LODs: ``"decimate"`` (simplify s0
        meshes) or ``"downsample"`` (re-mesh at lower resolutions).
    decimation_factor : int
        Face-count reduction factor between consecutive LODs.
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
        target_reduction: float = 0.99,
        num_workers: int = 10,
        remove_smallest_components: bool = True,
        n_smoothing_iter: int = 10,
        default_aggressiveness: int = 0.3,
        check_mesh_validity: bool = True,
        do_simplification: bool = True,
        do_analysis: bool = True,
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
        do_multires: bool = False,
        num_lods: int = 3,
        lod_0_box_size=None,
        target_faces_per_lod0_chunk: int = 25_000,
        downsample_method: str = "mode_suppress_zero",
        multires_strategy: str = "decimate",
        decimation_factor: int = 4,
        decimation_aggressiveness: int = 7,
        delete_decimated_meshes: bool = True,
        segment_properties_csv: str = None,
        segment_properties_columns: list = None,
        segment_properties_id_column: str = "Object ID",
        coordinate_units: str = "nm",
        voxel_size_nm: list = None,
    ):
        filename, dataset_name = split_dataset_path(input_path)
        self.segmentation_array = open_dataset(filename, dataset_name)
        self.output_directory = output_directory
        self.input_path = input_path
        self._dataset_path = os.path.join(filename, dataset_name) if dataset_name else filename
        self._swap_axes = input_path.rfind(".n5") > input_path.rfind(".zarr")

        # Get true (possibly non-integer) voxel size from the underlying data
        self.true_voxel_size = np.array(read_raw_voxel_size(self.segmentation_array))

        # Check if voxel_size is just defaults (1,1,1) and
        # try OME-NGFF multiscales metadata from the parent zarr group
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
        self.decimation_factor = decimation_factor
        self.decimation_aggressiveness = decimation_aggressiveness
        self.delete_decimated_meshes = delete_decimated_meshes
        self.segment_properties_csv = segment_properties_csv
        self.segment_properties_columns = segment_properties_columns
        self.segment_properties_id_column = segment_properties_id_column
        self.coordinate_units = coordinate_units

    def _default_block_shape_pixels(self, target_mb=128):
        """Choose a default block shape as a chunk-aligned multiple.

        Picks the largest integer multiple of the dataset's chunk shape
        whose memory footprint stays at or below ``target_mb``.  Larger
        blocks reduce the number of block boundaries (and therefore
        frozen boundary vertices during fixed-edge simplification).

        Parameters
        ----------
        target_mb : int or float
            Target memory budget per block in megabytes.

        Returns
        -------
        numpy.ndarray
            Block shape in voxels, as a multiple of the chunk shape.
        """
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

        Parameters
        ----------
        dirname : str
            Directory where per-segment block mesh PLYs are written.
        """
        blocks = dask_util.create_blocks(
            self.roi,
            self.segmentation_array,
            self.read_write_block_shape_pixels.copy(),
            padding=self.output_voxel_size_funlib,
        )

        worker_config = self._get_worker_config()
        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            _get_chunked_mesh_worker, dirname, worker_config
        )

        with dask_util.start_dask(self.num_workers, "generate chunked meshes", logger):
            with Timing_Messager("Generating chunked meshes", logger):
                b.compute()

    @staticmethod
    def simplify_and_smooth_mesh(
        mesh,
        target_reduction=0.99,
        n_smoothing_iter=10,
        remove_smallest_components=True,
        aggressiveness=0.3,
        do_simplification=True,
        check_mesh_validity=True,
        preserve_open_boundaries=False,
    ):
        """Simplify, smooth, and optionally repair a mesh.

        Applies quadric-error simplification followed by Taubin smoothing.
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

        Returns
        -------
        trimesh.Trimesh
            Processed mesh.
        """
        def get_cleaned_simplified_and_smoothed_mesh(
            mesh, target_reduction, aggressiveness, do_simplification
        ):
            if do_simplification:
                simplified_mesh = simplify_mesh(
                    mesh,
                    voxel_size=None,
                    target_reduction=target_reduction,
                    aggressiveness=aggressiveness,
                    verbose=False,
                    fix_edges=preserve_open_boundaries,
                )
            else:
                simplified_mesh = mesh
            del mesh

            if n_smoothing_iter > 0:
                ms = pymeshlab.MeshSet()
                ms.add_mesh(
                    pymeshlab.Mesh(
                        vertex_matrix=simplified_mesh.vertices,
                        face_matrix=simplified_mesh.faces,
                    )
                )
                if preserve_open_boundaries:
                    # Identify boundary vertices and save their positions
                    ms.compute_selection_from_mesh_border()
                    border_mask = ms.current_mesh().vertex_selection_array()
                    border_positions = simplified_mesh.vertices[border_mask].copy()

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

                simplified_mesh = trimesh.Trimesh(
                    vertices=verts, faces=m.face_matrix()
                )

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
            chunk_size = (
                self.read_write_block_shape_pixels * self.base_voxel_size_funlib
            )
            mesh = mesh.deduplicate_chunk_boundaries(
                chunk_size=chunk_size[::-1],
                offset=self.roi.offset[::-1],
            )
            if self.use_fixed_edge_simplification:
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
                stage_2_reduction, _ = staged_reductions(
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
                )

            if len(mesh.faces) == 0:
                _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        # Correct for difference between funlib voxel sizes (rounded) and actual
        if list(self.true_voxel_size) != list(self.output_voxel_size_funlib):
            scale = np.array(self.true_voxel_size[::-1]) / np.array(
                self.output_voxel_size_funlib[::-1]
            )
            offset_xyz = np.array(self.roi.offset[::-1], dtype=float)
            mesh.vertices -= offset_xyz
            mesh.vertices *= scale
            mesh.vertices += offset_xyz * scale

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
        mesh_ids = os.listdir(dirname)
        # Drop the zarr-backed array before dask serialises self — assembly
        # only reads PLY files, not the segmentation volume.
        saved_array = self.segmentation_array
        self.segmentation_array = None
        try:
            b = db.from_sequence(
                mesh_ids,
                npartitions=dask_util.guesstimate_npartitions(mesh_ids, self.num_workers),
            ).map(self._assemble_mesh)
            with dask_util.start_dask(self.num_workers, "assemble meshes", logger):
                with Timing_Messager("Assembling meshes", logger):
                    b.compute()
        finally:
            self.segmentation_array = saved_array
        if self.do_legacy_neuroglancer:
            write_ngmesh_metadata(f"{self.output_directory}/meshes")
        elif self.do_singleres_multires_neuroglancer:
            write_singleres_multires_metadata(f"{self.output_directory}/meshes")
        shutil.rmtree(dirname)

    def _generate_meshes_at_scale(self, output_mesh_dir, downsample_factor=None):
        """Generate meshes at a given downsampling level, writing PLYs to output_mesh_dir.

        This creates a temporary Meshify-like pipeline that:
        1. Reads the segmentation volume (with optional extra downsampling)
        2. Runs marching cubes per block
        3. Assembles block meshes into per-segment PLYs
        """
        # Save/restore state so we can temporarily override output + downsample
        orig_output = self.output_directory
        orig_downsample = self.downsample_factor
        orig_voxel_size = self.output_voxel_size_funlib
        orig_true_voxel = self.true_voxel_size.copy()
        orig_do_legacy = self.do_legacy_neuroglancer
        orig_do_singleres = self.do_singleres_multires_neuroglancer

        try:
            # Override to write PLYs (not neuroglancer format) to the scale dir
            self.output_directory = output_mesh_dir
            self.do_legacy_neuroglancer = False
            self.do_singleres_multires_neuroglancer = False

            if downsample_factor is not None:
                self.downsample_factor = downsample_factor
                self.output_voxel_size_funlib = Coordinate(
                    np.array(self.base_voxel_size_funlib) * downsample_factor
                )
                self.true_voxel_size = orig_true_voxel / (orig_downsample or 1) * downsample_factor

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
                if mesh_ext is None:
                    mesh_ext = ext
                try:
                    mesh_ids.append(int(root))
                    file_sizes.append(entry.stat(follow_symlinks=False).st_size)
                except ValueError:
                    continue

        if not mesh_ids:
            logger.warning("No meshes found at s0, skipping multires generation")
            return

        logger.info(f"Generating neuroglancer multires for {len(mesh_ids)} meshes with {self.num_lods} LODs")

        with dask_util.start_dask(self.num_workers, "multires creation", logger):
            with Timing_Messager("Generating multires meshes", logger):
                generate_all_neuroglancer_multires_meshes(
                    self.output_directory,
                    self.num_workers,
                    mesh_ids,
                    lods,
                    mesh_ext,
                    np.array(file_sizes, dtype=float),
                    self.lod_0_box_size,
                    target_faces_per_lod0_chunk=self.target_faces_per_lod0_chunk,
                )

        with Timing_Messager("Writing info and segment properties files", logger):
            multires_path = f"{self.output_directory}/multires"
            neuroglancer.write_segment_properties_file(
                multires_path,
                csv_path=self.segment_properties_csv,
                csv_columns=self.segment_properties_columns,
                csv_id_column=self.segment_properties_id_column,
            )
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
            with dask_util.start_dask(self.num_workers, "decimation", logger):
                with Timing_Messager("Generating decimated meshes", logger):
                    generate_decimated_meshes(
                        s0_dir,
                        self.output_directory,
                        lods,
                        mesh_ids,
                        mesh_ext,
                        self.decimation_factor,
                        self.decimation_aggressiveness,
                        self.num_workers,
                    )

    def _generate_multires_downsample(self, mesh_lods_dir, lods):
        """Strategy: downsample volume at each LOD, re-mesh."""
        for lod in lods:
            scale_dir = f"{mesh_lods_dir}/s{lod}"
            logger.info(f"Generating meshes at LOD {lod} "
                        f"(downsample factor {self._get_downsample_factor_for_lod(lod)})")

            if lod == 0:
                self._generate_meshes_at_scale(scale_dir, self.downsample_factor)
            else:
                ds_factor = self._get_downsample_factor_for_lod(lod)
                self._generate_meshes_at_scale(scale_dir, ds_factor)

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
