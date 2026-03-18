"""Mesh generation from segmentation volumes using zmesh and dask."""

from funlib.persistence import open_ds
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
from mesh_n_bone.meshify.downsample import (
    downsample_labels_3d_suppress_zero,
    downsample_labels_3d,
    downsample_binary_3d,
)

logger = logging.getLogger(__name__)

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
    """Compute per-stage reductions given how much of the total reduction
    each stage should contribute.

    Args:
        target_reduction_total: overall target reduction (e.g. 0.99)
        frac1, frac2: relative fractions of total simplification (e.g. 0.25, 0.75)
    """
    assert abs(frac1 + frac2 - 1.0) < 1e-6, "fractions must sum to 1"
    keep_total = 1 - target_reduction_total
    r1 = 1 - keep_total**frac1
    r2 = 1 - keep_total**frac2
    return r1, r2


class Meshify:
    """Create meshes from a segmentation volume using dask and zmesh."""

    def __init__(
        self,
        input_path: str,
        output_directory: str,
        total_roi: Roi = None,
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
        downsample_method: str = "mode_suppress_zero",
    ):
        # Try single-arg open_ds first (newer funlib.persistence),
        # fall back to two-arg form (older versions)
        try:
            self.segmentation_array = open_ds(input_path)
        except Exception:
            for file_type in [".n5", ".zarr"]:
                if file_type in input_path:
                    path_split = input_path.split(file_type + "/")
                    break
            self.segmentation_array = open_ds(
                path_split[0] + file_type, path_split[1]
            )
        self.output_directory = output_directory

        # Get true (possibly non-integer) voxel size from the underlying data
        try:
            from funlib.persistence.arrays.datasets import _read_attrs
            self.true_voxel_size, _, _ = _read_attrs(self.segmentation_array.data)
            self.true_voxel_size = np.array(self.true_voxel_size)
        except ImportError:
            # Newer funlib.persistence versions don't expose _read_attrs
            self.true_voxel_size = np.array(self.segmentation_array.voxel_size)
        if total_roi:
            self.total_roi = total_roi
        else:
            self.total_roi = self.segmentation_array.roi
        self.num_workers = num_workers

        if read_write_block_shape_pixels:
            self.read_write_block_shape_pixels = np.array(read_write_block_shape_pixels)
        else:
            self.read_write_block_shape_pixels = np.array(
                self.segmentation_array.chunk_shape
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
            if self.lod_0_box_size.size == 1:
                self.lod_0_box_size = np.full(3, self.lod_0_box_size.item())
        else:
            self.lod_0_box_size = None
        self.downsample_method = downsample_method
        self.input_path = input_path

    def _get_downsample_function(self):
        """Return the appropriate downsample function based on config."""
        methods = {
            "mode_suppress_zero": downsample_labels_3d_suppress_zero,
            "mode": downsample_labels_3d,
            "binary": downsample_binary_3d,
        }
        if self.downsample_method not in methods:
            raise ValueError(
                f"Unknown downsample_method '{self.downsample_method}'. "
                f"Choose from: {list(methods.keys())}"
            )
        return methods[self.downsample_method]

    @staticmethod
    def my_cloudvolume_concatenate(*meshes):
        vertex_ct = np.zeros(len(meshes) + 1, np.uint32)
        vertex_ct[1:] = np.cumsum([len(mesh) for mesh in meshes])
        vertices = np.concatenate([mesh.vertices for mesh in meshes])
        faces = np.concatenate(
            [mesh.faces + vertex_ct[i] for i, mesh in enumerate(meshes)]
        )
        normals = None
        return CloudVolumeMesh(vertices, faces, normals)

    def _get_chunked_mesh(self, block, tmpdirname):
        mesher = Mesher(self.output_voxel_size_funlib[::-1])
        segmentation_block = self.segmentation_array.to_ndarray(block.roi, fill_value=0)
        if segmentation_block.dtype.byteorder == ">":
            swapped_dtype = segmentation_block.dtype.newbyteorder()
            segmentation_block = segmentation_block.view(swapped_dtype).byteswap()
        if self.downsample_factor:
            ds_func = self._get_downsample_function()
            segmentation_block, _ = ds_func(
                segmentation_block, self.downsample_factor
            )

        block_offset = np.array(block.roi.get_begin())
        mesher.mesh(segmentation_block, close=False)
        for id in mesher.ids():
            mesh = mesher.get_mesh(id)
            os.makedirs(f"{tmpdirname}/{id}", exist_ok=True)

            if self.use_fixed_edge_simplification and self.do_simplification:
                mesh_tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
                stage_1_reduction, _ = staged_reductions(
                    self.target_reduction,
                    self.stage_1_reduction_fraction,
                    self.stage_2_reduction_fraction,
                )

                block_size_voxels = self.read_write_block_shape_pixels + 1
                block_size_world = (block_size_voxels * self.output_voxel_size_funlib)[
                    ::-1
                ]

                mesh_tri_simplified = simplify_mesh(
                    mesh_tri,
                    voxel_size=self.output_voxel_size_funlib,
                    target_reduction=stage_1_reduction,
                    block_size=block_size_world,
                    aggressiveness=self.default_aggressiveness,
                    verbose=False,
                    fix_edges=True,
                )
                mesh_tri_simplified.vertices += block_offset[::-1]

                mesh_simplified = CloudVolumeMesh(
                    mesh_tri_simplified.vertices,
                    mesh_tri_simplified.faces,
                    normals=None,
                )

                with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                    fp.write(mesh_simplified.to_ply())
            else:
                mesh.vertices += block_offset[::-1]
                with open(f"{tmpdirname}/{id}/block_{block.index}.ply", "wb") as fp:
                    fp.write(mesh.to_ply())

    @staticmethod
    def is_mesh_valid(mesh):
        return mesh.is_winding_consistent and mesh.is_watertight and mesh.volume > 0

    def get_chunked_meshes(self, dirname):
        blocks = dask_util.create_blocks(
            self.total_roi,
            self.segmentation_array,
            self.read_write_block_shape_pixels.copy(),
            padding=self.output_voxel_size_funlib,
        )

        b = db.from_sequence(blocks, npartitions=self.num_workers * 10).map(
            self._get_chunked_mesh, dirname
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
    ):
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
                    fix_edges=False,
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
                ms.apply_coord_taubin_smoothing(
                    lambda_=0.5,
                    mu=-0.53,
                    stepsmoothnum=n_smoothing_iter,
                )
                m = ms.current_mesh()
                simplified_mesh = trimesh.Trimesh(
                    vertices=m.vertex_matrix(), faces=m.face_matrix()
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
            do_simplification = output_trimesh_mesh.faces.shape[0] > 100
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
        """Repair mesh using PyMeshLab and return as trimesh."""
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices, face_matrix=faces))

        ms.meshing_remove_duplicate_faces()
        ms.meshing_remove_duplicate_vertices()
        ms.meshing_remove_unreferenced_vertices()

        ms.meshing_repair_non_manifold_edges(method="Split Vertices")
        ms.meshing_repair_non_manifold_edges(method="Remove Faces")

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
                offset=self.total_roi.offset[::-1],
            )

        if self.check_mesh_validity:
            try:
                vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
                faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
                del mesh
                mesh = Meshify.repair_mesh_pymeshlab(
                    vertices,
                    faces,
                    remove_smallest_components=self.remove_smallest_components,
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
                    self.check_mesh_validity,
                )
            else:
                mesh = Meshify.simplify_and_smooth_mesh(
                    mesh,
                    self.target_reduction,
                    self.n_smoothing_iter,
                    self.remove_smallest_components,
                    self.default_aggressiveness,
                    self.do_simplification,
                    self.check_mesh_validity,
                )

            if len(mesh.faces) == 0:
                _ = mesh.export(f"{self.output_directory}/meshes/{mesh_id}.ply")
                raise Exception(f"Mesh {mesh_id} contains no faces")
        except Exception as e:
            raise Exception(f"{mesh_id} failed, with error: {e}")

        # Correct for difference between funlib voxel sizes (rounded) and actual
        if list(self.true_voxel_size) != list(self.output_voxel_size_funlib):
            mesh.vertices -= self.total_roi.offset[::-1]
            mesh.vertices *= np.array(self.true_voxel_size[::-1]) / np.array(
                self.output_voxel_size_funlib[::-1]
            )
            mesh.vertices += self.total_roi.offset[::-1]

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
        from mesh_n_bone.util.neuroglancer import (
            write_ngmesh_metadata,
            write_singleres_multires_metadata,
        )

        os.makedirs(f"{self.output_directory}/meshes/", exist_ok=True)
        self.dirname = dirname
        mesh_ids = os.listdir(dirname)
        b = db.from_sequence(
            mesh_ids,
            npartitions=dask_util.guesstimate_npartitions(mesh_ids, self.num_workers),
        ).map(self._assemble_mesh)
        with dask_util.start_dask(self.num_workers, "assemble meshes", logger):
            with Timing_Messager("Assembling meshes", logger):
                b.compute()
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
        """Generate meshes at multiple scales by downsampling the volume, then
        create neuroglancer multiresolution output.

        For each LOD level (0, 1, ..., num_lods-1):
        1. Downsample the segmentation volume by 2^lod
        2. Run marching cubes to generate per-segment PLY meshes
        3. Write to mesh_lods/s{lod}/

        Then feed all scales into the multires decomposition pipeline.
        """
        from mesh_n_bone.multires.multires import generate_all_neuroglancer_multires_meshes
        from mesh_n_bone.util import neuroglancer

        mesh_lods_dir = f"{self.output_directory}/mesh_lods"
        os.makedirs(mesh_lods_dir, exist_ok=True)

        lods = list(range(self.num_lods))

        for lod in lods:
            scale_dir = f"{mesh_lods_dir}/s{lod}"
            logger.info(f"Generating meshes at LOD {lod} (downsample factor {self._get_downsample_factor_for_lod(lod)})")

            if lod == 0:
                # At LOD 0, use current downsample_factor (may be None)
                self._generate_meshes_at_scale(scale_dir, self.downsample_factor)
            else:
                ds_factor = self._get_downsample_factor_for_lod(lod)
                self._generate_meshes_at_scale(scale_dir, ds_factor)

        # Collect mesh IDs and file sizes from s0
        s0_mesh_dir = f"{mesh_lods_dir}/s0/meshes"
        mesh_ids = []
        file_sizes = []
        mesh_ext = None
        with os.scandir(s0_mesh_dir) as it:
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

        # The multires pipeline expects mesh_lods/s{lod}/{id}.ply
        # but _generate_meshes_at_scale writes to s{lod}/meshes/{id}.ply
        # So we need to restructure: move meshes up one level
        for lod in lods:
            scale_mesh_dir = f"{mesh_lods_dir}/s{lod}/meshes"
            scale_dir = f"{mesh_lods_dir}/s{lod}"
            if os.path.isdir(scale_mesh_dir):
                for f in os.listdir(scale_mesh_dir):
                    shutil.move(f"{scale_mesh_dir}/{f}", f"{scale_dir}/{f}")
                os.rmdir(scale_mesh_dir)

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
                )

        with Timing_Messager("Writing info and segment properties files", logger):
            multires_path = f"{self.output_directory}/multires"
            neuroglancer.write_segment_properties_file(multires_path)
            neuroglancer.write_info_file(multires_path)

        logger.info("Multi-scale multires pipeline complete")

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

        if self.do_analysis:
            from mesh_n_bone.analyze.analyze import AnalyzeMeshes

            analyze = AnalyzeMeshes(
                self.output_directory + "/meshes",
                self.output_directory + "/metrics",
                num_workers=self.num_workers,
            )
            analyze.analyze()
