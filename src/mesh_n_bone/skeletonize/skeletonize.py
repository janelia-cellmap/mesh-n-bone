"""Skeletonization of meshes using CGAL and dask."""

from pathlib import Path
import subprocess
import numpy as np
import os
import json
import logging
import pandas as pd
import dask.bag as db
import dask.dataframe as dd
import networkx as nx

from mesh_n_bone.util import dask_util
from mesh_n_bone.util.logging import Timing_Messager
from mesh_n_bone.util.neuroglancer import write_precomputed_annotations
from mesh_n_bone.skeletonize.skeleton import CustomSkeleton

logger = logging.getLogger(__name__)


class Skeletonize:
    """Skeletonize meshes using CGAL and dask.

    Orchestrates the full pipeline: reading meshes, calling the CGAL
    skeletonizer binary, pruning and simplifying the resulting skeletons,
    computing morphometric metrics, and writing neuroglancer-compatible
    outputs.

    Parameters
    ----------
    input_directory : str
        Directory containing input mesh files (PLY or neuroglancer Draco).
    output_directory : str
        Root directory for all outputs (CGAL skeletons, neuroglancer
        skeletons, and metrics CSV).
    num_workers : int, optional
        Number of dask workers for parallel processing.  Default is ``10``.
    min_branch_length_nm : float, optional
        Branches shorter than this value (nanometres) are pruned.
        Default is ``100``.
    simplification_tolerance_nm : float, optional
        Ramer-Douglas-Peucker tolerance (nanometres) used when
        simplifying skeletons.  Default is ``50``.
    base_loop_subdivision_iterations : int, optional
        Number of Loop subdivision iterations applied to the mesh
        before skeletonization.  Default is ``1``.
    neuroglancer_format : bool, optional
        If ``True``, input meshes are treated as neuroglancer
        precomputed format (possibly Draco-compressed).  Default is
        ``False``.
    """

    def __init__(
        self,
        input_directory: str,
        output_directory: str,
        num_workers: int = 10,
        min_branch_length_nm: float = 100,
        simplification_tolerance_nm: float = 50,
        base_loop_subdivision_iterations: int = 1,
        neuroglancer_format: bool = False,
    ):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.num_workers = num_workers
        self.min_branch_length_nm = min_branch_length_nm
        self.simplification_tolerance_nm = simplification_tolerance_nm
        self.base_loop_subdivision_iterations = base_loop_subdivision_iterations
        self.neuroglancer_format = neuroglancer_format

    @staticmethod
    def cgal_skeletonize_mesh(
        input_file: str,
        output_file: str,
        base_loop_subdivision_iterations: int = 1,
        neuroglancer_format: bool = False,
        timeout_seconds: int = 120,
    ) -> None:
        """Run the CGAL skeletonizer binary on a single mesh file.

        If the input is a Draco-compressed neuroglancer mesh, it is first
        decoded and dequantized to a temporary PLY file.  The binary is
        invoked with increasing Loop subdivision iterations (starting from
        *base_loop_subdivision_iterations*) until it succeeds or the
        maximum number of retries is exhausted.

        Parameters
        ----------
        input_file : str
            Path to the input mesh (PLY or neuroglancer Draco).
        output_file : str
            Path where the CGAL skeleton text file will be written.
        base_loop_subdivision_iterations : int, optional
            Starting number of Loop subdivision iterations.  Default is
            ``1``.  The timeout scales by ``4**(iterations - 1)``.
        neuroglancer_format : bool, optional
            When ``True``, attempt to read the mesh as neuroglancer
            precomputed / Draco format.  Default is ``False``.
        timeout_seconds : int, optional
            Base timeout for the subprocess call in seconds.  Default is
            ``120``.

        Raises
        ------
        FileNotFoundError
            If the CGAL skeletonizer binary cannot be located.
        Exception
            If skeletonization times out at every retry or the subprocess
            exits with a non-zero status.
        """
        import tempfile
        import DracoPy

        # Look for the binary relative to the package root
        # (works for both editable installs and cloned repos)
        pkg_root = Path(__file__).resolve().parents[2]
        exe = pkg_root / "cgal_skeletonize_mesh" / "skeletonize_mesh"
        if not exe.exists():
            # Also check repo root (one level above src/)
            repo_root = pkg_root.parent
            exe = repo_root / "cgal_skeletonize_mesh" / "skeletonize_mesh"
        if not exe.exists():
            raise FileNotFoundError(
                f"CGAL skeletonizer binary not found. "
                f"Build it with: pixi run -e build-cgal build-cgal\n"
                f"Searched: {pkg_root / 'cgal_skeletonize_mesh'}"
            )
        timeout_seconds = timeout_seconds * 4 ** (
            base_loop_subdivision_iterations - 1
        )

        temp_ply_path = None
        actual_input_file = input_file

        if neuroglancer_format:
            with open(input_file, "rb") as f:
                magic = f.read(5)

            if magic == b"DRACO":
                logger.info(
                    "Detected Draco-compressed mesh, decompressing to temporary PLY file"
                )
                with open(input_file, "rb") as f:
                    draco_data = f.read()

                mesh_obj = DracoPy.decode_buffer_to_mesh(draco_data)

                vertex_quantization_bits = 10
                info_file = os.path.join(os.path.dirname(input_file), "info")
                if os.path.exists(info_file):
                    try:
                        with open(info_file, "r") as f:
                            info = json.load(f)
                            vertex_quantization_bits = info.get(
                                "vertex_quantization_bits", 10
                            )
                    except Exception as e:
                        logger.warning(
                            f"Failed to read info file, using default: {e}"
                        )

                chunk_shape = [1.0, 1.0, 1.0]
                grid_origin = [0.0, 0.0, 0.0]
                vertex_offsets = [0.0, 0.0, 0.0]
                lod = 0

                import struct

                index_file = input_file + ".index"
                if os.path.exists(index_file):
                    try:
                        with open(index_file, "rb") as f:
                            chunk_shape = list(struct.unpack("<3f", f.read(12)))
                            grid_origin = list(struct.unpack("<3f", f.read(12)))
                            num_lods = struct.unpack("<I", f.read(4))[0]
                            _lod_scales = struct.unpack(
                                f"<{num_lods}f", f.read(4 * num_lods)
                            )
                            all_vertex_offsets = struct.unpack(
                                f"<{3 * num_lods}f", f.read(12 * num_lods)
                            )
                            vertex_offsets = list(all_vertex_offsets[0:3])
                    except Exception as e:
                        logger.warning(
                            f"Failed to read .index file, using defaults: {e}"
                        )

                vertices = np.array(mesh_obj.points, dtype=np.float32)

                quantization_max = (2**vertex_quantization_bits) - 1
                vertices /= quantization_max
                lod_scale = 2**lod
                vertices *= np.array(chunk_shape, dtype=np.float32) * lod_scale
                vertices += np.array(vertex_offsets, dtype=np.float32)
                vertices += np.array(grid_origin, dtype=np.float32)

                temp_ply_fd, temp_ply_path = tempfile.mkstemp(suffix=".ply", text=True)

                num_vertices = len(vertices)
                num_faces = len(mesh_obj.faces)

                with os.fdopen(temp_ply_fd, "w") as temp_ply:
                    temp_ply.write("ply\n")
                    temp_ply.write("format ascii 1.0\n")
                    temp_ply.write(f"element vertex {num_vertices}\n")
                    temp_ply.write("property float x\n")
                    temp_ply.write("property float y\n")
                    temp_ply.write("property float z\n")
                    temp_ply.write(f"element face {num_faces}\n")
                    temp_ply.write("property list uchar int vertex_indices\n")
                    temp_ply.write("end_header\n")

                    for vertex in vertices:
                        temp_ply.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
                    for face in mesh_obj.faces:
                        temp_ply.write(f"3 {face[0]} {face[1]} {face[2]}\n")

                actual_input_file = temp_ply_path
                neuroglancer_format = False
                logger.info(
                    f"Decompressed mesh: {num_vertices} vertices, {num_faces} faces"
                )

        max_iter = 5
        try:
            for loop_subdiv in range(base_loop_subdivision_iterations, max_iter + 1):
                cmd = [str(exe), actual_input_file, output_file, str(loop_subdiv)]
                if neuroglancer_format:
                    cmd.append("--neuroglancer_format")
                try:
                    subprocess.run(cmd, check=True, timeout=timeout_seconds)
                    return
                except subprocess.TimeoutExpired as e:
                    if loop_subdiv == max_iter:
                        raise Exception(
                            f"Skeletonization timed out after {timeout_seconds}s "
                            f"with {max_iter} loop_subdivisions"
                        ) from e
                    logger.info(
                        f"Timeout at iter={loop_subdiv}; retrying with iter={loop_subdiv+1}"
                    )
                except subprocess.CalledProcessError as e:
                    raise Exception(
                        f"Error skeletonizing {actual_input_file}; "
                        f"exit status {e.returncode}"
                    ) from e
                except FileNotFoundError as e:
                    raise Exception(
                        f"Could not find or execute skeletonizer at {exe}"
                    ) from e
        finally:
            if temp_ply_path is not None:
                try:
                    os.unlink(temp_ply_path)
                except Exception:
                    pass

    @staticmethod
    def read_skeleton_from_custom_file(filename):
        """Read a skeleton from the CGAL skeletonizer's custom text format.

        The file uses line prefixes: ``v`` for vertices (x y z radius),
        ``e`` for edges, and ``p`` for polylines.

        Parameters
        ----------
        filename : str
            Path to the skeleton text file.

        Returns
        -------
        CustomSkeleton
            Parsed skeleton with vertices, edges, radii, and polylines.
        """
        vertices = []
        edges = []
        radii = []
        polylines = []
        with open(filename, "r") as file:
            for line in file:
                data = line.strip().split()
                if data[0] == "v":
                    vertices.append((float(data[1]), float(data[2]), float(data[3])))
                    radii.append(float(data[4]))
                elif data[0] == "e":
                    edges.append((int(data[1]), int(data[2])))
                elif data[0] == "p":
                    polyline = np.array([])
                    for v in data[1:]:
                        polyline = np.append(polyline, vertices[int(v)])
                    polylines.append(polyline.reshape(-1, 3))
        return CustomSkeleton(vertices, edges, radii, polylines)

    def _get_skeleton_from_mesh(self, mesh):
        input_file = f"{self.input_directory}/{mesh}"
        mesh_id = mesh.split(".")[0]
        output_file = f"{self.output_directory}/cgal/{mesh_id}.txt"
        Skeletonize.cgal_skeletonize_mesh(
            input_file,
            output_file,
            self.base_loop_subdivision_iterations,
            self.neuroglancer_format,
        )

    def process_custom_skeleton_df(self, df):
        """Process a partition of skeleton IDs and return a metrics DataFrame.

        Intended to be used as a ``map_partitions`` callable with dask.
        Each row is read, pruned, simplified, and written to disk; the
        corresponding morphometric metrics are collected.

        Parameters
        ----------
        df : pandas.DataFrame
            Partition with at least an ``id`` column containing skeleton
            file names.

        Returns
        -------
        pandas.DataFrame
            Concatenated metrics for every skeleton in *df*.
        """
        results_df = []
        for row in df.itertuples():
            try:
                metrics = Skeletonize.process_custom_skeleton(
                    skeleton_path=f"{self.output_directory}/cgal/{row.id}",
                    output_directory=self.output_directory,
                    min_branch_length_nm=self.min_branch_length_nm,
                    simplification_tolerance_nm=self.simplification_tolerance_nm,
                )
            except Exception as e:
                raise Exception(
                    f"Error processing skeleton "
                    f"{self.output_directory}/cgal/{row.id}: {e}"
                )
            result_df = pd.DataFrame(metrics, index=[0])
            results_df.append(result_df)

        return pd.concat(results_df, ignore_index=True)

    @staticmethod
    def process_custom_skeleton(
        skeleton_path,
        output_directory,
        min_branch_length_nm=100,
        simplification_tolerance_nm=50,
    ):
        """Read, prune, simplify, and export a single skeleton.

        Writes both the full and simplified neuroglancer skeletons under
        ``<output_directory>/skeleton/{full,simplified}/<id>``.

        Parameters
        ----------
        skeleton_path : str
            Path to the CGAL skeleton text file.
        output_directory : str
            Root output directory.
        min_branch_length_nm : float, optional
            Minimum branch length for pruning (nanometres).  Default is
            ``100``.
        simplification_tolerance_nm : float, optional
            RDP tolerance for simplification (nanometres).  Default is
            ``50``.

        Returns
        -------
        dict
            Morphometric metrics keyed by ``"id"``,
            ``"lsp (nm)"``, ``"radius mean (nm)"``,
            ``"radius std (nm)"``, and ``"num branches"``.
        """
        skeleton_id = os.path.basename(skeleton_path).split(".")[0]
        custom_skeleton = Skeletonize.read_skeleton_from_custom_file(skeleton_path)

        custom_skeleton_pruned = custom_skeleton.prune(min_branch_length_nm)

        metrics = {"id": skeleton_id}
        metrics["lsp (nm)"] = Skeletonize.get_longest_shortest_path_distance(
            custom_skeleton_pruned
        )
        metrics["radius mean (nm)"] = np.mean(custom_skeleton_pruned.radii)
        metrics["radius std (nm)"] = np.std(custom_skeleton_pruned.radii)
        metrics["num branches"] = len(custom_skeleton_pruned.polylines)

        custom_skeleton_pruned_simplified = custom_skeleton_pruned.simplify(
            simplification_tolerance_nm
        )

        custom_skeleton.write_neuroglancer_skeleton(
            f"{output_directory}/skeleton/full/{skeleton_id}"
        )
        custom_skeleton_pruned_simplified.write_neuroglancer_skeleton(
            f"{output_directory}/skeleton/simplified/{skeleton_id}"
        )
        return metrics

    def get_skeletons_from_meshes(self):
        """Run CGAL skeletonization on all meshes in the input directory.

        Meshes are processed in parallel using dask.  Results are written
        to ``<output_directory>/cgal/``.
        """
        os.makedirs(f"{self.output_directory}/cgal", exist_ok=True)
        all_files = os.listdir(self.input_directory)

        if self.neuroglancer_format:
            meshes = []
            for f in all_files:
                if f.endswith(".index"):
                    base_name = f.rsplit(".index", 1)[0]
                    meshes.append(base_name)
            logger.info(
                f"Found {len(meshes)} neuroglancer mesh files "
                f"(filtered from {len(all_files)} total files)"
            )
        else:
            meshes = all_files

        b = db.from_sequence(
            meshes,
            npartitions=dask_util.guesstimate_npartitions(meshes, self.num_workers),
        ).map(self._get_skeleton_from_mesh)
        with dask_util.start_dask(self.num_workers, "create skeletons", logger):
            with Timing_Messager("Creating skeletons", logger):
                b.compute()

    def process_custom_skeletons(self):
        """Prune, simplify, and export all CGAL skeletons in parallel.

        Reads skeleton text files from ``<output_directory>/cgal/``,
        computes metrics, writes neuroglancer skeleton files, and saves a
        sorted CSV of morphometric measurements to
        ``<output_directory>/metrics/skeleton_metrics.csv``.
        """
        self.cgal_output_directory = f"{self.output_directory}/cgal/"
        skeleton_filenames = os.listdir(self.cgal_output_directory)
        metrics = ["lsp (nm)", "radius mean (nm)", "radius std (nm)", "num branches"]
        df = pd.DataFrame({"id": skeleton_filenames})
        for metric in metrics:
            df[metric] = -1.0

        ddf = dd.from_pandas(
            df,
            npartitions=dask_util.guesstimate_npartitions(len(df), self.num_workers),
        )

        meta = pd.DataFrame(columns=df.columns)
        ddf_out = ddf.map_partitions(self.process_custom_skeleton_df, meta=meta)
        with dask_util.start_dask(self.num_workers, "process skeletons", logger):
            with Timing_Messager("Processing skeletons", logger):
                results = ddf_out.compute()

        output_directory = f"{self.output_directory}/metrics"
        os.makedirs(output_directory, exist_ok=True)

        results.sort_values("lsp (nm)", ascending=False, inplace=True)
        results.to_csv(f"{output_directory}/skeleton_metrics.csv", index=False)

        skeleton_ids = [
            skeleton_filename.split(".txt")[0]
            for skeleton_filename in skeleton_filenames
        ]
        self._write_skeleton_metadata(
            f"{self.output_directory}/skeleton/full", skeleton_ids
        )
        self._write_skeleton_metadata(
            f"{self.output_directory}/skeleton/simplified", skeleton_ids
        )

    def _write_skeleton_metadata(self, output_directory, skeleton_ids):
        os.makedirs(output_directory, exist_ok=True)

        metadata = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "segment_properties": "segment_properties",
        }

        with open(os.path.join(output_directory, "info"), "w") as f:
            f.write(json.dumps(metadata))

        os.makedirs(f"{output_directory}/segment_properties", exist_ok=True)
        segment_properties = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": [str(skeleton_id) for skeleton_id in skeleton_ids],
                "properties": [
                    {
                        "id": "label",
                        "type": "label",
                        "values": [""] * len(skeleton_ids),
                    }
                ],
            },
        }
        with open(f"{output_directory}/segment_properties/info", "w") as f:
            f.write(json.dumps(segment_properties))

    @staticmethod
    def get_longest_shortest_path_distance(skeleton):
        """Compute the longest shortest-path distance in a skeleton.

        Builds a weighted graph from the skeleton and finds the pair of
        nodes whose shortest-path distance is maximal (i.e., the graph
        diameter in Euclidean-weighted terms).

        Parameters
        ----------
        skeleton : CustomSkeleton
            Skeleton whose vertices and edges define the graph.

        Returns
        -------
        float
            Maximum pairwise shortest-path distance in nanometres.
        """
        g = nx.Graph()
        g.add_nodes_from(range(len(skeleton.vertices)))
        g.add_edges_from(skeleton.edges)
        for edge in skeleton.edges:
            g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                np.array(skeleton.vertices[edge[0]])
                - np.array(skeleton.vertices[edge[1]])
            )
        node_distances = dict(nx.all_pairs_dijkstra_path_length(g, weight="weight"))
        max_distance = max(
            max(distance_dict.values()) for distance_dict in node_distances.values()
        )
        return max_distance

    def get_skeletons(self):
        """Run the full skeletonization pipeline.

        Creates the output directory, generates CGAL skeletons from all
        input meshes, then prunes, simplifies, exports, and computes
        metrics for each skeleton.
        """
        os.makedirs(self.output_directory, exist_ok=True)
        self.get_skeletons_from_meshes()
        self.process_custom_skeletons()
