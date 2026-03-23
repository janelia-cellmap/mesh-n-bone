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


def generate_neuroglancer_multires_mesh(
    id, num_subtask_workers, output_path, lods, original_ext, lod_0_box_size=None,
):
    """Create a complete multiresolution mesh for a single object."""
    with ExitStack() as stack:
        if num_subtask_workers > 1:
            client = stack.enter_context(worker_client())

        os.makedirs(f"{output_path}/multires", exist_ok=True)
        os.system(
            f"rm -rf {output_path}/multires/{id} {output_path}/multires/{id}.index"
        )

        grid_origin = np.ones(3) * np.inf
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
            if vertices is not None:
                grid_origin = np.minimum(
                    grid_origin, np.floor(vertices.min(axis=0) - 1)
                )

            if lod_0_box_size is None and current_lod == 0:
                distances_per_axis = np.ceil(
                    vertices.max(axis=0) - vertices.min(axis=0)
                )
                # Target ~25k faces per LOD 0 fragment (~30 KB Draco-
                # compressed at 10-bit quantization).  This balances
                # spatial selectivity against HTTP per-request overhead.
                heuristic_num_chunks = np.ceil(num_faces / 25_000)
                if heuristic_num_chunks == 1:
                    lod_0_box_size = distances_per_axis + 1
                else:
                    lod_0_box_size = (
                        np.ceil(
                            distances_per_axis
                            / np.ceil(heuristic_num_chunks ** (1 / 2))
                        )
                        + 1
                    )

            previous_num_faces = num_faces
        else:
            # Loop completed without break — all LODs are valid
            idx += 1

        lods = lods[:idx]

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


def generate_all_neuroglancer_multires_meshes(
    output_path, num_workers, ids, lods, original_ext, file_sizes, lod_0_box_size=None,
):
    """Generate all neuroglancer multiresolution meshes for all ids."""

    def get_number_of_subtask_workers(file_sizes, num_workers):
        total_file_size = np.sum(file_sizes)
        num_workers_per_byte = num_workers / total_file_size
        num_subtask_workers = np.ceil(file_sizes * num_workers_per_byte).astype(int)
        return num_subtask_workers

    num_subtask_workers = get_number_of_subtask_workers(file_sizes, num_workers)
    variable_args_list = []
    fixed_args_list = [output_path, lods, original_ext, lod_0_box_size]
    for idx, id in enumerate(ids):
        variable_args_list.append((id, num_subtask_workers[idx]))
    dask_util.compute_bag(
        generate_neuroglancer_multires_mesh,
        f"{output_path}/variable_args_to_multires.npy",
        variable_args_list,
        fixed_args_list,
        num_workers,
    )


def run_multires(config_path, num_workers):
    """Main entry point for the multires pipeline."""
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

    segment_properties_csv = optional_properties_settings["segment_properties_csv"]
    segment_properties_columns = optional_properties_settings["segment_properties_columns"]
    segment_properties_id_column = optional_properties_settings["segment_properties_id_column"]

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
                    )

            with Timing_Messager("Writing info and segment properties files", logger):
                multires_output_path = f"{output_path}/multires"
                neuroglancer.write_segment_properties_file(
                    multires_output_path,
                    csv_path=segment_properties_csv,
                    csv_columns=segment_properties_columns,
                    csv_id_column=segment_properties_id_column,
                )
                neuroglancer.write_info_file(multires_output_path)

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
