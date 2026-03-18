from contextlib import contextmanager, nullcontext
import os
import dask
from dask.distributed import Client, wait
import getpass
import tempfile
import shutil
from mesh_n_bone.util.logging import Timing_Messager, print_with_datetime
from datetime import datetime
import yaml
from yaml.loader import SafeLoader
import dask.bag as db
import numpy as np
import logging

logger = logging.getLogger(__name__)


def set_local_directory(cluster_type):
    """Sets local directory used for dask outputs."""
    local_dir = dask.config.get(f"jobqueue.{cluster_type}.local-directory", None)
    if local_dir:
        return

    user = getpass.getuser()
    local_dir = None
    for d in [f"/scratch/{user}", f"/tmp/{user}"]:
        try:
            os.makedirs(d, exist_ok=True)
        except OSError:
            continue
        else:
            local_dir = d
            dask.config.set({f"jobqueue.{cluster_type}.local-directory": local_dir})
            tempfile.tempdir = local_dir
            os.environ["TMPDIR"] = local_dir
            break

    if local_dir is None:
        raise RuntimeError(
            "Could not create a local-directory in any of the standard places."
        )


@contextmanager
def start_dask(num_workers, msg, logger, config=None):
    """Context manager used for starting/shutting down dask.

    When num_workers == 1, no cluster is created and no dask-config.yaml
    is required — work runs in the calling process via plain .compute().
    """
    if not config:
        if num_workers == 1:
            with dask.config.set(scheduler="synchronous"):
                yield
                return

        with open("dask-config.yaml") as f:
            config = yaml.load(f, Loader=SafeLoader)

    cluster_type = next(iter(config["jobqueue"]))
    dask.config.update(dask.config.config, config)

    set_local_directory(cluster_type)
    if cluster_type == "local":
        from dask.distributed import LocalCluster

        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    else:
        if cluster_type == "lsf":
            from dask_jobqueue import LSFCluster

            cluster = LSFCluster()
        elif cluster_type == "slurm":
            from dask_jobqueue import SLURMCluster

            cluster = SLURMCluster()
        elif cluster_type == "sge":
            from dask_jobqueue import SGECluster

            cluster = SGECluster()
        cluster.scale(num_workers)
    try:
        with Timing_Messager(f"Starting dask cluster for {msg}", logger):
            client = Client(cluster)
            try:
                client.wait_for_workers(num_workers, timeout=300)
            except TimeoutError:
                pass

        print_with_datetime(
            f"Check {client.cluster.dashboard_link} for {msg} status.", logger
        )
        yield client
    finally:
        client.shutdown()
        client.close()


def setup_execution_directory(config_path, logger):
    """Sets up the execution directory which is the config dir appended with
    the date and time."""
    config_path = config_path[:-1] if config_path[-1] == "/" else config_path
    timestamp = f"{datetime.now():%Y%m%d.%H%M%S}"
    execution_dir = f"{config_path}-{timestamp}"
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(config_path, execution_dir, symlinks=True)
    os.chmod(f"{execution_dir}/run-config.yaml", 0o444)
    print_with_datetime(f"Setup working directory as {execution_dir}.", logger)

    return execution_dir


def guesstimate_npartitions(elements, num_workers, scaling=10):
    if not isinstance(elements, int):
        elements = len(elements)
    approximate_npartitions = min(elements, num_workers * scaling)
    elements_per_worker = elements // approximate_npartitions
    actual_partitions = elements // elements_per_worker
    return actual_partitions


def compute_bag(fn, memmap_file_path, variable_args_list, fixed_args_list, num_workers):
    """Execute a function over variable args using dask bag with memory-mapped arrays.

    When num_workers == 1 (no distributed client), falls back to plain .compute().
    """
    np.save(memmap_file_path, variable_args_list)

    def partition_worker(indices, path, *fixed):
        arr = np.load(path, mmap_mode="r")
        for i in indices:
            fn(*arr[i], *fixed)
        return []

    bag = db.range(
        len(variable_args_list),
        npartitions=guesstimate_npartitions(variable_args_list, num_workers),
    )

    try:
        bag = bag.map_partitions(partition_worker, memmap_file_path, *fixed_args_list)
        if num_workers == 1:
            bag.compute()
        else:
            futures = bag.persist()

            [completed, _] = wait(futures)
            failed = [f for f in completed if f.exception() is not None]

            for completed_future in completed:
                completed_future.cancel()

            if failed:
                raise RuntimeError(f"Failed to compute {len(failed)} blocks: {failed}")
    except Exception as e:
        print("Compute raised an exception:", e)
        raise
    os.remove(memmap_file_path)


# -- Block-based processing utilities (from igneous-daskified) --

try:
    from dataclasses import dataclass
    from funlib.geometry import Roi
    from funlib.persistence import Array

    @dataclass
    class DaskBlock:
        index: int
        roi: Roi

    def create_blocks(
        roi: Roi,
        ds: Array,
        read_write_block_shape_pixels=None,
        padding=None,
    ):
        with Timing_Messager("Generating blocks", logger):
            block_size = read_write_block_shape_pixels
            if read_write_block_shape_pixels is None:
                block_size = ds.chunk_shape
            block_size *= ds.voxel_size
            num_expected_blocks = int(
                np.prod(
                    [np.ceil(roi.shape[i] / block_size[i]) for i in range(len(block_size))]
                )
            )
            block_rois = [None] * num_expected_blocks
            index = 0
            for dim_2 in range(roi.get_begin()[2], roi.get_end()[2], block_size[2]):
                for dim_1 in range(roi.get_begin()[1], roi.get_end()[1], block_size[1]):
                    for dim_0 in range(roi.get_begin()[0], roi.get_end()[0], block_size[0]):
                        block_roi = Roi((dim_0, dim_1, dim_2), block_size)
                        if padding:
                            block_roi = block_roi.grow(padding, padding)
                        block_rois[index] = DaskBlock(index, block_roi)
                        index += 1
            if index < len(block_rois):
                block_rois[index:] = []
        return block_rois

except ImportError:
    # funlib not available - block-based processing won't work
    # but multires pipeline doesn't need it
    pass
