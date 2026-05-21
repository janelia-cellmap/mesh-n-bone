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
    """Configure a writable local-directory for Dask worker spill and temp files.

    Tries ``/scratch/<user>`` then ``/tmp/<user>``, creating the directory if
    needed. Also sets ``tempfile.tempdir`` and the ``TMPDIR`` environment
    variable. Does nothing if the Dask config already has a local-directory
    for the given cluster type.

    Parameters
    ----------
    cluster_type : str
        Dask-jobqueue cluster type key (e.g. ``"lsf"``, ``"slurm"``,
        ``"local"``).

    Raises
    ------
    RuntimeError
        If no writable directory can be created.
    """
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
    """Context manager that starts a Dask cluster and yields a ``Client``.

    When ``num_workers == 1`` and no *config* is provided, no cluster is
    created -- work runs in the calling process with the synchronous
    scheduler. Otherwise a cluster is created from ``dask-config.yaml``
    (or the supplied *config* dict) and shut down on exit.

    Parameters
    ----------
    num_workers : int
        Number of workers to request.  ``1`` triggers local/synchronous
        mode when *config* is ``None``.
    msg : str
        Human-readable label used in log messages and the dashboard link.
    logger : logging.Logger
        Logger instance for status messages.
    config : dict or None
        Parsed Dask configuration dictionary.  If ``None``, the file
        ``dask-config.yaml`` in the current directory is loaded.

    Yields
    ------
    client : dask.distributed.Client or None
        A connected Dask client, or nothing (bare ``yield``) in
        synchronous mode.
    """
    if not config:
        if num_workers == 1:
            with dask.config.set(scheduler="synchronous"):
                yield
                return

        with open("dask-config.yaml") as f:
            config = yaml.load(f, Loader=SafeLoader)

    job_script_prologue = [
        "export NUMEXPR_MAX_THREADS=1",
        "export NUMEXPR_NUM_THREADS=1",
        "export MKL_NUM_THREADS=1",
        "export NUM_MKL_THREADS=1",
        "export OPENBLAS_NUM_THREADS=1",
        "export OPENMP_NUM_THREADS=1",
        "export OMP_NUM_THREADS=1",
    ]

    cluster_type = next(iter(config["jobqueue"]))
    dask.config.update(dask.config.config, config)

    set_local_directory(cluster_type)
    if cluster_type == "local":
        from dask.distributed import LocalCluster
        import socket

        hostname = socket.gethostname()
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    else:
        if cluster_type == "lsf":
            from dask_jobqueue import LSFCluster

            cluster = LSFCluster(
                job_script_prologue=job_script_prologue,
                scheduler_options={
                    "service_kwargs": {
                        "dashboard": {
                            "session_token_expiration": 60 * 60 * 24,
                        }
                    },
                },
            )
        elif cluster_type == "slurm":
            from dask_jobqueue import SLURMCluster

            cluster = SLURMCluster(
                job_script_prologue=job_script_prologue,
            )
        elif cluster_type == "sge":
            from dask_jobqueue import SGECluster

            cluster = SGECluster()
        cluster.scale(num_workers)
    try:
        with Timing_Messager(
            f"Starting {cluster_type} dask cluster for {msg} with {num_workers} workers",
            logger,
        ):
            client = Client(cluster)
            try:
                client.wait_for_workers(num_workers, timeout=300)
            except TimeoutError:
                pass

        dashboard_link = client.cluster.dashboard_link
        if cluster_type == "local":
            dashboard_link = dashboard_link.replace("127.0.0.1", hostname)
        print_with_datetime(
            f"Check {dashboard_link} for {msg} status.", logger
        )
        yield client
    finally:
        client.shutdown()
        client.close()


def setup_execution_directory(config_path, logger):
    """Create a timestamped copy of a configuration directory for execution.

    The new directory name is ``<config_path>-<YYYYmmdd.HHMMSS>``. The
    ``run-config.yaml`` and ``dask-config.yaml`` inside are made
    read-only to prevent accidental modification during a run.

    Parameters
    ----------
    config_path : str
        Path to the source configuration directory.
    logger : logging.Logger
        Logger instance for status messages.

    Returns
    -------
    str
        Absolute path to the newly created execution directory.
    """
    config_path = config_path[:-1] if config_path[-1] == "/" else config_path
    timestamp = f"{datetime.now():%Y%m%d.%H%M%S}"
    execution_dir = f"{config_path}-{timestamp}"
    execution_dir = os.path.abspath(execution_dir)
    shutil.copytree(config_path, execution_dir, symlinks=True)
    for name in ("run-config.yaml", "dask-config.yaml"):
        path = f"{execution_dir}/{name}"
        if os.path.exists(path):
            os.chmod(path, 0o444)
    print_with_datetime(f"Setup working directory as {execution_dir}.", logger)

    return execution_dir


def _load_dask_config():
    """Read ``dask-config.yaml`` from the cwd into a plain dict."""
    with open("dask-config.yaml") as f:
        return yaml.load(f, Loader=SafeLoader)


def run_with_oom_retry(
    work_fn,
    num_workers,
    phase_name,
    logger,
    max_retries=3,
    retry_on_oom=True,
):
    """Run a dask phase with optional retry on worker out-of-memory crashes.

    The phase's work happens inside *work_fn*, which is called as
    ``work_fn(num_workers, config)``. *work_fn* is responsible for
    starting its own dask cluster via
    :func:`start_dask(num_workers, ..., config=config)` and doing the
    compute inside that context.

    On a ``distributed.scheduler.KilledWorker`` (the typical dask
    symptom of a worker SIGKILL'd by the OS OOM-killer or LSF mem
    limit), we halve ``processes`` per slot in the in-memory dask
    config and halve *num_workers* to match — doubling the memory
    available to each worker while keeping the LSF slot count
    (and CPU budget) constant. Then we retry the phase, up to
    *max_retries* times.

    Parameters
    ----------
    work_fn : callable
        ``work_fn(num_workers: int, config: dict | None) -> Any``.
    num_workers : int
        Initial worker count.
    phase_name : str
        Label used in log lines (e.g. ``"assemble meshes"``).
    logger : logging.Logger
        Where retry messages are logged.
    max_retries : int
        Maximum number of OOM-triggered retries.  ``0`` disables retry.
    retry_on_oom : bool
        Set to ``False`` to skip the retry wrapper entirely (acts as a
        passthrough). Useful for synchronous (``num_workers==1``) runs.
    """
    if not retry_on_oom or max_retries < 1 or num_workers <= 1:
        return work_fn(num_workers, None)

    try:
        from distributed.scheduler import KilledWorker
    except ImportError:
        return work_fn(num_workers, None)

    config = _load_dask_config()
    cluster_type = next(iter(config.get("jobqueue", {})), None)
    if cluster_type not in ("lsf", "slurm", "sge"):
        # Local/in-process clusters don't have a processes/slot lever;
        # skip retry wrapping there.
        return work_fn(num_workers, config)

    workers = num_workers
    for attempt in range(max_retries + 1):
        try:
            return work_fn(workers, config)
        except KilledWorker as e:
            processes = int(config["jobqueue"][cluster_type].get("processes", 1) or 1)
            if attempt >= max_retries:
                logger.error(
                    "Phase '%s' hit worker OOM and exhausted %d retries "
                    "(processes/slot=%d). Increase per-slot memory in "
                    "dask-config.yaml or skip oversized segments with "
                    "max_num_blocks.",
                    phase_name, max_retries, processes,
                )
                raise
            if processes <= 1:
                logger.error(
                    "Phase '%s' hit worker OOM with processes/slot already at "
                    "1 — cannot halve further. Increase memory per slot "
                    "directly, or skip oversized segments via max_num_blocks.",
                    phase_name,
                )
                raise
            new_processes = max(1, processes // 2)
            new_workers = max(1, workers // 2)
            last_worker = getattr(e, "last_worker", None)
            logger.warning(
                "Phase '%s' worker OOM (retry %d/%d). Halving "
                "processes/slot %d→%d and workers %d→%d to give each worker "
                "more memory; LSF slot count and CPU budget unchanged. "
                "To find the segment that crashed: grep "
                "job-logs/LSFCluster-*.err for the last 'mesh_id=...' line "
                "from the killed worker at %s.",
                phase_name, attempt + 1, max_retries,
                processes, new_processes, workers, new_workers,
                last_worker or "(unknown address)",
            )
            config["jobqueue"][cluster_type]["processes"] = new_processes
            workers = new_workers


def effective_num_workers(requested, num_tasks, logger=None, phase_name=None):
    """Cap requested workers to the actual task count for a phase.

    Spinning up more workers than there are tasks just wastes cluster slots
    (especially on LSF/SLURM where each worker is its own job). Returns
    ``max(1, min(requested, num_tasks))``. When capping happens, a single
    info line is logged so the user knows why fewer workers were started.

    Parameters
    ----------
    requested : int
        Worker count the user asked for (typically ``self.num_workers``).
    num_tasks : int
        Number of independent tasks the phase will run.
    logger : logging.Logger, optional
        Where to log the cap message. No log if omitted.
    phase_name : str, optional
        Short phase name for the log line (e.g. ``"assemble meshes"``).
    """
    effective = max(1, min(int(requested), max(1, int(num_tasks))))
    if effective < requested and logger is not None:
        label = f" for {phase_name}" if phase_name else ""
        logger.info(
            "Capping workers%s to %d (matches task count); requested %d.",
            label, effective, requested,
        )
    return effective


def guesstimate_npartitions(elements, num_workers, scaling=10):
    """Estimate a reasonable number of Dask-bag partitions.

    Aims for roughly ``num_workers * scaling`` partitions, then rounds
    to produce evenly sized partitions.

    Parameters
    ----------
    elements : int or Sized
        Total number of elements (or a collection whose length is used).
    num_workers : int
        Number of Dask workers available.
    scaling : int, optional
        Multiplier applied to *num_workers* to set the target partition
        count.  Default is ``10``.

    Returns
    -------
    int
        Number of partitions.
    """
    if not isinstance(elements, int):
        elements = len(elements)
    approximate_npartitions = min(elements, num_workers * scaling)
    elements_per_worker = elements // approximate_npartitions
    actual_partitions = elements // elements_per_worker
    return actual_partitions


def compute_bag(fn, memmap_file_path, variable_args_list, fixed_args_list, num_workers):
    """Execute a function over a list of argument tuples using a Dask bag.

    The *variable_args_list* is saved to disk as a NumPy ``.npy`` file and
    memory-mapped by each worker so that large arrays are not serialised
    through the scheduler. When ``num_workers == 1`` the bag is computed
    synchronously.

    Parameters
    ----------
    fn : callable
        Function to call for each element. Called as
        ``fn(*variable_args[i], *fixed_args_list)``.
    memmap_file_path : str
        Path where the temporary ``.npy`` file is written.  Deleted after
        computation.
    variable_args_list : array_like
        Array of argument tuples, one per element.
    fixed_args_list : list
        Additional arguments appended to every call.
    num_workers : int
        Number of Dask workers.  ``1`` triggers synchronous execution.

    Raises
    ------
    RuntimeError
        If any partition raises an exception on a distributed cluster.
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

    @dataclass
    class DaskBlock:
        """A spatial block for distributed processing.

        Parameters
        ----------
        index : int
            Sequential block index.
        roi : funlib.geometry.Roi
            Region of interest covered by this block.
        """

        index: int
        roi: Roi

    def create_blocks(
        roi,
        ds,
        read_write_block_shape_pixels=None,
        padding=None,
    ):
        """Tile a region of interest into ``DaskBlock`` instances.

        Parameters
        ----------
        roi : funlib.geometry.Roi
            Region of interest to partition.
        ds : object
            Dataset with ``chunk_shape`` and ``voxel_size`` attributes
            (e.g. ``CellMapArray``).
        read_write_block_shape_pixels : numpy.ndarray or None
            Override block size in voxel units.  When ``None``, the
            dataset's chunk shape is used.
        padding : funlib.geometry.Coordinate or None
            Symmetric padding to grow each block ROI by.

        Returns
        -------
        list[DaskBlock]
            Blocks covering the entire *roi*.
        """
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
