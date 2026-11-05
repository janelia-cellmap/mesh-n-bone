"""Runs that cannot stall silently: progress lines and lost-worker checks
on every dask phase, walltime kills told apart from out-of-memory, retries
that skip finished blocks, and enough blocks to keep every worker busy."""

import logging
import os
import sys
import time
import types
from types import SimpleNamespace

import dask
import dask.bag as db
import numpy as np
import pytest
import trimesh
import zarr

from mesh_n_bone.meshify import meshify as meshify_module
from mesh_n_bone.meshify.meshify import Meshify
from mesh_n_bone.util import dask_util

# Shortened from real LSF job summaries of worker jobs.
RUNLIMIT_SUMMARY = (
    "Sender: LSF System <lsfadmin@h06u25>\n"
    "Subject: Job 155199487: <mesh-n-bone> in cluster <Janelia> Exited\n\n"
    "TERM_RUNLIMIT: job killed after reaching LSF run time limit.\n"
    "Exited with exit code 140.\n"
)
MEMLIMIT_SUMMARY = (
    "Subject: Job 1: <mesh-n-bone> in cluster <Janelia> Exited\n\n"
    "TERM_MEMLIMIT: job killed after reaching LSF memory usage limit.\n"
    "Exited with exit code 130.\n"
)
OWNER_SUMMARY = (
    "Subject: Job 2: <mesh-n-bone> in cluster <Janelia> Exited\n\n"
    "TERM_OWNER: job killed by owner.\n"
    "Exited with signal termination: 14.\n"
)
COMPLETED_SUMMARY = "Subject: Job 3: <mesh-n-bone> in cluster <Janelia> Done\n\nSuccessfully completed.\n"
EXITED_SUMMARY = "Subject: Job 4: <mesh-n-bone> in cluster <Janelia> Exited\n\nExited with exit code 1.\n"


def _summaries(directory, **texts):
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (directory / f"{name}.out").write_text(text)
    return directory


class TestLsfExitReasons:
    def test_counts_how_the_worker_jobs_ended(self, tmp_path):
        logs = _summaries(
            tmp_path, a=RUNLIMIT_SUMMARY, b=RUNLIMIT_SUMMARY, c=OWNER_SUMMARY,
            d=COMPLETED_SUMMARY, e=EXITED_SUMMARY,
        )
        (logs / "a.err").write_text("TERM_MEMLIMIT in a worker's stderr is not a summary")
        reasons = dask_util.lsf_exit_reasons(str(logs), since=time.time() - 60)
        assert reasons == {
            "TERM_RUNLIMIT": 2, "TERM_OWNER": 1, "completed": 1, "exit code 1": 1,
        }
        text = dask_util.describe_exit_reasons(reasons)
        assert text.startswith("2 TERM_RUNLIMIT (reached the worker walltime)")

    def test_summaries_from_before_the_phase_do_not_count(self, tmp_path):
        logs = _summaries(tmp_path, old=RUNLIMIT_SUMMARY, new=OWNER_SUMMARY)
        an_hour_ago = time.time() - 3600
        os.utime(logs / "old.out", (an_hour_ago, an_hour_ago))
        assert dask_util.lsf_exit_reasons(str(logs), since=time.time() - 60) == {"TERM_OWNER": 1}

    def test_no_log_directory_means_nothing_known(self, tmp_path):
        assert dask_util.lsf_exit_reasons(None, 0) == {}
        assert dask_util.lsf_exit_reasons(str(tmp_path / "missing"), 0) == {}


def _slow_double(x):
    time.sleep(0.05)
    return 2 * x


def _die(x):
    os._exit(1)


def _fail_on_three(x):
    if x == 3:
        raise ValueError("block 3 is broken")
    return x


@pytest.fixture
def local_client():
    from dask.distributed import Client, LocalCluster

    cluster = LocalCluster(
        n_workers=1, processes=False, threads_per_worker=1, dashboard_address=None,
    )
    client = Client(cluster)
    try:
        yield client
    finally:
        client.close()
        cluster.close()


class TestComputeWatchedOnACluster:
    def test_returns_what_compute_returns_and_logs_progress(self, local_client, caplog):
        bag = db.from_sequence(range(12), npartitions=12).map(_slow_double)
        with caplog.at_level(logging.INFO, logger="test-progress"):
            result = dask_util.compute_watched(
                bag, "doubling", logging.getLogger("test-progress"),
                poll_seconds=0.02, report_fraction=0.25,
            )
        assert result == [2 * x for x in range(12)]
        progress = [r.message for r in caplog.records if "tasks done" in r.message]
        assert progress, "no progress line was logged"
        assert all(m.startswith("doubling: ") and "/12 tasks done" in m for m in progress)
        assert "1 workers connected" in progress[0]

    def test_a_task_error_is_raised_as_compute_raises_it(self, local_client):
        bag = db.from_sequence(range(6), npartitions=6).map(_fail_on_three)
        with pytest.raises(ValueError, match="block 3 is broken"):
            dask_util.compute_watched(bag, "failing", poll_seconds=0.02)

    def test_a_dead_worker_surfaces_as_killed_worker(self):
        """The retry wrapper catches KilledWorker; a worker process that
        dies must still reach it through compute_watched."""
        from dask.distributed import Client, LocalCluster
        from distributed.scheduler import KilledWorker

        with dask.config.set({"distributed.scheduler.allowed-failures": 0}):
            cluster = LocalCluster(
                n_workers=1, processes=True, threads_per_worker=1,
                dashboard_address=None,
            )
            client = Client(cluster)
            try:
                bag = db.from_sequence(range(2), npartitions=2).map(_die)
                with pytest.raises(KilledWorker):
                    dask_util.compute_watched(bag, "dying", poll_seconds=0.05)
            finally:
                client.close()
                cluster.close()

    def test_without_a_client_it_is_plain_compute(self):
        bag = db.from_sequence(range(5), npartitions=2).map(lambda x: x + 1)
        with dask.config.set(scheduler="synchronous"):
            assert dask_util.compute_watched(bag, "local") == [1, 2, 3, 4, 5]


class _FakeFuture:
    def __init__(self, value):
        self.status = "pending"
        self.value = value


class _FakeClient:
    """Futures that finish when the test says so, and a worker count that
    follows a script: enough to drive compute_watched's loop without LSF."""

    def __init__(self, n, workers):
        self.futures = [_FakeFuture([i]) for i in range(n)]
        self.workers = list(workers)
        self.cancelled = False
        self.polls = 0

    def compute(self, parts):
        assert len(parts) == len(self.futures)
        return self.futures

    def nthreads(self):
        n = self.workers.pop(0) if len(self.workers) > 1 else self.workers[0]
        return {f"tcp://worker-{i}": 1 for i in range(n)}

    def cancel(self, futures):
        self.cancelled = True

    def gather(self, futures):
        return [f.value for f in futures]


@pytest.fixture
def fake_cluster(monkeypatch):
    """Install a _FakeClient; ``finish_after`` polls mark every future done."""
    state = {"client": None, "finish_after": None}

    def install(n, workers, finish_after=None):
        state["client"] = _FakeClient(n, workers)
        state["finish_after"] = finish_after
        return state["client"]

    def fake_wait(futures, timeout=None):
        client = state["client"]
        client.polls += 1
        if state["finish_after"] is not None and client.polls >= state["finish_after"]:
            for f in futures:
                f.status = "finished"
            return
        time.sleep(0.01)
        raise TimeoutError

    monkeypatch.setattr(dask_util, "_active_client", lambda: state["client"])
    monkeypatch.setattr(dask_util, "wait", fake_wait)
    yield install
    dask_util._ACTIVE_JOBQUEUE.clear()


class TestComputeWatchedWhenWorkersVanish:
    def test_fails_once_every_worker_has_been_gone_for_the_grace(self, fake_cluster, tmp_path, caplog):
        logs = _summaries(tmp_path / "job-logs", w0=RUNLIMIT_SUMMARY, w1=RUNLIMIT_SUMMARY)
        dask_util._remember_jobqueue("lsf", {"walltime": "01:00", "log-directory": str(logs)})
        client = fake_cluster(4, workers=[2, 2, 0])
        bag = db.from_sequence(range(4), npartitions=4)
        with caplog.at_level(logging.WARNING, logger="test-lost"):
            with pytest.raises(dask_util.WorkersLostError) as raised:
                dask_util.compute_watched(
                    bag, "chunked meshes", logging.getLogger("test-lost"),
                    poll_seconds=0.01, lost_grace_seconds=0.05,
                )
        message = str(raised.value)
        assert "All dask workers for chunked meshes have exited" in message
        assert "0 of 4 tasks done" in message
        assert "2 TERM_RUNLIMIT (reached the worker walltime)" in message
        assert "Their walltime is 01:00 (jobqueue.lsf.walltime in dask-config.yaml)" in message
        assert client.cancelled
        assert any("every worker has exited" in r.message for r in caplog.records)

    def test_a_worker_coming_back_resets_the_clock(self, fake_cluster):
        fake_cluster(3, workers=[1, 0, 0, 1, 1, 1], finish_after=8)
        bag = db.from_sequence(range(3), npartitions=3)
        assert dask_util.compute_watched(
            bag, "x", poll_seconds=0.01, lost_grace_seconds=0.2,
        ) == [0, 1, 2]

    def test_waiting_in_the_queue_is_not_an_error(self, fake_cluster, caplog):
        fake_cluster(3, workers=[0], finish_after=10)
        bag = db.from_sequence(range(3), npartitions=3)
        with caplog.at_level(logging.INFO, logger="test-queue"):
            result = dask_util.compute_watched(
                bag, "queued", logging.getLogger("test-queue"),
                poll_seconds=0.01, lost_grace_seconds=0.0, heartbeat_seconds=0.0,
            )
        assert result == [0, 1, 2]
        assert any("0/3 tasks done (0%), 0 workers connected" in r.message for r in caplog.records)


class _FakeKilledWorker(Exception):
    last_worker = "tcp://10.0.0.1:1234"


@pytest.fixture
def fake_killed_worker(monkeypatch):
    mod = types.ModuleType("distributed.scheduler")
    mod.KilledWorker = _FakeKilledWorker
    monkeypatch.setitem(sys.modules, "distributed.scheduler", mod)
    monkeypatch.setattr(dask_util, "_EXIT_SUMMARY_WAIT_SECONDS", 0.0)


def _lsf_config(log_directory):
    return {"jobqueue": {"lsf": {
        "ncpus": 8, "processes": 8, "cores": 8, "memory": "120GB",
        "walltime": "01:00", "log-directory": str(log_directory),
    }}}


class TestOomRetryTellsWalltimeFromMemory:
    def _killing(self, logs, summary, calls):
        def work_fn(workers, config):
            calls.append((workers, config["jobqueue"]["lsf"]["processes"]))
            if len(calls) == 1:
                if summary:
                    _summaries(logs, **{f"w{len(calls)}": summary})
                raise _FakeKilledWorker("killed")
            return "done"
        return work_fn

    def test_a_walltime_kill_is_reported_not_retried(self, fake_killed_worker, tmp_path, caplog):
        logs = tmp_path / "job-logs"
        logs.mkdir()
        calls = []
        with pytest.raises(dask_util.WorkersLostError, match=r"reached their walltime \(01:00\)"):
            dask_util.run_with_oom_retry(
                self._killing(logs, RUNLIMIT_SUMMARY, calls), 72,
                "generate chunked meshes", logging.getLogger("test"),
                config=_lsf_config(logs),
            )
        assert calls == [(72, 8)]

    def test_a_memory_kill_still_halves_the_workers(self, fake_killed_worker, tmp_path):
        logs = tmp_path / "job-logs"
        logs.mkdir()
        calls = []
        result = dask_util.run_with_oom_retry(
            self._killing(logs, MEMLIMIT_SUMMARY, calls), 72, "assembly",
            logging.getLogger("test"), config=_lsf_config(logs),
        )
        assert result == "done"
        assert calls == [(72, 8), (36, 4)]

    def test_an_unexplained_kill_is_treated_as_memory(self, fake_killed_worker, tmp_path):
        logs = tmp_path / "job-logs"
        logs.mkdir()
        calls = []
        dask_util.run_with_oom_retry(
            self._killing(logs, OWNER_SUMMARY, calls), 72, "assembly",
            logging.getLogger("test"), config=_lsf_config(logs),
        )
        assert calls == [(72, 8), (36, 4)]

    def test_an_earlier_phases_walltime_kill_does_not_count(self, fake_killed_worker, tmp_path):
        logs = _summaries(tmp_path / "job-logs", earlier=RUNLIMIT_SUMMARY)
        an_hour_ago = time.time() - 3600
        os.utime(logs / "earlier.out", (an_hour_ago, an_hour_ago))
        calls = []
        dask_util.run_with_oom_retry(
            self._killing(logs, None, calls), 72, "assembly",
            logging.getLogger("test"), config=_lsf_config(logs),
        )
        assert calls == [(72, 8), (36, 4)]


def _cube_volume(tmp_path):
    vol = np.zeros((32, 32, 32), dtype=np.uint32)
    vol[3:29, 3:29, 3:29] = 7  # crosses every one of the eight 16^3 blocks
    root = zarr.open_group(str(tmp_path / "vol.zarr"), mode="w")
    arr = root.create_array("labels/s0", data=vol, chunks=(16, 16, 16))
    arr.attrs["voxel_size"] = [8, 8, 8]
    arr.attrs["offset"] = [0, 0, 0]
    return str(tmp_path / "vol.zarr" / "labels" / "s0")


class _Crash(Exception):
    pass


class TestChunkedMeshRetrySkipsFinishedBlocks:
    def _mesh(self, path, out):
        Meshify(
            input_path=path, output_directory=out, num_workers=1,
            do_multires=False, do_analysis=False, check_mesh_validity=False,
            do_simplification=False, n_smoothing_iter=0,
            remove_smallest_components=False,
            read_write_block_shape_pixels=[16, 16, 16],
        ).get_meshes()
        return trimesh.load(os.path.join(out, "meshes", "7.ply"), process=False)

    def test_a_retry_meshes_only_the_blocks_that_did_not_finish(self, tmp_path, monkeypatch, caplog):
        path = _cube_volume(tmp_path)
        clean = self._mesh(path, str(tmp_path / "clean"))

        real_worker = meshify_module._get_chunked_mesh_worker
        real_retry = dask_util.run_with_oom_retry
        calls = []
        attempt = {"n": 0}

        def flaky_worker(block_index, tmpdirname, config):
            calls.append((attempt["n"], block_index))
            if attempt["n"] == 0 and len(calls) == 4:
                raise _Crash("worker died")
            return real_worker(block_index, tmpdirname, config)

        def retry_once(work_fn, workers, phase_name, logger, **kwargs):
            if phase_name != "generate chunked meshes":
                return real_retry(work_fn, workers, phase_name, logger, **kwargs)
            try:
                return work_fn(workers, None)
            except _Crash:
                attempt["n"] = 1
                return work_fn(workers, None)

        monkeypatch.setattr(meshify_module, "_get_chunked_mesh_worker", flaky_worker)
        monkeypatch.setattr(dask_util, "run_with_oom_retry", retry_once)
        with caplog.at_level(logging.INFO):
            retried = self._mesh(path, str(tmp_path / "retried"))

        finished_first = {b for a, b in calls[:3]}
        crashed = calls[3][1]
        second = {b for a, b in calls if a == 1}
        assert len(finished_first) == 3
        assert second == set(range(8)) - finished_first
        assert crashed in second
        assert any(
            "3 of 8 blocks finished in the earlier attempt; meshing the other 5" in r.message
            for r in caplog.records
        )
        assert len(retried.faces) == len(clean.faces)
        np.testing.assert_allclose(
            np.unique(np.round(retried.vertices, 3), axis=0),
            np.unique(np.round(clean.vertices, 3), axis=0),
        )

    def test_manifests_from_an_earlier_run_are_ignored(self, tmp_path):
        mdir = tmp_path / meshify_module._MANIFEST_SUBDIR
        mdir.mkdir()
        for name in ("block_0.npy", "block_1.npy", "block_2.npy.partial", "notes.txt"):
            (mdir / name).write_bytes(b"")
        an_hour_ago = time.time() - 3600
        os.utime(mdir / "block_1.npy", (an_hour_ago, an_hour_ago))
        assert meshify_module._blocks_finished_since(str(tmp_path), time.time() - 60) == {0}


class TestEnoughBlocksForEveryWorker:
    @staticmethod
    def _meshify(chunk, dtype, roi_shape, num_workers):
        m = object.__new__(Meshify)
        m.segmentation_array = SimpleNamespace(chunk_shape=tuple(chunk), dtype=np.dtype(dtype))
        m.roi = SimpleNamespace(shape=tuple(roi_shape))
        m.num_workers = num_workers
        return m

    def test_the_synapse_crop_gets_three_blocks_per_worker(self):
        # 15 GB workers -> a 1875 MB budget picks 768^3 blocks: 72 of them.
        one = self._meshify((128,) * 3, "uint32", (1466, 4125, 4125), 1)
        np.testing.assert_array_equal(one._default_block_shape_pixels(target_mb=1875), [768] * 3)
        many = self._meshify((128,) * 3, "uint32", (1466, 4125, 4125), 96)
        block = many._default_block_shape_pixels(target_mb=1875)
        np.testing.assert_array_equal(block, [384] * 3)
        count = np.prod(np.ceil(np.array((1466, 4125, 4125)) / block))
        assert count >= meshify_module.MIN_BLOCKS_PER_WORKER * 96

    def test_never_below_one_chunk(self):
        m = self._meshify((64,) * 3, "uint64", (64, 64, 64), 96)
        np.testing.assert_array_equal(m._default_block_shape_pixels(target_mb=500), [64] * 3)

    def test_a_big_volume_keeps_its_memory_sized_blocks(self):
        m = self._meshify((128,) * 3, "uint32", (8192, 8192, 8192), 96)
        np.testing.assert_array_equal(m._default_block_shape_pixels(target_mb=1875), [768] * 3)


class TestMultiresClearsOldFilesWithoutAShell:
    def test_no_process_per_segment(self, multires_mesh_dir, monkeypatch):
        from mesh_n_bone.multires.multires import generate_neuroglancer_multires_mesh

        multires = os.path.join(multires_mesh_dir, "multires")
        os.makedirs(multires, exist_ok=True)
        for stale in ("1", "1.index"):
            with open(os.path.join(multires, stale), "wb") as f:
                f.write(b"stale")

        def no_shell(command):
            raise AssertionError(f"spawned a shell: {command}")

        monkeypatch.setattr(os, "system", no_shell)
        generate_neuroglancer_multires_mesh(
            id=1, num_subtask_workers=1, output_path=multires_mesh_dir,
            lods=[0, 1], original_ext=".ply", lod_0_box_size=None,
        )
        with open(os.path.join(multires, "1.index"), "rb") as f:
            assert f.read() != b"stale"
