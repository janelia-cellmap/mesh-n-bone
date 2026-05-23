"""Unit tests for the OOM-retry wrapper and worker-cap helper."""

import logging

import pytest

from mesh_n_bone.util import dask_util


class _FakeKilledWorker(Exception):
    """Stand-in for ``distributed.scheduler.KilledWorker`` in tests."""
    def __init__(self, msg="killed", last_worker="tcp://10.0.0.1:1234"):
        super().__init__(msg)
        self.last_worker = last_worker


@pytest.fixture
def fake_distributed(monkeypatch):
    """Make ``run_with_oom_retry`` catch our ``_FakeKilledWorker``."""
    import sys
    import types
    mod = types.ModuleType("distributed.scheduler")
    mod.KilledWorker = _FakeKilledWorker
    monkeypatch.setitem(sys.modules, "distributed.scheduler", mod)
    monkeypatch.setitem(
        sys.modules,
        "distributed",
        types.ModuleType("distributed"),
    )
    yield


@pytest.fixture
def fake_dask_config(monkeypatch):
    """Replace _load_dask_config with an in-memory LSF config."""
    cfg = {
        "jobqueue": {
            "lsf": {
                "ncpus": 12,
                "processes": 12,
                "cores": 12,
                "memory": "180GB",
            }
        }
    }
    monkeypatch.setattr(
        dask_util,
        "_load_dask_config",
        lambda: {**cfg, "jobqueue": {"lsf": dict(cfg["jobqueue"]["lsf"])}},
    )


class TestEffectiveNumWorkers:
    def test_no_cap_when_tasks_exceed_workers(self):
        assert dask_util.effective_num_workers(10, 100) == 10

    def test_caps_to_task_count(self):
        assert dask_util.effective_num_workers(100, 5) == 5

    def test_min_one(self):
        assert dask_util.effective_num_workers(10, 0) == 1

    def test_logs_when_capped(self, caplog):
        logger = logging.getLogger("test")
        with caplog.at_level(logging.INFO, logger="test"):
            dask_util.effective_num_workers(100, 5, logger, "phaseX")
        assert any("phaseX" in r.message for r in caplog.records)
        assert any("5" in r.message for r in caplog.records)


class TestGuesstimateNPartitions:
    def test_avoids_dask_range_huge_final_partition(self):
        elements = 217620
        workers = 576

        direct_partitions = min(elements, workers * 10)
        direct_size = elements // direct_partitions
        direct_last_size = elements - (direct_partitions - 1) * direct_size
        assert direct_last_size == 4537

        partitions = dask_util.guesstimate_npartitions(elements, workers)
        size = elements // partitions
        last_size = elements - (partitions - 1) * size

        assert partitions == 5881
        assert size == 37
        assert last_size == 60
        assert last_size <= size * 2


class TestSetJobqueueProcesses:
    def test_preserves_lsf_scheduler_cpu_request(self):
        cfg = {
            "jobqueue": {
                "lsf": {
                    "ncpus": 12,
                    "processes": 12,
                    "cores": 12,
                }
            }
        }

        dask_util.set_jobqueue_processes(cfg, "lsf", 5)

        settings = cfg["jobqueue"]["lsf"]
        assert settings["processes"] == 5
        assert settings["cores"] == 5
        assert settings["ncpus"] == 12


class TestCloseDaskClient:
    def test_shutdown_assertion_is_warning_not_failure(self, caplog):
        class Client:
            def __init__(self):
                self.closed = False

            def shutdown(self):
                raise AssertionError("Status.running")

            def close(self):
                self.closed = True

        client = Client()

        with caplog.at_level(logging.WARNING, logger="test"):
            dask_util._close_dask_client(
                client,
                "assemble meshes",
                logging.getLogger("test"),
            )

        assert client.closed
        assert any("Dask shutdown failed" in r.message for r in caplog.records)


class TestRunWithOomRetry:
    def test_passthrough_when_disabled(self):
        calls = []
        def work(workers, cfg):
            calls.append((workers, cfg))
            return "ok"
        result = dask_util.run_with_oom_retry(
            work, num_workers=10, phase_name="p",
            logger=logging.getLogger("test"),
            retry_on_oom=False,
        )
        assert result == "ok"
        assert calls == [(10, None)]

    def test_single_attempt_success(self, fake_distributed, fake_dask_config):
        calls = []
        def work(workers, cfg):
            calls.append((workers, cfg["jobqueue"]["lsf"]["processes"]))
            return "ok"
        result = dask_util.run_with_oom_retry(
            work, num_workers=576, phase_name="assemble",
            logger=logging.getLogger("test"),
            max_retries=3,
        )
        assert result == "ok"
        assert len(calls) == 1
        assert calls[0] == (576, 12)

    def test_halves_processes_and_workers_on_oom(
        self, fake_distributed, fake_dask_config, caplog,
    ):
        attempts = []
        def work(workers, cfg):
            attempts.append((
                workers,
                cfg["jobqueue"]["lsf"]["processes"],
                cfg["jobqueue"]["lsf"]["cores"],
                cfg["jobqueue"]["lsf"]["ncpus"],
            ))
            if len(attempts) < 3:
                raise _FakeKilledWorker(
                    "Attempted to run task on N workers",
                )
            return "ok"

        with caplog.at_level(logging.WARNING, logger="test"):
            result = dask_util.run_with_oom_retry(
                work, num_workers=576, phase_name="assemble",
                logger=logging.getLogger("test"),
                max_retries=3,
            )
        assert result == "ok"
        # First try: 576 workers, processes=12
        # Retry 1:   288, 6
        # Retry 2:   144, 3
        assert attempts == [
            (576, 12, 12, 12),
            (288, 6, 6, 12),
            (144, 3, 3, 12),
        ]
        # Each retry produces a clearly-flagged warning
        warnings = [r.message for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("retry 1/3" in m for m in warnings)
        assert any("retry 2/3" in m for m in warnings)

    def test_uses_supplied_config(self, fake_distributed):
        calls = []
        cfg = {
            "jobqueue": {
                "lsf": {
                    "ncpus": 12,
                    "processes": 6,
                    "cores": 6,
                    "memory": "180GB",
                }
            }
        }

        def work(workers, cfg):
            calls.append((workers, cfg["jobqueue"]["lsf"]["processes"]))
            return "ok"

        result = dask_util.run_with_oom_retry(
            work, num_workers=12, phase_name="assemble",
            logger=logging.getLogger("test"),
            max_retries=3,
            config=cfg,
        )
        assert result == "ok"
        assert calls == [(12, 6)]
        assert cfg["jobqueue"]["lsf"]["processes"] == 6

    def test_gives_up_after_max_retries(
        self, fake_distributed, fake_dask_config,
    ):
        attempts = []
        def work(workers, cfg):
            attempts.append(workers)
            raise _FakeKilledWorker("always dies")

        with pytest.raises(_FakeKilledWorker):
            dask_util.run_with_oom_retry(
                work, num_workers=100, phase_name="p",
                logger=logging.getLogger("test"),
                max_retries=2,
            )
        # 1 initial + 2 retries = 3 attempts
        assert len(attempts) == 3

    def test_gives_up_when_processes_cannot_halve(self, fake_distributed, monkeypatch):
        cfg = {"jobqueue": {"lsf": {"processes": 1, "ncpus": 1, "cores": 1}}}
        monkeypatch.setattr(
            dask_util, "_load_dask_config",
            lambda: {"jobqueue": {"lsf": dict(cfg["jobqueue"]["lsf"])}},
        )
        attempts = []
        def work(workers, cfg):
            attempts.append(workers)
            raise _FakeKilledWorker()

        with pytest.raises(_FakeKilledWorker):
            dask_util.run_with_oom_retry(
                work, num_workers=4, phase_name="p",
                logger=logging.getLogger("test"),
                max_retries=5,
            )
        assert len(attempts) == 1  # bailed out immediately

    def test_skips_retry_for_non_jobqueue_clusters(self, fake_distributed, monkeypatch):
        cfg = {"jobqueue": {"local": {}}}
        monkeypatch.setattr(
            dask_util, "_load_dask_config",
            lambda: {"jobqueue": dict(cfg["jobqueue"])},
        )
        attempts = []
        def work(workers, cfg):
            attempts.append(workers)
            return "ok"
        dask_util.run_with_oom_retry(
            work, num_workers=4, phase_name="p",
            logger=logging.getLogger("test"),
            max_retries=3,
        )
        # No retry wrap for local cluster; called once.
        assert len(attempts) == 1
