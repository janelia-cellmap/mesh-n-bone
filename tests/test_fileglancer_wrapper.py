"""The Fileglancer manifest and its wrapper agree, and the wrapper puts
meshes where the OME-Zarr layout says."""

import importlib.util
import inspect
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def wrapper():
    spec = importlib.util.spec_from_file_location(
        "run_meshify", REPO / "fileglancer" / "run_meshify.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def manifest():
    with open(REPO / "runnables.yaml") as f:
        return yaml.safe_load(f)


def _parameters(items):
    """Every parameter of a runnable, sections flattened."""
    for item in items:
        if "section" in item:
            yield from _parameters(item["parameters"])
        else:
            yield item


def _zarr_root_with_labels(root, labels, version=3):
    """A zarr root whose labels/ group lists ``labels``, each a group dir."""
    labels_dir = root / "labels"
    labels_dir.mkdir(parents=True)
    if version == 3:
        (root / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group"}))
        (labels_dir / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "group",
            "attributes": {"ome": {"version": "0.5", "labels": labels}},
        }))
    else:
        (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (labels_dir / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (labels_dir / ".zattrs").write_text(json.dumps({"labels": labels}))
    for name in labels:
        (labels_dir / name / "s0").mkdir(parents=True)
    return root


class TestManifestMatchesWrapper:
    def test_every_flag_is_a_wrapper_option(self, wrapper, manifest):
        parser = wrapper._build_arg_parser()
        known = set(parser._option_string_actions)
        runnable = manifest["runnables"][0]
        flags = [p["flag"] for p in _parameters(runnable["parameters"])]
        assert flags, "the manifest lists no parameters"
        missing = [f for f in flags if f not in known]
        assert not missing, f"manifest flags the wrapper does not take: {missing}"
        assert runnable["command"].endswith("fileglancer/run_meshify.py")

    def test_enum_options_match_the_wrapper_choices(self, wrapper, manifest):
        parser = wrapper._build_arg_parser()
        for p in _parameters(manifest["runnables"][0]["parameters"]):
            if p["type"] == "enum":
                action = parser._option_string_actions[p["flag"]]
                assert set(p["options"]) == set(action.choices), p["flag"]
                if "default" in p:
                    assert p["default"] in action.choices, p["flag"]

    def test_booleans_are_sent_as_values(self, manifest):
        """Options whose Meshify default is True can only be turned off
        when Fileglancer sends an explicit false."""
        for p in _parameters(manifest["runnables"][0]["parameters"]):
            if p["type"] == "boolean":
                assert p.get("boolean_style") == "value", p["flag"]

    def test_form_defaults_are_meshify_defaults(self, manifest):
        from mesh_n_bone.meshify.meshify import Meshify

        defaults = {
            k: v.default for k, v in inspect.signature(Meshify).parameters.items()
        }
        form = {p["flag"]: p.get("default") for p in _parameters(manifest["runnables"][0]["parameters"])}
        assert form["--target-reduction"] == defaults["target_reduction"]
        assert form["--n-smoothing-iter"] == defaults["n_smoothing_iter"]
        assert form["--num-lods"] == defaults["num_lods"]
        assert form["--decimation-factor"] == defaults["decimation_factor"]
        assert form["--multires-strategy"] == defaults["multires_strategy"]
        assert form["--downsample-method"] == defaults["downsample_method"]
        assert form["--multires"] is defaults["do_multires"]
        assert form["--simplification"] is defaults["do_simplification"]
        assert form["--use-fixed-edge-simplification"] is defaults["use_fixed_edge_simplification"]
        assert form["--use-existing-scales"] is defaults["use_existing_scales"]
        assert form["--downsample-cascade"] is defaults["downsample_cascade"]
        assert form["--delete-intermediate-scales"] is defaults["delete_intermediate_scales"]
        assert form["--sharded"] is defaults["sharded"]
        assert form["--center-octree"] is defaults["center_octree"]

    def test_output_may_not_exist_yet(self, manifest):
        output = next(
            p for p in _parameters(manifest["runnables"][0]["parameters"])
            if p["flag"] == "--output"
        )
        assert output["type"] == "directory"
        assert output["exists"] is False
        assert output["required"] is False


class TestFormReadsOnItsOwn:
    """Fileglancer pre-fills defaults without marking them, repeats a field's
    name as the placeholder of an empty box, and shows a bare checkbox for a
    boolean without a description. The manifest has to carry the meaning."""

    @pytest.fixture(scope="class")
    def items(self, manifest):
        return manifest["runnables"][0]["parameters"]

    def test_every_field_and_section_is_described(self, items):
        for item in items:
            if "section" in item:
                assert item.get("description"), f"section {item['section']!r} has no description"
        for p in _parameters(items):
            assert p.get("description"), f"{p['flag']} has no description"

    def test_descriptions_state_the_default(self, items):
        for p in _parameters(items):
            if "default" not in p:
                continue
            text = p["description"]
            if p["type"] == "boolean":
                expected = "Default on" if p["default"] else "Default off"
            else:
                expected = str(p["default"])
            assert expected in text, f"{p['flag']}: {expected!r} not stated in {text!r}"

    def test_empty_text_fields_show_an_example(self, items):
        for p in _parameters(items):
            if p["type"] == "string":
                assert p.get("placeholder"), f"{p['flag']} has no placeholder"

    def test_blank_means_something_for_optional_fields(self, items):
        for p in _parameters(items):
            if p["type"] in ("file", "directory", "boolean", "enum") or "default" in p:
                continue
            assert "Blank" in p["description"], f"{p['flag']} does not say what blank means"

    def test_only_the_compute_section_starts_open(self, items):
        """A first run is the two path fields; every tuning section starts
        collapsed."""
        expanded = [item["section"] for item in items if "section" in item and not item.get("collapsed")]
        assert expanded == ["Compute"]

    def test_only_the_debug_option_is_hidden(self, items):
        hidden = [p["flag"] for p in _parameters(items) if p.get("hidden")]
        assert hidden == ["--keep-decimated-meshes"]

    def test_cross_references_name_real_fields_and_sections(self, items):
        names = {p["name"] for p in _parameters(items)}
        sections = {item["section"] for item in items if "section" in item}
        for phrase in ("Downsample method", "Downsample factor", "Target face reduction",
                       "Face count ratio", "Output directory"):
            assert any(n.startswith(phrase) for n in names), phrase
        for phrase in ("Output layout", "Levels of detail", "Resolution", "Simplification"):
            assert any(s.startswith(phrase) for s in sections), phrase


class TestWhereTheMeshesGo:
    """Blank output: a sibling of the input named after it plus the suffix.
    No convention is read; it works the same for a label group, an N5
    dataset or a precomputed folder."""

    @pytest.mark.parametrize("version", [3, 2])
    def test_label_group_is_meshed_next_to_itself(self, tmp_path, wrapper, version):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito", "nuc"], version)
        assert wrapper.plan_output(str(root / "labels" / "mito")) == str(root / "labels" / "mito_mesh")

    def test_the_suffix_is_a_parameter(self, tmp_path, wrapper):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        out = wrapper.plan_output(str(root / "labels" / "mito") + "/", mesh_suffix="_meshes")
        assert out == str(root / "labels" / "mito_meshes")

    def test_any_input_gets_a_sibling(self, wrapper):
        assert wrapper.plan_output("/data/cells.n5/seg") == "/data/cells.n5/seg_mesh"
        assert wrapper.plan_output("/data/seg.zarr/s0") == "/data/seg.zarr/s0_mesh"

    def test_explicit_output_wins(self, tmp_path, wrapper):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        assert wrapper.plan_output(str(root / "labels" / "mito"), output="/somewhere/else") == "/somewhere/else"
        assert wrapper.plan_output("https://example.org/seg.zarr/labels/mito", output="/out") == "/out"

    def test_remote_input_needs_an_output(self, wrapper):
        with pytest.raises(ValueError, match="remote input"):
            wrapper.plan_output("https://example.org/seg.zarr/labels/mito")

    def test_a_root_path_cannot_have_a_sibling(self, wrapper):
        with pytest.raises(ValueError, match="give an output directory"):
            wrapper.plan_output("/")


class TestConfigs:
    def test_booleans_and_values_reach_the_run_config(self, wrapper):
        args = wrapper._build_arg_parser().parse_args([
            "--input", "/in", "--output", "/out",
            "--multires", "false", "--sharded", "False", "--simplification", "true",
            "--strict", "true", "--keep-decimated-meshes", "true",
            "--center-octree", "false",
            "--use-existing-scales", "false", "--num-lods", "5",
            "--target-reduction", "0.9", "--downsample-method", "nearest",
            "--voxel-size-nm", "45,16,16", "--roi", "0,0,0,100,200,300",
            "--segment-properties-csv", "/props.csv",
        ])
        config = wrapper._build_run_config(args, "/in", "/out")
        assert config == {
            "input_path": "/in", "output_directory": "/out",
            "downsample_method": "nearest",
            "do_simplification": True, "target_reduction": 0.9,
            "check_mesh_validity": True, "do_analysis": True,
            "do_multires": False, "num_lods": 5, "use_existing_scales": False,
            "sharded": False, "delete_decimated_meshes": False,
            "center_octree": False,
            "voxel_size_nm": [45.0, 16.0, 16.0],
            "roi": {"begin": [0.0, 0.0, 0.0], "end": [100.0, 200.0, 300.0]},
            "segment_properties_csv": "/props.csv",
        }

    def test_unset_options_are_left_to_meshify(self, wrapper):
        args = wrapper._build_arg_parser().parse_args(["--input", "/in", "--output", "/out"])
        assert wrapper._build_run_config(args, "/in", "/out") == {
            "input_path": "/in", "output_directory": "/out",
        }

    def test_ids_go_on_the_command_line(self, wrapper):
        args = wrapper._build_arg_parser().parse_args(
            ["--input", "/in", "--output", "/out", "--ids", "1,2,3", "--num-workers", "7"],
        )
        assert wrapper._meshify_command("/cfg", args) == [
            "mesh-n-bone", "meshify", "/cfg", "-n", "7", "--ids", "1,2,3",
        ]

class TestWorkerProject:
    """The worker jobs are billed where the Fileglancer job is. LSF exposes
    a running job's project as LSB_PROJECT_NAME (the Cluster tab's -P, or
    the user's default), the same thing cellmap-flow reads; --lsf-project
    overrides it for runs by hand. There is no form field for it."""

    def test_the_form_has_no_project_field(self, manifest):
        assert "--lsf-project" not in {p["flag"] for p in _parameters(manifest["runnables"][0]["parameters"])}

    def test_inherits_the_job_project(self, wrapper):
        assert wrapper.resolve_project(None, {"LSB_PROJECT_NAME": "miaai"})[0] == "miaai"

    def test_explicit_project_wins(self, wrapper):
        project, source = wrapper.resolve_project("cellmap", {"LSB_PROJECT_NAME": "miaai"})
        assert (project, source) == ("cellmap", "--lsf-project")

    def test_no_project_outside_lsf(self, wrapper):
        assert wrapper.resolve_project(None, {})[0] is None
        assert "project" not in wrapper._build_dask_config(None)["jobqueue"]["lsf"]

    def test_asks_bjobs_when_the_variable_is_missing(self, wrapper):
        """Janelia's nodes did not set LSB_PROJECT_NAME for a Fileglancer
        job, so the job's own id is the second source."""
        asked = []

        def job_project(job_id):
            asked.append(job_id)
            return "cellmap"

        project, source = wrapper.resolve_project(None, {"LSB_JOBID": "155196857"}, job_project=job_project)
        assert (project, asked) == ("cellmap", ["155196857"])
        assert "bjobs" in source and "155196857" in source

    def test_says_loudly_when_the_job_has_no_project(self, wrapper):
        project, source = wrapper.resolve_project(None, {"LSB_JOBID": "1"}, job_project=lambda job_id: None)
        assert project is None
        assert "rejected" in source and "Extra Arguments" in source

    @pytest.mark.parametrize("stdout, code, expected", [
        ("cellmap   \n", 0, "cellmap"),
        ("  scicompsoft\n", 0, "scicompsoft"),
        ("default\n", 0, None),
        ("-\n", 0, None),
        ("", 0, None),
        ("cellmap\n", 1, None),
    ])
    def test_bjobs_output_is_read_as_a_project(self, wrapper, stdout, code, expected):
        def run(argv, **kwargs):
            assert argv[:4] == ["bjobs", "-noheader", "-o", "proj_name:64"]
            return subprocess.CompletedProcess(argv, code, stdout=stdout, stderr="")

        assert wrapper._project_of_job("7", run=run) == expected

    def test_bjobs_missing_or_hanging_means_no_project(self, wrapper):
        def missing(argv, **kwargs):
            raise FileNotFoundError("bjobs")

        def hanging(argv, **kwargs):
            raise subprocess.TimeoutExpired(argv, 60)

        assert wrapper._project_of_job("7", run=missing) is None
        assert wrapper._project_of_job("7", run=hanging) is None

    @pytest.mark.parametrize("typed", ["cellmap", "-P cellmap", " -Pcellmap ", "-P=cellmap", "-P  cellmap"])
    def test_a_typed_dash_p_is_stripped(self, wrapper, typed):
        """bsub rejects '#BSUB -P -P cellmap' silently, so a pasted
        '-P cellmap' is reduced to the name wherever it comes from."""
        assert wrapper.resolve_project(typed, {})[0] == "cellmap"
        assert wrapper.resolve_project(None, {"LSB_PROJECT_NAME": typed})[0] == "cellmap"
        assert wrapper._build_dask_config("cellmap")["jobqueue"]["lsf"]["project"] == "cellmap"

    @pytest.mark.parametrize("typed", ["", "  ", "-P", "-P "])
    def test_an_empty_project_is_none(self, wrapper, typed):
        assert wrapper.resolve_project(typed, {"LSB_PROJECT_NAME": typed})[0] is None

    def test_dry_run_writes_the_configs_with_the_job_project(self, tmp_path, wrapper, monkeypatch, capsys):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        monkeypatch.setenv("FG_WORK_DIR", str(tmp_path / "work"))
        monkeypatch.setenv("LSB_PROJECT_NAME", "miaai")
        wrapper.main(["--input", str(root / "labels" / "mito"), "--dry-run"])
        out = capsys.readouterr().out
        assert "LSF project: miaai (LSB_PROJECT_NAME" in out
        cfg = yaml.safe_load((tmp_path / "work" / "meshify-config" / "run-config.yaml").read_text())
        assert cfg["input_path"] == str(root / "labels" / "mito")
        assert cfg["output_directory"] == str(root / "labels" / "mito_mesh")
        dask = yaml.safe_load((tmp_path / "work" / "meshify-config" / "dask-config.yaml").read_text())
        assert dask["jobqueue"]["lsf"]["project"] == "miaai"
        assert not (root / "labels" / "mito_mesh").exists()


class TestWorkerWalltime:
    """The worker jobs get the Fileglancer job's own walltime, so a block
    can take as long as the driver waits for it, and they end with it."""

    @staticmethod
    def _job_script(work_dir, runlimit_line):
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "fg-mesh-n-bone-meshify.1.sh").write_text(
            "#!/bin/bash\n"
            "#BSUB -J fg-mesh-n-bone-meshify\n"
            "#BSUB -n 2\n"
            f"{runlimit_line}\n"
            '#BSUB -cwd "/somewhere"\n'
            "export FG_WORK_DIR=/somewhere\n"
        )

    @pytest.mark.parametrize("line, expected", [
        ("#BSUB -W 08:00", "08:00"),
        ('#BSUB -W "4:00"', "4:00"),
        ("#BSUB  -W 30", "30"),
    ])
    def test_reads_the_job_scripts_runlimit(self, tmp_path, wrapper, line, expected):
        self._job_script(tmp_path, line)
        assert wrapper.driver_walltime(tmp_path) == expected

    def test_no_script_or_no_runlimit_means_none(self, tmp_path, wrapper):
        assert wrapper.driver_walltime(tmp_path) is None
        self._job_script(tmp_path, "#BSUB -M 30720")
        assert wrapper.driver_walltime(tmp_path) is None

    def test_template_matches_the_apps_walltime_and_ends_with_the_driver(self, wrapper, manifest):
        lsf = wrapper._build_dask_config(None)["jobqueue"]["lsf"]
        app = manifest["runnables"][0]
        assert lsf["walltime"] == app["resources"]["walltime"]
        assert lsf["death-timeout"] == 60

    def test_dry_run_gives_the_workers_the_jobs_walltime(self, tmp_path, wrapper, monkeypatch, capsys):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        work = tmp_path / "work"
        # A walltime YAML would read as a number unless it is written as a string.
        self._job_script(work, "#BSUB -W 4:00")
        monkeypatch.setenv("FG_WORK_DIR", str(work))
        monkeypatch.setenv("LSB_PROJECT_NAME", "miaai")
        wrapper.main(["--input", str(root / "labels" / "mito"), "--dry-run"])
        assert "worker jobs' walltime: 4:00 (this job's -W)" in capsys.readouterr().out
        dask = yaml.safe_load((work / "meshify-config" / "dask-config.yaml").read_text())
        assert dask["jobqueue"]["lsf"]["walltime"] == "4:00"

    def test_dry_run_without_a_script_keeps_the_default(self, tmp_path, wrapper, monkeypatch, capsys):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        monkeypatch.setenv("FG_WORK_DIR", str(tmp_path / "work"))
        monkeypatch.setenv("LSB_PROJECT_NAME", "miaai")
        wrapper.main(["--input", str(root / "labels" / "mito"), "--dry-run"])
        assert "worker jobs' walltime: 08:00 (the dask-config default" in capsys.readouterr().out
