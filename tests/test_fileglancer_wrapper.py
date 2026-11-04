"""The Fileglancer manifest and its wrapper agree, and the wrapper puts
meshes where the OME-Zarr layout says."""

import importlib.util
import inspect
import json
import os
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

    def test_only_the_lod_and_compute_sections_start_open(self, items):
        expanded = [item["section"] for item in items if "section" in item and not item.get("collapsed")]
        assert expanded == ["Levels of detail", "Compute"]

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
    @pytest.mark.parametrize("version", [3, 2])
    def test_label_group_is_meshed_next_to_itself(self, tmp_path, wrapper, version):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito", "nuc"], version)
        runs = wrapper.plan_runs(str(root / "labels" / "mito"))
        assert runs == [("mito", str(root / "labels" / "mito"), str(root / "labels" / "mito_mesh"))]

    def test_array_inside_a_label_group(self, tmp_path, wrapper):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        path = str(root / "labels" / "mito" / "s0")
        runs = wrapper.plan_runs(path, mesh_suffix="_meshes")
        assert runs == [("mito", path, str(root / "labels" / "mito_meshes"))]

    @pytest.mark.parametrize("version", [3, 2])
    def test_zarr_root_is_meshed_label_by_label(self, tmp_path, wrapper, version):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito", "nuc"], version)
        runs = wrapper.plan_runs(str(root))
        assert runs == [
            ("mito", str(root / "labels" / "mito"), str(root / "labels" / "mito_mesh")),
            ("nuc", str(root / "labels" / "nuc"), str(root / "labels" / "nuc_mesh")),
        ]

    def test_explicit_output_wins(self, tmp_path, wrapper):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito"])
        runs = wrapper.plan_runs(str(root / "labels" / "mito"), output="/somewhere/else")
        assert runs == [(None, str(root / "labels" / "mito"), "/somewhere/else")]

    def test_other_inputs_need_an_output(self, tmp_path, wrapper):
        plain = tmp_path / "seg.zarr" / "s0"
        plain.mkdir(parents=True)
        with pytest.raises(ValueError, match="give an output directory"):
            wrapper.plan_runs(str(plain))
        with pytest.raises(ValueError, match="remote input"):
            wrapper.plan_runs("https://example.org/seg.zarr/labels/mito")


class TestConfigs:
    def test_booleans_and_values_reach_the_run_config(self, wrapper):
        args = wrapper._build_arg_parser().parse_args([
            "--input", "/in", "--output", "/out",
            "--multires", "false", "--sharded", "False", "--simplification", "true",
            "--strict", "true", "--keep-decimated-meshes", "true",
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

    def test_lsf_project_is_set_or_dropped(self, wrapper):
        parser = wrapper._build_arg_parser()
        with_project = wrapper._build_dask_config(
            parser.parse_args(["--input", "/in", "--lsf-project", "cellmap"]),
        )
        assert with_project["jobqueue"]["lsf"]["project"] == "cellmap"
        without = wrapper._build_dask_config(parser.parse_args(["--input", "/in"]))
        assert "project" not in without["jobqueue"]["lsf"]

    def test_dry_run_plans_every_label(self, tmp_path, wrapper, monkeypatch, capsys):
        root = _zarr_root_with_labels(tmp_path / "crop.zarr", ["mito", "nuc"])
        monkeypatch.setenv("FG_WORK_DIR", str(tmp_path / "work"))
        wrapper.main(["--input", str(root), "--dry-run", "--lsf-project", "miaai"])
        out = capsys.readouterr().out
        assert "2 run(s)" in out
        for name in ("mito", "nuc"):
            cfg = yaml.safe_load((tmp_path / "work" / "meshify-config" / name / "run-config.yaml").read_text())
            assert cfg["input_path"] == str(root / "labels" / name)
            assert cfg["output_directory"] == str(root / "labels" / f"{name}_mesh")
            dask = yaml.safe_load((tmp_path / "work" / "meshify-config" / name / "dask-config.yaml").read_text())
            assert dask["jobqueue"]["lsf"]["project"] == "miaai"
        assert not (root / "labels" / "mito_mesh").exists()
