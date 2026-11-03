"""Fileglancer wrapper for ``mesh-n-bone meshify``.

Fileglancer launches apps with flagged CLI arguments. ``mesh-n-bone
meshify`` reads a config directory holding ``run-config.yaml`` and
``dask-config.yaml``. This script bridges the two: it takes the form's
flags, writes one config directory per run under the job's work
directory, and invokes ``mesh-n-bone meshify`` on each.

Where the meshes go when no output is given: a label group inside an
OME-Zarr ``labels/`` folder (``.../labels/<name>``, or an array in it)
is meshed to ``.../labels/<name>_mesh`` next to it, and a zarr root
whose ``labels/`` group lists labels is meshed label by label, each to
its own ``_mesh`` folder. The suffix is a form field.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_MESH_SUFFIX = "_mesh"
_REMOTE_PREFIXES = ("http://", "https://", "s3://", "gs://")


def _bool(value):
    """``true``/``false`` (any case; also yes/no, 1/0) to a bool: Fileglancer
    emits every boolean field as ``--flag true`` or ``--flag false``."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ("true", "yes", "1", "on"):
        return True
    if v in ("false", "no", "0", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def _parse_triplet(value):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated numbers (z,y,x), got {value!r}"
        )
    return [float(p) for p in parts]


def _parse_roi(value):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "ROI must be 6 comma-separated values: z0,y0,x0,z1,y1,x1"
        )
    nums = [float(p) for p in parts]
    return {"begin": nums[:3], "end": nums[3:]}


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Fileglancer wrapper that runs mesh-n-bone meshify.",
    )
    parser.add_argument(
        "--input", required=True,
        help=(
            "A label volume (an OME-Zarr label group or array, an N5 or "
            "precomputed volume; local or http(s)/s3/gs), or a zarr root "
            "whose labels/ group lists the labels to mesh."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help=(
            "Where to write the meshes. Blank: next to the label, as "
            "labels/<name><suffix>."
        ),
    )
    parser.add_argument(
        "--mesh-suffix", default=DEFAULT_MESH_SUFFIX,
        help="Suffix of a mesh folder derived from its label's name.",
    )
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument(
        "--lsf-project", default=None,
        help="LSF project (-P) for the dask worker jobs; blank for none.",
    )

    # Mesh generation
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument(
        "--downsample-method",
        choices=["mode", "mode_suppress_zero", "binary", "nearest"], default=None,
    )

    # Simplification & smoothing
    parser.add_argument("--target-reduction", type=float, default=None)
    parser.add_argument("--n-smoothing-iter", type=int, default=None)
    parser.add_argument("--simplification", type=_bool, default=None)
    parser.add_argument("--use-fixed-edge-simplification", type=_bool, default=None)
    parser.add_argument(
        "--strict", type=_bool, default=None,
        help="Require watertight meshes and write the mesh-metrics CSV.",
    )

    # Multiresolution output
    parser.add_argument("--multires", type=_bool, default=None)
    parser.add_argument("--num-lods", type=int, default=None)
    parser.add_argument(
        "--multires-strategy", choices=["downsample", "decimate"], default=None,
    )
    parser.add_argument("--decimation-factor", type=int, default=None)
    parser.add_argument("--use-existing-scales", type=_bool, default=None)
    parser.add_argument("--downsample-cascade", type=_bool, default=None)
    parser.add_argument("--delete-intermediate-scales", type=_bool, default=None)
    parser.add_argument("--sharded", type=_bool, default=None)
    parser.add_argument("--keep-decimated-meshes", type=_bool, default=None)

    # Coordinates & selection
    parser.add_argument("--voxel-size-nm", type=_parse_triplet, default=None)
    parser.add_argument("--roi", type=_parse_roi, default=None)
    parser.add_argument(
        "--ids", default=None,
        help="Only these segment ids: a comma-separated list or a CSV path.",
    )
    parser.add_argument("--segment-properties-csv", default=None)

    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the runs and their configs without running anything.",
    )
    return parser


# --- which runs, and where their meshes go --------------------------------

def _group_attributes(path):
    """The attributes of a zarr group (v3 ``zarr.json`` or v2 ``.zattrs``),
    or None when *path* is not a local zarr group."""
    v3 = Path(path) / "zarr.json"
    if v3.is_file():
        with open(v3) as f:
            doc = json.load(f)
        return doc.get("attributes", {}) if doc.get("node_type") == "group" else None
    v2 = Path(path) / ".zattrs"
    if v2.is_file() and (Path(path) / ".zgroup").is_file():
        with open(v2) as f:
            return json.load(f)
    return None


def labels_listed_in(zarr_root):
    """The label names an OME-Zarr root lists in its ``labels/`` group
    (``attributes.ome.labels`` for v3, ``labels`` for v2), or None."""
    attrs = _group_attributes(Path(zarr_root) / "labels")
    if attrs is None:
        return None
    labels = attrs.get("ome", {}).get("labels") if isinstance(attrs.get("ome"), dict) else None
    if labels is None:
        labels = attrs.get("labels")
    return list(labels) if labels else None


def _label_folder(path):
    """``(labels_dir, name)`` when *path* is a label group under a
    ``labels/`` folder, or an array inside one; else None."""
    p = Path(os.path.normpath(path))
    for candidate in (p, p.parent):
        if candidate.parent.name == "labels" and candidate.name:
            return candidate.parent, candidate.name
    return None


def plan_runs(input_path, output=None, mesh_suffix=DEFAULT_MESH_SUFFIX):
    """The ``(name, input, output)`` runs for the form's input and output.

    An explicit output is one run. Otherwise a label under ``labels/`` is
    meshed next to itself, and a zarr root listing labels is one run per
    label. Anything else needs an output directory.
    """
    if output:
        return [(None, input_path, output)]
    if input_path.startswith(_REMOTE_PREFIXES):
        raise ValueError(
            "An output directory is needed for a remote input; the meshes "
            "are only placed next to a local label."
        )
    found = _label_folder(input_path)
    if found is not None:
        labels_dir, name = found
        return [(name, input_path, str(labels_dir / f"{name}{mesh_suffix}"))]
    labels = labels_listed_in(input_path)
    if labels:
        labels_dir = Path(input_path) / "labels"
        return [
            (name, str(labels_dir / name), str(labels_dir / f"{name}{mesh_suffix}"))
            for name in labels
        ]
    raise ValueError(
        f"{input_path} is neither a label under a labels/ folder nor a zarr "
        "root listing labels; give an output directory."
    )


# --- configs -----------------------------------------------------------------

def _build_run_config(args, input_path, output):
    config = {"input_path": input_path, "output_directory": output}

    if args.downsample_factor is not None:
        config["downsample_factor"] = args.downsample_factor
    if args.downsample_method is not None:
        config["downsample_method"] = args.downsample_method

    if args.simplification is not None:
        config["do_simplification"] = args.simplification
    if args.target_reduction is not None:
        config["target_reduction"] = args.target_reduction
    if args.n_smoothing_iter is not None:
        config["n_smoothing_iter"] = args.n_smoothing_iter
    if args.use_fixed_edge_simplification is not None:
        config["use_fixed_edge_simplification"] = args.use_fixed_edge_simplification
    if args.strict is not None:
        config["check_mesh_validity"] = args.strict
        config["do_analysis"] = args.strict

    if args.multires is not None:
        config["do_multires"] = args.multires
    if args.num_lods is not None:
        config["num_lods"] = args.num_lods
    if args.multires_strategy is not None:
        config["multires_strategy"] = args.multires_strategy
    if args.decimation_factor is not None:
        config["decimation_factor"] = args.decimation_factor
    if args.use_existing_scales is not None:
        config["use_existing_scales"] = args.use_existing_scales
    if args.downsample_cascade is not None:
        config["downsample_cascade"] = args.downsample_cascade
    if args.delete_intermediate_scales is not None:
        config["delete_intermediate_scales"] = args.delete_intermediate_scales
    if args.sharded is not None:
        config["sharded"] = args.sharded
    if args.keep_decimated_meshes is not None:
        config["delete_decimated_meshes"] = not args.keep_decimated_meshes

    if args.voxel_size_nm is not None:
        config["voxel_size_nm"] = args.voxel_size_nm
    if args.roi is not None:
        config["roi"] = args.roi
    if args.segment_properties_csv:
        config["segment_properties_csv"] = args.segment_properties_csv

    return config


_DASK_TEMPLATE = Path(__file__).resolve().parent / "dask-config.yaml"


def _build_dask_config(args):
    """``fileglancer/dask-config.yaml`` with the form's LSF project, or
    without a project when the field is blank."""
    config = yaml.safe_load(_DASK_TEMPLATE.read_text())
    if args.lsf_project:
        config["jobqueue"]["lsf"]["project"] = args.lsf_project
    else:
        config["jobqueue"]["lsf"].pop("project", None)
    return config


def _meshify_command(config_dir, args):
    cmd = ["mesh-n-bone", "meshify", str(config_dir), "-n", str(args.num_workers)]
    if args.ids:
        cmd += ["--ids", args.ids]
    return cmd


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    work_dir = Path(os.environ.get("FG_WORK_DIR", os.getcwd()))
    runs = plan_runs(args.input, args.output, args.mesh_suffix)
    dask_config = _build_dask_config(args)

    print(f"{len(runs)} run(s):", flush=True)
    for name, input_path, output in runs:
        print(f"  {name or 'meshify'}: {input_path} -> {output}", flush=True)

    failures = []
    for name, input_path, output in runs:
        config_dir = work_dir / "meshify-config" / (name or "meshify")
        config_dir.mkdir(parents=True, exist_ok=True)
        run_config = _build_run_config(args, input_path, output)
        (config_dir / "run-config.yaml").write_text(
            yaml.safe_dump(run_config, sort_keys=False)
        )
        (config_dir / "dask-config.yaml").write_text(
            yaml.safe_dump(dask_config, sort_keys=False)
        )
        cmd = _meshify_command(config_dir, args)
        print(f"+ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        if args.dry_run:
            print(yaml.safe_dump(run_config, sort_keys=False), flush=True)
            continue
        os.makedirs(output, exist_ok=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(f"{name or 'meshify'} FAILED with exit code {completed.returncode}",
                  file=sys.stderr, flush=True)
            failures.append((name or "meshify", completed.returncode))

    if failures:
        print(f"{len(failures)} of {len(runs)} run(s) failed: "
              + ", ".join(n for n, _ in failures), file=sys.stderr, flush=True)
        sys.exit(failures[0][1] or 1)


if __name__ == "__main__":
    main()
