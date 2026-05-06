"""Fileglancer wrapper for `mesh-n-bone meshify`.

Fileglancer launches apps with flagged CLI arguments. The `mesh-n-bone meshify`
CLI instead reads a config directory containing `run-config.yaml` and
`dask-config.yaml`. This script bridges the two: it accepts individual flags,
writes a temporary config directory inside the job's working directory, and
invokes `mesh-n-bone meshify` against it.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def _parse_triplet(value):
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"expected three comma-separated numbers, got {value!r}"
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
        description="Fileglancer wrapper that runs mesh-n-bone meshify."
    )

    parser.add_argument("--input", required=True, help="Path to .zarr/.n5 dataset")
    parser.add_argument("--output", required=True, help="Output directory")

    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument(
        "--downsample-method",
        choices=["mode", "mode_suppress_zero", "binary"],
        default=None,
    )

    parser.add_argument("--target-reduction", type=float, default=None)
    parser.add_argument("--n-smoothing-iter", type=int, default=None)
    parser.add_argument("--no-simplification", action="store_true")
    parser.add_argument("--no-validity-check", action="store_true")
    parser.add_argument("--use-fixed-edge-simplification", action="store_true")
    parser.add_argument("--no-analysis", action="store_true")

    parser.add_argument("--multires", action="store_true")
    parser.add_argument("--num-lods", type=int, default=None)
    parser.add_argument(
        "--multires-strategy", choices=["decimate", "downsample"], default=None
    )
    parser.add_argument("--decimation-factor", type=int, default=None)
    parser.add_argument("--keep-decimated-meshes", action="store_true")

    parser.add_argument("--voxel-size-nm", type=_parse_triplet, default=None)
    parser.add_argument("--roi", type=_parse_roi, default=None)
    parser.add_argument("--segment-properties-csv", default=None)

    parser.add_argument(
        "--lsf-project",
        default=None,
        help="Optional LSF project (-P) for child worker jobs.",
    )

    return parser


def _build_run_config(args):
    config = {
        "input_path": args.input,
        "output_directory": args.output,
    }

    if args.downsample_factor is not None:
        config["downsample_factor"] = args.downsample_factor
    if args.downsample_method is not None:
        config["downsample_method"] = args.downsample_method

    if args.no_simplification:
        config["do_simplification"] = False
    if args.target_reduction is not None:
        config["target_reduction"] = args.target_reduction
    if args.n_smoothing_iter is not None:
        config["n_smoothing_iter"] = args.n_smoothing_iter
    if args.no_validity_check:
        config["check_mesh_validity"] = False
    if args.use_fixed_edge_simplification:
        config["use_fixed_edge_simplification"] = True
    if args.no_analysis:
        config["do_analysis"] = False

    if args.multires:
        config["do_multires"] = True
    if args.num_lods is not None:
        config["num_lods"] = args.num_lods
    if args.multires_strategy is not None:
        config["multires_strategy"] = args.multires_strategy
    if args.decimation_factor is not None:
        config["decimation_factor"] = args.decimation_factor
    if args.keep_decimated_meshes:
        config["delete_decimated_meshes"] = False

    if args.voxel_size_nm is not None:
        config["voxel_size_nm"] = args.voxel_size_nm
    if args.roi is not None:
        config["roi"] = args.roi
    if args.segment_properties_csv:
        config["segment_properties_csv"] = args.segment_properties_csv

    return config


_DASK_TEMPLATE = Path(__file__).resolve().parent / "dask-config.yaml"


def _build_dask_config(args):
    """Reuse fileglancer/dask-config.yaml, optionally overriding LSF project."""
    config = yaml.safe_load(_DASK_TEMPLATE.read_text())
    if args.lsf_project:
        config["jobqueue"]["lsf"]["project"] = args.lsf_project
    else:
        config["jobqueue"]["lsf"].pop("project", None)
    return config


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    work_dir = Path(os.environ.get("FG_WORK_DIR", os.getcwd()))
    config_dir = work_dir / "meshify-config"
    config_dir.mkdir(parents=True, exist_ok=True)

    run_config = _build_run_config(args)
    dask_config = _build_dask_config(args)

    (config_dir / "run-config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    (config_dir / "dask-config.yaml").write_text(yaml.safe_dump(dask_config, sort_keys=False))

    cmd = [
        "mesh-n-bone",
        "meshify",
        str(config_dir),
        "-n",
        str(args.num_workers),
    ]
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    completed = subprocess.run(cmd, check=False)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
