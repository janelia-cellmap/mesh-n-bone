"""Fileglancer wrapper for ``mesh-n-bone meshify``.

Fileglancer launches apps with flagged CLI arguments. ``mesh-n-bone
meshify`` reads a config directory holding ``run-config.yaml`` and
``dask-config.yaml``. This script bridges the two: it takes the form's
flags, writes the config directory under the job's work directory, and
invokes ``mesh-n-bone meshify``.

Where the meshes go when no output is given: next to the input, in a
sibling named after it plus a suffix (``labels/mito`` is meshed to
``labels/mito_mesh``). The suffix is a form field.

Which LSF project the worker jobs are billed to: the project of the
Fileglancer job itself (the ``-P`` from Fileglancer's Cluster tab, or
the user's default), read from ``LSB_PROJECT_NAME`` when LSF sets it
and otherwise asked of ``bjobs`` for this job's id. ``--lsf-project``
overrides it for runs by hand.

How long the worker jobs may run: as long as this job, the ``-W`` in
the job script Fileglancer wrote, so no block outlives the driver's
patience and no worker is killed while the driver still waits for it.
Workers that lose the driver exit within a minute (``death-timeout``).
"""

import argparse
import os
import re
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
            "A label volume: an OME-Zarr label group or array, an N5 or "
            "precomputed volume; local or http(s)/s3/gs."
        ),
    )
    parser.add_argument(
        "--output", default=None,
        help="Where to write the meshes. Blank: next to the input, as <input><suffix>.",
    )
    parser.add_argument(
        "--mesh-suffix", default=DEFAULT_MESH_SUFFIX,
        help="Suffix of the mesh folder named after the input.",
    )
    parser.add_argument("--num-workers", type=int, default=48)
    parser.add_argument(
        "--lsf-project", default=None,
        help=(
            "LSF project (-P) for the dask worker jobs. Default: the project "
            "of the job this runs in (LSB_PROJECT_NAME, else bjobs on "
            "LSB_JOBID), else none."
        ),
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
    parser.add_argument(
        "--center-octree", type=_bool, default=None,
        help="Centre each mesh in its octree (click-to-focus lands on it) "
             "or align the octree to the mesh corner (no seams in small meshes).",
    )

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
        help="Print the run and its configs without running anything.",
    )
    return parser


# --- where the meshes go -----------------------------------------------------

def plan_output(input_path, output=None, mesh_suffix=DEFAULT_MESH_SUFFIX):
    """The output directory for the form's input and output.

    An explicit output is used as given. Otherwise the meshes go next to
    the input, in a sibling named after it plus *mesh_suffix*:
    ``.../labels/mito`` is meshed to ``.../labels/mito_mesh``. A remote
    input has no "next to", so it needs an output.
    """
    if output:
        return output
    if input_path.startswith(_REMOTE_PREFIXES):
        raise ValueError(
            "An output directory is needed for a remote input; without one "
            "the meshes go next to a local input."
        )
    source = Path(os.path.normpath(input_path))
    if not source.name or source.parent == source:
        raise ValueError(f"Cannot place meshes next to {input_path!r}; give an output directory.")
    return str(source.with_name(source.name + mesh_suffix))


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
    if args.center_octree is not None:
        config["center_octree"] = args.center_octree

    if args.voxel_size_nm is not None:
        config["voxel_size_nm"] = args.voxel_size_nm
    if args.roi is not None:
        config["roi"] = args.roi
    if args.segment_properties_csv:
        config["segment_properties_csv"] = args.segment_properties_csv

    return config


_DASK_TEMPLATE = Path(__file__).resolve().parent / "dask-config.yaml"


def _project_name(value):
    """The bare LSF project name from a typed value. People type
    ``-P cellmap`` where a name is asked for; dask-jobqueue would then write
    ``#BSUB -P -P cellmap`` and bsub rejects every worker job with nothing
    on stderr, so the run sits with no workers until its walltime."""
    name = (value or "").strip()
    if name.startswith("-P"):
        name = name[2:].lstrip(" =")
    return name.strip() or None


def _project_of_job(job_id, run=subprocess.run):
    """What LSF reports as the project of *job_id* (``bjobs -o proj_name``),
    or None when bjobs is unavailable, fails, or reports LSF's own
    placeholder ``default``. The wide column keeps long names from being
    truncated with a ``*``."""
    try:
        done = run(
            ["bjobs", "-noheader", "-o", "proj_name:64", str(job_id)],
            capture_output=True, text=True, timeout=60, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [line.strip() for line in (done.stdout or "").splitlines() if line.strip()]
    name = _project_name(lines[0]) if lines else None
    if done.returncode != 0 or name in (None, "default", "-"):
        return None
    return name


def resolve_project(cli_value=None, environ=None, job_project=_project_of_job):
    """``(project, source)`` for the worker jobs: an explicit
    ``--lsf-project`` first; else the project of the job this runs in,
    from ``LSB_PROJECT_NAME`` when LSF sets it, else from ``bjobs`` on
    ``LSB_JOBID``; else ``(None, ...)`` and bsub applies the user's
    default."""
    environ = os.environ if environ is None else environ
    project = _project_name(cli_value)
    if project:
        return project, "--lsf-project"
    project = _project_name(environ.get("LSB_PROJECT_NAME"))
    if project:
        return project, "LSB_PROJECT_NAME, the project of this job"
    job_id = environ.get("LSB_JOBID")
    if job_id:
        project = job_project(job_id)
        if project:
            return project, f"bjobs, the project of this job ({job_id})"
        return None, (
            f"none found for this job ({job_id}): LSB_PROJECT_NAME is unset and "
            "bjobs reports no project. LSF applies your default; an account that "
            "must pass -P will see every worker job rejected, so give -P under "
            "Cluster, Extra Arguments"
        )
    return None, "none given; LSF applies your default"


_BSUB_RUNLIMIT = re.compile(r"^#BSUB\s+-W\s+(\S+)", re.MULTILINE)


def driver_walltime(work_dir):
    """The ``-W`` run limit of the Fileglancer job script in *work_dir*
    (``"08:00"``), or ``None`` when no script there sets one.

    Fileglancer writes the job's ``#BSUB`` lines, including the walltime
    the form or the app's resources gave it, into a script in the job's
    work directory."""
    for script in sorted(Path(work_dir).glob("*.sh")):
        try:
            text = script.read_text(errors="replace")
        except OSError:
            continue
        match = _BSUB_RUNLIMIT.search(text)
        if match:
            return match.group(1).strip("\"'")
    return None


def _build_dask_config(project, walltime=None):
    """``fileglancer/dask-config.yaml`` with *project* for the worker jobs,
    or without a project when there is none, and with *walltime* for
    them when given."""
    config = yaml.safe_load(_DASK_TEMPLATE.read_text())
    if project:
        config["jobqueue"]["lsf"]["project"] = project
    else:
        config["jobqueue"]["lsf"].pop("project", None)
    if walltime:
        config["jobqueue"]["lsf"]["walltime"] = str(walltime)
    return config


def _meshify_command(config_dir, args):
    cmd = ["mesh-n-bone", "meshify", str(config_dir), "-n", str(args.num_workers)]
    if args.ids:
        cmd += ["--ids", args.ids]
    return cmd


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)

    work_dir = Path(os.environ.get("FG_WORK_DIR", os.getcwd()))
    output = plan_output(args.input, args.output, args.mesh_suffix)
    project, source = resolve_project(args.lsf_project)

    print(f"meshify: {args.input} -> {output}", flush=True)
    if os.environ.get("LSB_JOBID"):
        # One line that settles, from the log alone, what LSF told this job.
        print(
            f"LSF job {os.environ['LSB_JOBID']} in queue {os.environ.get('LSB_QUEUE', '?')}; "
            f"LSB_PROJECT_NAME={os.environ.get('LSB_PROJECT_NAME') or '<unset>'}",
            flush=True,
        )
    print(f"worker jobs' LSF project: {project or 'none'} ({source})", flush=True)
    walltime = driver_walltime(work_dir)
    if walltime:
        print(f"worker jobs' walltime: {walltime} (this job's -W)", flush=True)
    else:
        default = _build_dask_config(project)["jobqueue"]["lsf"]["walltime"]
        print(
            f"worker jobs' walltime: {default} (the dask-config default; "
            "no -W found in this job's script)",
            flush=True,
        )

    config_dir = work_dir / "meshify-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    run_config = _build_run_config(args, args.input, output)
    (config_dir / "run-config.yaml").write_text(yaml.safe_dump(run_config, sort_keys=False))
    (config_dir / "dask-config.yaml").write_text(
        yaml.safe_dump(_build_dask_config(project, walltime), sort_keys=False)
    )
    cmd = _meshify_command(config_dir, args)
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
    if args.dry_run:
        print(yaml.safe_dump(run_config, sort_keys=False), flush=True)
        return
    os.makedirs(output, exist_ok=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        print(f"meshify FAILED with exit code {completed.returncode}", file=sys.stderr, flush=True)
        sys.exit(completed.returncode or 1)


if __name__ == "__main__":
    main()
