"""Unified CLI for mesh-n-bone: meshification, multiresolution, skeletonization, and analysis."""

import argparse
import os
import sys
import logging
import importlib

logger = logging.getLogger(__name__)


def _setup_pymeshlab_ld_path():
    """Ensure pymeshlab's shared libraries are on ``LD_LIBRARY_PATH``.

    If the path is missing, the environment variable is updated and the
    current process is re-executed so the dynamic linker picks up the
    change before any pymeshlab code is imported.
    """
    try:
        spec = importlib.util.find_spec("pymeshlab")
        if spec and spec.origin:
            pymesh_lib_path = os.path.join(os.path.dirname(spec.origin), "lib")
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if pymesh_lib_path not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = (
                    f"{pymesh_lib_path}:{ld_path}" if ld_path else pymesh_lib_path
                )
                try:
                    sys.stdout.flush()
                    os.execl(sys.executable, sys.executable, *sys.argv)
                except OSError:
                    pass
    except (ModuleNotFoundError, AttributeError):
        pass


def _get_run_properties(args):
    """Set up execution directory and load the YAML run config.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments. Must contain ``config_path`` and
        ``num_workers`` attributes.

    Returns
    -------
    tuple[str, str, dict]
        ``(execution_directory, logpath, run_config)`` where
        *execution_directory* is the working directory for this run,
        *logpath* is the path to the log file, and *run_config* is
        the parsed YAML config with ``num_workers`` injected.
    """
    from mesh_n_bone.util.dask_util import setup_execution_directory
    from mesh_n_bone.config import read_generic_config

    execution_directory = setup_execution_directory(args.config_path, logger)
    logpath = f"{execution_directory}/output.log"
    run_config = read_generic_config(args.config_path)
    run_config["num_workers"] = args.num_workers
    return execution_directory, logpath, run_config


def _parse_roi_arg(roi_str):
    """Parse a ``--roi`` argument into a dict with ``begin`` and ``end`` keys.

    Accepts 6 comma-separated values:
    ``begin_0,begin_1,begin_2,end_0,end_1,end_2``.
    For *meshify* these are in ZYX order (matching the volume); for
    *to-neuroglancer* they are in XYZ order.

    Parameters
    ----------
    roi_str : str
        Comma-separated string of 6 numeric values.

    Returns
    -------
    dict
        ``{"begin": [b0, b1, b2], "end": [e0, e1, e2]}``

    Raises
    ------
    argparse.ArgumentTypeError
        If *roi_str* does not contain exactly 6 values.
    """
    values = [float(v) for v in roi_str.split(",")]
    if len(values) != 6:
        raise argparse.ArgumentTypeError(
            "ROI must be 6 comma-separated values: begin_0,begin_1,begin_2,end_0,end_1,end_2"
        )
    return {"begin": values[:3], "end": values[3:]}


def cmd_meshify(args):
    """Run the meshify pipeline.

    Generates triangle meshes from a segmentation volume stored in
    Zarr/N5 format, using marching cubes and optional simplification.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including ``config_path``, ``num_workers``, and
        optional ``roi``.
    """
    from mesh_n_bone.meshify.meshify import Meshify
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    if args.roi:
        run_config["roi"] = _parse_roi_arg(args.roi)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        meshify = Meshify(**run_config)
        meshify.get_meshes()


def cmd_multires(args):
    """Run the multiresolution mesh pipeline.

    Converts existing meshes into the Neuroglancer precomputed
    multiresolution format with multiple levels of detail.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including ``config_path``, ``num_workers``, and
        optional ``roi``.
    """
    from mesh_n_bone.multires.multires import run_multires

    roi = _parse_roi_arg(args.roi) if args.roi else None
    run_multires(args.config_path, args.num_workers, roi=roi)


def cmd_skeletonize(args):
    """Run the batch skeletonization pipeline.

    Extracts curve skeletons from meshes using CGAL's mean curvature
    skeleton algorithm, with optional Dask parallelization.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including ``config_path`` and ``num_workers``.
    """
    from mesh_n_bone.skeletonize.skeletonize import Skeletonize
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        skeletonize = Skeletonize(**run_config)
        skeletonize.get_skeletons()


def cmd_skeletonize_single(args):
    """Run CGAL skeletonization on a single mesh file.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including ``input_file``, ``output_file``,
        ``subdivisions``, and ``neuroglancer`` flag.
    """
    from mesh_n_bone.skeletonize.skeletonize import Skeletonize

    Skeletonize.cgal_skeletonize_mesh(
        args.input_file,
        args.output_file,
        base_loop_subdivision_iterations=args.subdivisions,
        neuroglancer_format=args.neuroglancer,
    )


def cmd_analyze(args):
    """Run the mesh analysis pipeline.

    Computes geometric properties (volume, surface area, curvature,
    thickness, etc.) for a collection of meshes and writes results to
    a Parquet file.

    Parameters
    ----------
    args : argparse.Namespace
        CLI arguments including ``config_path`` and ``num_workers``.
    """
    from mesh_n_bone.analyze.analyze import AnalyzeMeshes
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        analyze = AnalyzeMeshes(**run_config)
        analyze.analyze()


def main():
    """Entry point for the ``mesh-n-bone`` CLI.

    Parses command-line arguments and dispatches to the appropriate
    sub-command (meshify, to-neuroglancer, skeletonize,
    skeletonize-single, or analyze).
    """
    _setup_pymeshlab_ld_path()

    parser = argparse.ArgumentParser(
        prog="mesh-n-bone",
        description="Unified tool for mesh generation, multiresolution, skeletonization, and analysis.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # meshify
    p_meshify = subparsers.add_parser(
        "meshify", help="Generate meshes from segmentation volumes"
    )
    p_meshify.add_argument("config_path", help="Path to config directory")
    p_meshify.add_argument(
        "-n", "--num-workers", type=int, default=10, help="Number of dask workers"
    )
    p_meshify.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI to process (ZYX): begin_z,begin_y,begin_x,end_z,end_y,end_x",
    )
    p_meshify.set_defaults(func=cmd_meshify)

    # to-neuroglancer
    p_multires = subparsers.add_parser(
        "to-neuroglancer", help="Convert existing meshes to neuroglancer multiresolution format"
    )
    p_multires.add_argument("config_path", help="Path to config directory")
    p_multires.add_argument(
        "-n", "--num-workers", type=int, default=10, help="Number of dask workers"
    )
    p_multires.add_argument(
        "--roi",
        type=str,
        default=None,
        help="ROI to process (XYZ): begin_x,begin_y,begin_z,end_x,end_y,end_z",
    )
    p_multires.set_defaults(func=cmd_multires)

    # skeletonize
    p_skel = subparsers.add_parser(
        "skeletonize", help="Skeletonize meshes using CGAL"
    )
    p_skel.add_argument("config_path", help="Path to config directory")
    p_skel.add_argument(
        "-n", "--num-workers", type=int, default=10, help="Number of dask workers"
    )
    p_skel.set_defaults(func=cmd_skeletonize)

    # skeletonize-single
    p_skel_single = subparsers.add_parser(
        "skeletonize-single", help="Skeletonize a single mesh file"
    )
    p_skel_single.add_argument("input_file", help="Path to input mesh (e.g. .ply, .obj)")
    p_skel_single.add_argument("output_file", help="Path to write skeleton output")
    p_skel_single.add_argument(
        "--subdivisions",
        type=int,
        default=1,
        help="Number of loop subdivision iterations (default: 1)",
    )
    p_skel_single.add_argument(
        "--neuroglancer",
        action="store_true",
        help="Input mesh is in neuroglancer format",
    )
    p_skel_single.set_defaults(func=cmd_skeletonize_single)

    # analyze
    p_analyze = subparsers.add_parser(
        "analyze", help="Analyze mesh geometry (volume, curvature, etc.)"
    )
    p_analyze.add_argument("config_path", help="Path to config directory")
    p_analyze.add_argument(
        "-n", "--num-workers", type=int, default=10, help="Number of dask workers"
    )
    p_analyze.set_defaults(func=cmd_analyze)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
