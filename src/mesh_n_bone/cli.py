"""Unified CLI for mesh-n-bone: meshification, multiresolution, skeletonization, and analysis."""

import argparse
import os
import sys
import logging
import importlib

logger = logging.getLogger(__name__)


def _setup_pymeshlab_ld_path():
    """Ensure pymeshlab's shared libraries are on LD_LIBRARY_PATH."""
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
    """Set up execution directory and load config."""
    from mesh_n_bone.util.dask_util import setup_execution_directory
    from mesh_n_bone.config import read_generic_config

    execution_directory = setup_execution_directory(args.config_path, logger)
    logpath = f"{execution_directory}/output.log"
    run_config = read_generic_config(args.config_path)
    run_config["num_workers"] = args.num_workers
    return execution_directory, logpath, run_config


def cmd_meshify(args):
    """Run the meshify pipeline."""
    from mesh_n_bone.meshify.meshify import Meshify
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        meshify = Meshify(**run_config)
        meshify.get_meshes()


def cmd_multires(args):
    """Run the multiresolution mesh pipeline."""
    from mesh_n_bone.multires.multires import run_multires

    run_multires(args.config_path, args.num_workers)


def cmd_skeletonize(args):
    """Run the skeletonization pipeline."""
    from mesh_n_bone.skeletonize.skeletonize import Skeletonize
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        skeletonize = Skeletonize(**run_config)
        skeletonize.get_skeletons()


def cmd_skeletonize_single(args):
    """Run CGAL skeletonization on a single mesh file."""
    from mesh_n_bone.skeletonize.skeletonize import Skeletonize

    Skeletonize.cgal_skeletonize_mesh(
        args.input_file,
        args.output_file,
        base_loop_subdivision_iterations=args.subdivisions,
        neuroglancer_format=args.neuroglancer,
    )


def cmd_analyze(args):
    """Run mesh analysis pipeline."""
    from mesh_n_bone.analyze.analyze import AnalyzeMeshes
    from mesh_n_bone.util.logging import tee_streams

    execution_directory, logpath, run_config = _get_run_properties(args)
    with tee_streams(logpath):
        os.chdir(execution_directory)
        analyze = AnalyzeMeshes(**run_config)
        analyze.analyze()


def main():
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
    p_meshify.set_defaults(func=cmd_meshify)

    # multires
    p_multires = subparsers.add_parser(
        "multires", help="Create multiresolution neuroglancer meshes"
    )
    p_multires.add_argument("config_path", help="Path to config directory")
    p_multires.add_argument(
        "-n", "--num-workers", type=int, default=10, help="Number of dask workers"
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
