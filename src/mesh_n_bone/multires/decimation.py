import os
import numpy as np
import trimesh
import pyfqmr

from mesh_n_bone.util import mesh_io, dask_util


def pyfqmr_decimate(id, lod, input_path, output_path, ext, decimation_factor, aggressiveness):
    """Decimate a single mesh using pyfqmr and export the result as PLY.

    Loads a mesh from ``input_path``, reduces the face count by
    ``decimation_factor ** lod`` (with a minimum of 4 faces), and writes the
    simplified mesh to ``output_path/s{lod}/{id}.ply``.

    Parameters
    ----------
    id : str
        Identifier of the mesh segment (used for file naming).
    lod : int
        Level of detail. Higher values produce coarser meshes.
    input_path : str
        Directory containing the source mesh files.
    output_path : str
        Root directory for decimated mesh output. Results are written into
        a ``s{lod}`` subdirectory.
    ext : str
        File extension of the source mesh (e.g., ``".ply"``).
    decimation_factor : int
        Factor by which the face count is divided at each LOD level.
    aggressiveness : float
        Controls how aggressively pyfqmr simplifies the mesh. Higher values
        produce faster but lower-quality decimation.
    """
    vertices, faces = mesh_io.mesh_loader(f"{input_path}/{id}{ext}")
    desired_faces = max(len(faces) // (decimation_factor**lod), 1000)
    mesh_simplifier = pyfqmr.Simplify()
    mesh_simplifier.setMesh(vertices, faces)
    del vertices, faces
    mesh_simplifier.simplify_mesh(
        target_count=desired_faces,
        aggressiveness=aggressiveness,
        preserve_border=False,
        verbose=False,
    )
    vertices, faces, _ = mesh_simplifier.getMesh()
    del mesh_simplifier

    mesh = trimesh.Trimesh(vertices, faces)
    del vertices, faces
    _ = mesh.export(f"{output_path}/s{lod}/{id}.ply")


def generate_decimated_meshes(
    input_path, output_path, lods, ids, ext, decimation_factor, aggressiveness, num_workers,
):
    """Generate decimated meshes for all segment IDs across all LOD levels.

    For LOD 0 a symbolic link to the original ``input_path`` is created
    instead of copying files. For LOD >= 1 each mesh is decimated via
    ``pyfqmr_decimate`` and the work is distributed across Dask workers.

    Parameters
    ----------
    input_path : str
        Directory containing the original (LOD 0) mesh files.
    output_path : str
        Root output directory. Decimated meshes are placed under
        ``{output_path}/mesh_lods/s{lod}/``.
    lods : list of int
        LOD levels to generate (e.g., ``[0, 1, 2]``).
    ids : list of str
        Segment identifiers to process.
    ext : str
        File extension of the source meshes (e.g., ``".ply"``).
    decimation_factor : int
        Factor by which the face count is divided at each LOD level.
    aggressiveness : float
        pyfqmr aggressiveness parameter forwarded to ``pyfqmr_decimate``.
    num_workers : int
        Number of Dask workers to use for parallel processing.
    """
    variable_args_list = []
    fixed_args_list = [
        input_path,
        f"{output_path}/mesh_lods",
        ext,
        decimation_factor,
        aggressiveness,
    ]
    for current_lod in lods:
        if current_lod == 0:
            os.makedirs(f"{output_path}/mesh_lods/", exist_ok=True)
            if not os.path.exists(f"{output_path}/mesh_lods/s0"):
                os.system(
                    f"ln -s {os.path.abspath(input_path)}/ {os.path.abspath(output_path)}/mesh_lods/s0"
                )
        else:
            os.makedirs(f"{output_path}/mesh_lods/s{current_lod}", exist_ok=True)
            for id in ids:
                variable_args_list.append((id, current_lod))
    dask_util.compute_bag(
        pyfqmr_decimate,
        f"{output_path}/variable_args_to_decimate.npy",
        variable_args_list,
        fixed_args_list,
        num_workers,
    )


def delete_decimated_mesh_files(output_path, lods, ids, num_workers):
    """Delete intermediate decimated mesh files created during processing.

    For LOD 0 the symbolic link ``{output_path}/mesh_lods/s0`` is unlinked.
    For LOD >= 1 the individual ``.ply`` files are removed in parallel via
    Dask.

    Parameters
    ----------
    output_path : str
        Root output directory whose ``mesh_lods/`` subtree contains the
        decimated meshes to remove.
    lods : list of int
        LOD levels whose files should be deleted.
    ids : list of str
        Segment identifiers whose mesh files should be deleted.
    num_workers : int
        Number of Dask workers to use for parallel deletion.

    Raises
    ------
    ValueError
        If ``{output_path}/mesh_lods/s0`` exists but is not a symbolic link.
    """

    def delete_decimated_mesh_file(id, lod, output_path):
        os.remove(f"{output_path}/s{lod}/{id}.ply")

    variable_args_list = []
    fixed_args_list = [f"{output_path}/mesh_lods"]
    for current_lod in lods:
        if current_lod == 0:
            if not os.path.islink(f"{output_path}/mesh_lods/s0"):
                raise ValueError(
                    f"{output_path}/mesh_lods/s0 is not a link to the original meshes."
                )
            os.system(f"unlink {output_path}/mesh_lods/s0")
        else:
            for id in ids:
                variable_args_list.append((id, current_lod))
    dask_util.compute_bag(
        delete_decimated_mesh_file,
        f"{output_path}/variable_args_to_delete.npy",
        variable_args_list,
        fixed_args_list,
        num_workers,
    )
