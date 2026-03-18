import os
import numpy as np
import trimesh
import pyfqmr

from mesh_n_bone.util import mesh_io, dask_util


def pyfqmr_decimate(id, lod, input_path, output_path, ext, decimation_factor, aggressiveness):
    """Decimate a single mesh using pyfqmr."""
    vertices, faces = mesh_io.mesh_loader(f"{input_path}/{id}{ext}")
    desired_faces = max(len(faces) // (decimation_factor**lod), 4)
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
    """Generate decimated meshes for all ids over all lods."""
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
    """Delete intermediate decimated mesh files."""

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
