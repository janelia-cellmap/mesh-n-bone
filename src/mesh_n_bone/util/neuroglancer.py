import os
import io
import json
import struct
import numpy as np


def write_info_file(path):
    """Write info file for neuroglancer multilod draco meshes."""
    with open(f"{path}/info", "w") as f:
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 10,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 1,
            "segment_properties": "segment_properties",
        }
        json.dump(info, f)


def write_segment_properties_file(path):
    """Create segment properties dir/file so that all meshes are selectable
    based on id in neuroglancer."""

    def list_index_ids(path):
        with os.scandir(path) as it:
            return [
                entry.name.rsplit(".", 1)[0]
                for entry in it
                if entry.is_file() and entry.name.endswith(".index")
            ]

    segment_properties_directory = f"{path}/segment_properties"
    if not os.path.exists(segment_properties_directory):
        os.makedirs(segment_properties_directory)

    ids = list_index_ids(path)
    ids.sort(key=int)

    with open(f"{segment_properties_directory}/info", "w") as f:
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": ids,
                "properties": [
                    {"id": "label", "type": "label", "values": [""] * len(ids)}
                ],
            },
        }
        json.dump(info, f)


# -- Legacy neuroglancer mesh format (from igneous-daskified) --

def _write_ngmesh(vertices_xyz, faces, f_out):
    """Write vertices and faces to a binary file in ngmesh format."""
    f_out.write(np.uint32(len(vertices_xyz)))
    f_out.write(vertices_xyz.astype(np.float32, "C", copy=False))
    f_out.write(faces.astype(np.uint32, "C", copy=False))


def write_ngmesh(vertices_xyz, faces, f_out=None):
    """Write mesh in ngmesh format."""
    if f_out is None:
        with io.BytesIO() as bio:
            _write_ngmesh(vertices_xyz, faces, bio)
            return bio.getvalue()
    elif isinstance(f_out, str):
        with open(f_out, "wb") as f:
            _write_ngmesh(vertices_xyz, faces, f)
    else:
        _write_ngmesh(vertices_xyz, faces, f_out)


def write_ngmesh_metadata(meshdir):
    """Write metadata for legacy neuroglancer mesh format."""
    mesh_ids = [f.split(":0")[0] for f in os.listdir(meshdir) if ":0" in f]
    info = {
        "@type": "neuroglancer_legacy_mesh",
        "segment_properties": "./segment_properties",
    }

    with open(meshdir + "/info", "w") as f:
        f.write(json.dumps(info))

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [mesh_id for mesh_id in mesh_ids],
            "properties": [
                {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
            ],
        },
    }
    os.makedirs(meshdir + "/segment_properties", exist_ok=True)
    with open(meshdir + "/segment_properties/info", "w") as f:
        f.write(json.dumps(segment_properties))


# -- Single-resolution multires format (from igneous-daskified) --

def _to_stored_model_space(
    vertices, chunk_shape, grid_origin, fragment_positions,
    vertex_offsets, lod, vertex_quantization_bits,
):
    """Inverse of from_stored_model_space."""
    import fastremap

    vertices = vertices.astype(np.float32, copy=False)
    quant_factor = (2**vertex_quantization_bits) - 1

    stored_model = vertices - grid_origin - vertex_offsets
    stored_model /= chunk_shape * (2**lod)
    stored_model -= fragment_positions
    stored_model *= quant_factor
    stored_model = np.round(stored_model, out=stored_model)
    stored_model = np.clip(stored_model, 0, quant_factor, out=stored_model)

    dtype = fastremap.fit_dtype(np.uint64, value=quant_factor)
    return stored_model.astype(dtype)


def write_singleres_index_file(
    path, grid_origin, fragment_positions, fragment_offsets,
    current_lod, lods, chunk_shape,
):
    """Write index file for single-resolution multires format."""
    lods = [lod for lod in lods if lod <= current_lod]
    num_lods = len(lods)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0.0, 0.0, 0.0] for _ in range(num_lods)])
    num_fragments_per_lod = np.array([len(fragment_positions)])

    blocks = [
        chunk_shape.astype("<f").tobytes(),
        grid_origin.astype("<f").tobytes(),
        struct.pack("<I", num_lods),
        lod_scales.astype("<f").tobytes(),
        vertex_offsets.astype("<f").tobytes(order="C"),
        num_fragments_per_lod.astype("<I").tobytes(),
        np.asarray(fragment_positions).T.astype("<I").tobytes(order="C"),
        np.asarray(fragment_offsets).astype("<I").tobytes(order="C"),
    ]
    with open(f"{path}", "wb") as f:
        f.writelines(blocks)


def write_singleres_multires_files(
    vertices_xyz, faces, path, vertex_quantization_bits=10, draco_compression_level=10
):
    """Write single-resolution mesh in multires draco format."""
    import DracoPy

    grid_origin = np.min(vertices_xyz, axis=0)
    chunk_shape = np.max(vertices_xyz, axis=0) - grid_origin
    vertices_xyz = _to_stored_model_space(
        vertices_xyz,
        chunk_shape=chunk_shape,
        grid_origin=grid_origin,
        fragment_positions=np.array([[0, 0, 0]]),
        vertex_offsets=np.array([0, 0, 0]),
        lod=0,
        vertex_quantization_bits=vertex_quantization_bits,
    )

    quantization_origin = np.min(vertices_xyz, axis=0)
    quantization_range = np.max(vertices_xyz, axis=0) - quantization_origin
    quantization_range = np.max(quantization_range)
    try:
        res = DracoPy.encode(
            vertices_xyz,
            faces,
            quantization_bits=vertex_quantization_bits,
            quantization_range=quantization_range,
            compression_level=draco_compression_level,
            quantization_origin=quantization_origin,
        )
    except Exception:
        res = b""

    with open(path, "wb") as f:
        f.write(res)

    write_singleres_index_file(
        f"{path}.index",
        grid_origin=grid_origin,
        fragment_positions=[[0, 0, 0]],
        fragment_offsets=[len(res)],
        current_lod=0,
        lods=[0],
        chunk_shape=chunk_shape,
    )
    return res, vertices_xyz


def write_singleres_multires_metadata(meshdir):
    """Write metadata for single-resolution multires format."""
    mesh_ids = [f.split(".index")[0] for f in os.listdir(meshdir) if ".index" in f]
    info = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": 10,
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "lod_scale_multiplier": 1,
        "segment_properties": "segment_properties",
    }
    with open(f"{meshdir}/info", "w") as f:
        json.dump(info, f)

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": [mesh_id for mesh_id in mesh_ids],
            "properties": [
                {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
            ],
        },
    }
    os.makedirs(meshdir + "/segment_properties", exist_ok=True)
    with open(meshdir + "/segment_properties/info", "w") as f:
        f.write(json.dumps(segment_properties))


# -- Neuroglancer annotations (from igneous-daskified) --

def write_precomputed_annotations(
    output_directory, annotation_type, ids, coords,
    properties_dict, relationships_dict=None,
):
    """Write neuroglancer precomputed annotations."""
    os.makedirs(f"{output_directory}/spatial0", exist_ok=True)
    os.makedirs(f"{output_directory}/relationships", exist_ok=True)

    if annotation_type == "line":
        coords_to_write = 6
    else:
        coords_to_write = 3

    properties_values = [v for v in properties_dict.values()]
    for v in properties_values:
        assert len(v) == len(coords)

    with open(f"{output_directory}/spatial0/0_0_0", "wb") as outfile:
        total_count = len(coords)
        buf = struct.pack("<Q", total_count)
        flattened = np.column_stack((coords, *properties_values)).flatten()
        buf += struct.pack(
            f"<{(coords_to_write+len(properties_values))*total_count}f", *flattened
        )
        id_buf = struct.pack(f"<{total_count}Q", *ids)
        buf += id_buf
        outfile.write(buf)

    if relationships_dict:
        for relationship_id, corresponding_indices in relationships_dict.items():
            with open(
                f"{output_directory}/relationships/{relationship_id}", "wb"
            ) as outfile:
                total_count = len(corresponding_indices)
                buf = struct.pack("<Q", total_count)
                flattened = np.column_stack(
                    (
                        coords[corresponding_indices],
                        *(v[corresponding_indices] for v in properties_values),
                    )
                ).flatten()
                buf += struct.pack(f"<{(coords_to_write+1)*total_count}f", *flattened)
                id_buf = struct.pack(
                    f"<{total_count}Q", *ids[corresponding_indices]
                )
                buf += id_buf
                outfile.write(buf)

    max_extents = coords.reshape((-1, 3)).max(axis=0) + 1
    max_extents = [int(max_extent) for max_extent in max_extents]
    info = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": {"x": [1, "nm"], "y": [1, "nm"], "z": [1, "nm"]},
        "by_id": {"key": "by_id"},
        "lower_bound": [0, 0, 0],
        "upper_bound": max_extents,
        "annotation_type": annotation_type,
        "properties": [
            {"id": key, "type": "float32", "description": key}
            for key in properties_dict.keys()
        ],
        "relationships": [{"id": "cells", "key": "relationships"}],
        "spatial": [
            {
                "chunk_size": max_extents,
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": 1,
            }
        ],
    }

    with open(f"{output_directory}/info", "w") as info_file:
        json.dump(info, info_file)
