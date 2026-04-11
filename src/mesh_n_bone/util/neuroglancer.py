import os
import io
import json
import re
import struct
import numpy as np


def write_info_file(path):
    """Write the ``info`` JSON file for a Neuroglancer multi-LOD Draco mesh layer.

    The generated file uses 10-bit vertex quantization, an identity
    transform, and references a ``segment_properties`` sub-directory.

    Parameters
    ----------
    path : str
        Directory in which the ``info`` file is created.
    """
    with open(f"{path}/info", "w") as f:
        info = {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": 10,
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "lod_scale_multiplier": 1,
            "segment_properties": "segment_properties",
        }
        json.dump(info, f)


def _build_properties_from_csv(ids, csv_path, columns=None, id_column="Object ID"):
    """Build neuroglancer property definitions from a CSV file.

    Parameters
    ----------
    ids : list[str]
        Ordered segment IDs (strings) already discovered from index files.
    csv_path : str
        Path to a CSV with an ID column whose values correspond to
        segment IDs in the dataset.
    columns : list[str] or None
        If provided, only these column names (besides the ID column) are
        included as properties.  Otherwise every non-ID column is used.
    id_column : str
        Name of the CSV column containing segment IDs.  Defaults to
        ``"Object ID"``.

    Returns
    -------
    list[dict]
        Neuroglancer property definitions ready for the ``"properties"``
        array in the segment-properties info file.
    """
    import csv

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or id_column not in fieldnames:
            raise ValueError(
                f"CSV {csv_path} must have a '{id_column}' column"
            )

        all_columns = [c for c in fieldnames if c != id_column]
        if columns is not None:
            missing = set(columns) - set(all_columns)
            if missing:
                raise ValueError(
                    f"Columns {missing} not found in CSV. "
                    f"Available: {all_columns}"
                )
            use_columns = columns
        else:
            use_columns = all_columns

        # Build a lookup: csv id -> row values for selected columns
        csv_rows = {}
        for row in reader:
            csv_rows[row[id_column]] = {col: row[col] for col in use_columns}

    properties = []
    for col in use_columns:
        # Neuroglancer's query parser only allows [a-zA-Z][a-zA-Z0-9_]* in
        # field IDs for numeric constraints and tokenizes on spaces for
        # sort/filter expressions.  Strip unsupported characters.
        sanitized_id = re.sub(r"[^a-zA-Z0-9_]", "_", col)
        sanitized_id = re.sub(r"_+", "_", sanitized_id).strip("_")

        values_for_col = [csv_rows.get(seg_id, {}).get(col, "") for seg_id in ids]

        # Determine if this column is numeric
        numeric_values = []
        is_numeric = True
        for v in values_for_col:
            if v == "":
                numeric_values.append(0)
                continue
            try:
                numeric_values.append(float(v))
                continue
            except ValueError:
                pass
            is_numeric = False
            break

        if is_numeric:
            # Check if all values are integers
            all_int = all(
                v == "" or float(v) == int(float(v)) for v in values_for_col
            )
            if all_int:
                int_values = [int(nv) for nv in numeric_values]
                min_val = min(int_values) if int_values else 0
                max_val = max(int_values) if int_values else 0
                data_type = None
                if min_val >= 0:
                    if max_val <= 255:
                        data_type = "uint8"
                    elif max_val <= 65535:
                        data_type = "uint16"
                    elif max_val <= 4294967295:
                        data_type = "uint32"
                else:
                    if -128 <= min_val and max_val <= 127:
                        data_type = "int8"
                    elif -32768 <= min_val and max_val <= 32767:
                        data_type = "int16"
                    elif -2147483648 <= min_val and max_val <= 2147483647:
                        data_type = "int32"
                if data_type is not None:
                    properties.append({
                        "id": sanitized_id,
                        "type": "number",
                        "data_type": data_type,
                        "values": int_values,
                    })
                else:
                    # Values overflow int32/uint32 — use float32
                    properties.append({
                        "id": sanitized_id,
                        "type": "number",
                        "data_type": "float32",
                        "values": numeric_values,
                    })
            else:
                properties.append({
                    "id": sanitized_id,
                    "type": "number",
                    "data_type": "float32",
                    "values": numeric_values,
                })
        else:
            # Check if values form a small set of tags (with repeats)
            non_empty = [v for v in values_for_col if v != ""]
            unique_values = sorted(set(non_empty))
            if 1 < len(unique_values) <= 50 and len(unique_values) < len(non_empty):
                tag_indices = []
                for v in values_for_col:
                    if v == "":
                        tag_indices.append([])
                    else:
                        tag_indices.append([unique_values.index(v)])
                properties.append({
                    "id": sanitized_id,
                    "type": "tags",
                    "tags": unique_values,
                    "values": tag_indices,
                })
            else:
                # Use label type for the first string column, string for rest
                prop_type = "string"
                if not any(p["type"] in ("label", "string") for p in properties):
                    prop_type = "label"
                properties.append({
                    "id": sanitized_id,
                    "type": prop_type,
                    "values": values_for_col,
                })

    # Neuroglancer requires a label property to exist for selection to work.
    if not any(p["type"] == "label" for p in properties):
        properties.insert(0, {
            "id": "label", "type": "label", "values": [""] * len(ids),
        })

    return properties


def write_segment_properties_file(
    path, csv_path=None, csv_columns=None, csv_id_column="Object ID",
):
    """Create segment properties dir/file so that all meshes are selectable
    based on id in neuroglancer.

    Parameters
    ----------
    path : str
        Directory containing the ``.index`` files (e.g. ``output/multires``).
    csv_path : str or None
        Optional CSV file with an ID column and additional property
        columns.  When provided, the properties are written into the
        neuroglancer segment-properties info file.
    csv_columns : list[str] or None
        Subset of CSV columns to include.  ``None`` means all non-ID
        columns.
    csv_id_column : str
        Name of the CSV column containing segment IDs.  Defaults to
        ``"Object ID"``.
    """

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

    if csv_path is not None:
        properties = _build_properties_from_csv(
            ids, csv_path, csv_columns, id_column=csv_id_column,
        )
    else:
        properties = [
            {"id": "label", "type": "label", "values": [""] * len(ids)}
        ]

    with open(f"{segment_properties_directory}/info", "w") as f:
        info = {
            "@type": "neuroglancer_segment_properties",
            "inline": {
                "ids": ids,
                "properties": properties,
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
    """Write a mesh in Neuroglancer binary (ngmesh) format.

    The format consists of a ``uint32`` vertex count, followed by
    ``float32`` vertex positions, followed by ``uint32`` triangle indices.

    Parameters
    ----------
    vertices_xyz : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle face indices with shape ``(M, 3)``.
    f_out : str, file-like, or None
        Destination. If ``None``, the encoded bytes are returned. If a
        string, it is treated as a file path. Otherwise it must be a
        writable binary file object.

    Returns
    -------
    bytes or None
        The encoded mesh bytes when ``f_out`` is ``None``; otherwise
        ``None``.
    """
    if f_out is None:
        with io.BytesIO() as bio:
            _write_ngmesh(vertices_xyz, faces, bio)
            return bio.getvalue()
    elif isinstance(f_out, str):
        with open(f_out, "wb") as f:
            _write_ngmesh(vertices_xyz, faces, f)
    else:
        _write_ngmesh(vertices_xyz, faces, f_out)


def write_ngmesh_metadata(meshdir, csv_path=None, csv_columns=None, csv_id_column="Object ID"):
    """Write ``info`` and segment-properties files for the legacy mesh format.

    Discovers mesh IDs by scanning *meshdir* for files containing ``:0``
    in their name.

    Parameters
    ----------
    meshdir : str
        Directory containing the legacy mesh files.
    csv_path : str or None
        Optional CSV file for additional segment properties.  See
        ``_build_properties_from_csv`` for details.
    csv_columns : list[str] or None
        Subset of CSV columns to include.  ``None`` means all non-ID
        columns.
    csv_id_column : str
        Name of the CSV column containing segment IDs.  Defaults to
        ``"Object ID"``.
    """
    mesh_ids = [f.split(":0")[0] for f in os.listdir(meshdir) if ":0" in f]
    mesh_ids.sort(key=int)
    info = {
        "@type": "neuroglancer_legacy_mesh",
        "segment_properties": "./segment_properties",
    }

    with open(meshdir + "/info", "w") as f:
        f.write(json.dumps(info))

    if csv_path is not None:
        properties = _build_properties_from_csv(
            mesh_ids, csv_path, csv_columns, id_column=csv_id_column,
        )
    else:
        properties = [
            {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
        ]

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": mesh_ids,
            "properties": properties,
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
    """Write a ``.index`` file for the single-resolution multi-LOD Draco format.

    Parameters
    ----------
    path : str
        Output file path (typically ending in ``.index``).
    grid_origin : numpy.ndarray
        Origin of the fragment grid in model coordinates, shape ``(3,)``.
    fragment_positions : list[list[int]]
        Grid positions for each fragment, each a 3-element list.
    fragment_offsets : list[int]
        Byte offsets (sizes) for each fragment in the mesh data file.
    current_lod : int
        The LOD level being written.
    lods : list[int]
        All LOD levels present.
    chunk_shape : numpy.ndarray
        Size of a single LOD 0 chunk in model coordinates, shape ``(3,)``.
    """
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
    """Encode a single-resolution mesh as a multi-LOD Draco file with index.

    The mesh is quantized into stored-model-space, Draco-encoded, and
    written alongside a matching ``.index`` file so that Neuroglancer can
    load it as a single-LOD multi-resolution mesh.

    Parameters
    ----------
    vertices_xyz : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle face indices with shape ``(M, 3)``.
    path : str
        Output file path for the Draco-encoded mesh data.
    vertex_quantization_bits : int, optional
        Number of quantization bits per vertex coordinate.  Default is
        ``10``.
    draco_compression_level : int, optional
        Draco compression level (0--10).  Default is ``10``.

    Returns
    -------
    res : bytes
        The Draco-encoded mesh bytes.
    vertices_xyz : numpy.ndarray
        Quantized vertex positions in stored-model space.
    """
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


def write_singleres_multires_metadata(meshdir, csv_path=None, csv_columns=None, csv_id_column="Object ID"):
    """Write ``info`` and segment-properties files for single-resolution multi-LOD meshes.

    Discovers mesh IDs by scanning *meshdir* for ``.index`` files.

    Parameters
    ----------
    meshdir : str
        Directory containing the mesh and ``.index`` files.
    csv_path : str or None
        Optional CSV file for additional segment properties.  See
        ``_build_properties_from_csv`` for details.
    csv_columns : list[str] or None
        Subset of CSV columns to include.  ``None`` means all non-ID
        columns.
    csv_id_column : str
        Name of the CSV column containing segment IDs.  Defaults to
        ``"Object ID"``.
    """
    mesh_ids = [f.split(".index")[0] for f in os.listdir(meshdir) if ".index" in f]
    mesh_ids.sort(key=int)
    info = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": 10,
        "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
        "lod_scale_multiplier": 1,
        "segment_properties": "segment_properties",
    }
    with open(f"{meshdir}/info", "w") as f:
        json.dump(info, f)

    if csv_path is not None:
        properties = _build_properties_from_csv(
            mesh_ids, csv_path, csv_columns, id_column=csv_id_column,
        )
    else:
        properties = [
            {"id": "label", "type": "label", "values": [""] * len(mesh_ids)}
        ]

    segment_properties = {
        "@type": "neuroglancer_segment_properties",
        "inline": {
            "ids": mesh_ids,
            "properties": properties,
        },
    }
    os.makedirs(meshdir + "/segment_properties", exist_ok=True)
    with open(meshdir + "/segment_properties/info", "w") as f:
        f.write(json.dumps(segment_properties))


# -- Neuroglancer annotations (from igneous-daskified) --

def write_precomputed_annotations(
    output_directory, annotation_type, ids, coords,
    properties_dict, relationships_dict=None, coordinate_units="nm",
):
    """Write a Neuroglancer precomputed annotation layer to disk.

    Creates the ``spatial0/0_0_0`` binary file, optional relationship
    files, and an ``info`` JSON file describing the layer.

    Parameters
    ----------
    output_directory : str
        Directory to write the annotation layer into.
    annotation_type : str
        Neuroglancer annotation type, e.g. ``"point"`` or ``"line"``.
    ids : numpy.ndarray
        Unique annotation IDs, shape ``(N,)``, dtype ``uint64``.
    coords : numpy.ndarray
        Coordinates for each annotation. Shape ``(N, 3)`` for points or
        ``(N, 6)`` for lines (start + end).
    properties_dict : dict[str, numpy.ndarray]
        Mapping of property names to per-annotation ``float32`` value
        arrays, each of length *N*.
    relationships_dict : dict[str, numpy.ndarray] or None
        Optional mapping of relationship IDs to index arrays selecting a
        subset of annotations for each relationship.
    coordinate_units : str, optional
        Spatial unit string written into the ``info`` file.  Default is
        ``"nm"``.
    """
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
        "dimensions": {"x": [1, coordinate_units], "y": [1, coordinate_units], "z": [1, coordinate_units]},
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
