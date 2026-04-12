import numpy as np
from functools import cmp_to_key
import struct
import os
import trimesh
from collections import namedtuple


class Fragment:
    """A mesh fragment representing a chunk of a multi-LOD mesh.

    Stores vertices, faces, and the corresponding LOD 0 fragment positions
    for a single spatial chunk. Supports incremental updates as new
    sub-fragments are merged in.

    Parameters
    ----------
    vertices : numpy.ndarray
        Vertex positions with shape ``(N, 3)``.
    faces : numpy.ndarray
        Triangle face indices with shape ``(M, 3)``.
    lod_0_fragment_pos : list
        List of LOD 0 fragment grid positions associated with this fragment.
    """

    def __init__(self, vertices, faces, lod_0_fragment_pos):
        self.vertices = vertices
        self.faces = faces
        self.lod_0_fragment_pos = lod_0_fragment_pos

    def update_faces(self, new_faces):
        self.faces = np.vstack((self.faces, new_faces + len(self.vertices)))

    def update_vertices(self, new_vertices):
        self.vertices = np.vstack((self.vertices, new_vertices))

    def update_lod_0_fragment_pos(self, new_lod_0_fragment_pos):
        self.lod_0_fragment_pos.append(new_lod_0_fragment_pos)

    def update(self, new_vertices, new_faces, new_lod_0_fragment_pos):
        self.update_faces(new_faces)
        self.update_vertices(new_vertices)
        self.update_lod_0_fragment_pos(new_lod_0_fragment_pos)


CompressedFragment = namedtuple(
    "CompressedFragment", ["draco_bytes", "position", "offset", "lod_0_positions"]
)


def unpack_and_remove(datatype, num_elements, file_content):
    """Unpack values from the front of a binary buffer and return the remainder.

    Parameters
    ----------
    datatype : str
        A single-character ``struct`` format code (e.g. ``'I'``, ``'f'``).
    num_elements : int
        Number of elements to unpack.
    file_content : bytes
        Binary buffer to read from.

    Returns
    -------
    value : int, float, or numpy.ndarray
        The unpacked value (scalar when ``num_elements == 1``, otherwise an
        array).
    file_content : bytes
        The remaining bytes after the consumed portion.
    """
    datatype = datatype * num_elements
    output = struct.unpack(datatype, file_content[0 : 4 * num_elements])
    file_content = file_content[4 * num_elements :]
    if num_elements == 1:
        return output[0], file_content
    else:
        return np.array(output), file_content


def mesh_loader(filepath):
    """Load a mesh from disk, supporting standard formats and ngmesh.

    Files with no extension, ``.ngmesh``, or ``.ng`` are loaded as
    Neuroglancer binary meshes. All other extensions are delegated to
    ``trimesh.load``.

    Parameters
    ----------
    filepath : str
        Path to the mesh file.

    Returns
    -------
    vertices : numpy.ndarray or None
        Vertex positions with shape ``(N, 3)``, or ``None`` if the file
        does not exist or contains no mesh geometry.
    faces : numpy.ndarray or None
        Triangle face indices with shape ``(M, 3)``, or ``None``.
    """

    def _load_ngmesh(filepath):
        with open(filepath, mode="rb") as file:
            file_content = file.read()

        num_vertices, file_content = unpack_and_remove("I", 1, file_content)
        vertices, file_content = unpack_and_remove("f", 3 * num_vertices, file_content)
        num_faces = int(len(file_content) / 12)
        faces, file_content = unpack_and_remove("I", 3 * num_faces, file_content)

        vertices = vertices.reshape(-1, 3)
        faces = faces.reshape(-1, 3)

        return vertices, faces

    vertices = None
    faces = None

    if not os.path.isfile(filepath):
        return vertices, faces

    _, ext = os.path.splitext(filepath)
    if ext == "" or ext == ".ngmesh" or ext == ".ng":
        vertices, faces = _load_ngmesh(filepath)
    else:
        mesh = trimesh.load(filepath)
        if hasattr(mesh, "vertices"):
            vertices = mesh.vertices.copy()
            faces = mesh.faces.copy()
        del mesh

    return vertices, faces


def _cmp_zorder(lhs, rhs) -> bool:
    """Check if two values are in correct z-curve order."""

    def less_msb(x: int, y: int) -> bool:
        return x < y and x < (x ^ y)

    assert len(lhs) == len(rhs)
    msd = 2
    for dim in [1, 0]:
        if less_msb(lhs[msd] ^ rhs[msd], lhs[dim] ^ rhs[dim]):
            msd = dim
    return lhs[msd] - rhs[msd]


def zorder_fragments(fragments):
    """Sort fragments into Z-curve (Morton) order by their grid positions.

    Parameters
    ----------
    fragments : list[CompressedFragment]
        Fragments to sort; each must have a ``position`` attribute.

    Returns
    -------
    list[CompressedFragment]
        The same fragments in Z-curve order.
    """
    fragments, _ = zip(
        *sorted(
            zip(fragments, [fragment.position for fragment in fragments]),
            key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1])),
        )
    )
    return list(fragments)


def rewrite_index_with_empty_fragments(path, current_lod_fragments):
    """Rewrite an existing index file, inserting empty fragments for completeness.

    Neuroglancer requires that every parent fragment at a coarser LOD has all
    of its child fragments present (even if empty) so that LOD transitions
    work correctly. This function reads the current ``.index`` file, computes
    which child fragments are missing, inserts zero-length placeholders, and
    writes the updated index back.

    Parameters
    ----------
    path : str
        Base path for the mesh (without ``.index`` suffix). The index file is
        expected at ``path + ".index"``.
    current_lod_fragments : list[CompressedFragment]
        Newly created fragments for the next LOD level to be appended.
    """

    with open(f"{path}.index", mode="rb") as file:
        file_content = file.read()

    chunk_shape, file_content = unpack_and_remove("f", 3, file_content)
    grid_origin, file_content = unpack_and_remove("f", 3, file_content)
    num_lods, file_content = unpack_and_remove("I", 1, file_content)
    lod_scales, file_content = unpack_and_remove("f", num_lods, file_content)
    vertex_offsets, file_content = unpack_and_remove("f", num_lods * 3, file_content)

    num_fragments_per_lod, file_content = unpack_and_remove("I", num_lods, file_content)
    if type(num_fragments_per_lod) == int:
        num_fragments_per_lod = np.array([num_fragments_per_lod])

    all_current_fragment_positions = []
    all_current_fragment_offsets = []

    for lod in range(num_lods):
        fragment_positions, file_content = unpack_and_remove(
            "I", num_fragments_per_lod[lod] * 3, file_content
        )
        fragment_positions = fragment_positions.reshape((3, -1)).T
        fragment_offsets, file_content = unpack_and_remove(
            "I", num_fragments_per_lod[lod], file_content
        )
        if type(fragment_offsets) == int:
            fragment_offsets = np.array([fragment_offsets])
        all_current_fragment_positions.append(fragment_positions.astype(int))
        all_current_fragment_offsets.append(fragment_offsets.tolist())

    current_lod = num_lods
    num_lods += 1
    all_current_fragment_positions.append(
        np.asarray([fragment.position for fragment in current_lod_fragments]).astype(int)
    )
    all_current_fragment_offsets.append(
        [fragment.offset for fragment in current_lod_fragments]
    )

    all_missing_fragment_positions = []
    for lod in range(num_lods):
        all_required_fragment_positions = set()

        if lod == current_lod:
            for lower_lod in range(lod):
                all_required_fragment_positions_np = np.unique(
                    all_current_fragment_positions[lower_lod] // 2 ** (lod - lower_lod),
                    axis=0,
                ).astype(int)
                all_required_fragment_positions.update(
                    set(map(tuple, all_required_fragment_positions_np))
                )
        else:
            # For each new LOD fragment at position p, ALL children at
            # LOD `lod` must exist (even if empty) so neuroglancer can
            # properly replace the parent with its children.  Enumerate
            # every child position from p*scale to (p+1)*scale - 1.
            for fragment in current_lod_fragments:
                scale = 2 ** (current_lod - lod)
                base = (np.asarray(fragment.position) * scale).astype(int)
                for dx in range(scale):
                    for dy in range(scale):
                        for dz in range(scale):
                            all_required_fragment_positions.add(
                                (base[0] + dx, base[1] + dy, base[2] + dz)
                            )
        current_missing_fragment_positions = all_required_fragment_positions - set(
            map(tuple, all_current_fragment_positions[lod])
        )
        all_missing_fragment_positions.append(current_missing_fragment_positions)

    num_fragments_per_lod = []
    all_fragment_positions = []
    all_fragment_offsets = []
    for lod in range(num_lods):
        if len(all_missing_fragment_positions[lod]) > 0:
            lod_fragment_positions = list(all_missing_fragment_positions[lod]) + list(
                all_current_fragment_positions[lod]
            )
            lod_fragment_offsets = (
                list(np.zeros(len(all_missing_fragment_positions[lod])))
                + all_current_fragment_offsets[lod]
            )
        else:
            lod_fragment_positions = all_current_fragment_positions[lod]
            lod_fragment_offsets = all_current_fragment_offsets[lod]

        lod_fragment_offsets, lod_fragment_positions = zip(
            *sorted(
                zip(lod_fragment_offsets, lod_fragment_positions),
                key=cmp_to_key(lambda x, y: _cmp_zorder(x[1], y[1])),
            )
        )
        all_fragment_positions.append(lod_fragment_positions)
        all_fragment_offsets.append(lod_fragment_offsets)
        num_fragments_per_lod.append(len(all_fragment_offsets[lod]))

    num_fragments_per_lod = np.array(num_fragments_per_lod)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0.0, 0.0, 0.0] for _ in range(num_lods)])
    with open(f"{path}.index_with_empty_fragments", "ab") as f:
        f.write(chunk_shape.astype("<f").tobytes())
        f.write(grid_origin.astype("<f").tobytes())

        f.write(struct.pack("<I", num_lods))
        f.write(lod_scales.astype("<f").tobytes())
        f.write(vertex_offsets.astype("<f").tobytes(order="C"))

        f.write(num_fragments_per_lod.astype("<I").tobytes())

        for lod in range(num_lods):
            fragment_positions = np.array(all_fragment_positions[lod]).reshape(-1, 3)
            fragment_offsets = np.array(all_fragment_offsets[lod]).reshape(-1)

            f.write(fragment_positions.T.astype("<I").tobytes(order="C"))
            f.write(fragment_offsets.astype("<I").tobytes(order="C"))

    os.system(f"mv {path}.index_with_empty_fragments {path}.index")


def write_index_file(path, grid_origin, fragments, current_lod, lods, chunk_shape):
    """Write or update the ``.index`` file for a multi-LOD Draco mesh.

    If this is the first LOD or no index file exists yet, a new file is
    created. Otherwise the existing index is rewritten via
    ``rewrite_index_with_empty_fragments`` to incorporate the new LOD.

    Parameters
    ----------
    path : str
        Base path for the mesh (without extension).
    grid_origin : numpy.ndarray
        Origin of the fragment grid in model coordinates, shape ``(3,)``.
    fragments : list[CompressedFragment]
        Compressed mesh fragments for the current LOD.
    current_lod : int
        The LOD level being written.
    lods : list[int]
        All LOD levels that have been (or will be) generated.
    chunk_shape : numpy.ndarray
        Size of a single LOD 0 chunk in model coordinates, shape ``(3,)``.
    """
    lods = [lod for lod in lods if lod <= current_lod]

    num_lods = len(lods)
    lod_scales = np.array([2**i for i in range(num_lods)])
    vertex_offsets = np.array([[0.0, 0.0, 0.0] for _ in range(num_lods)])
    num_fragments_per_lod = np.array([len(fragments)])
    if current_lod == lods[0] or not os.path.exists(f"{path}.index"):
        with open(f"{path}.index", "wb") as f:
            f.write(chunk_shape.astype("<f").tobytes())
            f.write(grid_origin.astype("<f").tobytes())
            f.write(struct.pack("<I", num_lods))
            f.write(lod_scales.astype("<f").tobytes())
            f.write(vertex_offsets.astype("<f").tobytes(order="C"))
            f.write(num_fragments_per_lod.astype("<I").tobytes())
            f.write(
                np.asarray([fragment.position for fragment in fragments])
                .T.astype("<I")
                .tobytes(order="C")
            )
            f.write(
                np.asarray([fragment.offset for fragment in fragments])
                .astype("<I")
                .tobytes(order="C")
            )
    else:
        rewrite_index_with_empty_fragments(path, fragments)


def write_mesh_file(path, fragments):
    """Append Draco-encoded fragment bytes to a mesh file on disk.

    After writing, each fragment's ``draco_bytes`` is cleared (set to
    ``None``) to free memory, while the positional metadata is preserved.

    Parameters
    ----------
    path : str
        File path to write (opened in append mode).
    fragments : list[CompressedFragment]
        Fragments whose ``draco_bytes`` will be written sequentially.

    Returns
    -------
    list[CompressedFragment]
        The same fragments with ``draco_bytes`` set to ``None``.
    """
    with open(path, "ab") as f:
        for idx, fragment in enumerate(fragments):
            f.write(fragment.draco_bytes)
            fragments[idx] = CompressedFragment(
                None, fragment.position, fragment.offset, fragment.lod_0_positions
            )
    return fragments


def write_mesh_files(
    mesh_directory, object_id, grid_origin, fragments, current_lod, lods, chunk_shape
):
    """Write the mesh data and index files for a single segment.

    Fragments are sorted into Z-curve order, their Draco bytes are appended
    to the mesh file, and the index file is created or updated.

    Parameters
    ----------
    mesh_directory : str
        Directory where mesh files are stored.
    object_id : str
        Segment identifier used as the file name.
    grid_origin : numpy.ndarray
        Origin of the fragment grid in model coordinates, shape ``(3,)``.
    fragments : list[CompressedFragment]
        Compressed mesh fragments for the current LOD.
    current_lod : int
        The LOD level being written.
    lods : list[int]
        All LOD levels that have been (or will be) generated.
    chunk_shape : numpy.ndarray
        Size of a single LOD 0 chunk in model coordinates, shape ``(3,)``.
    """
    path = mesh_directory + "/" + object_id
    if len(fragments) > 0:
        fragments = zorder_fragments(fragments)
        fragments = write_mesh_file(path, fragments)
        write_index_file(path, grid_origin, fragments, current_lod, lods, chunk_shape)
