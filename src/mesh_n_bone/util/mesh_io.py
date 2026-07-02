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
    vertex_lod_0_fragment_pos : numpy.ndarray, optional
        Per-vertex source fragment positions with shape ``(N, 3)``.
    """

    def __init__(
        self, vertices, faces, lod_0_fragment_pos, vertex_lod_0_fragment_pos=None
    ):
        self.vertices = vertices
        self.faces = faces
        self.lod_0_fragment_pos = lod_0_fragment_pos
        if vertex_lod_0_fragment_pos is None:
            self.vertex_lod_0_fragment_pos = self._per_vertex_fragment_pos(
                vertices, lod_0_fragment_pos
            )
        else:
            vertex_lod_0_fragment_pos = np.asarray(
                vertex_lod_0_fragment_pos, dtype=np.int64
            )
            if vertex_lod_0_fragment_pos.shape != (len(vertices), 3):
                raise ValueError(
                    "vertex_lod_0_fragment_pos must have shape "
                    f"({len(vertices)}, 3)"
                )
            self.vertex_lod_0_fragment_pos = vertex_lod_0_fragment_pos

    @staticmethod
    def _per_vertex_fragment_pos(vertices, lod_0_fragment_pos):
        positions = np.asarray(lod_0_fragment_pos, dtype=np.int64)
        if positions.ndim == 1:
            position = positions
        else:
            position = positions[-1]
        return np.repeat(position.reshape(1, 3), len(vertices), axis=0)

    def update_faces(self, new_faces):
        self.faces = np.vstack((self.faces, new_faces + len(self.vertices)))

    def update_vertices(self, new_vertices):
        self.vertices = np.vstack((self.vertices, new_vertices))

    def update_lod_0_fragment_pos(self, new_lod_0_fragment_pos):
        self.lod_0_fragment_pos.append(new_lod_0_fragment_pos)

    def update_vertex_lod_0_fragment_pos(self, new_vertices, new_lod_0_fragment_pos):
        new_vertex_lod_0_fragment_pos = self._per_vertex_fragment_pos(
            new_vertices, new_lod_0_fragment_pos
        )
        self.vertex_lod_0_fragment_pos = np.vstack(
            (self.vertex_lod_0_fragment_pos, new_vertex_lod_0_fragment_pos)
        )

    def update(self, new_vertices, new_faces, new_lod_0_fragment_pos):
        self.update_faces(new_faces)
        self.update_vertices(new_vertices)
        self.update_lod_0_fragment_pos(new_lod_0_fragment_pos)
        self.update_vertex_lod_0_fragment_pos(new_vertices, new_lod_0_fragment_pos)


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
    """Append a new LOD's fragments and apply descent-based minimum empties.

    For an octree to render correctly, every real (non-empty) parent
    fragment must have all 8 sub-octants listed at the next finer LOD —
    either as real fragments or as ``offset=0`` empty placeholders.
    Missing children make NG's "fall back to parent" rule kick in,
    which over-renders the parent in regions the finer LOD doesn't
    cover.

    This function walks the octree top-down from the current top LOD:
    a position is *reachable* if it's the root, or if it's a sub-octant
    of a reachable REAL position at the LOD above. Reachable positions
    that are also real keep their fragment data; reachable positions
    with no real fragment get a 0-byte empty placeholder. Real
    fragments NOT in the reachable set are *orphans* (no real-parent
    chain to the root) — NG can never traverse to them, so they're
    dropped from both the index and the data file. This matches the
    hemibrain layout: only emit what NG can actually use.

    Both the ``.index`` file and the mesh data file are rewritten in
    place. The data file is reordered to z-order per LOD with empties
    contributing zero bytes, so cumulative offsets line up.

    Parameters
    ----------
    path : str
        Base path for the mesh (without ``.index`` suffix). The index
        file is expected at ``path + ".index"`` and the data file at
        ``path``.
    current_lod_fragments : list[CompressedFragment]
        Newly created fragments for the next LOD level to be appended.
    """

    with open(f"{path}.index", mode="rb") as file:
        idx_bytes = file.read()
    with open(path, mode="rb") as file:
        data_blob = file.read()

    o = 0
    chunk_shape = np.frombuffer(idx_bytes, dtype="<f4", count=3, offset=o); o += 12
    grid_origin = np.frombuffer(idx_bytes, dtype="<f4", count=3, offset=o); o += 12
    num_existing_lods = struct.unpack_from("<I", idx_bytes, o)[0]; o += 4
    o += 4 * num_existing_lods            # skip lod_scales
    o += 12 * num_existing_lods           # skip vertex_offsets
    nfpl_existing = list(struct.unpack_from(
        f"<{num_existing_lods}I", idx_bytes, o
    )); o += 4 * num_existing_lods

    existing_per_lod = []  # list of (positions ndarray, sizes ndarray) preserving order
    for lod in range(num_existing_lods):
        n = nfpl_existing[lod]
        pos = np.frombuffer(
            idx_bytes, dtype="<u4", count=n * 3, offset=o
        ).reshape(3, n).T.copy()
        o += 12 * n
        sz = np.frombuffer(
            idx_bytes, dtype="<u4", count=n, offset=o
        ).copy()
        o += 4 * n
        existing_per_lod.append((pos, sz))

    # Append the new LOD's fragments.
    new_lod_positions = np.asarray(
        [fragment.position for fragment in current_lod_fragments], dtype=np.uint32
    ).reshape(-1, 3)
    new_lod_sizes = np.asarray(
        [fragment.offset for fragment in current_lod_fragments], dtype=np.uint32
    ).reshape(-1)
    existing_per_lod.append((new_lod_positions, new_lod_sizes))
    num_lods = num_existing_lods + 1
    top = num_lods - 1

    # Identify REAL (non-empty) positions at every LOD. Empties from
    # prior incremental writes are filtered out — descent recomputes
    # them from the current real footprint.
    real_pos_per_lod = [
        set(map(tuple, p[s > 0].tolist())) for (p, s) in existing_per_lod
    ]

    # Reachability is the union of two passes:
    #   1. Top-down descent through REAL parents, which adds empty
    #      placeholders under each real parent so NG's octree carve-out
    #      stays complete (this was the only pass before).
    #   2. Bottom-up ancestor walk from every REAL position, which
    #      keeps real leaves listed even when an intermediate LOD is
    #      empty in their cell. Independently-meshed LODs disagree:
    #      mode-downsampling can drop a thin edge at LOD k+1 that still
    #      exists at LOD k. Without the bottom-up pass, the real LOD-k
    #      leaf is dropped as an "orphan" (no real parent chain) and
    #      NG can't reach it — the user sees a chunk-shaped hole at
    #      the finest LOD. With it, the empty intermediates are listed
    #      as 0-byte placeholders and NG's fall-back-to-parent rule
    #      lets it traverse down to the real leaf.
    reachable = [set() for _ in range(num_lods)]
    if real_pos_per_lod[top]:
        reachable[top] = real_pos_per_lod[top].copy()
        for k in range(top - 1, -1, -1):
            for (X, Y, Z) in reachable[k + 1] & real_pos_per_lod[k + 1]:
                for a in (0, 1):
                    for b in (0, 1):
                        for c in (0, 1):
                            reachable[k].add((2 * X + a, 2 * Y + b, 2 * Z + c))
    for k in range(num_lods):
        for (X, Y, Z) in real_pos_per_lod[k]:
            reachable[k].add((X, Y, Z))
            x, y, z = X, Y, Z
            for kk in range(k + 1, num_lods):
                x, y, z = x // 2, y // 2, z // 2
                reachable[kk].add((x, y, z))

    # Walk the existing data blob LOD-by-LOD in its current listed
    # order, extracting bytes for reachable real fragments. Drop bytes
    # for orphan (real-but-unreachable) fragments; ignore empties
    # (they contribute zero bytes anyway).
    new_data_segments = []
    new_positions_per_lod = []
    new_sizes_per_lod = []
    byte_off = 0
    for lod in range(num_lods):
        pos, sz = existing_per_lod[lod]
        kept = []  # (position_tuple, size, data_bytes)
        for i in range(len(pos)):
            s = int(sz[i])
            p = tuple(pos[i].tolist())
            if s > 0 and p in reachable[lod]:
                kept.append((p, s, data_blob[byte_off : byte_off + s]))
            byte_off += s
        # Add empties at sub-octant positions reachable but lacking a
        # real fragment — keeps NG's octree carve-out complete under
        # each real parent.
        real_kept = {p for (p, _, _) in kept}
        for p in reachable[lod]:
            if p not in real_kept:
                kept.append((p, 0, b""))
        # Z-curve order across the combined real + empty list.
        kept.sort(key=cmp_to_key(lambda a, b: _cmp_zorder(a[0], b[0])))
        new_positions_per_lod.append(
            np.array([p for (p, _, _) in kept], dtype=np.uint32).reshape(-1, 3)
        )
        new_sizes_per_lod.append(
            np.array([s for (_, s, _) in kept], dtype=np.uint32)
        )
        for (_, _, b) in kept:
            if b:
                new_data_segments.append(b)

    nfpl_new = np.array([len(p) for p in new_positions_per_lod], dtype=np.uint32)
    lod_scales = np.array([2 ** i for i in range(num_lods)], dtype=np.float32)
    vertex_offsets = np.zeros((num_lods, 3), dtype=np.float32)

    tmp_index_path = f"{path}.index_tmp"
    with open(tmp_index_path, "wb") as f:
        f.write(chunk_shape.astype("<f4").tobytes())
        f.write(grid_origin.astype("<f4").tobytes())
        f.write(struct.pack("<I", num_lods))
        f.write(lod_scales.tobytes())
        f.write(vertex_offsets.tobytes(order="C"))
        f.write(nfpl_new.tobytes())
        for pos, sz in zip(new_positions_per_lod, new_sizes_per_lod):
            f.write(pos.T.astype("<u4").tobytes(order="C"))
            f.write(sz.astype("<u4").tobytes())

    tmp_data_path = f"{path}.tmp"
    with open(tmp_data_path, "wb") as f:
        for seg in new_data_segments:
            f.write(seg)

    os.replace(tmp_index_path, f"{path}.index")
    os.replace(tmp_data_path, path)


def write_index_file(
    path, grid_origin, fragments, current_lod, lods, chunk_shape,
):
    """Write or update the ``.index`` file for a multi-LOD Draco mesh.

    If this is the first LOD or no index file exists yet, a new file is
    created. Otherwise the existing index is rewritten via
    ``rewrite_index_with_empty_fragments`` to incorporate the new LOD
    and apply descent-based minimum empties.

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
    mesh_directory, object_id, grid_origin, fragments, current_lod, lods, chunk_shape,
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
        write_index_file(
            path, grid_origin, fragments, current_lod, lods, chunk_shape,
        )
