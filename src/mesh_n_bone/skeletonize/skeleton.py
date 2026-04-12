"""Custom skeleton data structure with graph operations, pruning, and neuroglancer I/O."""

from dataclasses import dataclass
import struct
import numpy as np
import os
import logging
from neuroglancer.skeleton import Skeleton as NeuroglancerSkeleton
import fastremap
import networkx as nx
from pybind11_rdp import rdp

logger = logging.getLogger(__name__)


@dataclass
class Source:
    """Minimal source descriptor for neuroglancer skeleton encoding.

    Provides the ``vertex_attributes`` list expected by
    ``neuroglancer.skeleton.Skeleton.encode``.
    """

    vertex_attributes = []


class CustomSkeleton:
    """Skeleton representation with vertices, edges, radii, and polyline branches.

    Supports graph-based operations such as pruning short branches,
    simplification via the Ramer-Douglas-Peucker algorithm, and
    reading/writing neuroglancer precomputed skeleton format.

    Parameters
    ----------
    vertices : list of tuple of float, optional
        Vertex positions as ``(x, y, z)`` tuples.
    edges : list of tuple of int, optional
        Edges as ``(start_index, end_index)`` pairs referencing *vertices*.
        If the elements are coordinate tuples instead of integer indices,
        they are automatically resolved to indices.
    radii : list of float or None, optional
        Per-vertex radius values.  ``None`` means no radii are stored.
    polylines : list of array-like, optional
        Ordered sequences of vertex positions representing each branch of
        the skeleton.  When empty, polylines are derived automatically
        from the graph structure.
    """

    def __init__(self, vertices=[], edges=[], radii=None, polylines=[]):
        self.vertices = []
        self.edges = []
        self.radii = []
        self.polylines = []

        self.add_vertices(vertices, radii=radii)
        self.add_edges(edges)
        if not polylines:
            g = self.skeleton_to_graph()
            polylines = self.get_polylines_positions_from_graph(g)
        self.add_polylines(polylines)

    def _get_vertex_index(self, vertex):
        if type(vertex) is not tuple:
            vertex = tuple(vertex)
        return self.vertices.index(tuple(vertex))

    def add_vertex(self, vertex, radius=None):
        """Append a single vertex to the skeleton.

        Parameters
        ----------
        vertex : tuple of float or array-like
            ``(x, y, z)`` position.  Converted to a tuple if necessary.
        radius : float or None, optional
            Radius at this vertex.  Only appended when not ``None``.
        """
        if type(vertex) is not tuple:
            vertex = tuple(vertex)
        self.vertices.append(vertex)
        if radius:
            self.radii.append(radius)

    def add_vertices(self, vertices, radii):
        """Append multiple vertices (and optional radii) to the skeleton.

        Parameters
        ----------
        vertices : list of tuple of float
            Vertex positions.
        radii : list of float or None
            Per-vertex radii.  Pass ``None`` to skip radius storage.
        """
        if radii:
            for vertex, radius in zip(vertices, radii):
                self.add_vertex(vertex, radius)
        else:
            for vertex in vertices:
                self.add_vertex(vertex)
        self.vertices = self.vertices

    def add_edge(self, edge):
        """Append a single edge to the skeleton.

        Parameters
        ----------
        edge : tuple
            Either a pair of integer vertex indices or a pair of coordinate
            tuples.  Coordinate tuples are resolved to indices via
            ``_get_vertex_index``.
        """
        if not isinstance(edge[0], (int, np.integer)):
            edge_start_id = self._get_vertex_index(edge[0])
            edge_end_id = self._get_vertex_index(edge[1])
            edge = (edge_start_id, edge_end_id)
        self.edges.append(edge)

    def add_edges(self, edges):
        """Append multiple edges to the skeleton.

        Parameters
        ----------
        edges : list of tuple
            See ``add_edge`` for accepted formats.
        """
        for edge in edges:
            self.add_edge(edge)

    def add_polylines(self, polylines):
        """Append multiple polylines to the skeleton.

        Parameters
        ----------
        polylines : list of array-like
            Each polyline is a sequence of ``(x, y, z)`` positions.
        """
        for polyline in polylines:
            self.add_polyline(polyline)

    def add_polyline(self, polyline):
        """Append a single polyline to the skeleton.

        Parameters
        ----------
        polyline : array-like
            Ordered sequence of ``(x, y, z)`` positions forming one branch.
        """
        self.polylines.append(polyline)

    def simplify(self, tolerance_nm=200):
        """Simplify skeleton polylines using the Ramer-Douglas-Peucker algorithm.

        Each polyline is simplified independently; the resulting skeleton
        retains only the vertices that survive the RDP decimation.

        Parameters
        ----------
        tolerance_nm : float, optional
            Maximum perpendicular distance (in nanometres) a vertex may
            deviate before it is kept.  Default is ``200``.

        Returns
        -------
        CustomSkeleton
            A new skeleton containing only the simplified vertices,
            edges, radii, and polylines.
        """
        vertices = []
        radii = []
        edges = []
        simplified_polylines = []
        for polyline in self.polylines:
            simplified_polyline = rdp(polyline, epsilon=tolerance_nm)
            for vertex in simplified_polyline:
                vertices.append(tuple(vertex))
                if self.radii:
                    radii.append(self.radii[self._get_vertex_index(tuple(vertex))])
            simplified_polylines.append(simplified_polyline)
            edges.extend(list(zip(simplified_polyline, simplified_polyline[1:])))

        return CustomSkeleton(vertices, edges, radii, simplified_polylines)

    @staticmethod
    def find_branchpoints_and_endpoints(graph):
        """Classify graph nodes as branch points or endpoints.

        Parameters
        ----------
        graph : networkx.Graph
            Skeleton graph whose nodes will be classified.

        Returns
        -------
        branchpoints : list
            Nodes with degree > 2.
        endpoints : list
            Nodes with degree <= 1.
        """
        branchpoints = []
        endpoints = []
        for node in graph.nodes:
            degree = graph.degree[node]
            if degree <= 1:
                endpoints.append(node)
            elif degree > 2:
                branchpoints.append(node)
        return branchpoints, endpoints

    @staticmethod
    def get_polyline_from_subgraph(subgraph, all_graph_edges):
        """Extract an ordered polyline of node IDs from a subgraph.

        The subgraph should represent a single branch segment (a path or
        an isolated node).  Edges from *all_graph_edges* that connect the
        branch to adjacent branch points are prepended/appended so that
        the returned polyline includes those junctions.

        Parameters
        ----------
        subgraph : networkx.Graph
            A connected subgraph (typically a simple path) from which the
            Eulerian path is computed.
        all_graph_edges : list of tuple
            Complete edge list of the parent graph, used to find
            connections to adjacent branch points.

        Returns
        -------
        list
            Ordered node IDs forming the polyline.
        """
        if len(subgraph.nodes) == 1:
            node = list(subgraph.nodes)[0]
            path = [(node, node)]
        else:
            path = list(nx.eulerian_path(subgraph))
        start_node = path[0][0]
        end_node = path[-1][-1]
        prepended = False
        appended = False
        output_path = path.copy()
        for edge in all_graph_edges:
            if edge not in path and edge[::-1] not in path:
                if start_node in edge and not prepended:
                    output_path.insert(0, edge if edge[1] == start_node else edge[::-1])
                    prepended = True
                elif end_node in edge and not appended:
                    output_path.append(edge if edge[0] == end_node else edge[::-1])
                    appended = True

        if len(output_path) == 1 and output_path[0][0] == output_path[0][1]:
            polyline = [output_path[0][0]]
        else:
            for edge in output_path:
                if edge[0] == edge[1]:
                    output_path.remove(edge)
            polyline = [node for node, _ in output_path]
            polyline.append(output_path[-1][-1])

        return polyline

    @staticmethod
    def get_polylines_from_graph(g):
        """Decompose a skeleton graph into polylines of node IDs.

        Branch points are removed to split the graph into simple-path
        connected components.  Each component is converted to an ordered
        polyline, and any edges between branch points that are not yet
        covered are added as two-node polylines.

        Parameters
        ----------
        g : networkx.Graph
            Skeleton graph with at least node indices.

        Returns
        -------
        list of list
            Each inner list is an ordered sequence of node IDs forming
            one polyline branch.
        """
        polylines = []
        edges = list(g.edges)
        g_copy = g.copy()
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g_copy)
        g_copy.remove_nodes_from(branchpoints)

        for component in nx.connected_components(g_copy):
            g_sub = g_copy.subgraph(component)
            polyline = CustomSkeleton.get_polyline_from_subgraph(g_sub, edges)
            polylines.append(polyline)

        all_edges = []
        for polyline in polylines:
            all_edges.extend(list(zip(polyline[:-1], polyline[1:])))
        all_edges = [tuple(sorted(edge)) for edge in all_edges]
        for edge in edges:
            if edge not in all_edges:
                polylines.append([edge[0], edge[1]])
        return polylines

    @staticmethod
    def remove_smallest_qualifying_branch(g, min_branch_length_nm=200):
        """Remove the shortest terminal branch shorter than a threshold.

        Only branches that are terminal (exactly one end is a branch
        point) and shorter than *min_branch_length_nm* are candidates.
        Among those, the single shortest branch is removed.

        Parameters
        ----------
        g : networkx.Graph
            Skeleton graph.  Modified **in place** if a branch is removed.
        min_branch_length_nm : float, optional
            Length threshold in nanometres.  Default is ``200``.

        Returns
        -------
        removed_path : list or None
            Node IDs of the removed polyline, or ``None`` if nothing
            qualified for removal.
        g : networkx.Graph
            The (possibly modified) graph.
        """
        branchpoints, _ = CustomSkeleton.find_branchpoints_and_endpoints(g)
        current_min_branch_length_nm = np.inf
        current_min_branch_path = None

        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)

        for polyline_by_vertex_id in polylines_by_vertex_id:
            if (polyline_by_vertex_id[0] in branchpoints) ^ (
                polyline_by_vertex_id[-1] in branchpoints
            ):
                polyline_length_nm = 0
                for v1, v2 in zip(
                    polyline_by_vertex_id[:-1], polyline_by_vertex_id[1:]
                ):
                    polyline_length_nm += np.linalg.norm(
                        np.array(g.nodes[v1]["position_nm"])
                        - np.array(g.nodes[v2]["position_nm"])
                    )
                if (
                    polyline_length_nm < min_branch_length_nm
                    and polyline_length_nm < current_min_branch_length_nm
                ):
                    if len(set(polyline_by_vertex_id)) < g.number_of_nodes():
                        current_min_branch_length_nm = polyline_length_nm
                        current_min_branch_path = polyline_by_vertex_id

        if current_min_branch_path:
            g.remove_edges_from(
                list(zip(current_min_branch_path[:-1], current_min_branch_path[1:]))
            )
            g.remove_nodes_from(list(nx.isolates(g)))
        return current_min_branch_path, g

    def skeleton_to_graph(self):
        """Convert this skeleton to a ``networkx.Graph``.

        Each node stores ``position_nm`` (and ``radius`` when available).
        Edge weights are set to Euclidean distances between endpoints.

        Returns
        -------
        networkx.Graph
            Weighted graph representation of the skeleton.
        """
        g = nx.Graph()
        g.add_nodes_from(range(len(self.vertices)))
        for idx in range(len(self.vertices)):
            g.nodes[idx]["position_nm"] = self.vertices[idx]
            if self.radii:
                g.nodes[idx]["radius"] = self.radii[idx]

        g.add_edges_from(self.edges)
        for edge in self.edges:
            try:
                g[edge[0]][edge[1]]["weight"] = np.linalg.norm(
                    np.array(self.vertices[edge[0]]) - np.array(self.vertices[edge[1]])
                )
            except IndexError as e:
                logger.error(
                    f"IndexError for edge {edge} with {len(self.vertices)} vertices"
                )
                raise e
        return g

    @staticmethod
    def get_polylines_positions_from_graph(g):
        """Convert graph polylines from node IDs to position arrays.

        Parameters
        ----------
        g : networkx.Graph
            Skeleton graph whose nodes have a ``position_nm`` attribute.

        Returns
        -------
        list of numpy.ndarray
            Each array has shape ``(N, 3)`` containing the ``(x, y, z)``
            positions of the nodes along one polyline.
        """
        polylines_by_vertex_id = CustomSkeleton.get_polylines_from_graph(g)
        polylines = []
        for polyline_by_vertex_id in polylines_by_vertex_id:
            polylines.append(
                np.array(
                    [
                        np.array(g.nodes[vertex_id]["position_nm"])
                        for vertex_id in polyline_by_vertex_id
                    ]
                )
            )
        return polylines

    def graph_to_skeleton(self, g):
        """Create a new ``CustomSkeleton`` from a subgraph of this skeleton.

        Vertices and radii are pulled from ``self`` using the node indices
        present in *g*.  Edges are remapped to new contiguous indices.

        Parameters
        ----------
        g : networkx.Graph
            Subgraph whose node IDs correspond to indices in
            ``self.vertices`` and ``self.radii``.

        Returns
        -------
        CustomSkeleton
            New skeleton containing only the nodes and edges in *g*.
        """
        vertices = [self.vertices[idx] for idx in g.nodes]
        radii = [self.radii[idx] for idx in g.nodes]
        edges = fastremap.remap(
            np.array(g.edges), dict(zip(list(g.nodes), list(range(len(g.nodes)))))
        )
        edges = edges.tolist()
        polylines = CustomSkeleton.get_polylines_positions_from_graph(g)
        return CustomSkeleton(vertices, edges, radii, polylines)

    def prune(self, min_branch_length_nm=200):
        """Iteratively remove short terminal branches from the skeleton.

        Branches whose length is below *min_branch_length_nm* and that
        are terminal (connected to a branch point at only one end) are
        removed, shortest first, until no more qualifying branches remain.

        Parameters
        ----------
        min_branch_length_nm : float, optional
            Minimum branch length in nanometres.  Default is ``200``.

        Returns
        -------
        CustomSkeleton
            A new pruned skeleton.  If the skeleton has only one vertex
            the original instance is returned unchanged.
        """
        if len(self.vertices) == 1:
            return self
        g = self.skeleton_to_graph()
        current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
            g, min_branch_length_nm
        )
        while current_min_tick_path:
            current_min_tick_path, g = CustomSkeleton.remove_smallest_qualifying_branch(
                g, min_branch_length_nm
            )
        return self.graph_to_skeleton(g)

    @staticmethod
    def lineseg_dists(p, a, b):
        """Calculate distances from points to a line segment.

        Parameters
        ----------
        p : array-like, shape ``(N, 3)`` or ``(3,)``
            Query point(s).
        a : array-like, shape ``(3,)``
            Start of the line segment.
        b : array-like, shape ``(3,)``
            End of the line segment.

        Returns
        -------
        numpy.ndarray, shape ``(N,)``
            Euclidean distance from each point in *p* to the closest
            point on segment *a*--*b*.
        """
        p = np.atleast_2d(p)
        if np.all(a == b):
            return np.linalg.norm(p - a, axis=1)
        d = np.divide(b - a, np.linalg.norm(b - a))
        s = np.dot(a - p, d)
        t = np.dot(p - b, d)
        h = np.maximum.reduce([s, t, np.zeros_like(s)])
        c = np.linalg.norm(np.cross(p - a, d), axis=1)
        return np.hypot(h, c)

    def write_neuroglancer_skeleton(self, path):
        """Write the skeleton in neuroglancer precomputed binary format.

        Parent directories are created automatically.

        Parameters
        ----------
        path : str
            Output file path (no extension).
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            skel = NeuroglancerSkeleton(
                self.vertices, self.edges, vertex_attributes=None
            )
            encoded = skel.encode(Source())
            f.write(encoded)

    @staticmethod
    def read_neuroglancer_skeleton(path):
        """Read a neuroglancer precomputed binary skeleton file.

        Parameters
        ----------
        path : str
            Path to the binary skeleton file.

        Returns
        -------
        vertex_positions : numpy.ndarray, shape ``(N, 3)``
            Vertex positions as 32-bit floats.
        edges : numpy.ndarray, shape ``(M, 2)``
            Edge pairs as unsigned 32-bit integers.
        """
        with open(path, "rb") as f:
            data = f.read()

        offset = 0
        n_vertices, n_edges = struct.unpack_from("<II", data, offset)
        offset += 8

        num_vp_values = n_vertices * 3
        vertex_positions = np.frombuffer(
            data, dtype="<f4", count=num_vp_values, offset=offset
        )
        offset += vertex_positions.nbytes
        vertex_positions = vertex_positions.reshape((n_vertices, 3))

        num_edge_values = n_edges * 2
        edges = np.frombuffer(data, dtype="<u4", count=num_edge_values, offset=offset)
        offset += edges.nbytes
        edges = edges.reshape((n_edges, 2))

        return vertex_positions, edges
