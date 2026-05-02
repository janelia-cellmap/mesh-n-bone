"""Local HTTP server with CORS and Range support for viewing zarr volumes and meshes in neuroglancer."""

import json
import os
import socket
import sys
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote


class CORSHandler(SimpleHTTPRequestHandler):
    """HTTP handler that adds CORS headers and supports HTTP Range requests for neuroglancer."""

    _client_disconnect_errors = (
        BrokenPipeError,
        ConnectionAbortedError,
        ConnectionResetError,
    )

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")
        super().end_headers()

    def finish(self):
        try:
            super().finish()
        except self._client_disconnect_errors:
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        range_header = self.headers.get("Range")
        if not range_header:
            try:
                return super().do_GET()
            except self._client_disconnect_errors:
                return

        path = self.translate_path(self.path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return

        try:
            file_size = os.fstat(f.fileno()).st_size
            # Parse "bytes=start-end"
            byte_range = range_header.split("=", 1)[1]
            start_str, end_str = byte_range.split("-", 1)
            start = int(start_str) if start_str else 0
            end = int(end_str) if end_str else file_size - 1
            end = min(end, file_size - 1)
            length = end - start + 1

            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Content-Length", str(length))
            self.end_headers()

            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)
        except self._client_disconnect_errors:
            pass
        finally:
            f.close()

    def log_message(self, format, *args):
        pass


def _get_local_ip():
    """Return the local network IP address (not 127.0.0.1), or None on failure."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def _detect_zarr_scheme(full_path):
    """Return 'zarr3' for zarr v3 (zarr.json), 'zarr' for zarr v2 (.zarray/.zgroup)."""
    if os.path.exists(os.path.join(full_path, "zarr.json")):
        return "zarr3"
    return "zarr"


def _resolve_ome_ngff_group(directory, zarr_rel_path):
    """If zarr_rel_path points to an array inside an OME-NGFF group, return the group path.

    Walks up one level: if the parent directory has multiscales metadata (zarr v3
    zarr.json attributes or zarr v2 .zattrs), returns the parent-relative path.
    Otherwise returns zarr_rel_path unchanged.
    """
    full_path = os.path.join(directory, zarr_rel_path)
    parent_full = os.path.dirname(full_path)
    parent_rel = os.path.dirname(zarr_rel_path)

    # zarr v3: check parent zarr.json attributes
    parent_zarr_json = os.path.join(parent_full, "zarr.json")
    if os.path.exists(parent_zarr_json):
        try:
            with open(parent_zarr_json) as f:
                meta = json.load(f)
            if "multiscales" in meta.get("attributes", {}):
                return parent_rel
        except (json.JSONDecodeError, OSError):
            pass

    # zarr v2: check parent .zattrs
    parent_zattrs = os.path.join(parent_full, ".zattrs")
    if os.path.exists(parent_zattrs):
        try:
            with open(parent_zattrs) as f:
                attrs = json.load(f)
            if "multiscales" in attrs:
                return parent_rel
        except (json.JSONDecodeError, OSError):
            pass

    return zarr_rel_path


def _get_segment_ids(directory, meshes_path):
    """Read segment IDs from the precomputed segment_properties info, or infer from .index files."""
    seg_props = os.path.join(directory, meshes_path, "segment_properties", "info")
    if os.path.exists(seg_props):
        try:
            with open(seg_props) as f:
                props = json.load(f)
            return props["inline"]["ids"]
        except (KeyError, json.JSONDecodeError, OSError):
            pass
    # Fall back: any file named like an integer (mesh data files, not .index)
    mesh_dir = os.path.join(directory, meshes_path)
    ids = []
    for name in os.listdir(mesh_dir):
        if name.isdigit():
            ids.append(name)
    return sorted(ids, key=int)


def _build_neuroglancer_url(directory, host, port, zarr_path=None, meshes_path=None):
    """Build a neuroglancer URL with a single segmentation layer for the given data sources.

    Parameters
    ----------
    directory : str
        Absolute path to the served root directory (used for zarr format detection).
    host : str
        Hostname or IP to use in source URLs (e.g. ``localhost`` or ``192.168.1.5``).
    port : int
        HTTP server port.
    zarr_path : str or None
        Relative path (from server root) to a zarr dataset.
    meshes_path : str or None
        Relative path (from server root) to precomputed meshes.

    Returns
    -------
    str
        A neuroglancer URL.
    """
    base = f"http://{host}:{port}"
    sources = []

    if zarr_path:
        group_path = _resolve_ome_ngff_group(directory, zarr_path)
        scheme = _detect_zarr_scheme(os.path.join(directory, group_path))
        sources.append(f"{scheme}://{base}/{group_path}")

    if meshes_path:
        sources.append(f"precomputed://{base}/{meshes_path}")

    if not sources:
        return None

    segment_ids = _get_segment_ids(directory, meshes_path) if meshes_path else []

    layer = {
        "type": "segmentation",
        "source": sources if len(sources) > 1 else sources[0],
        "segments": segment_ids,
        "name": "segmentation",
    }

    state = {"layers": [layer]}
    return f"https://neuroglancer-demo.appspot.com/#!{quote(json.dumps(state))}"


def serve(directory, port=9015, zarr_path=None, meshes_path=None):
    """Serve *directory* over HTTP with CORS headers and Range request support.

    Parameters
    ----------
    directory : str
        Root directory to serve.
    port : int
        Port to listen on (default 9015).
    zarr_path : str or None
        Relative path within *directory* to a zarr/n5 dataset.
    meshes_path : str or None
        Relative path within *directory* to precomputed meshes.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    handler = partial(CORSHandler, directory=directory)
    server = HTTPServer(("0.0.0.0", port), handler)

    local_ip = _get_local_ip()

    print(f"Serving {directory}")
    print(f"  localhost:  http://localhost:{port}")
    if local_ip and local_ip != "127.0.0.1":
        print(f"  network:    http://{local_ip}:{port}")

    if zarr_path or meshes_path:
        ng_local = _build_neuroglancer_url(directory, "localhost", port, zarr_path, meshes_path)
        if ng_local:
            print(f"\nOpen in neuroglancer:\n  {ng_local}")


    print("\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
