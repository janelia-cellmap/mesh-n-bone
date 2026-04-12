"""Local HTTP server with CORS for viewing zarr volumes and meshes in neuroglancer."""

import os
import sys
import json
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import quote


class CORSHandler(SimpleHTTPRequestHandler):
    """HTTP handler that adds CORS headers for neuroglancer compatibility."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Length, Content-Range")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        pass


def _build_neuroglancer_url(port, zarr_path=None, meshes_path=None):
    """Build a neuroglancer URL with layers for the given data sources.

    Parameters
    ----------
    port : int
        Local HTTP server port.
    zarr_path : str or None
        Relative path (from server root) to a zarr dataset, e.g.
        ``data/volume.zarr/seg/s0``.
    meshes_path : str or None
        Relative path (from server root) to precomputed meshes, e.g.
        ``output/multires``.

    Returns
    -------
    str
        A neuroglancer URL.
    """
    base = f"http://localhost:{port}"
    layers = []

    if zarr_path:
        layers.append({
            "source": f"zarr://n5://{base}/{zarr_path}",
            "type": "segmentation",
            "name": "volume",
        })

    if meshes_path:
        layers.append({
            "source": f"precomputed://{base}/{meshes_path}",
            "type": "segmentation",
            "name": "meshes",
        })

    state = {"layers": layers}
    return f"https://neuroglancer-demo.appspot.com/#!{quote(json.dumps(state))}"


def serve(directory, port=9015, zarr_path=None, meshes_path=None):
    """Serve *directory* over HTTP with CORS headers.

    Parameters
    ----------
    directory : str
        Root directory to serve.
    port : int
        Port to listen on (default 9015).
    zarr_path : str or None
        Relative path within *directory* to a zarr/n5 dataset to show as
        an image layer. E.g. ``data/example.zarr/seg/s0``.
    meshes_path : str or None
        Relative path within *directory* to precomputed meshes to show as
        a segmentation layer. E.g. ``output/multires``.
    """
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a directory", file=sys.stderr)
        sys.exit(1)

    handler = partial(CORSHandler, directory=directory)
    server = HTTPServer(("", port), handler)

    print(f"Serving {directory} on http://localhost:{port}")

    if zarr_path or meshes_path:
        ng_url = _build_neuroglancer_url(port, zarr_path, meshes_path)
        print(f"\nOpen in neuroglancer:\n  {ng_url}")
    elif not zarr_path and not meshes_path:
        # Guess: if directory has a multires/ subfolder, use it
        if os.path.isdir(os.path.join(directory, "multires")):
            ng_url = _build_neuroglancer_url(port, meshes_path="multires")
            print(f"\nOpen in neuroglancer:\n  {ng_url}")

    print("\nPress Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()
