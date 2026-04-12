# mesh-n-bone

Unified tool for mesh generation, multiresolution mesh creation, skeletonization, and analysis — all parallelized with [Dask](https://dask.org/).

Produces meshes in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md) for viewing in [neuroglancer](https://github.com/google/neuroglancer).

![Demo](recording/recording.gif)

## Quick start

```bash
git clone https://github.com/janelia-cellmap/mesh-n-bone.git
cd mesh-n-bone
pixi install

# Create a small example zarr volume
pixi run python examples/create_example_volume.py

# Generate meshes and multiresolution output
pixi run mesh-n-bone meshify examples/meshify-config -n 1

# View volume and meshes in neuroglancer
pixi run mesh-n-bone serve examples --zarr data/example.zarr/seg/s0 --meshes output/multires
```

See [examples/](examples/) for the full walkthrough.

## Features

- **Meshify** — Generate meshes from `.zarr` / `.n5` segmentation volumes via marching cubes, with blockwise processing, chunk assembly, simplification, and optional on-the-fly downsampling
- **To-Neuroglancer** — Convert single-scale meshes into neuroglancer multiresolution Draco-compressed meshes with automatic LOD decimation
- **Skeletonize** — Extract skeletons from meshes using CGAL mean curvature flow, with pruning, simplification, and metrics
- **Analyze** — Compute mesh metrics: volume, surface area, curvature, thickness, principal inertia, oriented bounds

## Installation

### With pixi (recommended)

```bash
git clone https://github.com/janelia-cellmap/mesh-n-bone.git
cd mesh-n-bone
pixi install
```

### With pip

```bash
pip install git+https://github.com/janelia-cellmap/mesh-n-bone.git
```

### Building the CGAL skeletonizer (optional)

The skeletonization module requires a compiled C++ binary. Build it with:

```bash
pixi run -e build-cgal build-cgal
```

This uses a separate pixi environment with CGAL, Boost, and Eigen dependencies.

## Usage

All commands are available through the `mesh-n-bone` CLI:

```
mesh-n-bone <command> [options]
```

### Commands

#### `meshify` — Generate meshes from segmentation volumes

```bash
mesh-n-bone meshify CONFIG_PATH -n NUM_WORKERS [--roi begin_z,begin_y,begin_x,end_z,end_y,end_x]
```

Reads a `.zarr` or `.n5` segmentation volume, runs marching cubes per chunk, assembles across chunk boundaries (with boundary deduplication), optionally simplifies and smooths, and writes output as PLY or neuroglancer format.

Example meshify `run-config.yaml`:

```yaml
# ── Required ──
input_path: /path/to/segmentation.zarr/s0   # Path to zarr/n5 segmentation dataset
output_directory: /path/to/output            # Where to write output meshes
num_workers: 10                              # Number of dask workers

# ── Mesh generation ──
downsample_factor: 2             # Downsample volume by this factor before meshing (default: none)
downsample_method: mode          # Downsampling method: mode, mode_suppress_zero, or binary

# ── Simplification & smoothing ──
do_simplification: true          # Simplify meshes after assembly (default: true)
target_reduction: 0.99           # Fraction of faces to remove (default: 0.99)
n_smoothing_iter: 10             # Taubin smoothing iterations (default: 10)
check_mesh_validity: false       # Require watertight meshes (default: true; disable for ROI)
use_fixed_edge_simplification: true  # Preserve chunk boundary edges during simplification
do_analysis: false               # Compute mesh metrics CSV (default: true)

# ── Multiresolution output ──
do_multires: true                # Also generate neuroglancer multilod_draco output
num_lods: 3                      # Number of levels of detail
multires_strategy: decimate      # LOD strategy: decimate or downsample
decimation_factor: 4             # Face reduction factor per LOD (default: 4)
delete_decimated_meshes: true    # Remove intermediate LOD mesh files

# ── Coordinate system ──
voxel_size_nm: [1000, 1000, 1000]  # Voxel size in nm (ZYX); use when dataset units
                                    # are not nm (e.g. 1 um = 1000 nm). Only affects
                                    # mesh vertex scaling, not ROI coordinates.

# ── Region of interest (optional) ──
roi:                             # Restrict processing to this subregion
  begin: [100, 200, 300]         # Start coordinates in dataset world units (ZYX)
  end: [500, 600, 700]           # End coordinates in dataset world units (ZYX)
                                 # Boundary edges are preserved during simplification.
                                 # Can also be passed via CLI: --roi z0,y0,x0,z1,y1,x1
```

#### `to-neuroglancer` — Convert existing meshes to neuroglancer multiresolution format

```bash
mesh-n-bone to-neuroglancer CONFIG_PATH -n NUM_WORKERS [--roi begin_x,begin_y,begin_z,end_x,end_y,end_z]
```

Takes existing meshes (e.g. PLY files), decimates them at multiple LODs using pyfqmr, decomposes into spatial fragments, Draco-compresses, and writes the neuroglancer `multilod_draco` format. Use this when you already have single-scale meshes and just need the neuroglancer format.

Config directory must contain `run-config.yaml` and `dask-config.yaml`. Example `run-config.yaml`:

```yaml
required_settings:
  input_path: /path/to/meshes       # Directory containing LOD 0 mesh files (e.g. PLY)
  output_path: /path/to/output      # Where to write neuroglancer output
  num_lods: 6                       # Number of levels of detail to generate

optional_decimation_settings:
  box_size: 4                       # LOD 0 fragment size in world units (scalar or [x, y, z])
  skip_decimation: false            # Set true to reuse previously decimated meshes
  decimation_factor: 4              # Face reduction factor per LOD (default: 2)
  aggressiveness: 10                # pyfqmr decimation aggressiveness (default: 7)
  delete_decimated_meshes: true     # Remove intermediate LOD mesh files when done
  roi:                              # Only process meshes intersecting this region (XYZ)
    begin: [0, 0, 0]
    end: [1000, 1000, 1000]

optional_properties_settings:
  segment_properties_csv: /path/to/properties.csv  # CSV with per-segment metadata
  segment_properties_columns: [col1, col2]         # Which columns to include (default: all)
  segment_properties_id_column: "Object ID"        # CSV column with segment IDs
```

`box_size` can be a scalar (applied to all axes) or a 3-element list for per-axis control, which prevents degenerate triangles on elongated meshes.

#### `skeletonize` — Skeletonize meshes using CGAL

```bash
mesh-n-bone skeletonize CONFIG_PATH -n NUM_WORKERS
```

Runs CGAL mean curvature flow skeletonization on all meshes in a directory. Produces skeleton files, metrics (longest shortest path, radius statistics, branch counts), and neuroglancer skeleton format output.

#### `skeletonize-single` — Skeletonize a single mesh

```bash
mesh-n-bone skeletonize-single INPUT_FILE OUTPUT_FILE [--subdivisions N] [--neuroglancer]
```

#### `analyze` — Analyze mesh geometry

```bash
mesh-n-bone analyze CONFIG_PATH -n NUM_WORKERS
```

Computes per-mesh metrics using trimesh and pymeshlab: volume, surface area, curvature (mean, Gaussian, RMS, absolute), thickness (shape diameter function), principal inertia components, and oriented bounding box dimensions. Outputs a CSV.

#### `serve` — Serve data for neuroglancer viewing

```bash
mesh-n-bone serve PATH [--zarr ZARR_PATH] [--meshes MESHES_PATH] [--port PORT]
```

Starts a local HTTP server with CORS headers and prints a neuroglancer URL. Use `--zarr` and `--meshes` to specify relative paths within `PATH` to a zarr/n5 volume and precomputed meshes, respectively. Default port is 9015.

### Dask configuration

All pipeline commands use Dask for parallelism. The config directory must contain a `dask-config.yaml` specifying the cluster type. Supported: `local`, `lsf`, `slurm`, `sge`.

When running with `-n 1`, no cluster is created and no config file is needed — work runs synchronously in the calling process.

See [dask-jobqueue configuration](https://github.com/dask/dask-jobqueue/blob/main/dask_jobqueue/jobqueue.yaml) for all cluster options.

## Testing

```bash
pixi run -e test test
```

The test suite includes unit tests and integration tests covering:

- Full meshify pipeline from zarr volumes (cross-chunk assembly, watertightness, volume accuracy)
- Multiresolution decomposition and Draco compression
- Mesh decimation across multiple LODs
- Downsampling methods (mode, mode-suppress-zero, binary mode)
- Mesh analysis metrics (volume, area, curvature, thickness)
- Skeleton processing (pruning, simplification, longest shortest path)
- Watertightness preservation after simplification and repair
- Fixed-edge boundary-preserving simplification
- Neuroglancer format output (ngmesh, multilod_draco, annotations)

## Project structure

```
src/mesh_n_bone/
  cli.py                    # Unified CLI
  config.py                 # YAML config parsing
  meshify/                  # Volume → mesh generation
    meshify.py              # Main pipeline (zmesh, chunk assembly)
    downsample.py           # Numba JIT blockwise downsampling
    fixed_edge.py           # Boundary-preserving simplification
  multires/                 # Multiresolution mesh creation
    multires.py             # Pipeline orchestrator
    decomposition.py        # Spatial fragment decomposition + Draco
    decimation.py           # pyfqmr LOD decimation
  skeletonize/              # Mesh skeletonization
    skeletonize.py          # CGAL skeletonization pipeline
    skeleton.py             # Skeleton data structure + operations
  analyze/                  # Mesh analysis
    analyze.py              # Volume, curvature, thickness metrics
  util/                     # Shared utilities
    dask_util.py            # Dask cluster management
    mesh_io.py              # Mesh I/O, fragments, z-order
    neuroglancer.py         # Neuroglancer format writers
    logging.py              # Timing, logging, stream capture
cgal_skeletonize_mesh/      # C++ CGAL skeletonizer source + binary
tests/                      # Unit and integration tests
```
