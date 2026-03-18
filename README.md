# mesh-n-bone

Unified tool for mesh generation, multiresolution mesh creation, skeletonization, and analysis — all parallelized with [Dask](https://dask.org/).

Produces meshes in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md) for viewing in [neuroglancer](https://github.com/google/neuroglancer).

![Demo](recording/recording.gif)

## Features

- **Meshify** — Generate meshes from `.zarr` / `.n5` segmentation volumes via marching cubes, with blockwise processing, chunk assembly, simplification, and optional on-the-fly downsampling
- **Multires** — Convert single-scale meshes into neuroglancer multiresolution Draco-compressed meshes with automatic LOD decimation
- **Skeletonize** — Extract skeletons from meshes using CGAL mean curvature flow, with pruning, simplification, and metrics
- **Analyze** — Compute mesh metrics: volume, surface area, curvature, thickness, principal inertia, oriented bounds

## Installation

### With pixi (recommended)

```bash
git clone https://github.com/janelia-cellmap/multiresolution-mesh-creator.git
cd multiresolution-mesh-creator
pixi install
```

### With pip

```bash
pip install git+https://github.com/janelia-cellmap/multiresolution-mesh-creator.git
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
mesh-n-bone meshify CONFIG_PATH -n NUM_WORKERS
```

Reads a `.zarr` or `.n5` segmentation volume, runs marching cubes per chunk, assembles across chunk boundaries (with boundary deduplication), optionally simplifies and smooths, and writes output as PLY or neuroglancer format.

#### `multires` — Create multiresolution neuroglancer meshes

```bash
mesh-n-bone multires CONFIG_PATH -n NUM_WORKERS
```

Takes existing meshes (e.g. PLY files), decimates them at multiple LODs using pyfqmr, decomposes into spatial fragments, Draco-compresses, and writes the neuroglancer `multilod_draco` format.

Config directory must contain `run-config.yaml` and `dask-config.yaml`. Example `run-config.yaml`:

```yaml
required_settings:
  input_path: /path/to/meshes        # Directory with LOD 0 meshes
  output_path: /path/to/output       # Output directory
  num_lods: 6                        # Number of levels of detail

optional_decimation_settings:
  box_size: 4                        # LOD 0 box size (scalar or [x, y, z])
  skip_decimation: false             # Skip if decimated meshes already exist
  decimation_factor: 4               # Face reduction factor per LOD
  aggressiveness: 10                 # Decimation aggressiveness
  delete_decimated_meshes: true      # Clean up intermediate files
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
