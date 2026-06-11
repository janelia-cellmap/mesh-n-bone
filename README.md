# mesh-n-bone

Unified tool for mesh generation, multiresolution mesh creation, skeletonization, and analysis — all parallelized with [Dask](https://dask.org/).

Produces meshes in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md) for viewing in [neuroglancer](https://github.com/google/neuroglancer).

![Demo](recording/recording.gif)

## Try it in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/janelia-cellmap/mesh-n-bone/blob/master/examples/example.ipynb)

The [example notebook](examples/example.ipynb) runs the full pipeline end-to-end (build a zarr volume, generate meshes, view in neuroglancer) without any local setup.

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
pixi run mesh-n-bone serve examples --zarr data/example.zarr/seg --meshes output/multires
```

See [examples/](examples/) for the full walkthrough.

## Features

- **Meshify** — Generate meshes from `.zarr`, `.n5`, or [neuroglancer precomputed](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/volume.md) segmentation volumes (auto-detected; local, `http(s)://`, `gs://`, or `s3://`) via marching cubes, with blockwise processing, chunk assembly, simplification, and optional on-the-fly downsampling
- **To-Neuroglancer** — Convert single-scale meshes into neuroglancer multiresolution Draco-compressed meshes with automatic LOD decimation, optionally [sharded](https://github.com/google/neuroglancer/blob/master/src/datasource/precomputed/sharded.md) so thousands of meshes load with a handful of HTTP range requests instead of per-segment file fetches
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
pip install mesh-n-bone
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

Reads a local or HTTP(S) `.zarr` / `.n5` segmentation volume, runs marching cubes per chunk, assembles across chunk boundaries (with boundary deduplication), optionally simplifies and smooths, and writes output as PLY or neuroglancer format.

Example meshify `run-config.yaml`:

```yaml
# ── Required ──
input_path: /path/to/segmentation.zarr/s0   # Local path or HTTP(S) URL to zarr/n5 dataset
output_directory: /path/to/output            # Where to write output meshes

# ── All remaining fields are optional ──

# ── Mesh generation ──
downsample_factor: 2             # Downsample volume by this factor before meshing (default: none)
downsample_method: mode          # Downsampling method: mode, mode_suppress_zero, or binary (default: mode)

# ── Simplification & smoothing ──
do_simplification: true          # Simplify meshes after assembly (default: true)
target_reduction: 0.933          # Total fraction of faces to remove (default: 0.933, hemibrain-matched)
n_smoothing_iter: 2              # Taubin smoothing iterations (default: 2)
check_mesh_validity: false       # Require watertight meshes (default: true; disable for ROI)
# When smooth_before_simplify is true (default), Taubin smoothing runs on the
# DENSE assembled mesh first, then quadric decimation collapses the smooth
# surface to target_reduction. Empirically ~2x lower RMS deviation from the
# underlying continuous surface at the same final face count compared to
# the reverse order, because Taubin can fully average out the voxel staircase
# only when it has every nearby vertex available as a neighbor. Set false for
# the legacy decimate-then-smooth ordering.
smooth_before_simplify: true     # (default: true)
# When use_fixed_edge_simplification is true, simplification runs in TWO STAGES:
#   1. per-chunk pass, with block-boundary vertices pinned so they survive assembly
#   2. global pass on the assembled mesh, finishing off the remaining reduction
# stage_1_reduction_fraction sets how much of target_reduction happens in stage 1.
# When false, a single standard simplification pass runs after assembly.
use_fixed_edge_simplification: true  # (default: false)
stage_1_reduction_fraction: 0.5      # Share of target_reduction in the per-chunk stage (default: 0.5)
do_analysis: false               # Compute mesh metrics CSV (default: true)

# ── Multiresolution output ──
do_multires: true                # Also generate neuroglancer multilod_draco output (default: false)
num_lods: 4                      # Number of levels of detail (default: 4)
multires_strategy: downsample    # LOD strategy: decimate (face-decimate s0 mesh for higher LODs) or
                                 # downsample (re-mesh from progressively downsampled volumes,
                                 # better-looking coarse LODs) (default: downsample)
# IMPORTANT: in the downsample strategy, the volume is downsampled in-memory by
# the worker using `downsample_method` — pre-existing multiscale levels in the
# input zarr (s1, s2, ...) are NOT read for higher LODs. The whole multires
# pyramid is built from the single `input_path` you supply. If you'd rather mesh
# from a pre-computed coarse scale (e.g. hemibrain's `s2` at 32 nm), point
# `input_path` directly at that scale — but then *that* scale becomes LOD 0 and
# all coarser LODs are built by further downsampling it.
decimation_factor: 6             # Per-LOD face reduction factor. In decimate strategy: literal ratio
                                 # between LODs. In downsample strategy: target ratio used to derive
                                 # per-LOD target_reduction so the LOD-to-LOD face count drops by this
                                 # factor. (default: 6, hemibrain-matched)
delete_decimated_meshes: true    # Remove intermediate LOD mesh files (default: true)

# ── Sharded output (recommended for thousands of meshes) ──
sharded: true                    # Pack all meshes into a few <n>.shard files
                                 # instead of two files per segment (default: true)
# shard_bits: 4                  # Optional overrides — defaults are auto-sized
# minishard_bits: 6              # from the segment count via choose_shard_params.
# preshift_bits: 0
delete_unsharded_files: true     # Remove per-segment files after packing (default: true)

# ── Coordinate system ──
# Voxel size is read automatically from the dataset metadata (OME-NGFF or
# zarr attributes). Use voxel_size_nm only to override when the metadata is
# missing or incorrect. It affects mesh vertex scaling, not ROI coordinates.
voxel_size_nm: [1000, 1000, 1000]  # Override voxel size (ZYX)

# ── Segment properties ──
segment_properties_csv: /path/to/properties.csv  # CSV with per-segment metadata
segment_properties_columns: [col1, col2]         # Which columns to include (default: all)
segment_properties_id_column: "Object ID"        # CSV column with segment IDs (default: "Object ID")

# ── Region of interest ──
roi:                             # Restrict processing to this subregion
  begin: [100, 200, 300]         # Start coordinates in dataset world units (ZYX)
  end: [500, 600, 700]           # End coordinates in dataset world units (ZYX)
                                 # Boundary edges are preserved during simplification.
                                 # Can also be passed via CLI: --roi z0,y0,x0,z1,y1,x1

# ── Specific segment ids only (optional) ──
# When set, only the listed ids are meshified — every other voxel is
# zeroed before marching cubes via fastremap.mask_except, so zmesh does
# no work on unwanted objects. Blocks containing none of the targets
# are skipped entirely.
target_ids: [12345, 67890]                # YAML list of ids, OR
# target_ids: /path/to/ids.csv            # CSV file (first col, or "id"/"Object ID" col)
# target_ids: 12345                       # Single id
                                          # CLI: --ids 12345,67890  OR  --ids /path.csv
```

`input_path` accepts any of `.zarr`, `.n5`, or neuroglancer precomputed sources, from any of these locations:

- A local filesystem path (e.g. `/data/seg.zarr/s0`)
- `http(s)://...`
- `gs://bucket/path`
- `s3://bucket/path`

The format is auto-detected by probing for the relevant marker file (`info` → precomputed, `zarr.json` → zarr v3, `.zarray` → zarr v2, `attributes.json` → N5), so a path like `gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation` opens correctly with no extra prefix.

If no specific scale is provided, the highest-resolution scale is used by default — for OME-Zarr multiscales groups (root or subgroup) that's the first dataset listed in the metadata; for precomputed, scale index 0 from the `info` file. To use a different scale, either name the array directly (`.../seg.zarr/s2`) or, for precomputed, append the scale `key` from the `info` file (e.g. `gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/16.0x16.0x16.0`).

Example using the public hemibrain segmentation at native res:

```yaml
input_path: gs://neuroglancer-janelia-flyem-hemibrain/v1.0/segmentation/8.0x8.0x8.0
output_directory: /path/to/output
roi:
  begin: [155744, 155744, 155744]   # ZYX, nm
  end:   [163744, 163744, 163744]   # 1000 voxels × 8 nm per side
check_mesh_validity: false          # ROIs cut at the boundary
use_fixed_edge_simplification: true
do_multires: true
num_lods: 4
```

### Worker auto-cap and OOM retry

When a phase has fewer tasks than the requested `-n` workers (e.g. one block in a tiny ROI), meshify caps the cluster size for that phase to the task count so idle LSF/SLURM slots aren't requested. You'll see a one-line `Capping workers ...` log message.

If a worker dies from out-of-memory (typically a single huge segment hitting the per-worker memory cap), meshify automatically halves `processes`-per-slot in the in-memory dask config (so each remaining worker gets 2× memory while LSF slot count and CPU budget stay fixed) and retries the phase. Defaults: 3 retries, opt out with `retry_on_oom: false`, change retry count with `memory_retry_max: N`. Each retry logs a `WARNING` with the dead worker's address; the per-task `mesh_id` is also printed to worker stderr so you can grep `job-logs/LSFCluster-*.err` to identify the segment that caused the OOM.

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

sharding_settings:
  sharded: true                                    # Pack output into <n>.shard files (default: true)
  # shard_bits: 4                                  # Optional overrides — auto-sized by default
  # minishard_bits: 6
  # preshift_bits: 0
  delete_unsharded_files: true                     # Remove per-segment files after packing (default: true)
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

### Running on an HPC cluster

#### LSF (bsub)

To run on an LSF cluster, submit the driver process via `bsub`. The driver launches Dask workers as separate LSF jobs:

```bash
bsub -n 2 -P your_project_name mesh-n-bone meshify lsf-config -n 40
```

This submits a 2-slot driver job that creates a 40-worker Dask cluster. Each worker is launched as its own LSF job using the settings in `lsf-config/dask-config.yaml`.

With pixi:

```bash
bsub -n 2 -P your_project_name pixi run mesh-n-bone meshify lsf-config -n 40
```

An example LSF dask config is provided in [`lsf-config/dask-config.yaml`](lsf-config/dask-config.yaml):

```yaml
jobqueue:
  lsf:
    ncpus: 48
    processes: 40
    cores: 40
    memory: 720GB
    walltime: 01:00
    mem: 720000000000
    use-stdin: true
    log-directory: job-logs
    name: mesh-n-bone
    project: your_project_name
```

Update `project` to your LSF project/queue allocation and adjust `ncpus`, `memory`, and `walltime` for your cluster.

#### SLURM / SGE

The same pattern applies — submit the driver via your scheduler and set the cluster type in `dask-config.yaml`:

```yaml
# SLURM
jobqueue:
  slurm:
    cores: 40
    memory: 720GB
    walltime: "01:00:00"
    # ... other SLURM-specific options

# SGE
jobqueue:
  sge:
    cores: 40
    memory: 720GB
    # ... other SGE-specific options
```

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

## Acknowledgments

Thanks to [Luca Marconato](https://github.com/LucaMarconato) for the pixi configuration that informed macOS support and the neuroglancer serving approach ([#6](https://github.com/janelia-cellmap/mesh-n-bone/pull/6)).

## License

mesh-n-bone is distributed under the GNU General Public License v3.0 or later (GPL-3.0-or-later). See [LICENSE](LICENSE) for the full text.
