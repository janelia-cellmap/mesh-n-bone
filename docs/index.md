# mesh-n-bone

Unified tool for mesh generation, multiresolution mesh creation, skeletonization, and analysis — all parallelized with [Dask](https://dask.org/).

Produces meshes in the [neuroglancer precomputed format](https://github.com/google/neuroglancer/blob/master/src/neuroglancer/datasource/precomputed/meshes.md) for viewing in [neuroglancer](https://github.com/google/neuroglancer).

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

## Quick start

All commands are available through the `mesh-n-bone` CLI:

```bash
mesh-n-bone <command> [options]
```

See the [CLI Usage](cli.md) page for full command documentation, or the
[API Reference](api/meshify.md) for Python API details.
