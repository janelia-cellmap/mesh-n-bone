# Fileglancer integration

mesh-n-bone exposes a [Fileglancer](https://github.com/JaneliaSciComp/fileglancer) app via the `runnables.yaml` manifest at the repo root. Fileglancer reads that manifest, renders a form for the parameters defined there, and runs the resulting command on the cluster.

## Files

- [`../runnables.yaml`](../runnables.yaml) — the Fileglancer manifest. Defines one runnable, `meshify`, with form fields for input/output paths, simplification, multiresolution, ROI, etc.
- [`run_meshify.py`](run_meshify.py) — small Python wrapper. Fileglancer passes flagged arguments; the wrapper writes a `run-config.yaml` and `dask-config.yaml` into `$FG_WORK_DIR/meshify-config/` and invokes `mesh-n-bone meshify` against them.

The wrapper exists because `mesh-n-bone meshify` reads a config directory, while Fileglancer is built around individual CLI flags.

## How to register the app

On the Fileglancer **Apps** page, add the GitHub repo URL:

```
https://github.com/janelia-cellmap/mesh-n-bone
```

(Or a branch/tag-pinned URL while this lives outside `master`.)

Fileglancer clones the repo, looks up `runnables.yaml` at the root, and on first launch `pixi run` resolves the environment automatically before invoking the wrapper.

## Local sanity check

Run the wrapper directly to confirm it produces a valid config and invokes the CLI:

```bash
pixi run python fileglancer/run_meshify.py \
  --input examples/data/example.zarr/seg/s0 \
  --output /tmp/meshify-fg-test \
  --multires \
  --num-workers 1
```

`FG_WORK_DIR` defaults to the current working directory when not set, so the generated config lands in `./meshify-config/`.

## Adding more runnables

Each subcommand (`to-neuroglancer`, `skeletonize`, `analyze`) follows the same pattern: add a runnable entry in `runnables.yaml` and a matching `run_<subcommand>.py` wrapper that materializes its config directory.
