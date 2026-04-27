import yaml
from yaml.loader import SafeLoader
import numpy as np


def read_multires_config(config_path):
    """Read and validate the run config for the multiresolution pipeline.

    Loads ``run-config.yaml`` from *config_path*, extracts the
    ``required_settings`` section, and fills in defaults for optional
    decimation and segment-properties settings.

    Parameters
    ----------
    config_path : str
        Directory containing ``run-config.yaml``.

    Returns
    -------
    tuple[dict, dict, dict]
        ``(required_settings, optional_decimation_settings,
        optional_properties_settings)``

        *optional_decimation_settings* defaults:

        - ``box_size``: ``None`` (auto-computed per-axis heuristic)
        - ``skip_decimation``: ``False``
        - ``decimation_factor``: ``2``
        - ``aggressiveness``: ``7``
        - ``delete_decimated_meshes``: ``False``
        - ``roi``: ``None``
        - ``target_faces_per_lod0_chunk``: ``25000``

        *optional_properties_settings* defaults:

        - ``segment_properties_csv``: ``None``
        - ``segment_properties_columns``: ``None``
        - ``segment_properties_id_column``: ``"Object ID"``
    """
    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
        required_settings = config["required_settings"]
        optional_decimation_settings = config.get("optional_decimation_settings", {})

        if "box_size" not in optional_decimation_settings:
            optional_decimation_settings["box_size"] = None
        else:
            box_size = optional_decimation_settings["box_size"]
            if box_size is not None:
                box_size = np.atleast_1d(np.asarray(box_size, dtype=float))
                if box_size.shape == (1,):
                    box_size = np.repeat(box_size, 3)
                optional_decimation_settings["box_size"] = box_size
        if "skip_decimation" not in optional_decimation_settings:
            optional_decimation_settings["skip_decimation"] = False
        if "decimation_factor" not in optional_decimation_settings:
            optional_decimation_settings["decimation_factor"] = 2
        if "aggressiveness" not in optional_decimation_settings:
            optional_decimation_settings["aggressiveness"] = 7
        if "delete_decimated_meshes" not in optional_decimation_settings:
            optional_decimation_settings["delete_decimated_meshes"] = False
        if "roi" not in optional_decimation_settings:
            optional_decimation_settings["roi"] = None
        if "target_faces_per_lod0_chunk" not in optional_decimation_settings:
            from mesh_n_bone.multires.multires import (
                DEFAULT_TARGET_FACES_PER_LOD0_CHUNK,
            )
            optional_decimation_settings["target_faces_per_lod0_chunk"] = (
                DEFAULT_TARGET_FACES_PER_LOD0_CHUNK
            )

        optional_properties_settings = config.get("optional_properties_settings", {})
        if "segment_properties_csv" not in optional_properties_settings:
            optional_properties_settings["segment_properties_csv"] = None
        if "segment_properties_columns" not in optional_properties_settings:
            optional_properties_settings["segment_properties_columns"] = None
        if "segment_properties_id_column" not in optional_properties_settings:
            optional_properties_settings["segment_properties_id_column"] = "Object ID"

        return required_settings, optional_decimation_settings, optional_properties_settings


def read_generic_config(config_path):
    """Read a generic YAML run config (for meshify, skeletonize, analyze).

    Parameters
    ----------
    config_path : str
        Directory containing ``run-config.yaml``.

    Returns
    -------
    dict
        Parsed YAML config as a flat dictionary.
    """
    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config
