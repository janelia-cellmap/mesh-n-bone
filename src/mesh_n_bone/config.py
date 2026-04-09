import yaml
from yaml.loader import SafeLoader
import numpy as np


def read_multires_config(config_path):
    """Read run config for the multires pipeline."""
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

        optional_properties_settings = config.get("optional_properties_settings", {})
        if "segment_properties_csv" not in optional_properties_settings:
            optional_properties_settings["segment_properties_csv"] = None
        if "segment_properties_columns" not in optional_properties_settings:
            optional_properties_settings["segment_properties_columns"] = None
        if "segment_properties_id_column" not in optional_properties_settings:
            optional_properties_settings["segment_properties_id_column"] = "Object ID"

        return required_settings, optional_decimation_settings, optional_properties_settings


def read_generic_config(config_path):
    """Read a generic run config (for meshify, skeletonize, analyze)."""
    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config
