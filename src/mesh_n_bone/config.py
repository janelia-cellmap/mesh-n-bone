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
                if box_size.size == 1:
                    box_size = np.full(3, box_size.item())
                elif box_size.size != 3:
                    raise ValueError(
                        f"box_size must be a scalar or a 3-element list, got {box_size}"
                    )
                optional_decimation_settings["box_size"] = box_size
        if "skip_decimation" not in optional_decimation_settings:
            optional_decimation_settings["skip_decimation"] = False
        if "decimation_factor" not in optional_decimation_settings:
            optional_decimation_settings["decimation_factor"] = 2
        if "aggressiveness" not in optional_decimation_settings:
            optional_decimation_settings["aggressiveness"] = 7
        if "delete_decimated_meshes" not in optional_decimation_settings:
            optional_decimation_settings["delete_decimated_meshes"] = False

        return required_settings, optional_decimation_settings


def read_generic_config(config_path):
    """Read a generic run config (for meshify, skeletonize, analyze)."""
    with open(f"{config_path}/run-config.yaml") as f:
        config = yaml.load(f, Loader=SafeLoader)
    return config
