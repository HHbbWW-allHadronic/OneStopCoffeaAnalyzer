import importlib.resources
import logging.config
import os
from pathlib import Path

import yaml
from analyzer.configuration import CONFIG



def setup_logging(
    default_path=None,
    default_level=None,
    env_key="LOG_CFG",
):
    if default_path is None:
        default_path = Path(CONFIG.CONFIG_PATH) / "logging_config.yaml"
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value
    try:
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except OSError as e:
        pass
