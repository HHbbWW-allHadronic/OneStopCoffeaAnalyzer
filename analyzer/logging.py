import importlib.resources
import logging.config
import os
from pathlib import Path

import yaml
import rich.logging

import analyzer.resources


def setup_logging(
    default_path=None,
    default_level=None,
    env_key="LOG_CFG",
):
    if default_path is None:
        with importlib.resources.as_file(
            importlib.resources.files(analyzer.resources)
        ) as f:
            default_path = Path(f) / "logging_config.yaml"
    path = Path(default_path)
    value = os.getenv(env_key, None)
    if value:
        path = value


    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    if default_level is not None:
        print(default_level)
        logger = logging.getLogger("analyzer")
        logger.handlers = [rich.logging.RichHandler()]
        logger.setLevel(default_level)
        logger = logging.getLogger("distributed")
        logger.setLevel(default_level)

