import logging.config
from pathlib import Path

import yaml
from analyzer.configuration import CONFIG



def setup_logging(
    default_level=None,
):
    default_path = Path(CONFIG.CONFIG_PATH) / "logging_config.yaml"
    path = Path(default_path)
    try:
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except OSError as e:
        pass
    if default_level is not None:
        logger = logging.getLogger("analyzer")
        logger.setLevel(default_level)
        logger = logging.getLogger("distributed")
        logger.setLevel(default_level)
