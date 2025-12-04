from . import static
import logging
import importlib.resources 


logger = logging.getLogger("analyzer")

def setupLogging(
    default_level=None,
):
    import logging.config
    from pathlib import Path
    import yaml


    default_log_config_path = importlib.resources.files(static) / 'logging_config.yaml'
    with default_log_config_path.open("r") as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)


