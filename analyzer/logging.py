

def setup_logging(
    default_level=None,
):
    import logging.config
    import base64
    from pathlib import Path

    import yaml
    from analyzer.configuration import CONFIG
    import os
    env_config = os.environ.get("ANALYZER_LOG_CONFIG")
    if env_config:
        import json

        d = json.loads(base64.b64decode(env_config).decode("utf-8"))
        logging.config.dictConfig(d)
        return

    default_path = Path(CONFIG.CONFIG_PATH) / "logging_config.yaml"
    path = Path(default_path)
    try:
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    except OSError:
        pass
    if default_level is not None:
        logger = logging.getLogger("analyzer")
        logger.setLevel(default_level)
        logger = logging.getLogger("distributed")
        logger.setLevel(default_level)
