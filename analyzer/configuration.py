import os
from pathlib import Path

venv = Path(os.environ.get("VIRTUAL_ENV"))
venv_name = str(Path(os.environ.get("VIRTUAL_ENV")).stem)
__config = {}


class Config:
    APPLICATION_DATA = "/srv/.application_data"
    PHYSICS_DATA = APPLICATION_DATA + "/physics_data"
    ENV_LOCAL_APPLICATION_DATA = f"/srv/.application_data/envlocal/{venv_name}"
    APPLICATION_RESOURCES = "/srv/analyzer_resources"
    DATASET_PATHS = [str(Path(APPLICATION_RESOURCES) / "datasets")]
    ERA_PATHS = [str(Path(APPLICATION_RESOURCES) / "eras")]
    CONFIG_PATH = str(Path(APPLICATION_RESOURCES) / "configuration")
    DASK_CONFIG_PATH = str(Path(CONFIG_PATH) / "dask_config.yaml")
    PRETTY_MODE = True


CONFIG = Config()
