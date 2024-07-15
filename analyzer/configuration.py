import os
from pathlib import Path


def getConfiguration():
    venv = Path(os.environ.get("VIRTUAL_ENV"))
    venv_name = str(Path(os.environ.get("VIRTUAL_ENV")).stem)
    config = {}
    config["APPLICATION_DATA"]  = "/srv/.application_data"
    config["ENV_LOCAL_APPLICATION_DATA"]  = f"/srv/.application_data/envlocal/{venv_name}"
    config["DATASET_PATH"]  = None
    config["BASE_STATIC_RESOURCE_PATH"]  = None
    config["DASK_CONFIGURATION"]  = None
    config["LOGGING_CONFIGURATION"]  = None
    return config
