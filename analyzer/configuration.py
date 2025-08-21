from pathlib import Path


class Config:
    FILE_ROOTS = [("store", "store"), "store", "applocal"]
    APPLICATION_DATA = "/srv/.application_data"
    PHYSICS_DATA = APPLICATION_DATA + "/physics_data"
    # APPLICATION_RESOURCES = "/srv/analyzer_resources"
    APPLICATION_RESOURCES = "analyzer_resources"

    DATASET_PATHS = [str(Path(APPLICATION_RESOURCES) / "datasets")]
    ERA_PATHS = [str(Path(APPLICATION_RESOURCES) / "eras")]
    CONFIG_PATH = str(Path(APPLICATION_RESOURCES) / "configuration")
    DASK_CONFIG_PATH = str(Path(CONFIG_PATH) / "dask_config.yaml")
    STATIC_PATH = str(Path(APPLICATION_RESOURCES) / "static")
    STYLE_PATH = str(Path(APPLICATION_RESOURCES) / "styles")
    TEMPLATE_PATH = str(Path(APPLICATION_RESOURCES) / "templates")

    PRETTY_MODE = True

    DEFAULT_PARALLEL_PROCESSES = None
    DEFAULT_PARALLEL_THREADS = 8

    WARN_LOAD_FILE_NUMBER = 50


CONFIG = Config()
