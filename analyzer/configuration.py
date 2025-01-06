from pathlib import Path


class Config:
    FILE_ROOTS = ["store", "applocal"]
    APPLICATION_DATA = "/srv/.application_data"
    PHYSICS_DATA = APPLICATION_DATA + "/physics_data"
    #APPLICATION_RESOURCES = "/srv/analyzer_resources"
    APPLICATION_RESOURCES = "analyzer_resources"

    DATASET_PATHS = [str(Path(APPLICATION_RESOURCES) / "datasets")]
    ERA_PATHS = [str(Path(APPLICATION_RESOURCES) / "eras")]
    CONFIG_PATH = str(Path(APPLICATION_RESOURCES) / "configuration")
    DASK_CONFIG_PATH = str(Path(CONFIG_PATH) / "dask_config.yaml")
    STATIC_PATH = str(Path(APPLICATION_RESOURCES) / "static")
    STYLE_PATH = str(Path(APPLICATION_RESOURCES) / "styles")

    PRETTY_MODE = True

class LocalConfig:
    FILE_ROOTS = ["store", "applocal"]
    APPLICATION_DATA = ".application_data"
    PHYSICS_DATA = APPLICATION_DATA + "/physics_data"
    #APPLICATION_RESOURCES = "/srv/analyzer_resources"
    APPLICATION_RESOURCES = "analyzer_resources"

    DATASET_PATHS = [str(Path(APPLICATION_RESOURCES) / "datasets")]
    ERA_PATHS = [str(Path(APPLICATION_RESOURCES) / "eras")]
    CONFIG_PATH = str(Path(APPLICATION_RESOURCES) / "configuration")
    DASK_CONFIG_PATH = str(Path(CONFIG_PATH) / "dask_config.yaml")
    STATIC_PATH = str(Path(APPLICATION_RESOURCES) / "static")
    STYLE_PATH = str(Path(APPLICATION_RESOURCES) / "styles")

    PRETTY_MODE = True


CONFIG = LocalConfig()
