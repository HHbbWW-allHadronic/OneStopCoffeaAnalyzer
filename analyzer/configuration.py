from pathlib import Path
from attrs import define, field


@define
class ExecutionConfig:
    warn_ulimit_ratio: int = 10


@define
class GeneralConfig:
    pretty: bool = True
    do_safety_checks: bool = True
    use_compression: bool = True
    compression_lib: str = "lz4"
    suppress_coffea_warnings: bool = True


@define
class LocationConfig:
    # default_dataset_paths: list[str]
    # default_era_paths: list[str]
    # default_style_paths: list[str]
    # default_template_paths: list[str]
    cache_location: str = ".application_data/cache"


@define
class Config:
    general: GeneralConfig
    paths: LocationConfig
    execution: ExecutionConfig


CONFIG = Config(
    general=GeneralConfig(),
    execution=ExecutionConfig(),
    paths=LocationConfig(),
)
