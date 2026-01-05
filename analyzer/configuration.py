from pathlib import Path
from attrs import define, field


@define
class ExecutionConfig:
    warn_ulimit_ratio: int = 10


@define
class DatasetConfig:
    default_dataset_paths: list[str]
    default_era_paths: list[str]
    cache_datasets_by_mtime: bool = True
    cache_eras_by_mtime: bool = True


@define
class GeneralConfig:
    pretty: bool = True
    do_safety_checks: bool = True
    use_compression: bool = True
    compression_lib: str = "lz4"
    suppress_coffea_warnings: bool = True
    suppress_xrootd_warnings: bool = True
    base_data_path: str = ".application_data"


@define
class PostConfig:
    static_resource_path: str = "analyzer_resources/static"


@define
class CacheConfig:
    cache_subdir: str = "cache"


@define
class CondorConfig:
    temp_location: str = "condor"


@define
class Config:
    general: GeneralConfig
    condor: CondorConfig
    cache: CacheConfig
    execution: ExecutionConfig
    datasets: DatasetConfig
    post: PostConfig


CONFIG = Config(
    general=GeneralConfig(),
    execution=ExecutionConfig(),
    condor=CondorConfig(),
    cache=CacheConfig(),
    post=PostConfig(),
    datasets=DatasetConfig(
        ["analyzer_resources/datasets"],
        ["analyzer_resources/eras"],
    ),
)
