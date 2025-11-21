from pathlib import Path
from attrs import define, field
from cattrs import structure
from importlib import resources
from pathlib import Path


@define
class ExecutionConfig:
  warn_ulimit_ratio: int 

@define
class GeneralConfig:
  pretty: bool 
  do_safety_checks: bool 
  use_compression: bool 
  compression_lib: str 
  suppress_coffea_warnings: bool 

@define
class LocationConfig:
  default_dataset_paths: list[str] 
  default_era_paths: list[str] 
  default_style_paths: list[str]
  default_template_paths: list[str]
  cache_location: str = ".application_data"

    

@define
class Config:
    

CONFIG = None

def initConfig():
  global CONFIG



