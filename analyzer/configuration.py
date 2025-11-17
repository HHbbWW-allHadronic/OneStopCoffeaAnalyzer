from pathlib import Path
from attrs import define
from cattrs import structure
from importlib import resources



class Config:
  do_safety_checks: bool
  default_dataset_paths: list[str] 
  default_era_paths: list[str]
  default_style_paths: list[str]
  default_template_paths: list[str]
  cache_location: str
  use_compression: bool
  compression_lib: str
  pretty: bool
  suppress_coffea_warnings: bool
  warn_ulimit_ratio: int
    

CONFIG = None

def initConfig():
  global CONFIG



