from diskcache import Cache
from analyzer.configuration import CONFIG
from pathlib import Path

cache = Cache(directory=Path(CONFIG.general.base_data_path) / CONFIG.cache.cache_subdir)
