from diskcache import Cache
from analyzer.configuration import CONFIG
from pathlib import Path

cache = Cache(directory=Path(CONFIG.general.base_data_path) / CONFIG.cache.cache_subdir)

# def makeCached(cache_name, use_arguments=None):
#     memory = Memory(cache_name, verbose=0)
#     def decorator(func):
#         if use_arguments is not None:
#
#
#         return memory.cache
#     return decorator
