from diskcache import Cache
from analyzer.configuration import CONFIG

cache = Cache(directory=CONFIG.paths.cache_location)

# def makeCached(cache_name, use_arguments=None):
#     memory = Memory(cache_name, verbose=0)
#     def decorator(func):
#         if use_arguments is not None:
#             
# 
#         return memory.cache
#     return decorator
