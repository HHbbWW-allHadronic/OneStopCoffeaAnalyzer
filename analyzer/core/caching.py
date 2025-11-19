from diskcache import Cache

cache = Cache(directory="test_cache")

# def makeCached(cache_name, use_arguments=None):
#     memory = Memory(cache_name, verbose=0)
#     def decorator(func):
#         if use_arguments is not None:
#             
# 
#         return memory.cache
#     return decorator
