from joblib import Memory

memory = Memory("test_cache", verbose=0)


def makeCached():
    return memory.cache
