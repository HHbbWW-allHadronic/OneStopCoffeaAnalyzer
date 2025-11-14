from joblib import Memory

memory = Memory(location, verbose=0)


def makeCached():
    return memory.cache
