from pathlib import Path
import pickle as pkl
from analyzer.configuration import CONFIG
import lz4.frame


def runOrCacheMtimePaths(func, paths, cache_object_name, *args, **kwargs):
    if not CONFIG.USE_CACHE:
        return func(*args, **kwargs)

    cache_path = Path(CONFIG.CACHE_PATH) / f"{cache_object_name}.pklz4"
    if not cache_path.exists():
        ret = func(*args, **kwargs)
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        with lz4.frame.open(cache_path, "wb") as f:
            pkl.dump(ret, f)
    else:
        cache_time = cache_path.stat().st_mtime
        mtimes = [Path(p).stat().st_mtime for p in paths]
        if any(x > cache_time for x in mtimes):
            print("HERE1")
            ret = func(*args, **kwargs)
            cache_path.parent.mkdir(exist_ok=True, parents=True)
            with lz4.frame.open(cache_path, "wb") as f:
                pkl.dump(ret, f)
        else:
            print("HERE2")
            with lz4.frame.open(cache_path, "rb") as f:
                ret = pkl.load(f)
    return ret
