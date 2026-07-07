import importlib.util
import sys
import cloudpickle

DYNAMIC_MODULES_LOADED = False


def loadModuleFromPath(module_name, path):
    global DYNAMIC_MODULES_LOADED
    DYNAMIC_MODULES_LOADED = True
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[module_name] = mod
    cloudpickle.register_pickle_by_value(mod)
    return mod
