import importlib.util
import sys


def loadModuleFromPath(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = foo
    spec.loader.exec_module(foo)
