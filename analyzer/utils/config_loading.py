from pathlib import Path

from analyzer.utils.load import loadModuleFromPath
from analyzer.utils.yamlload import loadTemplateYaml


def loadConfigData(path, variable_name):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in (".yaml", ".yml"):
        return loadTemplateYaml(path)

    if suffix == ".py":
        module = loadModuleFromPath(f"_config_{path.stem}", str(path))
        if not hasattr(module, variable_name):
            available = [k for k in dir(module) if not k.startswith("_")]
            raise ValueError(
                f"Python config '{path}' has no variable '{variable_name}'. "
                f"Available names: {available}"
            )
        return getattr(module, variable_name)

    raise ValueError(
        f"Unsupported config file extension '{suffix}' for '{path}'. "
        f"Expected .yaml, .yml, or .py"
    )
