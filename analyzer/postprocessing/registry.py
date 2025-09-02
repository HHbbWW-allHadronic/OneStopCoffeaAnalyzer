import yaml
import copy
from analyzer.utils.structure_tools import deepMerge
from analyzer.utils.querying import pattern_expr_adapter

REGISTRY = {}


def registerPostprocessor(cls):
    REGISTRY[cls.__name__] = cls


def loadPostprocessors(file_path, root="Postprocessing", defaults_root="PostDefaults"):
    with open(file_path, "r") as f:
        d = yaml.safe_load(f)
    data = d[root]
    default = d.get(defaults_root, {})
    actions = []
    catalog_path = d.get("catalog_path")
    num_files = d.get("num_files")
    use_samples_as_datasets = d.get("use_samples_as_datasets", False)
    drop_patterns = pattern_expr_adapter.validate_python(
        d.get("drop_sample_patterns", [])
    )
    for post_action in data:
        final = deepMerge(copy.deepcopy(default), post_action)
        actions.append(REGISTRY[post_action["action"]](**final))
    return actions, catalog_path, drop_patterns, use_samples_as_datasets, num_files
