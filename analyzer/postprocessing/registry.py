import yaml
import copy
from analyzer.utils.structure_tools import deepMerge

REGISTRY = {}


def registerPostprocessor(cls):
    REGISTRY[cls.__name__] = cls



def loadPostprocessors(file_path, root="Postprocessing", defaults_root="Default"):
    with open(file_path, "r") as f:
        d = yaml.safe_load(f)
    data = d[root]
    default = d.get(defaults_root,{})
    actions = []
    for post_action in data:
        final=deepMerge(copy.deepcopy(default),post_action)
        actions.append(REGISTRY[post_action["action"]](**final))
    return actions
        


    
