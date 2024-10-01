import yaml

REGISTRY = {}


def registerPostprocessor(cls):
    REGISTRY[cls.__name__] = cls



def loadPostprocessors(file_path, root="Postprocessing"):
    with open(file_path, "r") as f:
        d = yaml.safe_load(f)
    data = d[root]
    actions = []
    for post_action in data:
        actions.append(REGISTRY[post_action["action"]](**post_action))
    return actions
        


    
