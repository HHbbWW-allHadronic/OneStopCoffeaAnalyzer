import jinja2

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader
import yaml
from pathlib import Path
import rich
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.lexer import count_newlines
from jinja2.lexer import Token
import re


def makeEnv(path):
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader([path.parent, path.parent.parent])
    )


def loadAndRender(path):
    env = makeEnv(path)
    template = env.get_template(path.name)
    return template.render()


def loadTemplateYaml(path):
    path = Path(path)
    rendered = loadAndRender(path)
    return yaml.safe_load(rendered)
