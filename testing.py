from analyzer.utils.querying import configureConverter, BasePattern
from analyzer.core.serialization import setupConverter
from rich import print
from cattrs.converters import Converter


conv = Converter()
setupConverter(conv)

d = {"key1": "pat1", "key2": "pat2"}


