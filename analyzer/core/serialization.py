from __future__ import annotations


from cattrs import Converter
from cattrs import Converter
from cattrs.strategies import use_class_methods

converter = Converter()
use_class_methods(converter, "_structure", "_unstructure")
