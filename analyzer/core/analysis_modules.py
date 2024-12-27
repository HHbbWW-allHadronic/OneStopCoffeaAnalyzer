"""
ModuleType.Selection
=========
(events, parameters, selection_registry):
selection_registery.add(name, mask, type="or")
selection_registery.add(name, mask, type="and")

ModuleType.Categorization
=========
(events, parameters, histogram_builder):
histogram_builder.addCategory(axis, vals)

ModuleType.Weight
=========
(events, parameters, weights):
variation={vname: (up,down)}
histogram_builder.addWeight(name, weight, variations)

ModuleType.Producer
=========
(events, parameters):
events[""] = ....

ModuleType.Histogram
=========
(events, parameters,  hist_manager):
hist_manager.add(
   Spec(name, axes, storage, description)
   fill_vals,
   no_scale=False,
   weights_to_apply={name: variations}
)
"""

import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Optional, Any
from pydantic import BaseModel, Field
import inspect


class ModuleType(str, enum.Enum):
    Selection = "Selection"
    Categorization = "Categorization"
    Weight = "Weight"
    Histogram = "Histogram"
    Producer = "Producer"


class AnalyzerModule(BaseModel):
    name: str
    type: ModuleType
    description: Optional[str] = None
    _function: Optional[Callable] = None

    def __call__(self, events, analyzer, *args, **kwargs):
        return self._function(events, analyzer, *args, **kwargs)

    def __eq__(self, other):
        return (self.name, self.type) == (other.name, other.type)

    def identity(self):
        return (self.name,)


class ConfiguredAnalyzerModule(BaseModel):
    module: AnalyzerModule
    config: dict[str, Any] = Field(default_factory=dict)

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs, **self.config)

    def __eq__(self, other):
        return (self.module, self.config) == (other.module, other.config)

    def identity(self):
        return self.module.identity + (self.config,)


@dataclass
class ModuleRepo:
    modules: defaultdict[ModuleType, dict[str, AnalyzerModule]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def getFunction(self, type, name):
        return self.modules[type][name]._function


    def get(self, type, name, configuration=None):
        config = configuration or {}
        return ConfiguredAnalyzerModule(module=self.modules[type][name], config=config)

    def register(self, first, *args, **kwargs):
        if isinstance(first, ModuleType):
            return lambda x: self.__registerFunction(x, first)
        elif isinstance(first, AnalyzerModule):
            return self.__registerModuleInstance(analyzer_module, *args, **kwargs)
        elif inspect.isclass(analyzer_module):
            return self.__registerModuleClass(first, *args, **kwargs)

    def __registerModuleInstance(self, analyzer_module: AnalyzerModule):
        type = analyzer_module.type
        name = analyzer_module.name
        self.modules[type][name] = analyzer_module
        return analyzer_module

    def __registerModuleClass(self, analyzer_module):
        self.__registerInstance(analyzer_module())
        return analyzer_module

    def __registerFunction(self, function: Callable, module_type: ModuleType):
        name = function.__name__
        analyzer_module = AnalyzerModule(
            name=name,
            type=module_type,
            description=function.__doc__,
        )
        self.modules[module_type][name] = analyzer_module
        self.modules[module_type][name]._function = function

        return function


MODULE_REPO = ModuleRepo()
