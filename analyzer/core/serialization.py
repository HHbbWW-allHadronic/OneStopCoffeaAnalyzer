from __future__ import annotations


from cattrs import Converter
from cattrs.strategies import use_class_methods, configure_union_passthrough
from typing import TypeVar, Type

converter = Converter()


T  = TypeVar("T")

# def structureListOrSingle(data: T | list[T], t: Type[T|list[T]] ,conv)-> list[T] | T:
#     if isinstance(data, list):
#         return [conv.structure(x, t) for x in data]
#     else:
#         return conv.structure(data, t)

def setupConverter(conv):
    import analyzer.core.analysis_modules
    import analyzer.core.event_collection
    import analyzer.core.executors.executor
    import analyzer.core.run_builders
    import analyzer.utils.querying
    import analyzer.modules.common.axis 

    # import analyzer.core.results
    import analyzer.core.datasets

    # conv.register_structure_hook(structureListOrSingle)

    #conv.register_structure_hook(int | float | None, lambda x, t: x)
    configure_union_passthrough(int | float , conv)
    configure_union_passthrough(str | int , conv)


    use_class_methods(conv, "_structure", "_unstructure")

    analyzer.utils.querying.configureConverter(conv)
    analyzer.core.analysis_modules.configureConverter(conv)
    analyzer.core.event_collection.configureConverter(conv)
    analyzer.core.executors.executor.configureConverter(conv)
    analyzer.core.run_builders.configureConverter(conv)
    # analyzer.core.results.configureConverter(conv)
    analyzer.core.datasets.configureConverter(conv)
    analyzer.modules.common.axis.configureConverter(conv)

    return conv
