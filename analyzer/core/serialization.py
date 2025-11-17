from __future__ import annotations


from cattrs import Converter
from cattrs import Converter
from cattrs.strategies import use_class_methods

converter = Converter()
use_class_methods(converter, "_structure", "_unstructure")

def setupConverter(conv):
    import analyzer.core.analysis_modules
    import analyzer.core.event_collection
    import analyzer.core.executors.executor
    import analyzer.core.results

    analyzer.core.analysis_modules.configureConverter(converter)
    analyzer.core.event_collection.configureConverter(converter)
    analyzer.core.executors.executor.configureConverter(converter)
    analyzer.core.results.configureConverter(converter)
