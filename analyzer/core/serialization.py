from __future__ import annotations


from cattrs import Converter
from cattrs import Converter
from cattrs.strategies import use_class_methods

converter = Converter()


def setupConverter(conv):
    import analyzer.core.analysis_modules
    import analyzer.core.event_collection
    import analyzer.core.executors.executor
    import analyzer.core.run_builders
    import analyzer.utils.querying

    # import analyzer.core.results
    import analyzer.core.datasets

    use_class_methods(conv, "_structure", "_unstructure")

    analyzer.core.analysis_modules.configureConverter(conv)
    analyzer.core.event_collection.configureConverter(conv)
    analyzer.core.executors.executor.configureConverter(conv)
    analyzer.core.run_builders.configureConverter(conv)
    # analyzer.core.results.configureConverter(conv)
    analyzer.core.datasets.configureConverter(conv)
    analyzer.utils.querying.configureConverter(conv)

    return conv
