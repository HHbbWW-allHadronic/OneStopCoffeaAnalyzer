Writing Custom Postprocessors
=============================

If the built-in postprocessors (:class:`~analyzer.postprocessing.basic_histograms.Histogram1D`, :class:`~analyzer.postprocessing.basic_histograms.RatioPlot`, etc.) do not meet your needs, you can write custom postprocessors and transforms.


Custom Postprocessor
--------------------

A postprocessor is a class that inherits from :class:`~analyzer.postprocessing.processors.BasePostprocessor` and implements :meth:`~analyzer.postprocessing.processors.BasePostprocessor.getRunFuncs`.

.. code-block:: python

    from attrs import define
    from analyzer.postprocessing.processors import BasePostprocessor
    import functools as ft

    @define
    class MyCustomPlot(BasePostprocessor):
        """A custom postprocessor that produces some output."""
        output_name: str

        def getRunFuncs(self, group, prefix=None):
            # 'group' is the data structured by the GroupBuilder.
            # It may be a list of ItemWithMeta, a dict, or other nested structure.
            from analyzer.utils.structure_tools import commonDict, dictToDot, dotFormat

            common_meta = commonDict(group)
            output_path = dotFormat(
                self.output_name, prefix=prefix, **dict(dictToDot(common_meta))
            )

            yield ft.partial(self._makePlot, group, output_path)

        @staticmethod
        def _makePlot(group, output_path):
            import matplotlib.pyplot as plt

            # Extract data from group and create your plot
            fig, ax = plt.subplots()

            # ... plotting logic ...
            # matplotlib calls, mplhep, etc

            fig.savefig(output_path)
            plt.close(fig)

Key points:

- :meth:`~analyzer.postprocessing.processors.BasePostprocessor.getRunFuncs` is a generator that yields callables (typically built using ``functools.partial``).
- Each callable will be executed by the postprocessing framework, potentially in parallel.
- The ``group`` parameter contains the data structured by the :class:`~analyzer.postprocessing.grouping.GroupBuilder` (selected, grouped, transformed, and sub-grouped).
- Since we use processes for parallelism, the callables must be picklable if ``--parallel`` is used.

Registration
^^^^^^^^^^^^

Custom postprocessors are automatically discovered through Python's subclass mechanism.
As long as your class inherits from :class:`~analyzer.postprocessing.processors.BasePostprocessor` and is imported before the postprocessing configuration is loaded, it will be available.

The ``name`` field in the YAML configuration must match the class name.
The class is resolved via ``cattrs`` tagged union with ``tag_name="name"``.

.. danger::

    If you are implementing your postprocessor in a new file it must be imported by the framework!


Custom Transform
----------------

Transforms operate on lists of ``ItemWithMeta`` objects and return modified lists.

.. code-block:: python

    from attrs import define
    from analyzer.postprocessing.transforms.registry import TransformHistogram
    from analyzer.utils.structure_tools import ItemWithMeta
    from analyzer.core.results import Histogram

    @define
    class MakeVariancesBigger(TransformHistogram):
        """Multiply all histogram values by a constant."""
        scale_factor: float

        def __call__(self, items: list[ItemWithMeta]):
            ret = []
            for item, meta in items:
                h = item.histogram.copy(deep=True)
                h.view()[...] *= self.scale_factor
                ret.append(ItemWithMeta(
                    Histogram(name=item.name, axes=item.axes, histogram=h),
                    metadata=meta,
                ))
            return ret

Transform base classes:

- :class:`~analyzer.postprocessing.transforms.registry.Transform`: Generic base class.
- :class:`~analyzer.postprocessing.transforms.registry.TransformHistogram`: For transforms operating on histograms.
- :class:`~analyzer.postprocessing.transforms.registry.TransformSavedColumns`: For transforms operating on saved column data.
- :class:`~analyzer.postprocessing.transforms.registry.TransformGroup`: For transforms operating on groups.

Like postprocessors, custom transforms are discovered via subclass resolution and the ``name`` field in YAML.


.. code-block:: yaml

    transforms:
      - name: MakeVariancesBigger
        scale_factor: 1.5
