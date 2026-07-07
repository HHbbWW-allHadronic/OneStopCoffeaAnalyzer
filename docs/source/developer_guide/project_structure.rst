Project Structure
=================

This page provides a map of the codebase for developers who need to understand or modify the framework internals.


Directory Layout
----------------

.. code-block:: text

    OneStopCoffea/
    +-- analyzer/                   # The framework code
    |   +-- core/                   # Core runtime and interfaces
    |   |   +-- analysis.py         # Analysis loading (loadAnalysis)
    |   |   +-- analysis_modules.py # Module base classes and lifecycle
    |   |   +-- analyzer.py         # Analyzer: pipeline execution engine
    |   |   +-- columns.py          # Column, TrackedColumns, provenance
    |   |   +-- datasets.py         # Dataset and sample definitions
    |   |   +-- era.py              # Era definitions
    |   |   +-- event_collection.py # FileSet, SourceDescription
    |   |   +-- executors/          # Executor implementations
    |   |   |   +-- executor.py     # Base Executor class
    |   |   |   +-- immediate_exec.py  # Single-process executor
    |   |   |   +-- dask_exec.py    # Dask and Condor executors
    |   |   |   \-- premade_executors.py # Pre-configured executor instances
    |   |   +-- param_specs.py      # ParameterSpec for dynamic parameters
    |   |   +-- results.py          # Result types and serialization
    |   |   +-- run_builders.py     # RunBuilder strategies
    |   |   +-- running.py          # High-level run/patch orchestration
    |   |   +-- serialization.py    # cattrs converter setup
    |   |   \-- caching.py          # Disk cache configuration
    |   |
    |   +-- modules/                # Analysis modules
    |   |   +-- common/             # Built-in modules
    |   |   |   +-- jets.py         # Jet filtering, corrections, combinatorics
    |   |   |   +-- selection.py    # SelectOnColumns, NObjFilter
    |   |   |   +-- histogram_builder.py # HistogramBuilder, SimpleHistogram
    |   |   |   +-- electrons.py    # Electron selection
    |   |   |   +-- muons.py        # Muon selection
    |   |   |   +-- bjet_sf.py      # B-jet scale factors
    |   |   |   +-- categories.py   # Category axis definitions
    |   |   |   +-- hlt_selection.py # Trigger selection
    |   |   |   +-- event_level_corrections.py # Pileup, prefiring, etc.
    |   |   |   +-- skimming.py     # Event skimming to files
    |   |   |   \-- ...
    |   |   \-- utils/              # Module utilities
    |   |
    |   +-- postprocessing/         # Postprocessing framework
    |   |   +-- running.py          # Postprocessor loading and execution
    |   |   +-- processors.py       # BasePostprocessor
    |   |   +-- basic_histograms.py # Histogram1D, RatioPlot, etc.
    |   |   +-- cutflows.py         # CutflowTable, PlotSelectionFlow
    |   |   +-- combine.py          # CombineDatacard
    |   |   +-- aggregate_plots.py  # Significance2D, etc.
    |   |   +-- grouping.py         # GroupBuilder
    |   |   +-- style.py            # StyleSet, StyleRule, CMS colors
    |   |   +-- transforms/         # Transform classes
    |   |   |   +-- registry.py     # Transform base classes
    |   |   |   +-- hist_transforms.py # Histogram transforms
    |   |   |   \-- data_transforms.py # Data transforms
    |   |   \-- plots/              # Plotting functions
    |   |       +-- plots_1d.py     # 1D plot rendering
    |   |       +-- plots_2d.py     # 2D plot rendering
    |   |       \-- common.py       # PlotConfiguration
    |   |
    |   +-- cli/                    # Command-line interface
    |   |   \-- cli.py              # Click commands
    |   |
    |   +-- utils/                  # Shared utilities
    |   |   +-- querying.py         # Pattern matching system
    |   |   +-- structure_tools.py  # Tree operations, metadata tools
    |   |   \-- yamlload.py         # Jinja2 + YAML loading
    |   |
    |   +-- configuration.py        # Global configuration (CONFIG)
    |   \-- logging.py              # Logging setup
    |
    +-- analyzer_resources/         # Data files
    |   +-- datasets/               # Dataset YAML definitions
    |   +-- eras/                   # Era YAML definitions
    |   \-- static/                 # Fonts, styles
    |
    +-- configurations/             # Example analysis configurations
    |   +-- example.yaml
    |   \-- single_stop/            # Single stop analysis configs
    |
    +-- docs/                       # Documentation
    \-- osca                        # Wrapper script


Key Files
---------

The most important files for understanding the framework:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - File
     - Role
   * - ``core/analysis_modules.py``
     - Module base class, lifecycle, ``should_run``, :class:`~analyzer.core.analysis_modules.ModuleAddition`
   * - ``core/columns.py``
     - :class:`~analyzer.core.columns.Column`, :class:`~analyzer.core.columns.TrackedColumns`, provenance tracking, :func:`~analyzer.core.columns.addSelection`
   * - ``core/analyzer.py``
     - Pipeline execution, caching logic, multi-run orchestration
   * - ``core/results.py``
     - All result types, serialization, :func:`~analyzer.core.results.mergeAndScale`
   * - ``core/run_builders.py``
     - :class:`~analyzer.core.run_builders.RunBuilder` strategies, systematic variation construction
   * - ``core/serialization.py``
     - ``cattrs`` converter setup, tagged union configuration
   * - ``utils/querying.py``
     - Pattern matching DSL (:class:`~analyzer.utils.querying.Pattern`, :class:`~analyzer.utils.querying.DeepPattern`, etc.)
   * - ``postprocessing/grouping.py``
     - :class:`~analyzer.postprocessing.grouping.GroupBuilder`: select, group, transform, subgroup pipeline
   * - ``cli/cli.py``
     - All CLI commands


The Serialization System
------------------------

The framework uses ``cattrs`` for converting between YAML/dict representations and Python objects.
This is the mechanism that makes the configuration-driven approach work.

A central ``Converter`` is set up in ``core/serialization.py``, with hooks registered for:

- **Tagged unions**: Module classes are resolved by ``module_name``, executors by ``executor_name``, run builders by ``strategy_name``, transforms by ``name``, and postprocessors by ``name``.
- **Custom types**: :class:`~analyzer.core.columns.Column`, :class:`~analyzer.utils.querying.Pattern`, :class:`~analyzer.core.analysis_modules.MetadataExpr`, :class:`~analyzer.core.datasets.SampleType`, etc., have custom structure/unstructure hooks.
- **Subclass resolution**: ``include_subclasses`` from ``cattrs.strategies`` automatically discovers all subclasses of base types.

This is why importing module files (via ``extra_module_paths``) before parsing the configuration is essential -- the subclass list is built at import time.


Library Dependencies
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Library
     - Purpose
   * - ``coffea`` / ``awkward``
     - Reading NanoAOD data and manipulating columnar event data.
   * - ``dask`` / ``distributed``
     - Distributed execution.
   * - ``attrs`` / ``cattrs``
     - Dataclass definitions and YAML deserialization.
   * - ``click``
     - CLI framework.
   * - ``hist``
     - Histogram objects.
   * - ``mplhep`` / ``matplotlib``
     - CMS-style plotting.
   * - ``lz4``
     - Compression for result files.
   * - ``diskcache``
     - Disk-based caching for dataset and provenance lookups.
   * - ``rich`` / ``textual``
     - Pretty printing and terminal UIs.
   * - ``jinja2``
     - Template rendering for configuration files.
