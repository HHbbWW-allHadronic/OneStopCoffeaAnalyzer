Architecture Overview
=====================

Before getting into the details, it is important to understand the goals of the framework and how they shape its design.


The Challenges
--------------

In the author's opinion, the three most difficult parts of designing an analysis system are systematics, multiple regions, and general bookkeeping.

**Systematics** are challenging because they can vastly expand the execution time and memory requirements of the system.
Weight systematics can be handled easily by computing varied event-level weights and producing copies of each result with the varied weights.
Object-level systematics, also called *shape systematics*, are much more challenging.
Since these manipulate actual objects in an event, any calculation using these objects downstream may yield a different result.
Therefore, to compute the effect of a shape systematic, any calculation done on an affected object must be recomputed.

**Multiple Regions** correspond to different event-level selections.
A naive approach is to run each region independently, but regions generally contain significant overlap and we would like to avoid needlessly redoing computations.

**Bookkeeping** is perhaps the most underappreciated challenge.
A Run 2 + Run 3 analysis might contain 8 individual eras, each with 10 datasets containing many samples, hundreds of possible signal MC files, all run over multiple selections with dozens of systematic uncertainties.
Scale factors and ML models may differ across eras.
It gets very challenging to manage this information, let alone use it effectively.

A final higher-level point is that of **speed**.
Any framework must be capable of giving results in a reasonable time-frame.
Iteration time is a major challenge -- if one is not careful, the inclusion of multiple regions and systematics can linearly increase execution time.


How the Architecture Addresses These
------------------------------------

**Bookkeeping** is handled by always keeping the metadata associated with a given input coupled to the output, and allowing for flexible queries of this metadata.
When a sample is processed, the output contains the complete metadata corresponding to said sample, including the information associated with its era, the exact chunks it ran over, and which correction files were used.
This means the results file contains extensive information that can be used in postprocessing steps, without relying on ad-hoc file-name matching or after-the-fact lookups.

**Systematics and multiple regions** are handled through the use of *analysis modules* that encapsulate some manipulation of the data and/or the production of some result.
A data *pipeline* simply consists of a chain of modules through which the events are passed.
Systematics are handled through *dynamic parameters*: a module can declare that it has parameters with multiple possible values (e.g., a JEC systematic with "central", "up", and "down" values).
A downstream module can then request a "multi-run," receiving multiple event collections corresponding to different parameter values and aggregating them into a single result (typically a histogram with a systematic variation axis).

**Efficiency** comes from aggressive caching.
Each module declares its input and output columns.
Before running, the framework checks whether the module has already been executed with the same inputs and parameters, and if so, returns the cached result.
This means that when re-running a pipeline with a different systematic, only the modules whose inputs actually changed are re-executed.


Key Abstractions
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Concept
     - Description
   * - **Analysis**
     - The top-level object loaded from a YAML configuration. Contains an Analyzer, dataset descriptions, executor definitions, and paths.
   * - **Analyzer**
     - Contains one or more named pipelines and a default run builder. Responsible for executing pipelines on chunks of data.
   * - **Pipeline**
     - An ordered sequence of modules. Events flow through each module in order.
   * - **Module**
     - A unit of analysis logic. Reads input columns, optionally modifies data or produces results, writes output columns. Three types: `AnalyzerModule`, :class:`~analyzer.core.analysis_modules.EventSourceModule`, :class:`~analyzer.core.analysis_modules.PureResultModule`.
   * - **TrackedColumns**
     - The data container passed between modules. Wraps a coffea NanoEvents array with provenance tracking and lazy column management.
   * - **Column**
     - A reference to a specific field in the event data (e.g., ``Jet.pt``, :class:`~analyzer.modules.common.jets.HT`, ``Selection.njets``).
   * - **ResultGroup**
     - A tree structure that collects analysis outputs (histograms, cutflows, arrays). Organized as ROOT  ->  dataset  ->  sample  ->  pipeline  ->  results.
   * - **Executor**
     - Responsible for running the analysis, potentially distributing across workers. Handles chunking, preprocessing, and result collection.
   * - **RunBuilder**
     - Determines how systematic variation runs are constructed from the dynamic parameters declared by modules.


Data Flow
---------

The below graphic shows a very simplified view of how data is processed.

.. graphviz::

    digraph dataflow {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#cbd5e1", fontname="Helvetica", penwidth=1.5];
        edge [color="#64748b", fontname="Helvetica", fontsize=10, penwidth=1.2];
        
        Executor [label="Executor\n(Chunking & Distribution)", fillcolor="#f1f5f9"];
        Analyzer [label="Analyzer\n(Pipeline Execution)", fillcolor="#f1f5f9"];
        LoadColumns [label="LoadColumns\n(Read NanoAOD)", fillcolor="#e0f2fe"];
        Module [label="Analysis Module\n(Filter, Calculate)", fillcolor="#e0f2fe"];
        Cache [shape=cylinder, fillcolor="#fef3c7", color="#fcd34d", label="Cache\n(Provenance Check)"];
        ResultGroup [label="ResultGroup\n(.result file)", fillcolor="#dcfce7", color="#86efac"];
        
        Executor -> Analyzer [label=" chunk task "];
        Analyzer -> LoadColumns [label=" starts pipeline "];
        LoadColumns -> Module [label=" TrackedColumns "];
        Module -> Cache [label=" Check execution key "];
        Cache -> Module [label=" Cached output (if hit) ", color="#f59e0b", fontcolor="#b45309"];
        Module -> ResultGroup [label=" Results "];
    }

1. The **Executor** receives the analysis configuration and a list of tasks (one per sample).
   For each task, it determines the files to process and how to chunk them.

2. For each chunk, the Executor calls :meth:`~analyzer.core.analyzer.Analyzer.run`, which iterates over the pipelines assigned to this dataset.

3. Within a pipeline, events flow through modules sequentially:

   a. The implicit :class:`~analyzer.modules.common.load_columns.LoadColumns` module loads the chunk from disk and creates a :class:`~analyzer.core.columns.TrackedColumns` object.
   b. Each subsequent module's ``__call__`` method is invoked:

      - The module's ``should_run`` condition is checked. If false, the module is skipped.
      - The module's :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs` are used to compute a cache key.
      - If a cached result exists, it is returned without re-executing.
      - Otherwise, :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run` is called, which reads from and writes to the :class:`~analyzer.core.columns.TrackedColumns`.
      - The module returns the (potentially modified) columns and a list of results.

   c. If a module returns a :class:`~analyzer.core.analysis_modules.ModuleAddition` (used by histogram builders), the framework may trigger a **multi-run**: re-executing the pipeline with different parameter values and passing all resulting event collections to the aggregator module.

4. Results from all modules in all pipelines are collected into a :class:`~analyzer.core.results.ResultGroup` tree.

5. The :class:`~analyzer.core.results.ResultGroup` is serialized (with LZ4 compression) and written to a ``.result`` file. 


.. note::

   The ``task`` nomenclature is poorly chosen and should probably be changed.


Configuration to Code Mapping
-----------------------------

Understanding how the YAML configuration maps to Python classes can help when debugging or extending the system.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - YAML Section
     - Python Class
   * - Top-level document
     - ``analyzer.core.analysis.Analysis``
   * - ``analyzer:``
     - ``analyzer.core.analyzer.Analyzer``
   * - ``analyzer.pipeline_name.[list item]``
     - Subclass of ``analyzer.core.analysis_modules.AnalyzerModule``
   * - ``analyzer.default_run_builder``
     - Subclass of ``analyzer.core.run_builders.RunBuilder``
   * - ``event_collections.[list item]``
     - ``analyzer.core.analysis.DatasetDescription``
   * - ``event_collections.[].dataset``
     - ``analyzer.utils.querying.Pattern``
   * - ``extra_executors.name``
     - Subclass of ``analyzer.core.executors.Executor``

The ``module_name`` field in module configurations corresponds to the class name of an :class:`~analyzer.core.analysis_modules.AnalyzerModule` subclass.
The framework uses ``cattrs`` tagged unions to automatically resolve the correct class based on this tag.


Caching in Detail
-----------------

Caching is fundamental to the framework's efficiency.
Without it, running an analysis with 20 systematic variations would naively require 20x the computation.
The combination of caching and input/output tracking turns the linear pipelines into an implicit DAG, so only the modules whose inputs actually change are re-executed.

The caching system works through **provenance tracking** in :class:`~analyzer.core.columns.TrackedColumns`:

1. Every column in :class:`~analyzer.core.columns.TrackedColumns` has a *provenance key* -- a hash that changes whenever the column is modified.
2. When a module runs, the framework computes an *execution key* from the module's identity, its dynamic parameters, and the provenance keys of its input columns.
3. Before running, the framework checks if this execution key exists in the cache. If so, the cached output is returned.
4. After running, the module's output columns receive provenance keys derived from the execution key.

This means that if you re-run a pipeline with a different JEC systematic:

- :class:`~analyzer.modules.common.load_columns.LoadColumns` returns cached results (same chunk, same metadata).
- Selection modules return cached results (same input events).
- ``JetCorrection`` re-runs (different systematic parameter).
- :class:`~analyzer.modules.common.jets.JetFilter` re-runs (its input column ``Jet`` changed).
- :class:`~analyzer.modules.common.event_level_corrections.PileupSF` returns cached results (it does not depend on ``Jet``).

This is why declaring correct :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs` and :meth:`~analyzer.core.analysis_modules.AnalyzerModule.outputs` in your modules is important: it is what enables the caching system to determine which modules need re-execution.
