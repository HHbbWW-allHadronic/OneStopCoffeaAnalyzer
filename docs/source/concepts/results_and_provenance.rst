Results and Provenance
======================

When your analysis runs, each module can produce *results* -- histograms, cutflows, event counts, saved arrays, etc. 
These results are collected into a tree structure and serialized to disk as ``.result`` files.
Understanding this system is important for both debugging your analysis and writing postprocessing configurations.


The Result Tree
---------------

Results are organized in a tree of :class:`~analyzer.core.results.ResultGroup` objects:

.. graphviz::

    digraph results_tree {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=folder, style="filled", fillcolor="#f8fafc", color="#cbd5e1", fontname="Helvetica", penwidth=1.5];
        edge [color="#64748b", fontname="Helvetica", fontsize=10, penwidth=1.2];
        
        ROOT [fillcolor="#f1f5f9"];
        Dataset [label="dataset: qcd_ht_2018", fillcolor="#e0f2fe", color="#7dd3fc"];
        Sample [label="sample: QCD_HT-500to700", fillcolor="#e0f2fe", color="#7dd3fc"];
        
        node [shape=note, fillcolor="#ffffff"];
        Provenance [label="_provenance\n(chunk tracking)"];
        
        node [shape=folder, fillcolor="#fef3c7", color="#fcd34d"];
        Pipelines [label="pipelines"];
        SignalRegion [label="SignalRegion"];
        
        node [shape=component, fillcolor="#dcfce7", color="#86efac"];
        HT [label="HT (Histogram)"];
        Selection [label="selection (SelectionFlow)"];
        
        ROOT -> Dataset;
        Dataset -> Sample;
        Sample -> Provenance;
        Sample -> Pipelines;
        Pipelines -> SignalRegion;
        SignalRegion -> HT;
        SignalRegion -> Selection;
    }

.. code-block:: text

    ROOT
    \-- dataset_name (e.g., "qcd_ht_2018")
        \-- sample_name (e.g., "QCD_HT-500to700_...")
            +-- _provenance   (tracks which file chunks were processed)
            \-- pipelines
                \-- pipeline_name (e.g., "SignalRegion")
                    +-- HT           (a Histogram)
                    +-- selection    (a SelectionFlow)
                    \-- ...

Each :class:`~analyzer.core.results.ResultGroup` is a named container that holds other results or sub-groups.
The ``_provenance`` entry is special -- it records exactly which file chunks were processed, which is used by the ``check`` and ``patch`` commands.


Result Types
------------

The framework provides several built-in result types.
Each type knows how to merge with another result of the same type (via ``+=``) and how to scale (for MC normalization).

.. list-table::
   :header-rows: 1
   :widths: 25 50 25

   * - Type
     - Description
     - Responds to :meth:`~analyzer.core.results.ResultBase.iscale`?
   * - :class:`~analyzer.core.results.Histogram`
     - A ``hist.Hist`` with axes. The standard output from :func:`~analyzer.modules.common.histogram_builder.makeHistogram`.
     - Yes (scales bin contents)
   * - :class:`~analyzer.core.results.UnscaledHistogram`
     - Like :class:`~analyzer.core.results.Histogram` but ignores scaling. Used for raw event counts.
     - No
   * - :class:`~analyzer.core.results.SelectionFlow`
     - Cutflow data: sequential cut yields, N-1 yields, and single-cut yields.
     - Yes
   * - :class:`~analyzer.core.results.RawSelectionFlow`
     - Like :class:`~analyzer.core.results.SelectionFlow` but not scaled. For unweighted cutflows.
     - No
   * - :class:`~analyzer.core.results.ScalableArray`
     - An awkward or numpy array that scales with luminosity.
     - Yes
   * - :class:`~analyzer.core.results.RawArray`
     - An array that is not scaled.
     - No
   * - :class:`~analyzer.core.results.SavedColumns`
     - A dictionary of named arrays (e.g., for saving event-level data to disk).
     - Partial (adds a "Scale" column)
   * - :class:`~analyzer.core.results.SavedFiles`
     - References to files written to disk by skimming modules.
     - No
   * - :class:`~analyzer.core.results.RawEventCount`
     - A simple event count, unscaled.
     - No
   * - :class:`~analyzer.core.results.ScaledEventCount`
     - A simple event count that scales with luminosity.
     - Yes


Loading Results
---------------

Results can be loaded in Python or browsed interactively.

**Python interface:**

.. code-block:: python

    from analyzer.core.results import loadResults

    results = loadResults(["output/**/*.result"])

    # Navigate the tree
    for dataset_name in results.keys():
        dataset = results[dataset_name]
        for sample_name in dataset.keys():
            sample = dataset[sample_name]
            # Access a specific result
            if "pipelines" in sample.keys():
                pipeline = sample["pipelines"]["SignalRegion"]
                ht_hist = pipeline["HT"]

**Interactive browser:**

.. code-block:: bash

    ./osca browse 'output/**/*.result'

The browser provides a terminal UI for navigating the result tree.
When run in this way, the samples will *not* be merged into datasets.
If you want to merge samples into datasets, then add the ``--merge-datasets`` flag.


Merging and Scaling
-------------------

When loading results from multiple ``.result`` files, the framework automatically merges them by adding together results with the same path in the tree.
Provenance tracking ensures that overlapping chunks are detected and raise an error.

For postprocessing, results typically need to be **merged and scaled** to physical units.
The :func:`~analyzer.core.results.mergeAndScale` function does this:

.. code-block:: python

    from analyzer.core.results import loadResults, mergeAndScale

    results = loadResults(["output/**/*.result"])
    scaled = mergeAndScale(results)

For each dataset, this function:

1. Iterates over all samples in the dataset.
2. For **MC samples**: computes a scale factor of ``lumi * cross_section / processed_events`` and applies it to all results.
3. For **Data samples**: computes ``expected_events / processed_events`` (should be 1.0 if fully processed).
4. Merges all scaled samples within a dataset into a single result.

After :func:`~analyzer.core.results.mergeAndScale`, the result tree has one entry per dataset (rather than one per sample), and histograms contain physically meaningful yields.


Serialization Format
--------------------

Result files use a custom packed format to allow for fast inspection of the file 
without needing to deserialize the entire file.

1. A magic identifier (``sstopresult``)
2. A 4-byte header giving the size of the peek data
3. A pickle of the result *summary* (lightweight, no large arrays)
4. An LZ4-compressed pickle of the full result data

This two-part format allows the ``check`` command to quickly peek at result metadata (provenance, structure) without decompressing the full data.
The ``browse`` command with ``--peek`` uses this same mechanism for fast inspection.


Checking Completeness
---------------------

The ``check`` command compares the provenance in your result files against the expected samples:

.. code-block:: bash

    ./osca check -c config/analysis.yaml output/**/*.result --only-bad

For each sample, it reports:

- Expected number of events (from the dataset definition)
- Found number of events (from the provenance in result files)
- Completion fraction

Samples with less than 100% completion can be reprocessed using the ``patch`` command.
