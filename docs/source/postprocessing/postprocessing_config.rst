Postprocessing Configuration
============================

This page describes the YAML configuration for the postprocessing system.


Top-Level Structure
-------------------

A postprocessing configuration file has the following top-level fields:

.. code-block:: yaml

    Postprocessing:
      processors:
        - name: Histogram1D
          ...
        - name: RatioPlot
          ...

      default_style_set:
        styles:
          - pattern: ...
            style: ...

      default_plot_config:
        cms_text: "Preliminary"
        image_type: ".png"

      drop_sample_pattern: ...
      do_merge_and_scale: true

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``processors``
     - List of processors to run. Each produces some output (plots, tables, datacards).
   * - ``default_style_set``
     - Default styling rules applied to all processors unless overridden.
   * - ``default_plot_config``
     - Default plot configuration (CMS label text, image format, etc.).
   * - ``drop_sample_pattern``
     - A pattern to exclude certain samples from the results before processing.
   * - ``do_merge_and_scale``
     - Whether to merge samples and apply MC scaling. Default ``true``.


Processors
----------

Each processor is identified by its ``name`` field and has processor-specific parameters, plus the common fields ``inputs``, ``structure``, and ``output_name``.

Histogram1D
^^^^^^^^^^^

Produces a 1D histogram plot.

.. code-block:: yaml

    - name: Histogram1D
      inputs:
        - "*/*/*/HT"
      scale: log                    # "log" or "linear"
      normalize: false
      show_stacked_unc: true
      structure:
        select: {type: Histogram, pipeline: "Signal*"}
        group: {"era.name": "*"}
        transforms:
          - name: SelectAxesValues
            select_axes_values: {"variation": "central"}
      output_name: "{prefix}/{era.name}/{name}.png"

If the ``structure`` uses ``subgroups`` with ``stacked`` and ``unstacked`` keys, the processor will draw stacked histograms (typically MC backgrounds) with unstacked overlays (typically signal or data).

.. code-block:: yaml

      structure:
        group: {"era.name": "*"}
        subgroups:
          stacked:
            select: {sample_type: MC, dataset_name: "!signal*"}
          unstacked:
            select:
              or_exprs:
                - dataset_name: "signal*"
                - sample_type: Data


RatioPlot
^^^^^^^^^

Produces a plot with a ratio panel below the main panel.

.. code-block:: yaml

    - name: RatioPlot
      inputs:
        - "*/*/*/HT"
      scale: log
      ratio_type: "poisson"         # "poisson", "poisson-ratio", "efficiency", "significance"
      ratio_ylim: [0, 2]
      ratio_hlines: [1.0]
      ratio_height: 0.5
      no_stack: false
      structure:
        select: {type: Histogram}
        group: {"era.name": "*"}
        subgroups:
          denominator:
            select: {sample_type: MC}
          numerator:
            select: {sample_type: Data}
        transforms:
          - name: SelectAxesValues
            select_axes_values: {"variation": "central"}
      output_name: "{prefix}/ratio_{era.name}_{name}.png"

The ``subgroups`` must define ``numerator`` and ``denominator``.

RatioOfRatiosPlot
^^^^^^^^^^^^^^^^^

Produces a double ratio plot.
The structure must define nested ``numerator`` and ``denominator`` subgroups, each containing their own ``numerator`` and ``denominator``.

Histogram2D
^^^^^^^^^^^

Produces a 2D histogram (color map) plot.

.. code-block:: yaml

    - name: Histogram2D
      inputs:
        - "*/*/*/mass_vs_ht"
      scale: log
      normalize: false
      structure:
        select: {type: Histogram}
        group: {"era.name": "*", "dataset_name": "*"}
      output_name: "{prefix}/2d_{era.name}_{dataset_name}_{name}.png"

CutflowTable
^^^^^^^^^^^^

Produces a cutflow table from :class:`~analyzer.core.results.SelectionFlow` results.

.. code-block:: yaml

    - name: CutflowTable
      inputs:
        - "*/*/*/selection"
      format: latex                 # "latex" or "csv"
      structure:
        select: {type: SelectionFlow}
        group: {"era.name": "*"}
      output_name: "{prefix}/cutflow_{era.name}.tex"

PlotSelectionFlow
^^^^^^^^^^^^^^^^^

Produces a visual bar chart of the cutflow.

CombineDatacard
^^^^^^^^^^^^^^^

Produces a Combine datacard for statistical analysis.

.. code-block:: yaml

    - name: CombineDatacard
      inputs:
        - "*/*/*/mass_histogram"
      channel: "{pipeline}"
      structure:
        select: {type: Histogram, sample_type: MC}
        group: {"era.name": "*"}
        subgroups:
          signal:
            select: {dataset_name: "signal*"}
          background:
            select: {dataset_name: "!signal*"}
      output_name: "{prefix}/combine_{era.name}_{dataset_name}"

Significance2D
^^^^^^^^^^^^^^

Produces a 2D significance scan plot (e.g., for mass plane scans).


The ``inputs`` Field
--------------------

The ``inputs`` field specifies which results to operate on using path patterns.
After merge-and-scale, results are organized as ``dataset/pipeline/result_name``, so the typical pattern has three components separated by ``/``:

.. code-block:: yaml

    inputs:
      - "*/*/*/HT"              # Any dataset, any pipeline, result named "HT"
      - "*/*/*/selection"       # Any SelectionFlow named "selection"

Multiple input patterns can be specified to combine different result types:

.. code-block:: yaml

    inputs:
      - "*/*/*/jet_pt"
      - "*/*/*/jet_eta"
      - "*/*/*/HT"

Each pattern selects a set of results, and all matched results are passed together to the ``structure`` for grouping and processing.


The ``structure`` Field (GroupBuilder)
--------------------------------------

The ``structure`` field configures how results are organized before being passed to the processor.
It implements a SQL-esque system for querying and manipulating results.
It is a :class:`~analyzer.postprocessing.grouping.GroupBuilder` with the following sub-fields:

``select``
^^^^^^^^^^

Filters results by metadata:

.. code-block:: yaml

    structure:
      select:
        type: Histogram
        pipeline: "Signal*"
        dataset_name: "!signal*"

Only results matching all conditions are kept.
See :doc:`../concepts/pattern_matching` for the full pattern syntax.

``group``
^^^^^^^^^

Groups results by captured metadata values:

.. code-block:: yaml

    structure:
      group: {"era.name": "*"}

This creates one group per unique era name.
Each group is then processed independently by the processor.

Multiple fields can be used for finer grouping:

.. code-block:: yaml

    structure:
      group: {"era.name": "*", "pipeline": "*"}

``transforms``
^^^^^^^^^^^^^^

Transformations applied to each group.
See :doc:`transforms_reference` for all available transforms.

.. code-block:: yaml

    structure:
      transforms:
        - name: SelectAxesValues
          select_axes_values: {"variation": "central"}
        - name: RebinAxes
          rebin: 2

``subgroups``
^^^^^^^^^^^^^

Subdivides each group into named sub-collections:

.. code-block:: yaml

    structure:
      group: {"era.name": "*"}
      subgroups:
        numerator:
          select: {sample_type: Data}
        denominator:
          select: {sample_type: MC}

The processor receives a dictionary with the subgroup names as keys.
This is commonly used by :class:`~analyzer.postprocessing.basic_histograms.RatioPlot` (requiring ``numerator``/``denominator``) and significance calculations (requiring ``signal``/``background``).

Subgroups can also be a list, in which case the processor receives a list of results.


The ``output_name`` Field
-------------------------

A template string for the output file path.
Metadata values are inserted using ``{key}`` syntax:
The available keys are those that are common to the group being processed, and therefore often correspond to the values passed to ``group``. 

.. code-block:: yaml

    output_name: "{prefix}/{era.name}/{pipeline}/{name}_plot.png"

Common grouping items (make sure that you have specified them in ``group``):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Variable
     - Description
   * - ``{prefix}``
     - The ``--prefix`` argument from the CLI.
   * - ``{era.name}``
     - The era name (e.g., "2018").
   * - ``{pipeline}``
     - The pipeline name (e.g., "SignalRegion").
   * - ``{name}``
     - The result name (e.g., "HT").
   * - ``{dataset_name}``
     - The dataset name (when grouping per-dataset).
   * - ``{dataset_title}``
     - The human-readable dataset title.

**Only metadata keys that are common across all items in a group are available as template variables.**
For example, if you group by ``era.name``, you can use ``{era.name}`` in the output path, but you cannot use ``{dataset_name}``, because it is not guaranteed to be the same for all items in the group.

