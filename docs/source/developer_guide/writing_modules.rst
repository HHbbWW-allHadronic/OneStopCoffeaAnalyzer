Writing Modules
===============

This page walks through how to write custom analysis modules.
If the built-in modules do not cover your use case, writing a custom module is the primary way to extend the framework.


Module Basics
-------------

A module is a Python class that:

1. Inherits from :class:`~analyzer.core.analysis_modules.AnalyzerModule` (or another module base class).
2. Uses the ``@define`` decorator from ``attrs``.
3. Implements :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs`, :meth:`~analyzer.core.analysis_modules.AnalyzerModule.outputs`, and :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run`.

.. graphviz::

    digraph lifecycle {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#cbd5e1", fontname="Helvetica", penwidth=1.5];
        edge [color="#64748b", fontname="Helvetica", fontsize=10, penwidth=1.2];
        
        Start [shape=oval, fillcolor="#f1f5f9", label="Module Execution"];
        Inputs [label="inputs()\nDetermine dependencies", fillcolor="#e0f2fe"];
        CacheCheck [shape=diamond, fillcolor="#fef3c7", color="#fcd34d", label="Cache\nHit?"];
        Run [label="run()\nProcess columns", fillcolor="#dbeafe", color="#60a5fa"];
        Outputs [label="outputs()\nDeclare new columns", fillcolor="#e0f2fe"];
        End [shape=oval, fillcolor="#f1f5f9", label="Return Columns & Results"];
        
        Start -> Inputs;
        Inputs -> CacheCheck;
        CacheCheck -> End [label=" Yes ", color="#10b981", fontcolor="#059669"];
        CacheCheck -> Run [label=" No "];
        Run -> Outputs;
        Outputs -> End;
    }

Here is a minimal example:

.. code-block:: python

    from attrs import define
    from analyzer.core.analysis_modules import AnalyzerModule
    from analyzer.core.columns import Column

    @define
    class MyJetCounter(AnalyzerModule):
        """Count jets and store the result as a column."""
        input_col: Column
        output_col: Column

        def inputs(self, metadata):
            return [self.input_col]

        def outputs(self, metadata):
            return [self.output_col]

        def run(self, columns, params):
            import awkward as ak
            jets = columns[self.input_col]
            n_jets = ak.num(jets, axis=1)
            columns[self.output_col] = n_jets
            return columns, []

This module reads a jet collection, counts the jets per event, and stores the count.

In the YAML configuration, it would be used as:

.. code-block:: yaml

    - module_name: MyJetCounter
      input_col: GoodJet
      output_col: NJet


Registering Your Module
-----------------------

The framework discovers modules through Python's import system.
For modules defined in the ``analyzer/modules/`` directory, they are auto-discovered.

For modules defined elsewhere, use the ``extra_module_paths`` field in the configuration:

.. code-block:: yaml

    extra_module_paths:
      - path/to/my_modules.py

The file will be imported before the configuration is parsed, making your module classes available for use in pipeline definitions.

The ``module_name`` field in the YAML must exactly match the class name.


Working with TrackedColumns
---------------------------

The ``columns`` argument to :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run` is a :class:`~analyzer.core.columns.TrackedColumns` object.
See :doc:`../concepts/columns_and_data` for the full API.

Common operations:

.. code-block:: python

    def run(self, columns, params):
        # Read a column
        jets = columns[self.input_col]

        # Access individual fields
        jet_pt = columns[Column("GoodJet.pt")]

        # Metadata
        is_mc = columns.metadata["sample_type"].value == "MC"

        # Write a column
        columns[self.output_col] = result

        return columns, []


Creating Selections
-------------------

To add a selection (boolean mask), use the :func:`~analyzer.core.columns.addSelection` helper:

.. code-block:: python

    from analyzer.core.columns import addSelection

    @define
    class MySelection(AnalyzerModule):
        input_col: Column
        selection_name: str
        threshold: float

        def inputs(self, metadata):
            return [self.input_col]

        def outputs(self, metadata):
            return [Column("Selection", self.selection_name)]

        def run(self, columns, params):
            import awkward as ak
            values = columns[self.input_col]
            mask = values > self.threshold
            addSelection(columns, self.selection_name, mask)
            return columns, []

The selection is stored under ``Selection.<name>`` and tracked in ``pipeline_data["Selections"]`` for later application by :class:`~analyzer.modules.common.selection.SelectOnColumns`.


Producing Histograms
--------------------

To produce a histogram from your module, use the :func:`~analyzer.modules.common.histogram_builder.makeHistogram` helper:

.. code-block:: python

    from analyzer.modules.common.histogram_builder import makeHistogram
    from analyzer.modules.common.axis import RegularAxis

    @define
    class MyHTHistogram(AnalyzerModule):
        input_col: Column

        def inputs(self, metadata):
            return [self.input_col, Column("Weights")]

        def outputs(self, metadata):
            return []

        def run(self, columns, params):
            ht = columns[self.input_col]
            hist_result = makeHistogram(
                "my_ht_histogram",
                columns,
                RegularAxis(name="HT", start=0, stop=3000, bins=60, unit="GeV"),
                ht,
            )
            return columns, [hist_result]

:func:`~analyzer.modules.common.histogram_builder.makeHistogram` returns a :class:`~analyzer.core.analysis_modules.ModuleAddition` that tells the framework to create a :class:`~analyzer.modules.common.histogram_builder.HistogramBuilder` sub-pipeline.
The histogram will automatically include:

- A ``variation`` axis with entries for each systematic variation.
- Any category axes defined in ``pipeline_data["categories"]``.
- Proper weighting from ``Weights`` columns.


Conditional Execution
---------------------

Add a ``should_run`` parameter to make your module conditionally executable:

.. code-block:: python

    from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr

    @define
    class MCSFCorrection(AnalyzerModule):
        should_run: MetadataExpr | None = MetadataExpr.IsSampleType("MC")

        # ... rest of module

The ``should_run`` field defaults to ``None`` (always run), but can be overridden in the YAML config or set to a sensible default in the class definition.

Common defaults:

- ``MetadataExpr.IsSampleType("MC")``: Only MC samples.
- ``MetadataExpr.IsSampleType("Data")``: Only Data.
- ``MetadataExpr.IsYear("2018")``: Only 2018 era.


Declaring Dynamic Parameters (Systematics)
------------------------------------------

If your module introduces a systematic variation, override :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.getParameterSpec`:

.. code-block:: python

    from analyzer.core.param_specs import ParameterSpec

    @define
    class MyCorrection(AnalyzerModule):
        input_col: Column
        output_col: Column

        def getParameterSpec(self, metadata):
            return {
                "my_systematic": ParameterSpec(
                    default_value="central",
                    possible_values={"central", "up", "down"},
                    tags={"shape_variation"},
                )
            }

        def run(self, columns, params):
            variation = params.get("my_systematic", "central")
            jets = columns[self.input_col]

            if variation == "up":
                jets = modify_up(jets)
            elif variation == "down":
                jets = modify_down(jets)

            columns[self.output_col] = jets
            return columns, []

        def inputs(self, metadata):
            return [self.input_col]

        def outputs(self, metadata):
            return [self.output_col]

See :doc:`../concepts/systematics` for more details on how the parameter spec and run builder systems interact.


Testing Your Module
-------------------

The fastest way to test a module is with the :class:`~analyzer.core.executors.immediate_exec.ImmediateExecutor` and a small event count:

.. code-block:: bash

    ./osca run -e imm-10000 \
      --max-sample-events 1000 \
      --filter-dataset 'qcd_ht_2018' \
      --log-level DEBUG \
      config/analysis.yaml test_output/

With ``--log-level DEBUG``, you can see exactly which modules are executing, what their cache keys are, and when cache hits occur.

You can also use standard Python debugging tools (``breakpoint()``, ``print()``) since the immediate executor runs in a single process.


Common Patterns
---------------

**Accessing era-specific corrections:**

.. code-block:: python

    def run(self, columns, params):
        era = columns.metadata["era"]
        correction_file = era["corrections"]["pileup_sf"]
        # Load and apply correction...

**Producing multiple results:**

.. code-block:: python

    def run(self, columns, params):
        result1 = makeHistogram("hist1", columns, axis1, data1)
        result2 = makeHistogram("hist2", columns, axis2, data2)
        return columns, [result1, result2]

**Working with multiple jet collections:**

.. code-block:: python

    @define
    class DijetMass(AnalyzerModule):
        jet_col: Column
        output_col: Column

        def run(self, columns, params):
            jets = columns[self.jet_col]
            j1, j2 = jets[:, 0], jets[:, 1]
            mass = (j1 + j2).mass
            columns[self.output_col] = mass
            return columns, []
