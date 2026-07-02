Systematics
===========

Handling systematic uncertainties is one of the core capabilities of the framework.
The system is designed so that shape and weight systematics can be included without linearly multiplying the execution time, thanks to the caching system described in :doc:`architecture`.


How Systematics Work
--------------------

Systematics are handled through **dynamic parameters**.
A module can declare that it has a parameter with multiple possible values -- for example, a JEC correction module might declare a ``jec_systematic`` parameter with values ``central``, ``up``, and ``down``.

When a downstream module (typically a histogram builder) requests a "multi-run," the framework:

1. Collects the :class:`~analyzer.core.param_specs.ParameterSpec` from all modules in the pipeline.
2. Uses the :class:`~analyzer.core.run_builders.RunBuilder` to determine which parameter combinations to execute.
3. Re-runs the pipeline once for each combination, producing a set of event collections.
4. Passes all event collections to the requesting module (a :class:`~analyzer.core.analysis_modules.PureResultModule`), which aggregates them into a single result with a ``variation`` axis.

.. graphviz::

    digraph systematics {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="#f8fafc", color="#cbd5e1", fontname="Helvetica", penwidth=1.5];
        edge [color="#64748b", fontname="Helvetica", fontsize=10, penwidth=1.2];
        
        InputChunk [shape=oval, fillcolor="#f1f5f9", label="Input Chunk\n(TrackedColumns)"];
        RunBuilder [shape=diamond, fillcolor="#fef3c7", color="#fcd34d", label="RunBuilder\nGenerates Variations"];
        
        PipelineCentral [label="Pipeline (Central)", fillcolor="#e0f2fe", color="#7dd3fc"];
        PipelineJECUp [label="Pipeline (JEC Up)", fillcolor="#fee2e2", color="#fca5a5"];
        PipelineJECDn [label="Pipeline (JEC Down)", fillcolor="#dcfce7", color="#86efac"];
        
        HistogramBuilder [label="HistogramBuilder\n(PureResultModule)", fillcolor="#f3e8ff", color="#d8b4fe"];
        FinalResult [shape=note, fillcolor="#ffffff", label="Histogram\n(with variation axis)"];
        
        InputChunk -> RunBuilder;
        RunBuilder -> PipelineCentral;
        RunBuilder -> PipelineJECUp;
        RunBuilder -> PipelineJECDn;
        
        PipelineCentral -> HistogramBuilder;
        PipelineJECUp -> HistogramBuilder;
        PipelineJECDn -> HistogramBuilder;
        
        HistogramBuilder -> FinalResult;
    }

The caching system ensures that modules whose inputs did not change across variations are not re-executed.


``ParameterSpec``
-----------------

Modules declare dynamic parameters by overriding :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.getParameterSpec`:

.. code-block:: python

    from analyzer.core.param_specs import ParameterSpec

    def getParameterSpec(self, metadata):
        return {
            "jec_systematic": ParameterSpec(
                default_value="central",
                possible_values={"central", "JES_up", "JES_down"},
                tags={"shape_variation"},
            )
        }

The key fields of :class:`~analyzer.core.param_specs.ParameterSpec`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``default_value``
     - The value used when no variation is requested.
   * - ``possible_values``
     - The set of all allowed values for this parameter.
   * - ``tags``
     - A set of tags classifying this parameter. Common tags: ``weight_variation``, ``shape_variation``.
   * - ``driven_by``
     - A dictionary mapping driver parameter names to mapping functions. See "Driven Parameters" below.

The ``tags`` field is how the :class:`~analyzer.core.run_builders.RunBuilder` knows which parameters are weight systematics vs shape systematics.


Run Builders
------------

A :class:`~analyzer.core.run_builders.RunBuilder` determines which parameter combinations to actually execute.
It receives the complete :class:`~analyzer.core.param_specs.ParameterSpec` from all modules in the pipeline and returns a list of ``(name, parameter_dict)`` tuples.

The ``default_run_builder`` in your configuration selects the strategy:

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: CompleteSysts

Available strategies:

**NoSystematics**
    Always returns just ``[("central", {})]``. Only the default parameter values are used.

**CompleteSysts**
    Runs central + all weight variations + all shape variations.
    Each variation is run independently (one parameter changed at a time from central).

**WeightsOnly**
    Runs central + weight variations only. Shape systematics are skipped.

**SignalOnlySysts**
    Full systematics for datasets where ``"signal"`` appears in the dataset name or ``is_signal`` is set in metadata.
    Central-only for everything else.

**LimitSysts**
    Like :class:`~analyzer.core.run_builders.CompleteSysts` but filtered: only variations whose names match a given pattern are included.

    .. code-block:: yaml

        default_run_builder:
          strategy_name: LimitSysts
          systs: "JES*"

**LimitSystsBackground**
    Like :class:`~analyzer.core.run_builders.LimitSysts` but signal datasets always get central-only.

**UnscaledOnly**
    Returns ``[("UNSCALED", {})]``. Used to produce unweighted histograms.

You can also combine multiple builders using the ``+`` operator in Python, though this is not directly exposed in YAML.


Driven Parameters
-----------------

Sometimes one systematic is correlated with another.
For example, certain b-tagging systematics only apply when a specific JES variation is active.

This is handled through the ``driven_by`` field of :class:`~analyzer.core.param_specs.ParameterSpec`:

.. code-block:: python

    def getParameterSpec(self, metadata):
        return {
            "btag_sf_systematic": ParameterSpec(
                default_value="central",
                possible_values={"central", "jes_up", "jes_down"},
                tags={"shape_variation"},
                driven_by={
                    "jec_systematic": lambda jec_val: {
                        "JES_up": "jes_up",
                        "JES_down": "jes_down",
                    }.get(jec_val),
                },
            )
        }

When the ``jec_systematic`` parameter is set to ``JES_up``, the ``btag_sf_systematic`` is automatically set to ``jes_up``.
Driven parameter values are excluded from the independent variation list, so they are not varied on their own.


Weight vs Shape Systematics
---------------------------

The distinction between weight and shape systematics is important:

**Weight systematics** modify only the event weight.
Examples: pileup scale factors, b-tagging weights, L1 prefiring corrections.
These are cheap to compute because they only require re-weighting, not re-processing.

**Shape systematics** modify actual physics objects.
Examples: JEC/JER variations, MET corrections.
These are more expensive because any downstream computation depending on the modified objects must be re-run.

The framework handles both types uniformly through the dynamic parameter system, but the :class:`~analyzer.core.run_builders.RunBuilder` uses the tags to decide which to include.
A module that produces a weight variation should tag its parameter with ``weight_variation``, while one that modifies object quantities should use ``shape_variation``.
