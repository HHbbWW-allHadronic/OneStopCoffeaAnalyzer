Analysis Configuration
======================

This is the most important page in this documentation from the user perspective.
This framework is primarily declarative, you say *what* you want to do rather than *how* you want to do it.
The analysis configuration file is where you define *what* your analysis does -- which modules to run, which datasets to process, how to handle systematics, and how to execute the computation.
Understanding how to write these files is essential to using the framework.

The configuration is a YAML file, optionally enhanced with Jinja2 templates for reuse.
When loaded, the framework first renders any Jinja2 templates, then parses the resulting YAML and constructs the internal Python objects that drive the analysis.


Overview of Top-Level Sections
------------------------------

A configuration file is divided into the following top-level sections:

.. code-block:: yaml

    # The analyzer itself: pipelines and their modules
    analyzer:
      default_run_builder: ...
      pipeline_name_1:
        - module_name: ...
        - module_name: ...
      pipeline_name_2:
        - module_name: ...

    # Which datasets to process with which pipelines
    event_collections:
      - dataset: 'pattern*'
        pipelines: [pipeline_name_1, pipeline_name_2]

    # Additional executors beyond the built-in ones
    extra_executors:
      my_executor:
        executor_name: ImmediateExecutor
        chunk_size: 10000

    # Paths to user-defined module packages
    extra_module_paths:
      - path/to/my_modules.py

    # Additional dataset/era definition directories
    extra_dataset_paths:
      - path/to/my_datasets/
    extra_era_paths:
      - path/to/my_eras/

    # Regex patterns for xrootd file location ranking
    location_priorities: [".*(T0|T1|T2).*", "eos"]

    # Postprocessing configuration (can also be in a separate file)
    Postprocessing:
      processors: ...

Let's examine each of these sections in detail.


The ``analyzer`` Block
----------------------

This is the top level heading describging the "analyzer" object, which contains the core logic for processing data and producing outputs.

The ``default_run_builder`` Field
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``default_run_builder`` controls how systematic variations are constructed when a module requests a multi-run (see :doc:`../concepts/systematics` for details).
It is specified by ``strategy_name``:

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: NoSystematics

Available strategies:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Strategy
     - Description
   * - :class:`~analyzer.core.run_builders.NoSystematics`
     - Run only the central value. Good for testing and initial development.
   * - :class:`~analyzer.core.run_builders.CompleteSysts`
     - Run central + all weight variations + all shape variations.
   * - :class:`~analyzer.core.run_builders.WeightsOnly`
     - Run central + weight variations only (no shape systematics).
   * - :class:`~analyzer.core.run_builders.SignalOnlySysts`
     - Full systematics for signal datasets, central-only for backgrounds.
   * - :class:`~analyzer.core.run_builders.LimitSysts`
     - Like :class:`~analyzer.core.run_builders.CompleteSysts`, but filtered by a pattern. Requires a ``systs`` field with a pattern to match against variation names.
   * - :class:`~analyzer.core.run_builders.LimitSystsBackground`
     - Like :class:`~analyzer.core.run_builders.LimitSysts`, but signal datasets always get central-only.
   * - :class:`~analyzer.core.run_builders.UnscaledOnly`
     - Run with the special ``UNSCALED`` tag (produces unweighted histograms).

Example with :class:`~analyzer.core.run_builders.LimitSysts`:

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: LimitSysts
        systs: "JES*"   # Only run JES-related systematics


Defining Pipelines
^^^^^^^^^^^^^^^^^^

Every key under ``analyzer:`` other than ``default_run_builder`` defines a **pipeline**.
A pipeline is an ordered sequence of modules that events are processed through.
Pipelines correspond more or less to analysis regions.
Each pipeline can define different selections, corrections, and result outputs.
For example, the code below defines two pipelines, named ``my_signal_pipeline`` and ``my_control_pipeline``.

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: NoSystematics

      my_signal_pipeline:
        - module_name: GoldenLumi
        - module_name: JetFilter
          input_col: Jet
          output_col: GoodJet
          min_pt: 30
          max_abs_eta: 2.4

      my_control_pipeline:
        - module_name: JetFilter
          input_col: Jet
          output_col: GoodJet
          min_pt: 200
          max_abs_eta: 2.4

Each item in the list is a module configuration.
The ``module_name`` field identifies which module class to instantiate, and all other fields are passed as configuration parameters to that module's constructor.

.. note::
   A :class:`~analyzer.modules.common.load_columns.LoadColumns` module is automatically prepended to every pipeline by the framework.
   It is responsible for loading the raw event data from the input files.


Configuring Modules
^^^^^^^^^^^^^^^^^^^

Each module is an ``attrs`` class, and its fields become the YAML configuration parameters.
The only required field is ``module_name``, which identifies the module class.
All other fields are specific to the module.

For example, the :class:`~analyzer.modules.common.jets.JetFilter` module has the following configurable fields:

.. code-block:: yaml

    - module_name: JetFilter
      input_col: Jet          # Which column of jets to read
      output_col: GoodJet     # Where to store the filtered jets
      min_pt: 30              # Minimum pT cut
      max_abs_eta: 2.4        # Maximum |eta| cut
      include_jet_id: True    # Apply jet ID
      include_pu_id: False    # Apply pileup jet ID

See the :doc:`builtin_modules` for the complete list of built-in modules and their parameters.
You can also use ``./osca search-modules "query"`` to search for available modules from the command line.


Conditional Module Execution with ``should_run``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Any module can have a ``should_run`` field that controls whether it executes based on the dataset metadata.
If the condition evaluates to ``False``, the module is skipped entirely.

This is useful when the same pipeline processes both Data and MC, or multiple eras that require different corrections.
For most modules this is already set to a sensible default.
For example, for all scale factor modules, ``should_run`` defaults to ``sample_type: MC``, and does not need to be set by the user.

.. code-block:: yaml

    - module_name: PileupSF
      should_run:
        sample_type: MC    # Only run for MC, not Data

    - module_name: JetEtaPhiVeto
      input_col: GoodJet
      eta_range: [-3.2, -1.2]
      phi_range: [-1.77, -0.87]
      should_run:
        year: "2018"       # Only apply this veto in 2018

Available conditions:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Condition
     - Description
   * - ``sample_type: MC`` or ``sample_type: Data``
     - Match on the sample type.
   * - ``year: "2018"``
     - Match on the era name.
   * - ``run: 2``
     - Match on Run 2 vs Run 3 (2016/17/18 are Run 2).
   * - ``require_all: [...]``
     - AND: all sub-conditions must be true.
   * - ``require_any: [...]``
     - OR: at least one sub-condition must be true.
   * - ``require_not: ...``
     - NOT: the sub-condition must be false.

These can be nested arbitrarily, though it is rarely needed:

.. code-block:: yaml

    - module_name: SomeCorrectionNeededForNon2016MC
      should_run:
        require_all:
          - sample_type: MC
          - require_not:
              year: "2016"


The ``event_collections`` Block
-------------------------------

This section maps datasets to pipelines.
It is a list of entries, each specifying a dataset **pattern** and the list of pipelines to run on matching datasets.

.. code-block:: yaml

    event_collections:
      - dataset: 'data_JetHT_*'
        pipelines: [my_control_pipeline]

      - dataset: 'qcd_*'
        pipelines: [my_signal_pipeline, my_control_pipeline]

      - dataset: 'signal_2018_312_*'
        pipelines: [my_signal_pipeline]

The ``dataset`` field uses the pattern matching system (see :doc:`../concepts/pattern_matching`).
The simplest usage is glob-style matching: ``*`` matches anything, ``?`` matches a single character.

Each dataset that matches a pattern will be processed with all the specified pipelines.

.. warning::
   Each dataset must match **exactly one** entry in ``event_collections``.
   If a dataset matches multiple entries, the framework will raise an error.


The ``extra_executors`` Block
-----------------------------

The framework comes with many premade executors (see :doc:`executors`), but you can also define additional ones in your configuration:

.. code-block:: yaml

    extra_executors:
      my_fast_test:
        executor_name: ImmediateExecutor
        chunk_size: 5000

      my_condor:
        executor_name: LPCCondorDask
        chunk_size: 100000
        min_workers: 10
        max_workers: 200
        worker_memory: "4GB"
        container: "/cvmfs/unpacked.cern.ch/..."

You can then use these with ``./osca run -e my_fast_test ...``.
The ``executor_name`` field identifies the executor class, and additional fields are its configuration parameters.

The ``extra_module_paths`` Field
--------------------------------

If you have written custom analysis modules (see :doc:`../developer_guide/writing_modules`) in an external location, you need to tell the framework where to find them.
This field takes a list of Python file paths.

.. code-block:: yaml

    extra_module_paths:
      - modules/my_analysis_modules.py
      - modules/my_special_corrections.py

These files are loaded and their module classes registered before the configuration is parsed, so you can use your custom modules just like built-in ones in the pipeline definitions.

.. note::
   The paths are relative to the working directory, not the configuration file.


The ``extra_dataset_paths`` and ``extra_era_paths`` Fields
----------------------------------------------------------

By default, the framework loads datasets from ``analyzer_resources/datasets/`` and eras from ``analyzer_resources/eras/``.
These fields allow you to add additional directories:

.. code-block:: yaml

    extra_dataset_paths:
      - my_private_datasets/

    extra_era_paths:
      - my_custom_eras/

This is useful when you have private samples or custom era configurations that should not be checked into the main repository.


The ``location_priorities`` Field
---------------------------------

When accessing remote files via xrootd, files may be replicated across multiple sites.
This field provides a list of regex patterns used to rank the sites, with earlier patterns being preferred:

.. code-block:: yaml

    location_priorities: [".*(T0|T1|T2).*", "eos"]

In this example, Tier-0/1/2 sites are preferred, followed by EOS, and then anything else.


Using Jinja2 Templates
----------------------

Configuration files are rendered as Jinja2 templates before YAML parsing.
This is a very useful feature for avoiding duplication when you have many pipelines that share common module sequences.

A Jinja2 **macro** defines a reusable block of YAML:

.. code-block:: yaml+jinja

    {% macro common_cleanup() -%}
    - module_name: GoldenLumi
    - module_name: VetoMap
      input_col: Jet
    - module_name: VetoMapFilter
      input_col: Jet
      output_col: Jet
    - module_name: NoiseFilter
    {%- endmacro %}

    {% macro jet_objects() -%}
    - module_name: JetFilter
      input_col: Jet
      output_col: GoodJet
      min_pt: 30
      max_abs_eta: 2.4
    - module_name: HT
      input_col: GoodJet
    {%- endmacro %}

    analyzer:
      default_run_builder:
        strategy_name: NoSystematics

      signal_pipeline:
        {{ common_cleanup() | indent(4) }}
        {{ jet_objects()  | indent(4) }}
        - module_name: NObjFilter
          input_col: GoodJet
          selection_name: njets
          min_count: 4

      control_pipeline:
        {{ common_cleanup() | indent(4) }}
        {{ jet_objects() | indent(4) }}
        - module_name: NObjFilter
          input_col: GoodJet
          selection_name: njets
          min_count: 2

Macros can also be defined in a separate base file and included using Jinja2's ``{% include %}`` or by placing them in a file that other configs extend.
The template search path includes both the directory containing the config file and its parent directory.
Note that the ``indent`` filter is needed to ensure that the intendation is properly applied.

This system supersedes the older YAML anchor (``&anchor`` / ``*anchor``) approach and is much more flexible -- macros can take arguments, use conditionals, and be organized across multiple files.


Complete Annotated Example
--------------------------

Here is a complete, realistic configuration that demonstrates all the major features:

.. code-block:: yaml+jinja

    {% macro common_cleanup() -%}
    - module_name: GoldenLumi
    - module_name: VetoMap
      input_col: Jet
    - module_name: VetoMapFilter
      input_col: Jet
      output_col: Jet
    - module_name: NoiseFilter
    {%- endmacro %}

    analyzer:
      # Use complete systematics (central + all weight + all shape variations)
      default_run_builder:
        strategy_name: CompleteSysts

      # Signal region pipeline
      SignalRegion:
        # Data quality cleanup (shared across pipelines)
        {{ common_cleanup() | indent(4) }}

        # JEC/JER corrections (these declare shape systematics)
        - module_name: JetScaleCorrections
          input_col: Jet
          output_col: Jet
        - module_name: JetResolutionCorrections
          input_col: Jet
          genjet_col: GenJet
          output_col: Jet

        # Object definitions
        - module_name: JetFilter
          input_col: Jet
          output_col: GoodJet
          include_pu_id: True
          include_jet_id: True
          min_pt: 30
          max_abs_eta: 2.4
        - module_name: ElectronMaker
          input_col: Electron
          output_col: GoodElectron
          working_point: "veto"
          min_pt: 10
          max_abs_eta: 2.4

        # Event-level quantities
        - module_name: HT
          input_col: GoodJet
        - module_name: Count
          input_col: GoodJet
          output_col: NJet

        # Selections (these create boolean masks, not applied yet)
        - module_name: NObjFilter
          input_col: GoodJet
          selection_name: njets
          min_count: 4
          max_count: 6
        - module_name: NObjFilter
          input_col: GoodElectron
          selection_name: zero_electron
          min_count: 0
          max_count: 0

        # Apply all pending selections at once
        - module_name: SelectOnColumns
          sel_name: selection

        # MC weights (only run for MC via should_run)
        - module_name: PileupSF
          should_run:
            sample_type: MC
        - module_name: BJetShapeSF
          input_col: GoodJet
          should_run:
            sample_type: MC

        # Produce a histogram (this triggers the systematics multi-run)
        - module_name: SimpleHistogram
          hist_name: HT
          input_cols: [HT]
          axes:
            - name: HT
              start: 0
              stop: 3000
              bins: 60
              unit: GeV

    # Map datasets to pipelines
    event_collections:
      - dataset: 'data_JetHT_*'
        pipelines: [SignalRegion]
      - dataset: 'qcd_*'
        pipelines: [SignalRegion]
      - dataset: 'signal_2018_*'
        pipelines: [SignalRegion]

    # Prefer Tier-1/2 sites for xrootd
    location_priorities: [".*(T0|T1|T2).*", "eos"]


Understanding the Selection Pattern
-----------------------------------

One of the more subtle aspects of the configuration is how selections work.
Modules like :class:`~analyzer.modules.common.selection.NObjFilter`, :class:`~analyzer.modules.common.event_level_corrections.GoldenLumi`, and :class:`~analyzer.modules.common.jets.VetoMap` do **not** immediately filter events.
Instead, they create boolean mask columns under a special ``Selection`` namespace.

The :class:`~analyzer.modules.common.selection.SelectOnColumns` module then applies all pending selections at once by AND-ing the masks and filtering the events.
This design has several advantages:

1. **Cutflow tracking**: :class:`~analyzer.modules.common.selection.SelectOnColumns` can record how many events pass each cut individually, producing a cutflow table.
2. **N-1 plots**: Since selections are stored as separate masks, it is easy to compute N-1 efficiencies. This is not possible if cuts are applied immediately.
3. **Flexibility**: You can apply selections at different points in the pipeline (e.g., a preselection before corrections, then a final selection after).

.. code-block:: yaml

    # These modules create boolean masks:
    - module_name: NObjFilter
      input_col: GoodJet
      selection_name: njets
      min_count: 4

    - module_name: NObjFilter
      input_col: GoodElectron
      selection_name: zero_electron
      max_count: 0

    # At this point NO FILTERING of events has occured.  All events are still
    # present in the event collection.

    # This module applies all pending masks and filters events:
    - module_name: SelectOnColumns
      sel_name: preselection    # Name for the cutflow
      save_cutflow: True        # Save cutflow results

    # At this point all subsequent modules will only operate on the filtered events.
    

You can also specify which selections to apply if you only want a subset:

.. code-block:: yaml

    - module_name: SelectOnColumns
      sel_name: preselection
      selection_names: [njets, zero_electron]   # Only apply these two


For more details on how this works internally, see :doc:`../concepts/columns_and_data`.
In general, however, this is rarely used, and it is easier to simply let the the module track which cuts have been applied.
