Analysis Configuration
======================

This framework defines physics analyses entirely through YAML configuration files.
An analysis is composed of one or *pipelines*, each of which is comprised of one or more *modules*.
In addition to the core analysis logic, we must additionally define the *datasets* which are processed and the *executor* used to process them.


Overview
--------

A configuration file is divided into the following logical sections:

- (Optional) Common module blocks (anchors)
- Analysis pipelines
- Dataset-to-pipeline mapping
- Execution backends
- Other configuration options


A Note on YAML Anchors and Reuse
---------------------

Common analysis logic can be factored into reusable blocks using YAML anchors:

.. code-block:: yaml

   common_cleanup: &common_cleanup
     - module_name: GoldenLumi
     - module_name: VetoMap

Anchors allow sharing logic across multiple pipelines without duplication.
They are expanded inline at runtime.
Note that these are purely a convience feature and can be ignored if you wish.

Simple Example
---------------

The following snippet shows a complete (though very uninteresting) analysis.

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: NoSystematics
      simple_pipeline:
        - module_name: GoldenLumi
        - module_name: VetoMap
          input_col: Jet
        - module_name: NoiseFilter
        - module_name: VetoMapFilter
          input_col: Jet
          output_col: Jet
        - module_name: SelectOnColumns
          sel_name: pre_selection
        - module_name: JetFilter
          input_col: FatJet
          output_col: GoodFatJet
          min_pt: 200
          max_abs_eta: 2.4
    
    
    event_collections:
      - dataset: 'data_JetHT_2018'
        pipelines: [simple_pipeline]
      
    extra_executors:
      test:
        executor_name: ImmediateExecutor
        chunk_size: 10000

    location_priorities: [".*(T0|T1|T2).*","eos"]

There are 4 top levels headings:

- The configuration of the analyzer itself.
- A mapping between datasets and the pipelines they should be processed with.
- A list of additional excutors to make available.
- A list of locations to prioritize when retrieving remote files through xrootd.


The most important item is the analyzer.
The ``default_run_builder`` parameter defines the default strategy to use for systematics.
All other subheadings are the names of pipelines followed by the modules that make them up.
For example, in this analysis we define a single pipeline called ``simple_pipeline`` which contains 5 modules that perform basic cleaning on the events.
The heart of an analysis is the modules that make up the pipelines, and understanding how they work is key.
Even in this simple example, we can identify a number of important features, that generalize even to complex analyses with potentially dozens of modules:

- The name of module to be used is always given by the property ``module_name``.
- Modules are configurable, all properties other than ``module_name`` are configuration parameters.
- Many modules have configurable input and/or output columns.
- The act of creating a boolean mask and the act of applying the selection are separate. In this example, the first three modules create boolean filters that are then applied only when the ``SelectOnColumns`` module is run.

