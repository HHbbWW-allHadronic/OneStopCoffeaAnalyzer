Quick Start
===========

This tutorial walks through a complete analysis workflow -- from configuration to plots -- using a simple example.
By the end, barring any complications, you will have run an analysis, inspected the results, and produced a histogram.


Step 1: The Example Configuration
---------------------------------


We are doing to work with a very simple 
Let us start by looking at the example configuration at ``configurations/example.yaml``:

.. code-block:: yaml

    analyzer:
      default_run_builder:
        strategy_name: NoSystematics

      MyPipeline:
        - module_name: JetFilter
          input_col: Jet
          output_col: GoodJet
          include_pu_id: False
          include_jet_id: False
          min_pt: 30
          max_abs_eta: 2.4

        - module_name: BQuarkMaker
          input_col: GoodJet
          output_col: MedB
          working_point: M

        - module_name: HT
          input_col: GoodJet

        - module_name: Count
          input_col: GoodJet
          output_col: NJet

        - module_name: NObjFilter
          input_col: GoodJet
          selection_name: njets
          min_count: 4

        - module_name: SelectOnColumns
          sel_name: "selection"

        - module_name: SimpleHistogram
          hist_name: HT
          input_cols: [HT]
          axes:
            - name: HT
              start: 0
              stop: 3000
              bins: 60
              unit: GeV

    location_priorities: [".*(T0|T1|T2).*","eos"]

    event_collections:
      - dataset: 'signal_2018_312_*15*0'
        pipelines: [MyPipeline]

Let us break this down:

- **``analyzer``**: Defines one pipeline called ``MyPipeline`` with :class:`~analyzer.core.run_builders.NoSystematics` (central values only).
- **``MyPipeline``**: A sequence of modules that:
  1. Filters jets to those with pT > 30 GeV and :math:`|\eta|` < 2.4.
  2. Identifies medium b-tagged jets.
  3. Computes HT (scalar sum of jet pT).
  4. Counts the number of jets.
  5. Requires at least 4 jets (creates a selection mask).
  6. Produces a histogram of HT.
- **``event_collections``**: Processes signal datasets matching the pattern ``signal_2018_312_*15*0``.

The histogram is filled with all events passing the selection.


Step 2: Run the Analysis
------------------------

Run the example with a small event count for a quick test:

.. code-block:: bash

    ./osca run -e imm-10000 \
      --max-sample-events 10000 \
      configurations/example.yaml \
      test_output/

This processes at most 10000 events from each matching dataset sample using the local single-process executor (``imm-10000``).

You should see log output indicating which datasets and samples are being processed, followed by the result files being saved.


Step 3: Check the Results
-------------------------

Verify that result files were produced:

.. code-block:: bash

    ls test_output/

You should see one or more ``.result`` files, named after the dataset and sample.

Browse them interactively:

.. code-block:: bash

    ./osca browse 'test_output/*.result'

This opens a TUI where you can navigate the result tree and inspect the HT histogram.

.. note::
    
    The histogram rendering in the browser is quite rough, and should not be used for real analysis decisions.
    The browser should be used for quickly checking that outputs look roughtly as you expect.
    Making "real" plots should be done using the postprocessing configuration explained below.
    

Step 4: Write a Postprocessing Configuration
--------------------------------------------

Create a file called ``my_postprocessing.yaml``:

.. code-block:: yaml

    Postprocessing:
      processors:
        - name: Histogram1D
          inputs:
            - "*/*/*/HT"
          scale: log
          structure:
            select:
              type: Histogram
            group: {"era.name": "*"}
            transforms:
              - name: SelectAxesValues
                select_axes_values: {"variation": "central"}
          output_name: "{prefix}/HT_{era.name}.png"

This configuration tells the postprocessing system to:

1. Find all results matching the path ``*/*/*/HT`` (any dataset, any sample, any pipeline, result named "HT").
2. Select only :class:`~analyzer.core.results.Histogram` type results.
3. Group by era name.
4. Select only the "central" variation from the histogram's variation axis.
5. Produce a 1D histogram plot saved as a PNG.


.. note::
    
    The variation axis is added by default even for situations where no systematic variations exist.
    It is therefore advisable to always select the "central" variation byd efault.




Step 5: Produce the Plot
------------------------

Run the postprocessing:

.. code-block:: bash

    ./osca postprocess my_postprocessing.yaml \
      'test_output/*.result' \
      --prefix plots/

Check the output:

.. code-block:: bash

    ls plots/

You should see a plot file like ``HT_2018.png``.


What's Next
-----------

- See the :doc:`../user_guide/analysis_configuration` page for a comprehensive guide to writing configurations.
- See the :doc:`../concepts/architecture` page to understand how the framework processes data.
- See the :doc:`../developer_guide/writing_modules` page when you need custom analysis logic.
- See the :doc:`../postprocessing/postprocessing_config` page for more postprocessing options.
