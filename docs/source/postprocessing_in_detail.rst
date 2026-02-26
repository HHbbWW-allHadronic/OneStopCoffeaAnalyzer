Postprocessing Configuration
============================

The postprocessing framework allows you to define complex plotting workflows using a YAML configuration file.

Example Configuration
---------------------
Below are some examples of Postprocessing configurations demonstrating different available processors.

Ratio Plot
^^^^^^^^^^
The `RatioPlot` processor takes the numerator and denominator inputs and creates a 1D ratio plot. We can perform selections using metadata.

.. postprocessor-example:: postprocessing_examples/demo_ratioplot.yaml

1D Histogram
^^^^^^^^^^^^
The `Histogram1D` processor creates stacked 1D distributions.

.. postprocessor-example:: postprocessing_examples/demo_histogram1d.yaml

Cutflow Table
^^^^^^^^^^^^^
The `CutflowTable` processor creates a LaTeX cutflow table.

.. postprocessor-example:: postprocessing_examples/demo_cutflow.yaml

Applying Histogram Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can apply transforms to the axes of the underlying histograms before they move onto processors. The `transforms` field in the structure definition allows you to `SelectAxesValues` for a systematic variation, `SliceAxes` to narrow down the range (or drop bins), `RebinAxes`, etc.

**Slicing Axes Example**
The `SliceAxes` transformation uses dictionary items mirroring a normal python list slice: `(start, stop)` inside the dimension to cut limits out of generated graphs.

.. postprocessor-example:: postprocessing_examples/demo_sliceaxes.yaml

**Rebinning Axes Example**
The `RebinAxes` combines grouped adjacent elements out of a plot into a single continuous bin block.

.. postprocessor-example:: postprocessing_examples/demo_rebinaxes.yaml


Dropping Samples
^^^^^^^^^^^^^^^^
Sometimes, especially when dealing with many similar signal mass points, you may only want to process a subset of them to avoid plotting hundreds of overlapping distributions or running out of memory.
You can achieve this in the top-level configuration using ``drop_sample_pattern``. This utilizes the same querying pattern language as other parts of the analyzer, allowing you to filter out specific samples *before* any postprocessing actions are run:

.. code-block:: yaml

    Postprocessing:
      ...
      drop_sample_pattern:
        or_exprs:
          - sample_name: "*50to100*"
          - sample_name: "*100to200*"
          - sample_name: "*200to300*"

Running Postprocessing
----------------------
You can execute a Postprocessing configuration block using the CLI via the `postprocess` subcommand.

.. code-block:: bash

    ./osca postprocess \
      config/analysis.yaml \
      full_output/**/*.result \
      --parallel 4 \
      --prefix output_figures/

This command takes your master configuration file, gathers the specified ``.result`` files, and runs the defined postprocessors (like ``RatioPlot`` or ``CutflowTable``). The ``--prefix`` option allows you to designate an output base directory for all the resulting files.
