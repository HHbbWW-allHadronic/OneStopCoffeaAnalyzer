CLI Reference
=============

The primary entry point for the framework is the ``analyzer`` module.
You can run commands using the ``./osca`` wrapper, which handles container setup and virtual environment management automatically:

.. code-block:: bash

    ./osca <command> [options]

This is equivalent to ``python -m analyzer <command>`` when already inside the correct environment.


``run``
-------

Run the analysis on datasets.

.. code-block:: bash

    ./osca run [OPTIONS] CONFIG OUTPUT

**Arguments:**

- ``CONFIG``: Path to the YAML analysis configuration file.
- ``OUTPUT``: Directory where ``.result`` files will be written.

**Options:**

.. list-table::
   :widths: 35 65

   * - ``-e, --executor NAME``
     - Executor to use (e.g., ``imm-10000``, ``lpc-dask-condor-4G-100000``).
   * - ``--max-sample-events N``
     - Maximum events to process per sample. Useful for testing.
   * - ``--filter-dataset PATTERN``
     - Only process datasets matching this pattern.
   * - ``--filter-sample PATTERN``
     - Only process samples matching this pattern.
   * - ``--log-level LEVEL``
     - Set logging level (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).

**Examples:**

.. code-block:: bash

    # Quick local test
    ./osca run -e imm-10000 --max-sample-events 10000 config/analysis.yaml test_output/

    # Full production run on Condor
    ./osca run -e lpc-dask-condor-4G-100000 config/analysis.yaml full_output/

    # Only process signal datasets
    ./osca run -e imm-10000 --filter-dataset 'signal*' config/analysis.yaml output/


``check``
---------

Check the completeness of processed output files against the expected samples.

.. code-block:: bash

    ./osca check [OPTIONS] FILES...

**Options:**

.. list-table::
   :widths: 35 65

   * - ``-c, --config PATH``
     - Analysis configuration (needed to know the expected samples).
   * - ``--only-bad``
     - Only show incomplete or missing samples.
   * - ``--filter-dataset PATTERN``
     - Filter which datasets to check.
   * - ``--filter-sample PATTERN``
     - Filter which samples to check.

**Example:**

.. code-block:: bash

    ./osca check -c config/analysis.yaml full_output/**/*.result --only-bad


``patch``
---------

Resubmit failed or incomplete jobs.
This reads the provenance from existing result files, determines what is missing, and runs only those tasks.

.. code-block:: bash

    ./osca patch [OPTIONS] FILES...

**Options:**

.. list-table::
   :widths: 35 65

   * - ``-c, --config PATH``
     - Analysis configuration.
   * - ``-e, --executor NAME``
     - Executor to use for reprocessing.
   * - ``-o, --output PATH``
     - Output directory for patched results.
   * - ``--filter-dataset PATTERN``
     - Filter which datasets to patch.
   * - ``--filter-sample PATTERN``
     - Filter which samples to patch.

**Example:**

.. code-block:: bash

    ./osca patch -c config/analysis.yaml -e lpc-dask-condor-4G-100000 \
      -o full_output full_output/**/*.result


``postprocess``
---------------

Run postprocessing to generate plots, tables, and other outputs.

.. code-block:: bash

    ./osca postprocess [OPTIONS] CONFIG FILES...

**Arguments:**

- ``CONFIG``: Path to the postprocessing YAML configuration.
- ``FILES``: Input ``.result`` files.

**Options:**

.. list-table::
   :widths: 35 65

   * - ``--parallel N``
     - Number of parallel processes for plot generation.
   * - ``--prefix PATH``
     - Output prefix directory for generated files.
   * - ``--include-sidecar``
     - Include sidecar data in plots.

**Example:**

.. code-block:: bash

    ./osca postprocess config/postprocessing.yaml full_output/**/*.result \
      --parallel 4 --prefix plots/


``browse``
----------

Interactively browse the result tree in a terminal UI.

.. code-block:: bash

    ./osca browse FILES...

**Example:**

.. code-block:: bash

    ./osca browse full_output/**/*.result


``merge``
---------

Merge multiple ``.result`` files into a single file.
Useful for reducing the number of files after patching, or grouping results.

.. code-block:: bash

    ./osca merge [OPTIONS] FILES...

**Options:**

.. list-table::
   :widths: 35 65

   * - ``-o, --output PATH``
     - Output path for the merged file.
   * - ``--group-by FIELDS``
     - Group files before merging. Specifying ``dataset`` merges all samples within a dataset.


``list datasets``
-----------------

List all datasets known to the framework.

.. code-block:: bash

    ./osca list datasets [OPTIONS]

**Options:**

.. list-table::
   :widths: 35 65

   * - ``--filter PATTERN``
     - Only show datasets matching this pattern.

**Example:**

.. code-block:: bash

    ./osca list datasets --filter '*JetHT*'


``list samples``
----------------

List all samples across all datasets.

.. code-block:: bash

    ./osca list samples [OPTIONS]

**Options:**

.. list-table::
   :widths: 35 65

   * - ``--filter PATTERN``
     - Only show samples matching this pattern.


``search-modules``
------------------

Search for available analysis modules by name or description.

.. code-block:: bash

    ./osca search-modules QUERY

**Example:**

.. code-block:: bash

    ./osca search-modules "Jet"     # Find all jet-related modules
    ./osca search-modules "SF"      # Find scale factor modules


``export-adl``
--------------

Export an analysis configuration as an ADL (Analysis Description Language) representation.

.. code-block:: bash

    ./osca export-adl CONFIG

For more information on ADL see the `twiki <https://twiki.cern.ch/twiki/bin/view/LHCPhysics/ADL>`_. 

``describe_analysis``
---------------------

Print a summary of an analysis configuration: what pipelines it defines, which datasets it matches, and what modules are in each pipeline.

.. code-block:: bash

    ./osca describe_analysis CONFIG


``cache``
---------

Manage the disk cache used for dataset and column provenance lookups.

.. code-block:: bash

    ./osca cache list    # Show cache contents
    ./osca cache clear   # Clear the cache
