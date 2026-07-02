Running an Analysis
===================

Once you have constructed your configuration file, added your needed datasets, and written and custum modules, it is time to run!
This page covers the practical workflow of running an analysis, from initial testing through full-scale production and result patching.
Postprocessing, the process of creating plots and tables from the analyzer outputs, will be covered in :doc:`../postprocessing/postprocessing`.


Development Workflow
--------------------

The recommended workflow for developing an analysis is:

1. **Test Locally**: Run on a single dataset with a small event count to verify your pipeline works.
2. **Go to production**: Switch to a distributed executor for full processing.
3. **Check and patch**: Verify completeness and reprocess any failures.
3. **Postprocess and Plot**: Run postprocessing to produce plots and tables.


Step 1: Quick Test
^^^^^^^^^^^^^^^^^^

Use the :class:`~analyzer.core.executors.immediate_exec.ImmediateExecutor` with a small event limit:

.. code-block:: bash

    ./osca run -e imm-10000 \
      --max-sample-events 10000 \
      --filter-dataset 'qcd_ht_2018' \
      config/analysis.yaml test_output/

This runs a single dataset locally, processing at most 10000 events.
It is fast and lets you use standard Python debugging tools, such as breakpoints.

.. tip::
   When developing, add ``--log-level DEBUG`` to see detailed information about module execution, caching, and parameter resolution.


Step 2: Larger Local Test
^^^^^^^^^^^^^^^^^^^^^^^^^

Increase the event count and use the local Dask executor:

.. code-block:: bash

    ./osca run -e local-dask-4G-100000 \
      --max-sample-events 100000 \
      config/analysis.yaml local_output/


Step 3: Production
^^^^^^^^^^^^^^^^^^

For full-scale processing at the LPC, switch to Condor:

.. code-block:: bash

    ./osca run -e lpc-dask-condor-4G-100000 \
      config/analysis.yaml full_output/

This distributes work across many workers using dask-condor.


Step 4: Check Completeness
^^^^^^^^^^^^^^^^^^^^^^^^^^

After a production run, check that all samples were fully processed:

.. code-block:: bash

    ./osca check -c config/analysis.yaml full_output/**/*.result --only-bad

This compares the provenance in result files against the expected samples and reports any incomplete or missing entries.


Step 5: Patch Failures
^^^^^^^^^^^^^^^^^^^^^^

If some samples are incomplete, use ``patch`` to reprocess only the missing portions:

.. code-block:: bash

    ./osca patch -c config/analysis.yaml \
      -e lpc-dask-condor-4G-100000 \
      -o full_output \
      full_output/**/*.result

The ``patch`` command reads the provenance from existing result files, determines which file chunks are missing, and submits only those for processing.
The new results are written alongside the existing ones.


Step 6: Postprocess
^^^^^^^^^^^^^^^^^^^

Generate plots and tables using a postprocessing configuration:

.. code-block:: bash

    ./osca postprocess config/postprocessing.yaml \
      full_output/**/*.result \
      --parallel 4 \
      --prefix plots/

See the :doc:`../postprocessing/postprocessing` section for details on writing postprocessing configurations.


The ``./osca`` Wrapper
----------------------

The ``./osca`` script is a convenience wrapper that handles environment setup automatically.
When run outside a container, it:

1. Launches an Apptainer/Singularity container with the correct image.
2. On first run inside the container, creates a ``uv`` virtual environment and syncs dependencies.
3. Activates the virtual environment.
4. Runs ``python3 -m analyzer`` with your arguments.

On the LPC, it additionally detects the environment and installs LPC-specific and Condor dependencies.
After the first run, the virtual environment is cached and subsequent calls skip the setup step.

You can also work directly inside the container manually:

.. code-block:: bash

    apptainer shell /cvmfs/unpacked.cern.ch/.../coffea-dask-almalinux9:2025.10.2-py3.12
    source .venv/bin/activate
    python -m analyzer run ...


Filtering Datasets and Samples
------------------------------

The ``--filter-dataset`` and ``--filter-sample`` options accept patterns (see :doc:`../concepts/pattern_matching`) and are available on ``run``, ``check``, and ``patch`` commands:

.. code-block:: bash

    # Only process JetHT datasets
    ./osca run -e imm-10000 --filter-dataset '*JetHT*' config.yaml output/

    # Only process a specific sample
    ./osca run -e imm-10000 --filter-sample 'QCD_HT-500to700*' config.yaml output/

These are useful for debugging a specific dataset or rerunning a subset of your analysis.


Browsing Results
----------------

The ``browse`` command provides an interactive terminal UI for exploring result files:

.. code-block:: bash

    ./osca browse full_output/**/*.result

You can navigate the result tree, inspect histogram axes and bin contents, view cutflow tables, and examine metadata.


Debugging Tips
--------------

- Use ``--log-level DEBUG`` to see module execution order, cache hits/misses, and parameter resolution.
- Test locally with ``imm-10000`` for debugging. These run in a single process where breakpoints and print statements work.
