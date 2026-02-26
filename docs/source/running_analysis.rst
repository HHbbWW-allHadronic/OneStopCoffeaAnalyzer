Running the Analyzer
====================

The analysis framework provides a CLI for executing
configured analyses on datasets. The primary entry point for processing data is the ``run`` command. 
For convenience, you can use the ``./osca`` script (which wraps ``uv run python -m analyzer``) to run any CLI commands.

Key Commands
------------

Here are the most common commands you will use:

- ``./osca run``: Execute the analyzer on datasets
- ``./osca check``: Check the status of processed output files
- ``./osca patch``: Resubmit failed or missing jobs
- ``./osca browse``: Interactively browse results


Analysis Workflow
-----------------

A standard and robust workflow for running your analysis job involves several stages:
test locally -> run on condor -> check for failures -> patch failures -> check again.

1. Test Locally
^^^^^^^^^^^^^^^

Before submitting thousands of jobs to a cluster, it is highly recommended that you do a limited event run locally to catch any bugs in your analyzer code or configuration. The ``immediate`` or ``single-process-local`` execution patterns are perfect for this, combined with ``--max-sample-events``.

.. code-block:: bash

   ./osca run \
     -e imm-10000 \
     --max-sample-events 10000 \
     config/analysis.yaml \
     test_output/

2. Run on Condor
^^^^^^^^^^^^^^^^

Once you've verified the analysis works locally on a small subset of events, you can launch the full dataset processing using Condor (via Dask).

*Note: Make sure you are in a ``tmux`` or ``screen`` session before launching this command! 
The Dask scheduler runs on the local node, and without a persistent session, processing will crash if your SSH connection drops.*

.. code-block:: bash

   ./osca run \
     -e dask-condor-lpc-4G-100000 \
     config/analysis.yaml \
     full_output/

3. Check Results
^^^^^^^^^^^^^^^^

After the condor jobs finish (or if some failed), use the ``check`` command to see the status of the output files. You can pass the ``--only-bad`` flag to easily identify any samples that failed or are missing.

.. code-block:: bash

   ./osca check \
     -c config/analysis.yaml \
     full_output/**/*.result \
     --only-bad

4. Patch Failed Jobs
^^^^^^^^^^^^^^^^^^^^

If you found missing or corrupted results in the previous step, use the ``patch`` command. This command looks at the directory and resubmits processing only for the chunks/samples that did not complete successfully.

.. code-block:: bash

   ./osca patch \
     -c config/analysis.yaml \
     -e dask-condor-lpc-4G-100000 \
     -o full_output \
     full_output/**/*.result

*(You can also use a local executor for patching if only a few small pieces failed.)*

5. Final Check
^^^^^^^^^^^^^^

Run the ``check`` command once more to confirm that the patch successfully completed and your dataset has been processed fully.

.. code-block:: bash

   ./osca check \
     -c config/analysis.yaml \
     full_output/**/*.result


