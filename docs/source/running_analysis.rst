Running the Analyzer
====================

The analysis framework provides a cli for executing
configured analyses on datasets. The primary entry point for processing data is the ``run`` command.


Basic Usage
-----------

The general syntax is:

.. code-block:: bash

   uv run python -m analyzer run [OPTIONS] INPUT OUTPUT

Where:

- ``INPUT`` is the path to the analysis configuration file
- ``OUTPUT`` is the directory where results will be written. **Ensure this directory is empty before starting**

Command-Line Options
--------------------

The following options are supported:

.. option:: -e, --executor TEXT

   **Required.**

   Name of the executor configuration to use. This must correspond to an
   executor defined in the ``extra_executors`` section of the configuration
   file, or one of the premade executors.

.. option:: --max-sample-events INTEGE

   Optional limit on the number of events processed *per sample*.
   This is primarily intended for debugging.
   If not specified, all available events are processed.


Example
-------

Local Test Run
^^^^^^^^^^^^^^

Run the analyzer locally using a simple executor:

.. code-block:: bash

   uv run python -m analyzer run \
     -e single-process-local \
     config/analysis.yaml \
     output/

Limiting Events
^^^^^^^^^^^^^^^

To process only a limited number of events per dataset.
This is primarily useful for debugging locally.

.. code-block:: bash

   uv run python -m analyzer run \
     -e single-process-local \
     --max-sample-events 10000 \
     config/analysis.yaml \
     output/

Distributed Execution
^^^^^^^^^^^^^^^^^^^^^

Make sure you are in a tmux session before launching this command.
The dask scheduler runs on a local node.
Without tmux (or a similar tool), the processing will crash if the ssh connection is closed.


Additionally, it is highly recommended that you first do a limited event run locally to check for any bugs.

To run using a distributed executor (e.g. Condor + Dask):

.. code-block:: bash

   uv run python -m analyzer run \
     -e dask-condor-lpc-4G-100000 \
     config/analysis.yaml \
     output/

