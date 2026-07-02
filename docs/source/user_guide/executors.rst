Executors
=========

An executor is responsible for actually running your analysis -- taking the list of tasks (one per sample) and processing them, potentially in a distributed manner.
The choice of executor affects performance, debugging ease, and resource usage.

For testing and debugging, you should use a local single threaded executor.
When the analysis needs to scale up, you should pivot to a distributed executor.

Executors generally have a predefined chunk size -- how many events are processed simultaneously -- and a memory limit. 


Premade Executors
-----------------

The framework ships with many premade executors following a naming convention:

.. code-block:: text

    {type}-{memory}-{chunk_size}

For example, ``imm-10000`` is an immediate executor with chunk size 10000, and ``lpc-dask-condor-4G-100000`` is an LPC Condor Dask executor with 4GB worker memory and chunk size 100000.

You select an executor with the ``-e`` flag:

.. code-block:: bash

    ./osca run -e imm-10000 config/analysis.yaml output/


Executor Types
--------------

``ImmediateExecutor``
^^^^^^^^^^^^^^^^^^^^^

Runs everything in a single process on the local machine.
This is the best choice for testing and debugging.

- Python debuggers and print statements work normally.
- Only limited by the local machine's memory and CPU.

Premade variants:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Name
     - Details
   * - ``imm-1000``
     - Chunk size 1000. Very small, for quick smoke tests.
   * - ``imm-testing``
     - Chunk size 1000, without deepcopy of the analyzer (slightly faster).
   * - ``imm-10000``
     - Chunk size 10000. Good for development.
   * - ``imm-100000``
     - Chunk size 100000. For larger local tests.
   * - ``imm-400000``
     - Chunk size 400000. Full-scale local processing.


``LocalDaskExecutor``
^^^^^^^^^^^^^^^^^^^^^

Runs on the local machine using Dask for multi-process parallelism.

Key parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``chunk_size``
     - Number of events per chunk.
   * - ``min_workers`` / ``max_workers``
     - Number of worker processes.
   * - ``worker_memory``
     - Memory limit per worker (e.g., ``"4GB"``).
   * - ``timeout``
     - Maximum seconds before a worker times out.
   * - ``reduction_factor``
     - Factor for reducing results before accumulation. Useful for memory-intensive analyses.


``LPCCondorDask``
^^^^^^^^^^^^^^^^^

Distributes work across HTCondor workers at the LPC.
This is what you use for full-scale production processing.

- Can scale to hundreds of workers.
- Requires being on an LPC node.
- Requires specifying a container image.

Key parameters:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``chunk_size``
     - Number of events per chunk.
   * - ``min_workers`` / ``max_workers``
     - Range of worker count. Workers scale up to ``max_workers``.
   * - ``worker_memory``
     - Memory per worker (default ``"4GB"``).
   * - ``timeout``
     - Timeout in seconds.
   * - ``reduction_factor``
     - Factor for intermediate result reduction.
   * - ``container``
     - Path to the Apptainer/Singularity container image.


Choosing an Executor
--------------------

Use this decision tree:

1. **Writing and debugging code?** Use ``imm-10000`` with ``--max-sample-events 10000``.
   This processes a small number of events from a single chunk, giving you fast feedback.

2. **Testing on more data?** Use ``local-dask-4G-100000`` to utilize your local machine's cores or patching the last few failed jobs.

3. **Running production?** Use ``lpc-dask-condor-4G-100000`` or similar.
   Start with default memory and increase if you see memory-related failures.

4. **Memory issues?** Try increasing ``worker_memory`` (e.g., ``6GB``, ``8GB``) or using ``reduction_factor: 2``.

5. **Many small tasks?** A smaller ``chunk_size`` can help distribute work more evenly.
   A larger ``chunk_size`` reduces overhead but uses more memory per task.


Defining Custom Executors
-------------------------

You can define custom executors in your analysis configuration:

.. code-block:: yaml

    extra_executors:
      my_test:
        executor_name: ImmediateExecutor
        chunk_size: 5000

      my_condor:
        executor_name: LPCCondorDask
        chunk_size: 100000
        min_workers: 10
        max_workers: 300
        worker_memory: "6GB"
        timeout: 1800
        container: "/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12"

Then use them with:

.. code-block:: bash

    ./osca run -e my_condor config/analysis.yaml output/
