Installation
============

This project supports several installation workflows depending on the host and expected scale-out solution.
The recommended environment and package manager is ``uv``, which provides a much faster alternative to pip/poetry.

We strongly recommend using the provided ``setup.sh`` script to run within an Apptainer container. This ensures a consistent environment and prevents dependency conflicts, which is especially important if you plan to submit jobs to HTCondor. Alternatively, you can install the environment directly on your local machine.

Container Installation (Recommended)
------------------------------------

The easiest way to get started, particularly on the Fermilab LPC cluster, is by using the provided ``setup.sh`` script. This script automatically:

1. Launches an AlmaLinux 9 Apptainer container.
2. Mounts appropriate host directories (automatically detecting if you are on the LPC).
3. Creates a Python virtual environment using ``uv`` (inheriting container system packages).
4. Installs the project dependencies via ``uv sync`` (adding ``lpc`` and ``condor`` extras if on the LPC).

Prerequisites
^^^^^^^^^^^^^

Ensure the following are available on your system:

- Apptainer (or Singularity)
- Access to ``/cvmfs`` (for the base container image)
- ``uv`` installed and available in your ``PATH`` inside the container (e.g., loaded via your ``~/.bashrc``)

If you haven't installed ``uv`` yet, you can do so using the official installer on your host machine:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Running the Setup Script
^^^^^^^^^^^^^^^^^^^^^^^^

Simply execute the script from the project root:

.. code-block:: bash

   ./setup.sh

Once the script finishes, you will be dropped into a shell inside the container with the virtual environment ready to use for analysis.


Local Installation (Without Container)
--------------------------------------

If you prefer to run on your local machine without a container, you can install the environment directly using ``uv``.

Install ``uv``
^^^^^^^^^^^^^^

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Restart your shell or ensure ``uv`` is on your ``PATH``. Verify the installation:

.. code-block:: bash

   uv --version

Create the environment
^^^^^^^^^^^^^^^^^^^^^^

From the project root directory, create a virtual environment and sync the dependencies:

.. code-block:: bash

   uv venv 
   uv sync --no-managed-python --link-mode copy

.. note:: 
   If you are on the LPC and installing locally without ``setup.sh``, you may want to add ``--extra lpc --extra condor`` to the sync command.


Verify the Installation
-----------------------

Regardless of the installation method chosen, run the following command to verify that the environment was set up successfully:

.. code-block:: bash

   uv run python3 -m analyzer --help
