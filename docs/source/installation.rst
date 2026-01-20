Installation
============

This project supports several installation workflows depending on the host and expected scale-out solution.
The recommended project manager is ``uv``, which provides a much faster alternative to pip/poetry.
If you expect to need to run on condor, or if you just want to ensure a consistent environment, we recommend that you use the ``setup.sh`` script to activate an apptainer container before running the following steps.


Optionally Enabling the Container (LPC Only)
---------------------------------------------

Ensure the following are available:

- Apptainer (or Singularity)
- Access to ``/cvmfs``

Run the script ``setup.sh`` to launch the container.


Local Installation with ``uv`` 
-------------------------------


Install ``uv``
^^^^^^^^^^^^^^

Install ``uv`` using the official installer:

.. code-block:: bash

   curl -LsSf https://astral.sh/uv/install.sh | sh

Restart your shell or ensure ``uv`` is on your ``PATH``.

Verify the installation:

.. code-block:: bash

   uv --version

Create the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

From the project root directory:

.. code-block:: bash

   uv venv 
   uv sync  --no-managed-python  --link-mode copy


Verify the installation
^^^^^^^^^^^^^^^^^^^^^^^

Run the following command to verify that the installation was successful.

.. code-block:: bash

   uv run python3 -m analyzer --help
