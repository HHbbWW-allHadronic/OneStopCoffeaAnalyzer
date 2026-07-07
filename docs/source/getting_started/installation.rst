Installation
============

Prerequisites
-------------

If you are just doing local work, you can simply work with the framework in your host OS.
However, if you intend to do distributed job submission, then to ensure a consistent software environment, you will want to use a container.
The framework generally runs inside an Apptainer (Singularity) container to ensure consistency between workers nodes and your local environment.

You will need:

- **Git** for cloning the repository.
- **Apptainer** (or Singularity) available on your system. On the LPC, this is already available.
- **uv** for Python package management. Install it from `astral.sh <https://docs.astral.sh/uv/getting-started/installation/>`_ if not already available.

The container image is hosted on CVMFS and does not need to be downloaded manually.


Setup
-----

Clone the repository:

.. code-block:: bash

    git clone https://github.com/UMN-CMS/OneStopCoffea.git
    cd OneStopCoffea

That's it.
The ``./osca`` wrapper script handles the rest -- it will automatically launch the container, create a virtual environment on first run, sync dependencies, and execute your command.


First Run
---------

To verify the installation, run a quick test using the example configuration:

.. code-block:: bash

    ./osca --help

On the first invocation, ``./osca`` will:

1. Launch the Apptainer container.
2. Create a ``.venv`` directory with a Python virtual environment.
3. Install all dependencies via ``uv sync``.

This may take a couple minutes on the first run. 
Subsequent runs skip the setup step and go directly to execution.

Check that you get a help message and no errors occur.


LPC-Specific Notes
------------------

On the LPC (``cmslpc`` hosts), the ``./osca`` script automatically:

- Detects the LPC environment.
- Installs LPC and Condor-specific dependencies.
- Binds the necessary filesystem paths (scratch space, CVMFS, Condor configuration).

No additional configuration is needed.


Manual Environment Setup
------------------------

If you prefer not to use the ``./osca`` wrapper, you can set up the environment manually:

.. code-block:: bash

    # Enter the container
    apptainer shell /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask-almalinux9:2025.10.2-py3.12

    # Inside the container:
    uv venv --system-site-packages
    uv sync
    source .venv/bin/activate
    python -m analyzer run ...


Troubleshooting Installation
----------------------------

**"uv: command not found"**

Install ``uv`` following the instructions at `astral.sh <https://docs.astral.sh/uv/getting-started/installation/>`_.
On the LPC, you can typically install it with:

.. code-block:: bash

    curl -LsSf https://astral.sh/uv/install.sh | sh

Then add ``$HOME/.cargo/bin`` to your ``PATH`` (or source your ``.bashrc``).


**Container image not found**

Ensure CVMFS is mounted and the path exists:

.. code-block:: bash

    ls /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/

If CVMFS is not available then the modify the osca script to change the apptainer bindings.


**Permission errors in .venv**

If you see permission errors, try removing the ``.venv`` directory and letting ``./osca`` recreate it:

.. code-block:: bash

    rm -rf .venv
    ./osca run ...

