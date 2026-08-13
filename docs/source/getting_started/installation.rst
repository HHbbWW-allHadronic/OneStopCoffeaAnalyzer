Installation
============

Prerequisites
-------------

If you are just doing local work, you can simply work with the framework in your host OS, and can install it like any other python project.
However, if you intend to do distributed job submission, then to ensure a consistent software environment, you will want to use a container.
The framework generally runs inside an Apptainer (Singularity) container to ensure consistency between workers nodes and your local environment.

You will need:

- **Git** for cloning the repository (and also for tracking your own work!).
- **Apptainer** (or Singularity) available on your system. This should be available on most scientific computing clusters. If not, please complain to your local system administrator.
- **uv** A most excellent Python package manager. Install it from `astral.sh <https://docs.astral.sh/uv/getting-started/installation/>`_ if not already available.

The default container image is hosted on CVMFS and does not need to be downloaded manually.
If you don't have access to CVMFS you could use another container that is available on both your interactive machine and the remote workers.
Ideally choose a container that has some of the heavier dependencies already installed, such as pytorch, xrootd, etc.


Setup
-----

Clone the repository:

.. code-block:: bash

    git clone https://github.com/UMN-CMS/OneStopCoffea.git
    cd OneStopCoffea


The ``./osca`` wrapper will automatically launch the container, create a virtual environment on first run, sync dependencies, and execute your command.
If you prefer to handle things yourself, you can manually set up your container and install the packages using uv manually.
Note though that you will need to be careful when dealing with distributed execution!


First Run
---------

To verify the installation, check if you can run the program:

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

.. note::

   In the future this should be extended to other common CMS hosts, especially lxplus.


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


Make sure it is added to your ``PATH``.


**Container image not found**

Ensure CVMFS is mounted and the path exists:

.. code-block:: bash

    ls /cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/

If CVMFS is not available then the modify the osca script to change the apptainer bindings.
Also make sure to propagate any relevant container changes to the executors.


**Other environment errors**

Try removing the ``.venv`` directory and letting ``./osca`` recreate it:

.. code-block:: bash

    rm -rf .venv
    ./osca --help

