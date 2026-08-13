OneStopCoffea Analyzer
======================

The OneStopCoffea Analyzer (OSCA) is a modular, columnar analysis framework built on `coffea <https://github.com/CoffeaTeam/coffea>`_ and `dask <https://dask.org/>`_.
It aims to simplify the more tedious aspects of doing HEP analyses -- dataset management, systematics, distributed execution, and postprocessing -- while providing enough flexibility to meet a variety of analysis goals.


Key Features
------------

* **Modular Pipeline Architecture**: Analyses are composed of reusable modules chained into pipelines. Many common objects and corrections already have modules implemented.
* **Configuration-Driven**: Both the core analysis and postprocessing workflows are defined entirely through YAML configuration files, with Jinja2 template support for reuse across similar pipelines.
* **Automatic Systematics**: Shape and weight systematics are handled automatically. The framework automatically re-runs only the affected portions of the pipeline, using caching to avoid redundant computation.
* **Distributed Execution**: "Easily" switch from local execution to distributed computation using dask. Currently supports local clusters and htcondor on the lpc.
* **Robust Job Management**: Support for checking result completeness, patching failed jobs, and merging outputs.
* **Postprocessing Framework**: A separate configuration-driven system for producing a range of outputs including, 1D and 2D histograms, ratio plots, cutflow tables, and Combine datacards from result files.


How the System Works
--------------------

The analysis flow looks like this:

1. Write a **YAML configuration** describing your analysis: which modules to run for each region, which datasets to process in each region, and how to handle systematics.
2. The framework loads your configuration, resolves your datasets, and builds an **Analyzer** containing one or more **pipelines** -- ordered sequences of modules.
3. An **Executor** runs each dataset sample through the pipelines, chunking the data as needed and optionally distributing across workers.
4. Results are collected into a tree structure (**ResultGroup**) and serialized to ``.result`` files.
5. The **postprocessing**  system reads these result files, transforms them as necessary, and produces plots, tables, or datacards according to a separate YAML configuration.

For a more detailed look at the architecture, see the :doc:`concepts/architecture` page.


Where to Start
--------------

* For installation help see :doc:`getting_started/installation`.
* For getting your toes wet see: :doc:`getting_started/quickstart` pages.
* For a details on how write the analysis configurations files see :doc:`user_guide/analysis_configuration`.
* To extend the system with your own modules see the :doc:`developer_guide/writing_modules` page.
* For a reference on the command line system see :doc:`user_guide/cli_reference`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started/getting_started
   concepts/concepts
   user_guide/user_guide
   postprocessing/postprocessing
   gallery/index
   examples/examples
   developer_guide/developer_guide

