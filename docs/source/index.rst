OneStopCoffea Analyzer
======================

The OneStopCoffea Analyzer (OSCA) is a modular, columnar analysis framework built on `coffea <https://github.com/CoffeaTeam/coffea>`_ and `dask <https://dask.org/>`_.
It aims to simplify the more tedious aspects of doing HEP analyses -- dataset management, systematics, distributed execution, and postprocessing -- while providing enough flexibility to meet a variety of analysis goals.


Key Features
------------

* **Modular Pipeline Architecture**: Analyses are composed of independent, reusable modules chained into pipelines. Common tasks like jet filtering, b-tagging, and histogram production are already provided.
* **Configuration-Driven**: Both the core analysis and postprocessing workflows are defined entirely through YAML configuration files, with Jinja2 template support for reuse across similar pipelines.
* **Automatic Systematics**: Shape and weight systematics are handled through dynamic parameters. The framework automatically re-runs only the affected portions of the pipeline, using aggressive caching to avoid redundant computation.
* **Distributed Execution**: Easily switch from local execution to full-scale distributed computation.
* **Robust Job Management**: Built-in support for checking result completeness, patching failed jobs, and merging outputs.
* **Postprocessing Framework**: A separate configuration-driven system for producing publication-quality plots, ratio plots, cutflow tables, and Combine datacards from result files.


How the System Works
--------------------

At a high level, the analysis flow looks like this:

1. You write a **YAML configuration** describing your analysis: which modules to run, which datasets to process, and how to handle systematics.
2. The framework loads your configuration, resolves your datasets, and builds an **Analyzer** containing one or more **pipelines** -- ordered sequences of modules.
3. An **Executor** runs each dataset sample through the pipelines, chunking the data as needed and optionally distributing across workers.
4. Results are collected into a tree structure (**ResultGroup**) and serialized to ``.result`` files.
5. **Postprocessing** reads these result files, merges and scales them, and produces plots, tables, or datacards according to a separate YAML configuration.

For a more detailed look at the architecture, see the :doc:`concepts/architecture` page.


Where to Start
--------------

* **New to the framework?** Start with the :doc:`getting_started/installation` and :doc:`getting_started/quickstart` pages.
* **Setting up an analysis?** The :doc:`user_guide/analysis_configuration` page is the most comprehensive reference for writing configuration files.
* **Writing custom modules?** See the :doc:`developer_guide/writing_modules` page.
* **How to actually run things?** Check the :doc:`user_guide/cli_reference`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started/getting_started
   concepts/concepts
   user_guide/user_guide
   postprocessing/postprocessing
   examples/examples
   developer_guide/developer_guide

