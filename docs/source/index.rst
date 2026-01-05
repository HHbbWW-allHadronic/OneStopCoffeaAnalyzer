SingleStopCoffea
===========================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   
   installation
   datasets_and_meta
   analysis_structure
   analysis_configuration



The SingleStop Analyzer (SSA) is a modular, columnar analysis framework built on coffea and dask.
It aims to simplify the more tedious aspects of doing analyses while providing enough flexibility to meet a variety of analysis goals.
It has a number of features, including:

* Automatic scale out with dask.
* Easy handling of datasets.
* Composable, prebuilt modules to accomplish a variety of common tasks.
* Flexible handling of systematics, including arbitrarily composed shape and weight systematics.
* Automatic handling of MC weights.
* Analysis recovery and patching of failed jobs.
* Quick results inspection utilities.
* A configuration driven postprocessing framework with support for a variety of common tasks.


 






