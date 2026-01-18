# Introduction

See the [documentation](https://umn-cms.github.io/SingleStopCoffea/index.html) for more information.

The OneStopCoffea Analyzer (OSCA) is a modular, columnar analysis framework built on coffea and dask.
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
