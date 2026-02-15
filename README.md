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


# Todos
 
 - [ ] Convert slow column provenance dict to a more efficient structure, like a trie.
 - [ ] Move more of the postprocessing backend to mplhep 1.0
 - [ ] Add postprocessor for ABCD slicing
 - [ ] Improve postprocessor structuring to allow for more complex multi-result transforms and multi-stage postprocessing.
 - [ ] Add executor for UMN cluster
 - [ ] Work to ensure systematic name conformance
 - [ ] Improve documentation, dynamically generated plots, etc.
 - [ ] Include simple complete analysis template
 - [ ] More tests
 - [ ] Add more modules for electrons, muons, photons, MET, etc.
 - [ ] Include more tools for manipulating the cache, ie removing known bad sites, etc
 - [ ] Improve result browser, render histograms, show approx sizes.
 - [ ] In the immediate executor show warnings about possible issues, ie especially large histograms etc to help catch these before full condor runs.
 - [ ] More general ML modules
 - [ ] Lots more tests

 
 
