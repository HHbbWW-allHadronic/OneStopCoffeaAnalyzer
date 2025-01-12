# Introduction


This the repository for the single-stop analyzer, used by the UMN-CMS group's single-stop analysis group.
This repository contains a python package designed to allow the definition and execution the single-stop analysis.
This includes
- Definitions of all used dataset
- Region definitions
- Descriptions of histograms and other analysis artifacts.
- Handling of MC weights.
- Handling of systematics, both for scale and shape.
- Automatic scale-out with dask and condor
- Postprocessing utilities for creating plots, scale factors, and more


## Installation

To begin, clone the repository to your desired location 

``` shell
git clone git@github.com:UMN-CMS/SingleStopCoffea.git
```

Then follow the instructions to get set up.

### On a system with CVMFS

If you have access to CVMFS, the easiest way to get started is to simple run 

``` shell
source setup.sh
```

This will run a setup script that will create a complete environment, and use this same environment is used on worker nodes.

If this is the first time you have run the analyzer, you will also to populate the replica cache using

``` shell
analyzer generate-replicas
```
This may take some time as we find all locations for the files in our datasets.


# Running the Analyzer

You can run the complete analysis in a single command:

``` shell
python3 -m analyzer run configurations/<YOUR_CONFIG>.yaml -o results/my_results_file.pkl -e <EXECUTOR_CHOICE>
```
This will run the analysis defined by the configuration file `<YOUR_CONFIG>.yaml` using the chosen executor
Of course, this will be very slow, since the complete analysis is processing billions of events.
You can speed things up by specifying a distributed computation system.
To run on condor, using 100 workers, each with 4GB you can run

``` shell
analyzer run-analysis configurations/single_stop_complete.yaml -o results/my_results_file.pkl -t lpccondor -w 100 -m 4GB
```

While developing, you may instead want to use a different configuration
``` shell
analyzer run-analysis configurations/my_personal_configuration.yaml -o results/my_results_file.pkl -t local
```

We will discuss much more about configuration in <Configuration in Depth>.

# Inspecting and Processing Results

Processing is done using the `post-processor` subcommand.

``` shell
analyzer post-process configuration/single_stop_post.yaml
```

The post-processor will automatically handle things like scaling of MC histograms. 
Like the main analysis, post-processing is configuration driven. 
The configuraiton yaml defines what histograms and other graphs should be produced, how they should be saved, etc. 


# Configuration In Depth

# Writing new Modules

