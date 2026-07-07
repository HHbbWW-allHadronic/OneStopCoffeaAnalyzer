# OneStopCoffea

The OneStopCoffea Analyzer (OSCA) is a modular, columnar analysis framework built on [coffea](https://github.com/CoffeaTeam/coffea) and [dask](https://dask.org/). It aims to simplify the more tedious aspects of doing analyses while providing enough flexibility to meet a variety of analysis goals.

See the full [documentation](https://umn-cms.github.io/OneStopCoffea/index.html) for more information.

## Key Features
* Automatic scale-out with dask.
* Composable, prebuilt modules to accomplish a variety of common tasks.
* Flexible handling of systematics, including arbitrarily composed shape and weight systematics.
* Automatic handling of MC weights.
* A configuration-driven postprocessing framework with support for plots and tables.
* Analysis recovery, patching of failed jobs, and quick results inspection utilities.

## Configuration-Driven System

The framework defines both the core physics analysis and postprocessing workflows entirely through YAML configuration files.

### 1. Main Analysis Configuration
An analysis is composed of pipelines, each containing a sequence of configurable modules. The master configuration defines the pipelines, maps them to datasets, and specifies the execution backend. 

* **Pipelines & Modules**: Modules handle selections, filtering, or transformations (e.g., `GoldenLumi`, `JetFilter`). Modules are chained together into pipelines.
* **Datasets**: You link datasets to pipelines and choose which datasets to process.
* **Executors**: Choose the computational backend, controlling local execution or distributed scale-out (e.g., `DaskExecutor`, `ImmediateExecutor`).

### 2. Postprocessing Configuration
Postprocessing operates on the output `.result` files, producing plots, tables, and other formats using a separate YAML file.

* **Processors**: Map inputs to final outputs like `RatioPlot`, `Histogram1D`, or `CutflowTable`.
* **Transforms**: Alter underlying histogram axes prior to processing (e.g., `SliceAxes`, `RebinAxes`, `SelectAxesValues`).

## Command Line Interface (CLI)

The primary entry point for the framework is the `analyzer` module. It is recommended to run this within the `.nocontainer_venv` environment or use the `./osca` wrapper script. 

Activate the environment before running commands:
```bash
source .nocontainer_venv/bin/activate
```

### Key Commands and Examples

**1. `run`**: Execute the analyzer on datasets.
Run a small test locally before submitting full jobs:
```bash
python -m analyzer run \
  -e imm-10000 \
  --max-sample-events 10000 \
  config/analysis.yaml \
  test_output/
```

Run the full dataset via Dask on Condor:
```bash
python -m analyzer run \
  -e dask-condor-lpc-4G-100000 \
  config/analysis.yaml \
  full_output/
```

**2. `check`**: Check the status of processed output files.
Use `--only-bad` to identify failed or missing samples.
```bash
python -m analyzer check \
  -c config/analysis.yaml \
  full_output/**/*.result \
  --only-bad
```

**3. `patch`**: Resubmit failed or missing jobs.
```bash
python -m analyzer patch \
  -c config/analysis.yaml \
  -e dask-condor-lpc-4G-100000 \
  -o full_output \
  full_output/**/*.result
```

**4. `postprocess`**: Run postprocessing configurations to generate plots and figures.
```bash
python -m analyzer postprocess \
  config/postprocessing.yaml \
  full_output/**/*.result \
  --parallel 4 \
  --prefix output_figures/
```

**5. `browse`**: Interactively view results.
```bash
python -m analyzer browse full_output/**/*.result
```

**6. `list datasets` and `list samples`**: Query and view available datasets and samples from the configuration.
Use `--filter` to search for specific patterns.
```bash
python -m analyzer list datasets
python -m analyzer list samples --filter "*JetHT*"
```

**7. `search-modules`**: Search for available modules to use in your analysis pipeline.
```bash
python -m analyzer search-modules "Jet"
```
