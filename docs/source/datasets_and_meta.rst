=======================
 Datasets and Metadata
=======================

The OneStopCoffea Analyzer separates dataset definitions and era-specific configurations from the core code. This allows for flexible management of different data-taking periods and MC campaigns.

The configuration is loaded from two locations:

1. **Eras**: `analyzer_resources/eras/`
2. **Datasets**: `analyzer_resources/datasets/`

Eras
----

Era files (e.g., `2018.yaml`) define global metadata and corrections for a specific data-taking period.

Key fields include:

* `name`: The era name (e.g., '2018').
* `lumi`: Integrated luminosity in fb^-1.
* `trigger_names`: Mapping of abstract trigger names (e.g., `HT`) to concrete HLT paths.
* `jet_corrections`: Paths to JEC/JER files and definitions of systematics.
* `json_pog`: Paths to JSON POG correction files.

See `analyzer_resources/eras/` for examples.


Datasets
--------

Dataset files are organized by era (e.g., `analyzer_resources/datasets/2018/`). They group related samples (like different HT bins of QCD) into a single logical dataset.

Each file is a YAML list containing one or more dataset definitions.

Common fields:


- `name`: Unique identifier for the dataset.
- `title`: Human-readable title.
- `sample_type`: `Data` or `MC`.
- `era`: The era this dataset belongs to.
- `samples`: A list of specific file sets (DAS paths or file lists).


Data Example
~~~~~~~~~~~~

.. code-block:: yaml

  - name: data_JetHT_2018
    title: Data
    sample_type: Data
    era: '2018'
    samples:
      - name: Run2018A-UL2018_MiniAODv2_NanoAODv9-v2
        n_events: 171484635
        das_path: /JetHT/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD
      - name: Run2018B-UL2018_MiniAODv2_NanoAODv9-v1
        n_events: 78255208
        das_path: /JetHT/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD
      - name: Run2018C-UL2018_MiniAODv2_NanoAODv9-v1
        n_events: 70027804
        das_path: /JetHT/Run2018C-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD
      - name: Run2018D-UL2018_MiniAODv2_NanoAODv9-v2
        n_events: 356976276
        das_path: /JetHT/Run2018D-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD



Background MC Example
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

    - name: wjets_to_qq_2018
      title: WJetsQQ
      sample_type: MC
      era: '2018'
      samples:
        - name: WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8
          n_events: 14494966
          das_path: /WJetsToQQ_HT-200to400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM
          x_sec: 2546000.0
        - name: WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8
          n_events: 9335298
          das_path: /WJetsToQQ_HT-400to600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM
          x_sec: 276500.0
        - name: WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8
          n_events: 13633226
          das_path: /WJetsToQQ_HT-600to800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM
          x_sec: 59250.0
        - name: WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8
          n_events: 13581343
          das_path: /WJetsToQQ_HT-800toInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM
          x_sec: 28750.0
    



Metadata Access
---------------

In an `AnalyzerModule`, the `columns.metadata` dictionary provides access to the merged configuration:

1. **Era Metadata**: Everything from the era YAML file is available under the `era` key.
   
   .. code-block:: python
   
       # Access trigger mapping defined in 2018.yaml
       trigger_name = columns.metadata["era"]["trigger_names"]["HT"]

2. **Dataset Metadata**: Top-level keys like `sample_type` and `x_sec` come from the dataset definition.

   .. code-block:: python
   
       if columns.metadata["sample_type"] == "MC":
           weight = columns.metadata["x_sec"]
