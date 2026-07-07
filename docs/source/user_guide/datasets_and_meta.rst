Datasets and Metadata
=====================

The framework's dataset system provides a structured way to describe the data and MC samples your analysis will process.
Datasets, samples, and eras are defined in YAML files and loaded into repositories that the framework queries at runtime.


File Organization
-----------------

By default, dataset definitions live under ``analyzer_resources/datasets/`` and era definitions under ``analyzer_resources/eras/``.
Both directories are scanned recursively for ``.yaml`` files.

You can add additional directories using ``extra_dataset_paths`` and ``extra_era_paths`` in your analysis configuration.

A typical layout:

.. code-block:: text

    analyzer_resources/
    +-- datasets/
    |   +-- 2018/
    |   |   +-- qcd_inclusive.yaml
    |   |   +-- data_jetht.yaml
    |   |   +-- signal_312.yaml
    |   |   \-- signal_313.yaml
    |   \-- 2017/
    |       +-- qcd_inclusive.yaml
    |       +-- data_jetht.yaml
    |       +-- signal_312.yaml
    |       \-- signal_313.yaml
    \-- eras/
        +-- 2017.yaml
        +-- 2018.yaml
        \-- 2022.yaml


Dataset Definition
------------------

A dataset YAML file contains a list of datasets.
Each dataset has one or more samples:

.. code-block:: yaml

    - name: qcd_inclusive_2018
      title: QCD
      sample_type: MC
      era: '2018'
      samples:
        - name: QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8
          n_events: 14394786
          das_path: /QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM
          x_sec: 1127000.0
        - name: QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8
          n_events: 84434559
          das_path: /QCD_HT100to200_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM
          x_sec: 23590000000.0

          ...



Dataset Fields
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``name``
     - Yes
     - Unique identifier for the dataset. Used in pattern matching and result file naming.
   * - ``title``
     - Yes
     - Human-readable title. Used in plot legends and labels, can include TeX math.
   * - ``era``
     - Yes
     - Name of the era this dataset belongs to. Must match an era name.
   * - ``sample_type``
     - Yes
     - Either ``MC`` or ``Data``. Determines scaling behavior.
   * - ``samples``
     - Yes
     - List of samples in this dataset.
   * - ``other_data``
     - No
     - Arbitrary dictionary of extra metadata. Accessible in modules and postprocessing.


Sample Fields
^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Required
     - Description
   * - ``name``
     - Yes
     - Unique name within the dataset.
   * - ``n_events``
     - Yes
     - Total number of events in the sample. Used for scaling.
   * - ``x_sec``
     - MC only
     - Cross section in fb (or whatever unit is consistent with the era's luminosity).
   * - ``source``
     - Yes
     - Describes where to find the event data. See "Source Descriptions" below.


Source Descriptions
-------------------

The ``source`` field tells the framework where to find the actual event files.
There are two types:

``DasCollection``
^^^^^^^^^^^^^^^^^

Resolves files from CMS DAS (Data Aggregation System) via Rucio:

.. code-block:: yaml

    source:
      das_path: /QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM

The framework queries Rucio for the file replicas and their site locations.
The ``location_priorities`` configuration controls which sites are preferred.

``FileListCollection``
^^^^^^^^^^^^^^^^^^^^^^^

Specifies files directly:

.. code-block:: yaml

    source:
      files:
        - /store/user/me/custom_ntuple_1.root
        - /store/user/me/custom_ntuple_2.root

This is useful for privately produced samples.


Simplified Format
-----------------

For datasets with a single sample, the sample fields can be inlined directly into the dataset:

.. code-block:: yaml

    - name: data_JetHT_2018A
      title: Data
      sample_type: Data
      era: '2018'
      n_events: 171484635
      das_path: /JetHT/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD

This is equivalent to having a single sample with the same name as the dataset.


The ``other_data`` Field
------------------------

The ``other_data`` field allows you to attach arbitrary metadata to a dataset.
This metadata is accessible in module :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run` methods via ``columns.metadata["other_data"]`` and in postprocessing pattern matching.

This is commonly used for signal MC to attach mass points and other physics parameters:

.. code-block:: yaml

    - name: signal_2018_312_1000_400_combined
      title: $m_{\tilde{t} } = 1000\ \mathrm{GeV}, m_{\tilde{\chi}^{\pm}} = 400\ \mathrm{GeV}$
      sample_type: MC
      era: '2018'
      other_data:
        stop_mass: 1000
        chargino_mass: 400
        coupling: '312'
      samples:
      - name: signal_2018_312_1000_400_combined
        n_events: 56000
        x_sec: 385.32
        files:
        - root://cmseos.fnal.gov//store/user/ckapsiak/SingleStop/official_samples/2018/signal_312_1000_400_plus.root
        - root://cmseos.fnal.gov//store/user/ckapsiak/SingleStop/official_samples/2018/signal_312_1000_400_minus.root


Era Definition
--------------

Era files define the run-era-specific parameters:

.. code-block:: yaml

    - name: '2018'
      energy: 13
      lumi: 59.83

      trigger_names:
        HT: PFHT1050
        AK8SingleJetPt: AK8PFJet400_TrimMass30

      golden_json: https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions18/13TeV/Legacy_2018/Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt

      ...



The era metadata is accessible in modules via ``columns.metadata["era"]`` and can be used for era-dependent corrections, trigger selection, and postprocessing grouping.


The ``DatasetRepo``
-------------------

At runtime, all dataset YAML files are loaded into a :class:`~analyzer.core.datasets.DatasetRepo` -- a dictionary-like object keyed by dataset name.
The repo is then queried by the ``event_collections`` patterns in your analysis configuration to determine which datasets to process.

.. warning::
   Dataset names must be unique across all files. Loading two files that define a dataset with the same name will raise an error.
