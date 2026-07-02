=======================================
Higgs to 4 Leptons (H->4L) Analysis
=======================================

Intro
-----
This document walks through a complete end-to-end physics analysis using the OneStopCoffea framework, based on the `CMS OpenData H$\\rightarrow$4L tutorial <http://doi.org/10.7483/OPENDATA.CMS.F7HD.P3K4>`_.
A user will interact primarily with analyzer modules and postprocessors, and the vast majority of their efforts will be in programming modules to suit their needs when the existing systems are lacking.
Here, we demonstrate how to write bespoke modules when the premade solutions are insufficient.
That is particularly true in this case, as the outreach dataset does not exactly mirror the NanoAOD format expected by some of our existing modules.

The code is provided in the ``examples/higgs_to_4l/`` directory.

The Analysis Configuration
--------------------------
The top level description of an analysis is declarative.
This is handled by a central YAML configuration, in thise case ``examples/higgs_to_4l/configuration.yaml``.

Pipelines and Modules
^^^^^^^^^^^^^^^^^^^^^
The configuration declares three pipelines: ``FourMuon``, ``FourElectron``, and ``TwoElectronTwoMuon``. 
These pipelines are seaprate analysis regions, corresponding to the 3 possible leptonic ZZ decay channels detectable by CMS as leptons. 

* We begin with our custom ``OpenDataMuonMaker`` or ``OpenDataElectronMaker``.
* For MC samples, we apply a scale factor module.
* Depending on the channel we select for the desired number of leptons, among other properties.
* We then use these leptons to reconstruct the Z and H bosons.
* Finally, the :class:`~analyzer.modules.common.histogram_builder.SimpleHistogram` module simply books histograms for ``HiggsMass``, ``Z1Mass``, and ``Z2Mass``.


Custom Modules and Physics Logic
--------------------------------
While the framework provides many common modules, it is all but guaranteed that something will be missing.
In ``examples/higgs_to_4l/higgs_to_4l.py``, we define the custom :class:`~analyzer.core.analysis_modules.AnalyzerModule` classes specific to this example.

Postprocessing
--------------
The psotprocessing configuration in ``examples/higgs_to_4l/postprocessor.yaml`` shows a complex configuration for producing a "model plot," which shows how background and signal compare to data.
There are a number of complex features showcased, such as transforms and subgrouping.

The final output of the postprocessor is hown below.

.. image:: /_static/higgs_reco.png
   :alt: Higgs Mass Reconstruction
   :align: center
   :width: 800px
