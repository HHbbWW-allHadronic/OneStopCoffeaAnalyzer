Built-in Modules
================

The framework provides a library of built-in modules for common analysis tasks.
These are organized by category below.

You can search for modules from the command line:

.. code-block:: bash

    ./osca search-modules "Jet"


Data Quality
------------

- :class:`~analyzer.modules.common.event_level_corrections.GoldenLumi`: Filters events based on the golden JSON luminosity list. Applies only to Data.
- :class:`~analyzer.modules.common.event_level_corrections.NoiseFilter`: Applies standard CMS noise filters (MET filters).
- :class:`~analyzer.modules.common.jets.VetoMap`: Marks jets in vetoed detector regions.
- :class:`~analyzer.modules.common.jets.VetoMapFilter`: Removes jets flagged by the veto map.


Jet Modules
-----------

- :class:`~analyzer.modules.common.jets.JetFilter`: Filters jets by pT, :math:`\eta`, jet ID, and pileup jet ID. The primary jet selection module.
- :class:`~analyzer.modules.common.jets.JetScaleCorrections`: Applies Jet Energy Scale (JEC) corrections. Declares shape systematics.
- :class:`~analyzer.modules.common.jets.JetResolutionCorrections`: Applies Jet Energy Resolution (JER) smearing. Declares shape systematics.
- :class:`~analyzer.modules.common.jets.JetEtaPhiVeto`: Vetoes jets in specific :math:`\eta`-\ :math:`\phi` regions (used for hot ECAL regions).
- :class:`~analyzer.modules.common.jets.JetCombos`: Computes invariant masses for combinations of jets.
- :class:`~analyzer.modules.common.bquarks.BQuarkMaker`: Identifies b-tagged jets at a given working point (L, M, T).
- :class:`~analyzer.modules.common.jets.HT`: Computes HT (scalar sum of jet pT).
- :class:`~analyzer.modules.common.column_tools.Count`: Counts objects in a collection.
- :class:`~analyzer.modules.singlestop.selections.VecPt`: Selection based on vector sum pT.


Lepton Modules
--------------

- :class:`~analyzer.modules.common.electrons.ElectronMaker`: Selects electrons by working point, pT, and :math:`\eta`.
- :class:`~analyzer.modules.common.muons.MuonMaker`: Selects muons by ID, isolation, pT, and :math:`\eta`.


Trigger Selection
-----------------

- :class:`HLTSelection <analyzer.modules.common.hlt_selection.SimpleHLT>`: Selects events passing specified HLT triggers.
- :class:`~analyzer.modules.common.selection.SelectAllTriggers`: Creates a cutflow of all triggers for efficiency studies.


Selection and Filtering
-----------------------

- :class:`~analyzer.modules.common.selection.NObjFilter`: Creates a selection mask based on the number of objects in a collection.
- :class:`~analyzer.modules.common.selection.SelectOnColumns`: Applies all pending selection masks and records cutflow information.


Event-Level Corrections (MC)
----------------------------

- :class:`~analyzer.modules.common.event_level_corrections.PileupSF`: Applies pileup scale factors.
- :class:`~analyzer.modules.common.event_level_corrections.L1PrefiringSF`: Applies L1 prefiring corrections.
- :class:`~analyzer.modules.common.event_level_corrections.PosNegGenWeight`: Handles generator weight sign.
- :class:`~analyzer.modules.common.jets.PileupJetIdSF`: Applies pileup jet ID scale factors.
- :class:`~analyzer.modules.common.bjet_sf.BJetShapeSF`: Applies b-jet shape scale factors. Declares weight systematics.


Categories
----------

- :class:`~analyzer.modules.common.categories.SimpleCategory`: Bins events into categories based on a continuous variable.


Histogramming
-------------

- :class:`~analyzer.modules.common.histogram_builder.SimpleHistogram`: Produces a histogram from one or more input columns.
- :class:`~analyzer.modules.common.histogram_builder.HistogramBuilder`: Internal module that aggregates multi-variation event collections into a single histogram.


Data Export
-----------

- :class:`SaveColumns <analyzer.modules.common.save_arrays.SaveCols>`: Saves specified columns to the result tree.
- :class:`Skim <analyzer.modules.common.skimming.SaveEvents>`: Writes selected events to ROOT files.


Gen-Level
---------

- :class:`GenMatching <analyzer.modules.common.gen_matching.VectorMatching>`: Performs generator-level particle matching.
