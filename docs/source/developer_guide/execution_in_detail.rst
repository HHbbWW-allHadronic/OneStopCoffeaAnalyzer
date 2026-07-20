Execution in Detail
===================

Consider the following user-specified pipeline:

* ``Selection1``: Creates a selection mask on events.
* ``Selection2``: Creates another mask on events.
* :class:`~analyzer.modules.common.selection.SelectOnColumns`: Uses the previously created masks to filter events.
* :class:`~analyzer.modules.common.jets.JetScaleCorrections` (Jet Correction): Does some correction on jets; it declares dynamic parameters called ``systematic`` with possible values ``central``, ``up``, and ``down``.
* :class:`~analyzer.modules.common.jets.JetFilter`: Filters jets with some parameters.
* :class:`~analyzer.modules.common.event_level_corrections.PileupSF` (PileupScaleFactor): Adds some weight to the events.
* ``JetPtHistogram``: Creates a histogram of jet :math:`p_T`.

Let's examine how a single chunk of events is processed:

* The first module in the pipeline is always a :class:`~analyzer.modules.common.load_columns.LoadColumns` module.
  It is added implicitly by the framework to any user-specified pipeline; the user does not need to add it.
  It has dynamic parameters ``chunk`` and ``metadata``.
  Of course, it has no event input.
  When it runs, it loads the requested chunk and creates a :class:`~analyzer.core.columns.TrackedColumns` object.
* Each of the selections runs. In the backend, each selection adds a column to events which is the boolean mask of the requested selection. Note that no actual slicing of the array has been performed yet.
* :class:`~analyzer.modules.common.selection.SelectOnColumns` runs, which actually filters the events.
* ``JetCorrection`` (using :class:`~analyzer.modules.common.jets.JetScaleCorrections`) runs using the default systematic ``"central"``.
* :class:`~analyzer.modules.common.jets.JetFilter` runs, changing the shape of jets.
* :class:`~analyzer.modules.common.event_level_corrections.PileupSF` runs, adding a column ``Weights.pileup_sf``.
* ``JetPtHistogram`` runs and returns a :class:`~analyzer.core.analysis_modules.ModuleAddition` result, with a certain :class:`~analyzer.core.run_builders.RunBuilder` and a certain module :class:`~analyzer.modules.common.histogram_builder.HistogramBuilder`.
* This :class:`~analyzer.core.analysis_modules.ModuleAddition` requests a multi-run with systematics on the jet corrections. This starts a new pipeline for each systematic :math:`S`:

  * We start from the beginning with :class:`~analyzer.modules.common.load_columns.LoadColumns`. However, since the chunk is the same, we can use the cached result.
  * This holds true for ``Selection1``, ``Selection2``, and :class:`~analyzer.modules.common.selection.SelectOnColumns`, where we can use the cached result.
  * The dynamic parameter ``systematic`` of ``JetCorrection`` (corresponding to :class:`~analyzer.modules.common.jets.JetScaleCorrections`) changes to :math:`S`, so the correction module is re-run.
  * :class:`~analyzer.modules.common.jets.JetFilter` is re-run, since it depends on the column Jet, which was changed.
  * :class:`~analyzer.modules.common.event_level_corrections.PileupSF` uses the cached result, since it does not depend on Jet.
  * ``JetPtHistogram`` re-runs and produces another :class:`~analyzer.core.analysis_modules.ModuleAddition`, which is ignored (results are ignored within a nulti-run).
  * The three sets of events corresponding to the ``up``, ``down``, and ``central`` systematics are passed to the :class:`~analyzer.modules.common.histogram_builder.HistogramBuilder` module, which returns a single :class:`~analyzer.core.results.Histogram` containing an axis called ``"systematic"``, and any number of other data axes.
  * This ends the multi-run.

* The results are returned.
    

This process is visualized below.

.. graphviz::

    digraph AnalyzerDataFlow {
        // General Graph Settings
        rankdir=TB;
        bgcolor="white";
        fontname="Arial";
        node [fontname="Arial", shape=box, style=filled, fillcolor=white, color=black, penwidth=1.2];
        edge [color=black, penwidth=1.2];

        // --- Main Run (Sequential Pipeline) ---
        main_start [label="Start Main Run", shape=oval, fillcolor="#eeeeee"];
        load       [label="LoadColumns\n(implicit, chunk, metadata)"];
        sel1       [label="Selection1"];
        sel2       [label="Selection2"];
        select_on  [label="SelectOnColumns"];
        jet_corr   [label="JetCorrection\n(systematic='central')"];
        jet_filt   [label="JetFilter"];
        pu_sf      [label="PileupScaleFactor"];
        jet_hist   [label="JetPtHistogram\n(Outputs ModuleAddition)", shape=note, fillcolor="#ffffcc"];

        // Main Run Connections
        main_start -> load -> sel1 -> sel2 -> select_on -> jet_corr -> jet_filt -> pu_sf -> jet_hist;

        // --- Multi-Run Trigger ---
        multi_run_trigger [label="Multi-Run Triggered\n(Requests up, central, down variations)", shape=diamond, fillcolor="#f2f2f2", width=4];
        jet_hist -> multi_run_trigger;

        // --- Systematic Variations ---
        // Using rank=same to force the three branches to sit side-by-side
        subgraph cluster_multirun {
            label="Multi-Run Execution";
            style=dashed;
            color=gray;
            fontname="Arial";
            
            // Up Branch
            up_start   [label="S='up'", shape=diamond, fillcolor="#e6f2ff"];
            up_load    [label="LoadColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            up_sel     [label="Selections 1 & 2\nSelectOnColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            up_jet     [label="JetCorrection\n[Re-run]", fillcolor="#d9ead3"];
            up_filt    [label="JetFilter\n[Re-run]", fillcolor="#d9ead3"];
            up_pu      [label="PileupScaleFactor\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            up_hist    [label="JetPtHistogram\n[Re-run, Ignored]", fillcolor="#d9ead3"];
            
            up_start -> up_load -> up_sel -> up_jet -> up_filt -> up_pu -> up_hist;

            // Central Branch (Fully Cached)
            cen_start  [label="S='central'", shape=diamond, fillcolor="#e6f2ff"];
            cen_load   [label="LoadColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            cen_sel    [label="Selections 1 & 2\nSelectOnColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            cen_jet    [label="JetCorrection\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            cen_filt   [label="JetFilter\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            cen_pu     [label="PileupScaleFactor\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            cen_hist   [label="JetPtHistogram\n[Cached, Ignored]", style="dashed,filled", color=gray, fontcolor=gray];
            
            cen_start -> cen_load -> cen_sel -> cen_jet -> cen_filt -> cen_pu -> cen_hist;

            // Down Branch
            down_start [label="S='down'", shape=diamond, fillcolor="#e6f2ff"];
            down_load  [label="LoadColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            down_sel   [label="Selections 1 & 2\nSelectOnColumns\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            down_jet   [label="JetCorrection\n[Re-run]", fillcolor="#d9ead3"];
            down_filt  [label="JetFilter\n[Re-run]", fillcolor="#d9ead3"];
            down_pu    [label="PileupScaleFactor\n[Cached]", style="dashed,filled", color=gray, fontcolor=gray];
            down_hist  [label="JetPtHistogram\n[Re-run, Ignored]", fillcolor="#d9ead3"];
            
            down_start -> down_load -> down_sel -> down_jet -> down_filt -> down_pu -> down_hist;
            
            {rank=same; up_start; cen_start; down_start;}
            {rank=same; up_load; cen_load; down_load;}
            {rank=same; up_sel; cen_sel; down_sel;}
            {rank=same; up_jet; cen_jet; down_jet;}
            {rank=same; up_filt; cen_filt; down_filt;}
            {rank=same; up_pu; cen_pu; down_pu;}
            {rank=same; up_hist; cen_hist; down_hist;}
        }

        // Connections from trigger to branches
        multi_run_trigger -> up_start;
        multi_run_trigger -> cen_start;
        multi_run_trigger -> down_start;

        // --- Aggregation ---
        hist_builder [label="HistogramBuilder\n(Aggregates {up, central, down})", shape=box3d, fillcolor="#b4c7e7"];
        end_run      [label="Return Results", shape=oval, fillcolor="#eeeeee"];

        // Connections from branches to aggregation
        up_hist   -> hist_builder;
        cen_hist  -> hist_builder;
        down_hist -> hist_builder;

        hist_builder -> end_run;
    }