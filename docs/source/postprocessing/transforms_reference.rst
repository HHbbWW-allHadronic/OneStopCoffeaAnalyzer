Transforms Reference
====================

Transforms modify results within a postprocessing group before they are passed to the processor.
They are specified in the ``transforms`` field of a :class:`~analyzer.postprocessing.grouping.GroupBuilder`.
Transformations can perform a variety manipulations, from simply slicing histogram axes to doing complex rescaling of systematics.


Histogram Transforms
--------------------

These operate on :class:`~analyzer.core.results.Histogram` results and modify the underlying ``hist.Hist`` object.

SelectAxesValues
^^^^^^^^^^^^^^^^

Select specific values from categorical or string axes.
This is the most commonly used transform -- nearly every postprocessing configuration uses it to select the "central" variation.

.. code-block:: yaml

    - name: SelectAxesValues
      select_axes_values:
        variation: central

You can select from multiple axes and multiple values simultaneously:

.. code-block:: yaml

    - name: SelectAxesValues
      select_axes_values:
        variation: [central, JES_up, JES_down]
        HT_Cat: [500, 1000]

When multiple values are specified for an axis, the transform produces one output item per combination of values.

MergeAxes
^^^^^^^^^

Sum over one or more axes, collapsing them.
This is commonly used to merge binned category axes like ``HT_Cat`` into a single bin.

.. code-block:: yaml

    - name: MergeAxes
      merge_axis_names: [HT_Cat]

SplitAxes
^^^^^^^^^

Split a histogram along one or more axes, producing one result per bin value.
The opposite of :class:`~analyzer.postprocessing.transforms.hist_transforms.MergeAxes`.

.. code-block:: yaml

    - name: SplitAxes
      split_axis_names: [variation]

You can optionally limit which values are split using a pattern:

.. code-block:: yaml

    - name: SplitAxes
      split_axis_names: [variation]
      limit_pattern: "central"

RebinAxes
^^^^^^^^^

Rebin histogram axes by an integer factor.

.. code-block:: yaml

    # Rebin all axes by factor 2
    - name: RebinAxes
      rebin: 2

    # Rebin specific axes by different factors
    - name: RebinAxes
      rebin:
        HT: 4
        jet_pt: 2

SliceAxes
^^^^^^^^^

Slice histogram axes to a sub-range.
Values are specified as ``[low, high]`` in the axis's coordinate space (not bin indices).

.. code-block:: yaml

    - name: SliceAxes
      slices:
        HT: [500, 2000]       # Keep only HT between 500 and 2000
        jet_pt: [30, null]     # Keep jet_pt >= 30 (null = no upper bound)

MultiSliceAxes
^^^^^^^^^^^^^^

Produce multiple slices of an axis in one step.
Specify the start, stop, and number of bins, and the transform produces one result per adjacent pair of bin edges.

.. code-block:: yaml

    - name: MultiSliceAxes
      multi_slices:
        mass: [500, 2500, 5]    # [start, stop, n_bins]

SumHistograms
^^^^^^^^^^^^^

Sum together histograms matching a pattern, producing a single combined histogram.
Items not matching the pattern are passed through unchanged.

.. code-block:: yaml

    - name: SumHistograms
      sum_match_pattern:
        dataset_name: "qcd*"
      new_meta_fields:
        dataset_name: "Total QCD"
        dataset_title: "QCD (combined)"

FormatTitle
^^^^^^^^^^^

Set the display title for each result using a template string:

.. code-block:: yaml

    - name: FormatTitle
      title_format: "{dataset_title} ({era.name})"

SetStyle
^^^^^^^^

Override the plot style for all items passing through this transform:

.. code-block:: yaml

    - name: SetStyle
      style:
        plottype: step
        color: red
        linewidth: 2

StatMaker
^^^^^^^^^

Compute summary statistics (integral, mean, median, standard deviation) for each histogram and attach them to the metadata.
Useful for including statistics in plot labels or output file names.

.. code-block:: yaml

    - name: StatMaker

OrBinaryAxes
^^^^^^^^^^^^

Combine multiple binary (0/1) categorical axes using logical OR.

.. code-block:: yaml

    - name: OrBinaryAxes
      or_axis_names: [trigger_A, trigger_B]

NormalizeSystematicByProjection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Normalize systematic variations so that their total yield matches the nominal, preserving the shape difference.

.. code-block:: yaml

    - name: NormalizeSystematicByProjection
      normalize_within: [mass]
      pre_sf_name: "central"

ABCDTransformer
^^^^^^^^^^^^^^^

Transform a 2D histogram into a 1D categorical histogram with ABCD regions based on x and y cuts.
The cut values are read from a CSV file.

.. code-block:: yaml

    - name: ABCDTransformer
      csv_path: "config/abcd_cuts.csv"
      x_axis_name: "jet_pt"
      y_axis_name: "met"
      target_axis_name: "region"
      key_format: "{dataset_name}_{era.name}"


Data Transforms
---------------

These operate on :class:`~analyzer.core.results.SavedColumns` results rather than histograms.

MaskData
^^^^^^^^

Apply a boolean mask to saved column data.
The mask is specified as a Python expression evaluated with the column names available as variables.

.. code-block:: yaml

    - name: MaskData
      mask: "jet_pt > 30"

AddData
^^^^^^^

Add a new computed column to the saved data.
The expression has access to existing column names.

.. code-block:: yaml

    - name: AddData
      new_col: "jet_ratio"
      func: "jet_pt / HT"

MakeHistogram
^^^^^^^^^^^^^

Create a histogram from saved column data (as opposed to the histograms produced during the analysis).

.. code-block:: yaml

    - name: MakeHistogram
      histogram_name: "jet_pt_from_saved"
      column_axis_mapping:
        jet_pt:
          name: jet_pt
          start: 0
          stop: 500
          bins: 50
      weight_col: "Scale"    # Optional: column to use as weights


Applying Transforms Conditionally
---------------------------------

Any transform can be restricted to specific items using the ``should_run`` field:

.. code-block:: yaml

    - name: RebinAxes
      rebin: 4
      should_run:
        dataset_name: "signal*"

In this example, only signal histograms are rebinned; all other items pass through unchanged.
