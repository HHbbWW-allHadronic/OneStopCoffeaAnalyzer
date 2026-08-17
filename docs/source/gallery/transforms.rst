==========
Transforms
==========

Transforms modify results before they are plotted.
They are executed in the order they appear in the configuration.

Rebinning and Slicing
---------------------

The :class:`~analyzer.postprocessing.transforms.rebin.RebinAxes` and :class:`~analyzer.postprocessing.transforms.slicing.SelectAxesValues`  do more or less what they say.
Rebin simply rebins the given axes, merging every $N$ bins together. 
Select picks a particular value from the chosen axis.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Rebinning
   :category: Transformations
   :columns: 2

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/Z1Mass"
         structure:
           group: {"dataset_name": "opendata_ZZTo4mu"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
             - name: RebinAxes
               rebin: {Z1_mass: 5}
         output_name: "{prefix}/rebin_{dataset_name}.png"


Slicing
-------

The :class:`~analyzer.postprocessing.transforms.hist_transforms.SliceAxes` transform can restrict a histogram to a specific range of values.


.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Slicing
   :category: Transformations

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/Z2Mass"
         structure:
           group: {"dataset_name": "opendata_ZZTo4mu"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
             - name: SliceAxes
               slices: {Z2_mass: [40, 100]}
         output_name: "{prefix}/sliced_{dataset_name}.png"

Statistics and Formatting
-------------------------

The :class:`~analyzer.postprocessing.transforms.hist_transforms.StatMaker` transform calculates statistics (ie mean, median, std) for a histogram and stores them in the metadata.
They can then be used downstream, for example by :class:`~analyzer.postprocessing.transforms.hist_transforms.FormatTitle`.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Statistics Formatting
   :category: Transformations

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/Z1Mass"
         structure:
           group: {"dataset_name": "opendata_ZZTo4mu"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
             - name: StatMaker
             - name: FormatTitle
               title_format: "Z1 Mass (Mean: {stats.mean} GeV, Std: {stats.std} GeV)"
         output_name: "{prefix}/stats_{dataset_name}.png"

Splitting Axes
--------------

The :class:`~analyzer.postprocessing.transforms.hist_transforms.SplitAxes` transform can split a single multi-dimensional histogram into multiple independent histograms 
based on the bins of a specific axis.
This is especially useful for user-generated categorical axes, a common example would be the multiplicity of some physics object like jets or lepton.




.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Splitting Axes
   :category: Transformations

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/Z1Mass"
         structure:
           group: {"dataset_name": "opendata_ZZTo4mu"}
           transforms:
             - name: SplitAxes
               split_axis_names: ["variation"]
         output_name: "{prefix}/split_{dataset_name}_{axis_params.variation}.png"


.. tip::

    This can be combined various grouping mechanics to, for example, split the systematics then plot them as a ratio.