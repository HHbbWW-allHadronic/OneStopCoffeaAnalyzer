=================
Stacked Histogram
=================

Generic brand histogram.
With ``subgroups``, it can show stacked backgrounds with an  overlays.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Stacked Histogram
   :category: 1D Histograms

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     default_style_set:
       styles:
         - pattern:
             sample_type: Data
           style:
             plottype: errorbar
             color: black
             marker: 'o'
         - pattern:
             name: '*'
           style:
             plottype: fill
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/HiggsMass"
         structure:
           group: {"era.name": "*", "name": "*"}
           subgroups:
             stacked:
               select: {dataset_name: "opendata_ZZTo*"}
               transforms:
                 - name: SumHistograms
                   sum_match_pattern: {pipeline: "*"}
                   new_meta_fields: {title: "ZZ$\\rightarrow$4L (MC)"}
             unstacked:
               select: {dataset_name: "*Double*"}
               transforms:
                 - name: SumHistograms
                   sum_match_pattern: {pipeline: "*"}
                   new_meta_fields: {title: "Data"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
         output_name: "{prefix}/hist1d_{era.name}_{name}.png"


Unstacked Shape Comparison
--------------------------

It is also possible to use :class:`~analyzer.postprocessing.basic_histograms.Histogram1D` to overlay shapes without stacking.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Signal Shape Comparison
   :category: 1D Histograms

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     default_style_set:
       styles:
         - pattern:
             name: '*'
           style:
             plottype: step
             linewidth: 2
     processors:
       - name: Histogram1D
         inputs:
           - "*/*/*/Z1Mass"
         structure:
           group: {"dataset_name": "*SMHiggsToZZTo4L*"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
             - name: FormatTitle
               title_format: "{pipeline} Pipeline"
         output_name: "{prefix}/signals_{dataset_name}_{name}.png"
