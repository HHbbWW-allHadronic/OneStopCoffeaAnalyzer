==========
Ratio Plot
==========

The :class:`~analyzer.postprocessing.basic_histograms.RatioPlot` processor adds a ratio panel below the main histogram.
The ``subgroups`` must define ``numerator`` and ``denominator``.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Ratio Plot
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
       - name: RatioPlot
         inputs:
           - "*/*/*/Z1Mass"
         structure:
           group: {"era.name": "*", "name": "*"}
           subgroups:
             denominator:
               select: {dataset_name: "opendata_ZZTo*"}
               transforms:
                 - name: SumHistograms
                   sum_match_pattern: {pipeline: "*"}
                   new_meta_fields: {title: "ZZ$\\rightarrow$4L (MC)"}
             numerator:
               select: {dataset_name: "*Double*"}
               transforms:
                 - name: SumHistograms
                   sum_match_pattern: {pipeline: "*"}
                   new_meta_fields: {title: "Data"}
           transforms:
             - name: SelectAxesValues
               select_axes_values: {variation: [central]}
         output_name: "{prefix}/ratio_{era.name}_{name}.png"
