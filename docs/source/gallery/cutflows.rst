========
Cutflows
========

The :class:`~analyzer.postprocessing.cutflows.PlotSelectionFlow`  can be used to visualize selections.

.. postprocess::
   :results: docs/source/_data/output_higgs
   :title: Plot Selection Flow
   :category: Cutflows
   :columns: 2

   Postprocessing:
     default_plot_config:
       cms_text: "Open Data"
       image_type: ".png"
     processors:
       - name: PlotSelectionFlow
         inputs:
           - "*/*/*/higgs_valid_selection"
         scale: log
         structure:
           group: {"dataset_name": "*"}
         output_name: "{prefix}/cutflow_{dataset_name}.png"

