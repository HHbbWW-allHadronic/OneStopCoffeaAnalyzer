Styling
=======

The postprocessing system uses a pattern-based styling system to control the appearance of plots.
Styles can be set at the postprocessor level, per-processor, or per-item via transforms.


Style Sets
----------

A :class:`~analyzer.postprocessing.style.StyleSet` is a list of :class:`~analyzer.postprocessing.style.StyleRule` objects.
Each rule has a ``pattern`` and a ``style``.
When the postprocessor needs to style a result, it checks each rule in order and uses the first matching style, based on the metadata of the object being processed.

.. code-block:: yaml

    default_style_set:
      styles:
        - pattern:
            dataset_name: 'data*'
          style:
            plottype: errorbar
            color: black
            marker: 'o'

        - pattern:
            dataset_name: 'signal*'
          style:
            plottype: step
            linewidth: 2

        - pattern:
            sample_type: MC
          style:
            plottype: fill

Rules are checked in order -- put more specific patterns before general ones.
The ``pattern`` field uses the same matching system as :doc:`../concepts/pattern_matching`.


Style Properties
----------------

The ``style`` object controls how a histogram is rendered.
Generally hese options correspond exactly to the matplotlib function options.
Note though that not every style option will affect every plot.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Property
     - Default
     - Description
   * - ``plottype``
     - ``step``
     - How to draw the histogram: ``step`` (line), ``fill`` (filled area), ``band`` (error band), ``errorbar`` (points with error bars), ``scatter_z`` (scatter with color).
   * - ``color``
     - auto
     - The color. Accepts any matplotlib color string or the CMS palette names (see below).
   * - ``linestyle``
     - ``None``
     - Matplotlib line style: ``"-"``, ``"--"``, ``":"``, ``"-."``.
   * - ``alpha``
     - ``None``
     - Opacity (0.0 to 1.0).
   * - ``marker``
     - ``"o"``
     - Marker style for errorbar plots.
   * - ``markersize``
     - ``5``
     - Marker size.
   * - ``fill_hatching``
     - ``None``
     - Hatch pattern for filled histograms (e.g., ``"//"``, ``"\\\\"``, ``"xx"``).
   * - ``line_width``
     - ``None``
     - Line width override.
   * - ``yerr``
     - ``True``
     - Whether to show error bars.
   * - ``legend``
     - ``True``
     - Whether to show in the legend.
   * - ``legend_font``
     - ``None``
     - Font size for the legend entry.
   * - ``y_min``
     - ``None``
     - Minimum y-axis value.

When ``color`` is not specified, colors are automatically assigned from the default color cycle.


Plot Types
----------

``step``
    A step line (standard histogram outline). This is the default.

``fill``
    A filled histogram. Commonly used for stacked MC backgrounds.

``errorbar``
    Data points with vertical error bars. Commonly used for data.

``band``
    A shaded error band around the histogram.

``scatter_z``
    A scatter plot with a color axis. Used for 2D significance scans.


CMS Color Palette
-----------------

The framework includes the official CMS recommended color palette.
These can be used by name in style definitions:

.. list-table::
   :header-rows: 1
   :widths: 25 25

   * - Name
     - Hex
   * - ``cms-blue``
     - ``#5790fc``
   * - ``cms-blue-dark``
     - ``#3f90da``
   * - ``cms-orange``
     - ``#f89c20``
   * - ``cms-red``
     - ``#e42536``
   * - ``cms-purple``
     - ``#964a8b``
   * - ``cms-purple-dark``
     - ``#7a21dd``
   * - ``cms-gray``
     - ``#9c9ca1``
   * - ``cms-cyan``
     - ``#92dadd``
   * - ``cms-brown``
     - ``#a96b59``
   * - ``cms-gold``
     - ``#b9ac70``

These are registered as named matplotlib colors and can be used directly:

.. code-block:: yaml

    style:
      color: cms-blue


Plot Configuration
------------------

Beyond styles, there are also configuration options that affect the overall appearance and rendering of the plots.
The ``default_plot_config`` (or per-processor ``plot_configuration``) controls global plot properties:

.. code-block:: yaml

    default_plot_config:
      cms_text: "Preliminary"
      image_type: ".png"
      cms_text_color: Black
      legend_num_cols: 2

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``cms_text``
     - Text shown next to the CMS logo (e.g., "Preliminary", "Work in Progress", "Supplementary").
   * - ``image_type``
     - Output image format: ``".png"``, ``".pdf"``, ``".svg"``.
   * - ``cms_text_color``
     - Color of the CMS text.
   * - ``legend_num_cols``
     - Number of columns in the plot legend.

The framework uses ``mplhep`` with the CMS style applied by default.
Additionally, certain options support list values. 
If a list is provided, the system will distinct plots, one for each combination of values.
For example, if ``image_type`` is ``[".pdf", ".svg"]``, the system will produce both a PDF and an SVG file.
Another frequently useful option is to set the ``cms_text`` field to ``["Preliminary", "Private Work"]`` to generate plots with different annotations.


Overriding Styles Per-Processor
-------------------------------

Each processor can have its own ``style_set`` that overrides the default:

.. code-block:: yaml

    processors:
      - name: Histogram1D
        style_set:
          styles:
            - pattern:
                dataset_name: "signal*"
              style:
                plottype: step
                color: cms-red
                linewidth: 3
        ...

If a processor does not specify a ``style_set``, the ``default_style_set`` is used.


Overriding Styles Per-Item
--------------------------

The :class:`~analyzer.postprocessing.transforms.hist_transforms.SetStyle` transform can override the style for specific items within a group:

.. code-block:: yaml

    transforms:
      - name: SetStyle
        style:
          plottype: step
          color: blue
        should_run:
          dataset_name: "signal*"

This is useful when you want fine-grained control over individual items in a plot.
