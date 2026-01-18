=====================
 Writing New Modules
=====================

The OneStopCoffea Analyzer framework is built around modular units of analysis called "Analyzer Modules". Users can extend the framework by creating their own custom modules.

Basic Structure
---------------

All analyzer modules must inherit from `analyzer.core.analysis_modules.AnalyzerModule` and use the `attrs` library for defining parameters.

Here is a minimal example of a custom module:

.. code-block:: python

    from analyzer.core.analysis_modules import AnalyzerModule
    from analyzer.core.columns import Column
    from attrs import define
    import awkward as ak

    @define
    class MyCustomModule(AnalyzerModule):
        """
        A custom module that computes the square of a column.
        
        Parameters
        ----------
        input_col : Column
            The input column to process.
        output_col : Column
            The output column where results will be stored.
        """
        input_col: Column
        output_col: Column
        
        def inputs(self, metadata):
            """
            Declare required input columns.
            """
            return [self.input_col]

        def outputs(self, metadata):
            """
            Declare output columns produced by this module.
            """
            return [self.output_col]

        def run(self, columns, params):
            """
            Execute the analysis logic.
            
            Parameters
            ----------
            columns : TrackedColumns
                The columnar data container.
            params : dict
                Runtime parameters (e.g. systematics).
                
            Returns
            -------
            columns : TrackedColumns
                The updated columns object.
            results : list
                A list of any additional results (like histograms).
            """
            # Access the data
            data = columns[self.input_col]
            
            # Perform computation
            result = data ** 2
            
            # Store the result
            columns[self.output_col] = result
            
            # Return modified columns and an empty list of side-results
            return columns, []

Core Components
---------------

Subclasses of `AnalyzerModule` must implement three key methods:

1. **inputs(self, metadata)**:
   Returns a list of `Column` objects that this module requires. This allows the framework to manage dependencies and load only necessary data.

2. **outputs(self, metadata)**:
   Returns a list of `Column` objects that this module produces. This is used for checking if requirements of downstream modules are met.

3. **run(self, columns, params)**:
   The core logic. It receives the data in `columns` and any runtime `params`.

Metadata and Configuration
--------------------------

Modules can inspect `columns.metadata` to change their behavior based on the dataset (e.g. Data vs MC, different eras).

.. code-block:: python

    def run(self, columns, params):
        if columns.metadata["sample_type"] == "MC":
             # Apply MC-specific corrections
             pass
        return columns, []
