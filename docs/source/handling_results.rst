Results and Postprocessing
==========================


CLI Browser
^^^^^^^^^^^
The analyzer includes a terminal-based result browser.

.. code-block:: bash

   uv run python -m analyzer browse 'OUTPUT_DIR/*'

Python Interface
^^^^^^^^^^^^^^^^
You can also load results directly in Python:

.. code-block:: python

   from analyzer.core.results import loadResults
   
   results = loadResults(["path/to/results/*"])

