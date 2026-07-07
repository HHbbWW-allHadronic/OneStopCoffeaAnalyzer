Tips and Troubleshooting
========================

Common Issues
-------------

**"Module not found" error when loading configuration**

Your custom module class is not registered.
Make sure you have added the file containing your module to ``extra_module_paths`` in the configuration, and that the module file can be imported without errors.

.. code-block:: yaml

    extra_module_paths:
      - modules/my_modules.py

Also verify that the ``module_name`` in your YAML exactly matches the class name of your module.


**"More than one matching pattern" error**

A dataset matched more than one entry in ``event_collections``.
Each dataset must match exactly one entry.
Make your patterns more specific, or use negation (``!signal*``) to exclude datasets from a pattern.


**Results appear empty or have zero entries**

Check that your ``event_collections`` dataset pattern actually matches the datasets you intend.
Use ``./osca list datasets`` to see what datasets are available, and ``./osca describe_analysis config.yaml`` to see what the config will process.

Also verify that your selections are not cutting all events.
Run with ``--max-sample-events 10000`` and ``--log-level DEBUG`` to see event counts at each step.


**xrootd timeout or file access errors**

These are typically infrastructure issues.
Try adjusting ``location_priorities`` to prefer different sites, or retry later.
If a specific sample consistently fails, check whether its files are available on the sites you are using.


**"Overlapping provenance" error when merging**

This means two result files contain data from the same file chunks.
This can happen if you accidentally ran the same job twice.
Remove the duplicate result files before merging.


**Memory issues with large chunk sizes**

Reduce the ``chunk_size`` of your executor, or use an executor with more ``worker_memory``.
If you have many systematic variations, consider using ``reduction_factor: 2`` to reduce memory pressure from intermediate results.


Performance Tips
----------------

- Use :class:`~analyzer.core.run_builders.NoSystematics` during development and switch to :class:`~analyzer.core.run_builders.CompleteSysts` only for production.
- The ``--max-sample-events`` flag is your best friend during development. Start with 10000 events and increase as needed.
- When running postprocessing with many plots, use ``--parallel 4`` (or however many cores you have) to parallelize plot generation.
- If your analysis produces many large histograms, consider using fewer bins or limiting the number of systematics during development.


Useful Resources
----------------

- `coffea documentation <https://coffeateam.github.io/coffea/>`_
- `awkward-array documentation <https://awkward-array.readthedocs.io/>`_
- `hist documentation <https://hist.readthedocs.io/>`_
- `dask documentation <https://docs.dask.org/>`_
- `attrs documentation <https://www.attrs.org/>`_
- `Jinja2 template documentation <https://jinja.palletsprojects.com/>`_
- `CMS NanoAOD documentation <https://cms-nanoaod-integration.web.cern.ch/autoDoc/>`_
