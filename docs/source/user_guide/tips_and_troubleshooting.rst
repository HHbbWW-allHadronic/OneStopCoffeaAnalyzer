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


**xrootd timeout errors and friends**

These are typically infrastructure issues, and there is not a ton to be done about them.
Try adjusting ``location_priorities`` to prefer different sites, or retry later.
If a specific sample consistently fails, check whether its files are available on the sites you are using.


**"Overlapping provenance" error when merging**

This means two result files contain data from the same file chunks.
This can happen if you accidentally ran the same job twice.
Remove the duplicate result files before merging.
You can use standard bash tools like find to check the modification time and delete files newer or older than a certain time.


**Memory issues with large chunk sizes**

Reduce the ``chunk_size`` of your executor, or use an executor with more ``worker_memory``.
If you have many systematic variations or fine binning (both of which make for large histograms), consider using ``reduction_factor: 2`` to reduce memory pressure from intermediate results.


Performance Tips
----------------

- Use :class:`~analyzer.core.run_builders.NoSystematics` during development and switch to :class:`~analyzer.core.run_builders.CompleteSysts` only for production.
- The ``--max-sample-events`` flag is your best friend during development. Start with 10000 events and increase as needed.
- When running postprocessing with many plots, use ``--parallel 4`` (or however many cores you have) to parallelize plot generation.
- If your analysis produces many large histograms, consider using fewer bins or limiting the number of systematics during development. If your histograms are huge then things will be much more brittle.



Useful Resources
----------------

- `coffea documentation <https://coffeateam.github.io/coffea/>`_
- `awkward-array documentation <https://awkward-array.readthedocs.io/>`_
- `hist documentation <https://hist.readthedocs.io/>`_
- `dask documentation <https://docs.dask.org/>`_
- `attrs documentation <https://www.attrs.org/>`_
- `Jinja2 template documentation <https://jinja.palletsprojects.com/>`_
- `CMS NanoAOD documentation <https://cms-nanoaod-integration.web.cern.ch/autoDoc/>`_
- `HLT Info <https://cmshltinfo.app.cern.ch/>`_ -- Get list of available triggers for different years
- `PPD Homepage <https://cms-info.web.cern.ch/coordination/physics-performance-datasets-ppd/>`_
- `HLT Config <https://cmshltcfg.app.cern.ch/>`_ -- See how different triggers are defined in code
- `CMSSW Source Search <https://cmssdt.cern.ch/lxr/>`_ -- Explore the CMSSW code base
- `FNAL LPC Monitoring <https://landscape.fnal.gov/monitor/d/c9450043/lpc-batch-summary>`_ -- Dashboard for LPC Condor
- `GRASP <https://cms-pdmv-prod.web.cern.ch/grasp>`_ -- Find MC Samples
- `XSECDB <https://xsecdb-xsdb-official.app.cern.ch/>`_ -- Find cross sections of different processes
- `DAS <https://cmsweb.cern.ch/das/>`_ -- Explore CMS datasets
- `Site Status <https://cmssst.web.cern.ch/siteStatus/summary.html>`_ -- See status of different sites on the grid
- `CMSOnline <https://cmsonline.cern.ch/webcenter/portal/cmsonline>`_ -- As close to the control room as you can get without being there
- `OMS <https://cmsoms.cern.ch/>`_ -- Information about runs
- `CAT <https://cms-analysis.docs.cern.ch/>`_ -- Information about analysis tools
- `New iCMS <https://icms.cern.ch/tools/>`_ -- Updated interface for queries about CMS analyses and other information
