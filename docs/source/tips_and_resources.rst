Tips and Resources
==================

Debugging
---------

Limit Events
^^^^^^^^^^^^
When testing a new configuration, limit the number of processed events to speed up execution:

.. code-block:: bash

   uv run python -m analyzer run -e imm-1000--max-sample-events 1000 config.yaml output/


Useful Links
------------

- `HLT Info <https://cmshltinfo.app.cern.ch/>`_: Get list of available triggers for different years
- `PPD Homepage <https://cms-info.web.cern.ch/coordination/physics-performance-datasets-ppd/>`_
- `HLT Config <https://cmshltcfg.app.cern.ch/>`_: See how different triggers are defined in code
- `CMSSW Source Search <https://cmssdt.cern.ch/lxr/>`_: Explore the CMSSW code base
- `FNAL LPC Monitoring <https://landscape.fnal.gov/monitor/d/c9450043/lpc-batch-summary>`_: Dashboard for LPC Condor
- `GRASP <https://cms-pdmv-prod.web.cern.ch/grasp>`_: Find MC Samples
- `XSECDB <https://xsecdb-xsdb-official.app.cern.ch/>`_: Find cross sections of different processes
- `DAS <https://cmsweb.cern.ch/das/>`_: Explore CMS datasets
- `Site Status <https://cmssst.web.cern.ch/siteStatus/summary.html>`_: See status of different sites on the grid
- `CMSOnline <https://cmsonline.cern.ch/webcenter/portal/cmsonline>`_: As close to the control room as you can get without being there
- `OMS <https://cmsoms.cern.ch/>`_: Information about runs
- `CAT <https://cms-analysis.docs.cern.ch/>`_: Information about analysis tools
- `New iCMS <https://icms.cern.ch/tools/>`_: Updated interface for queries about CMS analyses and other information
