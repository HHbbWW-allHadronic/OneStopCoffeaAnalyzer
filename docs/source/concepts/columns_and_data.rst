Columns and Data
================

The :class:`~analyzer.core.columns.TrackedColumns` object is the central data container in the framework.
It is what modules receive as input and return as output (potentially along with results).
Understanding how it works is important both for writing modules and for debugging unexpected behavior.


The ``Column`` Class
--------------------

A :class:`~analyzer.core.columns.Column` represents a path to a specific field in the event data.
If you are familiar with NanoAOD, a :class:`~analyzer.core.columns.Column` is essentially a reference like ``Jet.pt`` or ``FatJet.msoftdrop``.

Columns are constructed from dot-delimited strings or tuples:

.. code-block:: python

    from analyzer.core.columns import Column

    col = Column("Jet.pt")        # Equivalent to Column(("Jet", "pt"))
    col = Column("HT")            # A top-level column
    col = Column("Selection.njets")  # A selection mask

Columns support a few useful operations:

- **Containment**: :class:`~analyzer.core.columns.Column` returns ``True``. This means "Jet" is a parent of "Jet.pt".
- **Concatenation**: :class:`~analyzer.core.columns.Column` gives :class:`~analyzer.core.columns.Column`.
- **Extraction**: ``col.extract(events)`` navigates the events array to retrieve the data at that path.


``TrackedColumns``
------------------

:class:`~analyzer.core.columns.TrackedColumns` wraps a coffea NanoEvents array and adds three important capabilities:

1. **Provenance tracking** for caching (see :doc:`architecture`).
2. **Lazy column writes** to avoid potentially expensive array synchronization.
3. **Input/output enforcement** to catch bugs where modules access columns they did not declare as inputs.

Reading and Writing Columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Inside a module's :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run` method, you interact with :class:`~analyzer.core.columns.TrackedColumns` through dictionary-like access:

.. code-block:: python

    def run(self, columns, params):
        # Read a column
        jets = columns[Column("Jet")]       # Or: columns["Jet"]
        jet_pt = columns[Column("Jet.pt")]  # Or: columns["Jet.pt"]

        # Write a column
        columns[Column("GoodJet")] = filtered_jets
        columns["HT"] = ak.sum(jet_pt, axis=1)

        return columns, []

When you write a column, :class:`~analyzer.core.columns.TrackedColumns` updates the provenance for that column and all its parents, which is how the caching system knows what changed.

.. note::
   Writes are lazy by default -- the data is not immediately synchronized to the underlying events array.
   If you need to read the raw ``events`` array directly (which you generally should not), call :meth:`~analyzer.core.columns.TrackedColumns.flush` first.

Accessing the Raw Events
^^^^^^^^^^^^^^^^^^^^^^^^

In rare cases, you may need access to the underlying awkward array:

.. code-block:: python

    events = columns.events   # Triggers a flush of lazy columns


.. danger:: 
    
    Manipulating the ``events`` object directly (e.g. using ``columns.events``) can lead to unexpected behavior.
    Always prefer using the ``columns[col]`` syntax.


Filtering Events
^^^^^^^^^^^^^^^^

To remove events from the collection (e.g., after applying a selection), use :meth:`~analyzer.core.columns.TrackedColumns.filter`:

.. code-block:: python

    mask = columns["Selection.njets"]
    columns.filter(mask)

This removes events where the mask is ``False`` from both the underlying events array and any lazy columns.
It also updates the provenance of all columns.

.. warning::
   :meth:`~analyzer.core.columns.TrackedColumns.filter` mutates the :class:`~analyzer.core.columns.TrackedColumns` in place.
   After filtering, ``columns.events`` will have fewer entries.

.. danger:: 
    
    In practive you generally should not filter directly, instead write selections (see below) to achieve the same effect.


The Selection Pattern
---------------------

Selections are a critical part of doing an analysis.
Rather than having each module immediately filter events, the framework uses a *deferred selection* approach:

1. **Modules create boolean masks** and store them under the ``Selection`` namespace using the :func:`~analyzer.core.columns.addSelection` helper:

   .. code-block:: python

       from analyzer.core.columns import addSelection

       def run(self, columns, params):
           jets = columns[self.input_col]
           count = ak.num(jets, axis=1)
           mask = count >= self.min_count
           addSelection(columns, self.selection_name, mask)
           return columns, []

   This stores the mask at ``Selection.<name>`` and registers it in ``columns.pipeline_data["Selections"]`` as unapplied.

2. :class:`~analyzer.modules.common.selection.SelectOnColumns` applies all pending selections** by AND-ing the masks and filtering events.
   It also records a cutflow (how many events pass each successive cut) and N-1 efficiencies.

This pattern the advantage that **N-1 plots** can be computed since the individual masks are preserved until the selection is applied.


The ``pipeline_data["Selections"]`` dictionary tracks which selections have been applied (``True``) and which are still pending (``False``).
When :class:`~analyzer.modules.common.selection.SelectOnColumns` runs without explicit ``selection_names``, it applies all pending selections.


Metadata
--------

Every :class:`~analyzer.core.columns.TrackedColumns` object carries a ``metadata`` dictionary that contains information about the current sample:

.. code-block:: python

    def run(self, columns, params):
        meta = columns.metadata

        # Dataset-level metadata
        sample_type = meta["sample_type"]    # SampleType.MC or SampleType.Data
        dataset_name = meta["dataset_name"]
        sample_name = meta["sample_name"]
        x_sec = meta.get("x_sec")           # Cross section (MC only)
        n_events = meta["n_events"]          # Total events in sample

        # Era-level metadata (nested under "era")
        era_name = meta["era"]["name"]       # e.g., "2018"
        lumi = meta["era"]["lumi"]           # Integrated luminosity
        triggers = meta["era"]["trigger_names"]

This metadata is constructed by merging the dataset definition, sample definition, and era configuration.
It is available in both :meth:`~analyzer.core.analysis_modules.AnalyzerModule.run` and :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs`/:meth:`~analyzer.core.analysis_modules.AnalyzerModule.outputs`, so modules can change their behavior based on the dataset they are processing.


Pipeline Data
-------------

:class:`~analyzer.core.columns.TrackedColumns` has a ``pipeline_data`` dictionary for storing arbitrary state that modules want to share within a pipeline.
This is used by the framework for a few specific purposes:

- ``pipeline_data["Selections"]``: tracks which selections have been applied.
- ``pipeline_data["categories"]``: stores category axes for histogram production.

You can also use it for your own purposes in custom modules, though you should be aware that it is deep-copied when the pipeline branches for systematic variations.


``ColumnCollection``
--------------------

When a module declares its :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs` or :meth:`~analyzer.core.analysis_modules.AnalyzerModule.outputs`, it returns a list of :class:`~analyzer.core.columns.Column` objects (or a :class:`~analyzer.core.columns.ColumnCollection`).
A :class:`~analyzer.core.columns.ColumnCollection` is a set of columns that supports containment checks:

.. code-block:: python

    from analyzer.core.columns import ColumnCollection

    coll = ColumnCollection(["Jet", "MET"])
    coll.contains(Column("Jet.pt"))   # True -- Jet is a parent of Jet.pt
    coll.contains(Column("Electron")) # False

This is used by the framework to enforce that modules only read from their declared inputs and write to their declared outputs.
If a module tries to access a column it did not declare, a ``RuntimeError`` is raised.

A special return value of ``"EVENTS"`` from :meth:`~analyzer.core.analysis_modules.BaseAnalyzerModule.inputs` or :meth:`~analyzer.core.analysis_modules.AnalyzerModule.outputs` means the module operates on the entire event record and no input/output restrictions are applied.
