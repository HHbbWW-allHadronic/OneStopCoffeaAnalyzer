Pattern Matching
================

The framework uses a pattern matching system in several places:

- The ``dataset`` field in ``event_collections`` to select which datasets to process.
- The ``--filter-dataset`` and ``--filter-sample`` CLI options.
- The ``select`` and ``group`` fields in postprocessing configurations.
- Style rules in postprocessing.
- The ``drop_sample_pattern`` in postprocessing.
- The ``systs`` field in :class:`~analyzer.core.run_builders.LimitSysts` run builders.

We summarize here the basics of the system. 
Ideally it would be nice to switch perhaps to a more robust pattern matcher in the future.

Basic Patterns
--------------

The most basic pattern matches against a simple scalar value, almost always a string.
The simplest and most common form is a **glob pattern** -- a string with ``*`` as a wildcard:

.. code-block:: yaml

    dataset: 'signal_2018_*'     # Matches any dataset starting with "signal_2018_"
    dataset: '*JetHT*'           # Matches any dataset containing "JetHT"
    dataset: 'qcd_ht_2018'      # Exact match (no wildcards)

Glob patterns are the default mode.
They use standard shell-style matching: ``*`` matches any sequence of characters, ``?`` matches a single character.


Negation
--------

Prefixing a pattern with ``!`` negates it:

.. code-block:: yaml

    dataset_name: '!signal*'     # Matches anything that does NOT start with "signal"


Regex Patterns
--------------

For more complex matching, prefix the pattern with ``re:`` to use a regular expression:

.. code-block:: yaml

    dataset_name: 're:signal_2018_312_(1000|1500)_.*'

This matches ``signal_2018_312_1000_...`` and ``signal_2018_312_1500_...`` but nothing else.


Deep Patterns (Metadata Matching)
---------------------------------

For matching against more sophisticated data, we use a dictionary matching system, which we refer to as a deep pattern.
When matching against structured metadata (a dictionary), you can specify patterns for specific keys.
This is the most common usage in postprocessing configurations:

.. code-block:: yaml

    select:
      type: Histogram            # Match results of type "Histogram"
      sample_type: MC            # Match MC samples only
      pipeline: "Signal*"        # Match pipelines starting with "Signal"
      dataset_name: "!signal*"   # Exclude signal datasets

Each key-value pair becomes a :class:`~analyzer.utils.querying.DeepPattern`: the key specifies the path in the metadata dictionary, and the value is a pattern matched against the value at that path.

When multiple keys are specified, they are combined with AND -- all must match.

Nested keys are supported using dot notation:

.. code-block:: yaml

    group: {"era.name": "*"}     # Group by the "name" field under the "era" key


Compound Patterns
-----------------

For more complex logic, you can use explicit AND, OR, and NOT:

**OR** (match any):

.. code-block:: yaml

    drop_sample_pattern:
      or_exprs:
        - sample_name: "*50to100*"
        - sample_name: "*100to200*"
        - sample_name: "*200to300*"

**AND** (match all):

.. code-block:: yaml

    some_pattern:
      and_exprs:
        - dataset_name: "qcd*"
        - era.name: "2018"

**NOT** (invert):

.. code-block:: yaml

    some_pattern:
      not_expr:
        dataset_name: "signal*"


Capture and Grouping
--------------------

In postprocessing, patterns are used not only for filtering but also for **grouping**.
The ``group`` field specifies a pattern where ``*`` wildcards act as capture groups:

.. code-block:: yaml

    structure:
      group: {"era.name": "*"}

This groups results by their era name -- all results with ``era.name = "2018"`` are placed in one group, those with ``era.name = "2017"`` in another, etc.

The capture value becomes part of the metadata that can be used in output file naming:

.. code-block:: yaml

    output_name: "{prefix}/{era.name}/plot.png"

For more details on how grouping works in postprocessing, see :doc:`../postprocessing/postprocessing_overview`.


Summary of Pattern Syntax
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Syntax
     - Meaning
   * - ``"*"``
     - Match anything (glob wildcard)
   * - ``"signal*"``
     - Glob: starts with "signal"
   * - ``"!signal*"``
     - Negated glob: does NOT start with "signal"
   * - ``"re:pattern"``
     - Regular expression matching
   * - ``{key: pattern}``
     - Deep pattern: match ``pattern`` against ``metadata[key]``
   * - ``{"a.b": pattern}``
     - Nested deep pattern: match against ``metadata["a"]["b"]``
   * - ``or_exprs: [...]``
     - OR: match any sub-pattern
   * - ``and_exprs: [...]``
     - AND: match all sub-patterns
   * - ``not_expr: ...``
     - NOT: invert the sub-pattern
