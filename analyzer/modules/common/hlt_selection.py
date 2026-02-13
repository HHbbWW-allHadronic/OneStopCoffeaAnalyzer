from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
import operator as op
from analyzer.utils.querying import Pattern, PatternMode
from attrs import define, field
from analyzer.utils.querying import BasePattern
import awkward as ak

from analyzer.core.columns import addSelection
import functools as ft


@define
class SimpleHLT(AnalyzerModule):
    """
    Select events based on HLT triggers.

    Parameters
    ----------
    triggers : list[str]
        List of trigger names to select.
    selection_name : str
        Name of the selection to be added to the columns.
    """

    triggers: list[str]
    selection_name: str = "PassHLT"

    def run(self, columns, params):
        metadata = columns.metadata
        trigger_names = metadata["era"]["trigger_names"]
        hlt = columns["HLT"]
        pass_trigger = ft.reduce(
            op.or_, (hlt[trigger_names[name]] for name in self.triggers)
        )
        addSelection(columns, self.selection_name, pass_trigger)
        return columns, []

    def inputs(self, metadata):
        return [Column(("HLT"))]

    def outputs(self, metadata):
        return [Column(f"Selection.{self.selection_name}")]


@define
class SaveHLT(AnalyzerModule):
    """
    Save HLT triggers to the output columns.
    """

    triggers: list[str]
    save_name: str = "SavedHLT"

    def run(self, columns, params):
        metadata = columns.metadata
        trigger_names = metadata["era"]["trigger_names"]
        hlt = columns["HLT"]
        for name in self.triggers:
            columns[f"{self.save_name}.{name}"] = hlt[trigger_names[name]]
        return columns, []

    def inputs(self, metadata):
        return [Column(("HLT"))]

    def outputs(self, metadata):
        return [Column(f"{self.save_name}.{name}") for name in self.triggers]


@define
class ComplexHLTConfig:
    pattern: BasePattern
    triggers: list[str]
    veto: list[str] = field(factory=list)


@define
class ComplexHLT(AnalyzerModule):
    """
    Analyzer module applying complex HLT-based selections with dataset-dependent
    trigger logic.

    This module evaluates High-Level Trigger (HLT) decisions using a configurable
    set of trigger and veto definitions. The configuration is selected dynamically
    based on the dataset name using regular-expression pattern matching.

    For a matched configuration:
      - All configured trigger paths are OR-combined to form the *pass* condition.
      - All configured veto paths are OR-combined to form the *veto* condition.
      - The final selection is defined as::

            pass_trigger AND (NOT veto_trigger)


    Parameters
    ----------
    trigger_config : list[ComplexHLTConfig]
        Ordered list of trigger configurations. Each configuration must define
        a regular-expression pattern used to match the dataset name, along with
        trigger and veto path names. The first matching configuration is used.
    selection_name : str, optional
        Name of the selection written to the output columns. The selection is stored
        under ``Selection.<selection_name>``. Default is ``"PassHLT"``.

    Raises
    ------
    ValueError
        If no trigger configuration matches the dataset name.


    """

    trigger_config: list[ComplexHLTConfig]
    selection_name: str = "PassHLT"

    def run(self, columns, params):
        metadata = columns.metadata
        dataset_name = metadata.get("dataset_name", "")

        trigger_names = metadata["era"]["trigger_names"]
        hlt = columns["HLT"]

        pass_trigger = None
        veto_trigger = None

        matched_config = None

        for conf in self.trigger_config:
            if conf.pattern.match(dataset_name):
                matched_config = conf
                break
        if matched_config is None:
            raise ValueError(
                f"No matching trigger config found for dataset {dataset_name}"
            )

        triggers = matched_config.triggers
        vetos = matched_config.veto

        pass_trigger = ft.reduce(
            op.or_, (hlt[trigger_names[name]] for name in triggers)
        )

        if vetos:
            veto_trigger = ft.reduce(
                op.or_, (hlt[trigger_names[name]] for name in vetos)
            )

        final_selection = pass_trigger
        if veto_trigger is not None:
            final_selection = final_selection & (~veto_trigger)

        addSelection(columns, self.selection_name, final_selection)
        return columns, []

    def inputs(self, metadata):
        return [Column("HLT")]

    def outputs(self, metadata):
        return [Column(f"Selection.{self.selection_name}")]
