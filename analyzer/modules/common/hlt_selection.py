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
class ComplexHLTConfig:
    pattern: BasePattern
    triggers: list[str]
    veto: list[str] = field(factory=list)


@define
class ComplexHLT(AnalyzerModule):
    """
    Select events based on HLT triggers, with different triggers for different datasets.
    Also supports vetoing events based on other triggers (e.g. for removing overlap).

    Parameters
    ----------
    trigger_config : list[ComplexHLTConfig]
        List of configuration objects for triggers.
    selection_name : str
        Name of the selection to be added to the columns.
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

        if triggers:
            pass_trigger = ft.reduce(
                op.or_, (hlt[trigger_names[name]] for name in triggers)
            )

        if vetos:
            veto_trigger = ft.reduce(
                op.or_, (hlt[trigger_names[name]] for name in vetos)
            )

        if pass_trigger is None:
            pass_trigger = ak.full_like(hlt[next(iter(trigger_names.values()))], False)

        final_selection = pass_trigger
        if veto_trigger is not None:
            final_selection = final_selection & (~veto_trigger)

        addSelection(columns, self.selection_name, final_selection)
        return columns, []

    def inputs(self, metadata):
        return [Column("HLT")]

    def outputs(self, metadata):
        return [Column(f"Selection.{self.selection_name}")]
