from __future__ import annotations
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
)
from attrs import define

ResultSet = list[list[ItemWithMeta]]


@define
class GroupBuilder:
    group: BasePattern
    select: BasePattern | None = None
    subgroups: list[GroupBuilder] | dict[str, GroupBuilder] | None = None
    transforms: None = None

    def apply(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]
        gathered = gatherByCapture(self.group, items)
        groups:  ResultSet = [g.items for g in gathered if g.capture is not NO_MATCH]

        for transform in self.transforms or []:
            groups = [transform(g) for g in groups]

        if self.subgroups is None:
            if len(groups) == 1:
                return groups[0]
            else:
                return groups

        ret = []
        for group_items in groups:
            if isinstance(self.subgroups, dict):
                r = {}
                for x, y in self.subgroups.items():
                    r[x] = y.apply(group_items)
            elif isinstance(self.subgroups, list):
                r = []
                for x in self.subgroups:
                    r.append(x.apply(group_items))
            ret.append(r)

        return ret

@define
class Histogram1D:
    inputs: GroupBuilder

    # style_set: str | StyleSet
    # output_name: str
    # 
    # to_stack: PatternExpression | None = None
    # 
    # scale: Literal["log", "linear"] = "linear"
    # normalize: bool = False
    # plot_configuration: PlotConfiguration | None = None

    def getRunFunc(self, group):
        common_meta = commonDict(group)

        for name, sector_pipeline in it.product(self.histogram_names, pipelines):
            histograms = sector_pipeline.getHists(name)
            if not histograms:
                return
            provenance = histograms[0].provenance
            output_path = doFormatting(self.output_name, **provenance.allEntries())
            stacked_hists = None
            if self.to_stack is not None:
                stacked_hists = [
                    x for x in histograms if self.to_stack.match(x.sector_parameters)
                ]
                histograms = [
                    x
                    for x in histograms
                    if not self.to_stack.match(x.sector_parameters)
                ]

            pc = self.plot_configuration.makeFormatted(**provenance.allEntries())
            yield ft.partial(
                plotOne,
                histograms,
                provenance,
                output_path,
                scale=self.scale,
                style_set=self.style_set,
                normalize=self.normalize,
                plot_configuration=pc,
                stacked_hists=stacked_hists,
            )
