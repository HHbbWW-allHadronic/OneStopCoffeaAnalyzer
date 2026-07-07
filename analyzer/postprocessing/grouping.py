from __future__ import annotations

from typing import TypeVar, Literal
from analyzer.utils.querying import BasePattern, gatherByCapture, CaptureSet, NO_MATCH
from analyzer.utils.structure_tools import flatten
from analyzer.utils.structure_tools import (
    ItemWithMeta,
)
from attrs import define, field, Factory
from .transforms.registry import Transform

ResultSet = list[list[ItemWithMeta]]

T = TypeVar("T")


def _itemSummary(item: ItemWithMeta) -> str:
    meta = item.metadata
    parts = []
    for key in ("name", "dataset_name", "pipeline", "sample_type"):
        val = meta.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return ", ".join(parts) if parts else repr(dict(meta))


@define
class InputTrace:
    count: int = 0
    items: list[str] = Factory(list)

    def record(self, items: list[ItemWithMeta]):
        self.count = len(items)
        self.items = [_itemSummary(x) for x in items]


@define
class SelectTrace:
    pattern: str | None = None
    selected_count: int = 0
    dropped_count: int = 0
    selected_items: list[str] = Factory(list)
    dropped_items: list[str] = Factory(list)

    def record(
        self,
        pattern: BasePattern,
        selected: list[ItemWithMeta],
        dropped: list[ItemWithMeta],
    ):
        self.pattern = repr(pattern)
        self.selected_count = len(selected)
        self.dropped_count = len(dropped)
        self.selected_items = [_itemSummary(x) for x in selected]
        self.dropped_items = [_itemSummary(x) for x in dropped]


@define
class CaptureGroupTrace:
    key: str = ""
    count: int = 0
    items: list[str] = Factory(list)

    @classmethod
    def fromCaptureSet(cls, cs: CaptureSet) -> CaptureGroupTrace:
        t = cls(key=repr(cs.capture), count=len(cs.items))
        t.items = [_itemSummary(x) for x in cs.items]
        return t


@define
class GroupingTrace:
    pattern: str | None = None
    groups: list[CaptureGroupTrace] = Factory(list)

    def record(
        self,
        pattern: BasePattern,
        capture_sets: list[CaptureSet],
    ):
        self.pattern = repr(pattern)
        self.groups = [CaptureGroupTrace.fromCaptureSet(cs) for cs in capture_sets]


@define
class TransformStepTrace:
    name: str = ""
    post_count: int = 0


@define
class GroupTrace:
    input: InputTrace = Factory(InputTrace)
    select: SelectTrace | None = None
    grouping: GroupingTrace | None = None
    transforms: list[TransformStepTrace] = Factory(list)

    subgroup_traces: list[dict[str, GroupTrace] | list[GroupTrace]] = Factory(list)

    def recordInput(self, items: list[ItemWithMeta]):
        self.input.record(items)

    def recordSelect(
        self,
        pattern: BasePattern,
        selected: list[ItemWithMeta],
        dropped: list[ItemWithMeta],
    ):
        self.select = SelectTrace()
        self.select.record(pattern, selected, dropped)

    def recordGrouping(
        self,
        pattern: BasePattern,
        capture_sets: list[CaptureSet],
    ):
        self.grouping = GroupingTrace()
        self.grouping.record(pattern, capture_sets)

    def recordTransform(self, transform, total_count: int):
        self.transforms.append(
            TransformStepTrace(
                name=type(transform).__name__,
                post_count=total_count,
            )
        )

    def makeChildTrace(self) -> GroupTrace:
        return GroupTrace()


def applyTransform(transform, items):
    ret = []
    to_transform = []
    for item in items:
        if transform.should_run is not None and not transform.should_run.match(
            item.metadata
        ):
            ret.append(item)
        else:
            to_transform.append(item)
    ret.extend(transform(to_transform))
    return ret


@define
class GroupBuilder:
    """
    Constructs and applies a sequence of operations to filter, group, and
    transform collections of items based on their metadata.

    Attributes:
        group (BasePattern | None): A pattern used to capture and group items.
            If provided, items are gathered by the capture groups defined in
            this pattern.
        select (BasePattern | None): A pattern used to filter the initial items.
            Only items whose metadata matches this pattern are kept.
        subgroups (list[GroupBuilder] | dict[str, GroupBuilder] | None): Nested
            group builders to apply recursively to the resulting groups. If a
            dictionary, the output will be a dictionary with corresponding keys.
            If a list, the output will be a list of applied results.
        transforms (list[Transform] | None): A list of transformation functions
            to apply to each formed group.
    """

    group: BasePattern | None = None
    select: BasePattern | None = None
    subgroups: list[GroupBuilder] | dict[str, GroupBuilder] | None = None
    transforms: list[Transform | list[Transform]] | None = None

    def apply(self, items, trace: GroupTrace | None = None):
        """
        Applies the selection, grouping, transformations, and subgroup
        operations to the given items.

        Args:
            items: A list of items (typically objects with metadata) to be
                processed.
            trace: If provided, populated in-place with a trace of every
                decision made during the pipeline.

        Returns:
            The processed groups. The exact return type depends on the structure
            of ``subgroups``:

            - If ``subgroups`` is None, returns the list of transformed groups.
            - If ``subgroups`` is a dict, returns a list of dictionaries containing
              the subgroup results.
            - If ``subgroups`` is a list, returns a list of lists containing the
              subgroup results.
        """
        transforms = self.transforms or []
        transforms = flatten(transforms)

        if trace is not None:
            trace.recordInput(items)

        # 1. Filter items: Only keep items whose metadata matches
        if self.select is not None:
            selected = [x for x in items if self.select.match(x.metadata)]
            if trace is not None:
                dropped = [x for x in items if not self.select.match(x.metadata)]
                trace.recordSelect(self.select, selected, dropped)
            items = selected

        if self.group:
            # Take remaining items and form groups based on the capture pattern
            gathered = gatherByCapture(self.group, items)
            valid = [g for g in gathered if g.capture is not NO_MATCH]

            if trace is not None:
                trace.recordGrouping(self.group, valid)

            groups: ResultSet = [g.items for g in valid]
            # Groups are now a list[list[ItemWithMeta]]

            for transform in transforms:
                groups = [applyTransform(transform, g) for g in groups]
                if trace is not None:
                    trace.recordTransform(transform, sum(len(g) for g in groups))
        else:
            # No grouping specified: Treat all filtered items as one large single group
            groups = items

            # Apply transformations sequentially to the entire single group
            for transform in transforms:
                groups = applyTransform(transform, groups)
                if trace is not None:
                    trace.recordTransform(transform, len(groups))

        # If no subgroup operations are defined, we are done
        if self.subgroups is None:
            return groups

        ret = []
        for group_items in groups:
            # If subgroups is a dict, apply each named GroupBuilder
            # recursively and return a dict
            if isinstance(self.subgroups, dict):
                r = {}
                sub_traces = {} if trace is not None else None
                for x, y in self.subgroups.items():
                    child_trace = trace.makeChildTrace() if trace is not None else None
                    r[x] = y.apply(group_items, trace=child_trace)
                    if sub_traces is not None:
                        sub_traces[x] = child_trace
                if trace is not None:
                    trace.subgroup_traces.append(sub_traces)
            # If subgroups is a list, apply each GroupBuilder
            # recursively and return a list
            elif isinstance(self.subgroups, list):
                r = []
                sub_traces = [] if trace is not None else None
                for x in self.subgroups:
                    child_trace = trace.makeChildTrace() if trace is not None else None
                    r.append(x.apply(group_items, trace=child_trace))
                    if sub_traces is not None:
                        sub_traces.append(child_trace)
                if trace is not None:
                    trace.subgroup_traces.append(sub_traces)
            ret.append(r)

        return ret

    def explain(self, items) -> GroupTrace:
        trace = GroupTrace()
        self.apply(items, trace=trace)
        return trace

    def _applySimple(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]
        if self.group:
            gathered = gatherByCapture(self.group, items)
            groups: ResultSet = [g.items for g in gathered if g.capture is not NO_MATCH]
        else:
            groups = items
        return groups


def configureConverter(conv):
    base_transform_hook = conv.get_structure_hook(Transform)
    base_list_transform_hook = conv.get_structure_hook(list[Transform])

    @conv.register_structure_hook
    def _(data, t) -> list[Transform] | Transform:
        if isinstance(data, list):
            return base_list_transform_hook(data, list[Transform])
        else:
            return base_transform_hook(data, Transform)

    @conv.register_structure_hook
    def _(data, t) -> list[GroupBuilder] | dict[str, GroupBuilder] | None:
        if data is None:
            return None
        if isinstance(data, list):
            return [conv.structure(x, GroupBuilder) for x in data]
        if isinstance(data, dict):
            return {k: conv.structure(v, GroupBuilder) for k, v in data.items()}
        else:
            raise RuntimeError()
