from __future__ import annotations

from typing import TypeVar
from analyzer.utils.querying import BasePattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import flatten
from analyzer.utils.structure_tools import (
    ItemWithMeta,
)
from attrs import define
from .transforms.registry import Transform

ResultSet = list[list[ItemWithMeta]]

T = TypeVar("T")


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

    def apply(self, items):
        """
        Applies the selection, grouping, transformations, and subgroup
        operations to the given items.

        Args:
            items: A list of items (typically objects with metadata) to be
                processed.

        Returns:
            The processed groups. The exact return type depends on the structure
            of `subgroups`:
            - If `subgroups` is None, returns the list of transformed groups.
            - If `subgroups` is a dict, returns a list of dictionaries containing
              the subgroup results.
            - If `subgroups` is a list, returns a list of lists containing the
              subgroup results.
        """
        transforms = self.transforms or []
        transforms = flatten(transforms)

        # 1. Filter items: Only keep items whose metadata matches
        if self.select is not None:
            items = [x for x in items if self.select.match(x.metadata)]

        if self.group:
            # Take remaining items and form groups based on the capture pattern
            gathered = gatherByCapture(self.group, items)

            groups: ResultSet = [g.items for g in gathered if g.capture is not NO_MATCH]
            # Groups are now a list[list[ItemWithMeta]]

            for transform in transforms:
                groups = [applyTransform(transform, g) for g in groups]
        else:
            # No grouping specified: Treat all filtered items as one large single group
            groups = items

            # Apply transformations sequentially to the entire single group
            for transform in transforms:
                groups = applyTransform(transform, groups)

        # If no subgroup operations are defined, we are done
        if self.subgroups is None:
            return groups

        ret = []
        for group_items in groups:
            # If subgroups is a dict, apply each named GroupBuilder
            # recursively and return a dict
            if isinstance(self.subgroups, dict):
                r = {}
                for x, y in self.subgroups.items():
                    r[x] = y.apply(group_items)
            # If subgroups is a list, apply each GroupBuilder
            # recursively and return a list
            elif isinstance(self.subgroups, list):
                r = []
                for x in self.subgroups:
                    r.append(x.apply(group_items))
            ret.append(r)

        return ret


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
