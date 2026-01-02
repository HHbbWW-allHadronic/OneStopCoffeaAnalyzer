from __future__ import annotations
from email.errors import InvalidMultipartContentTransferEncodingDefect
from analyzer.utils.querying import BasePattern, Pattern, gatherByCapture, NO_MATCH
from analyzer.utils.structure_tools import (
    deepWalkMeta,
    SimpleCache,
    ItemWithMeta,
    commonDict,
)
from attrs import define


@define
class MetaLeaf:
    results: list[ItemWithMeta]
    transforms: None = None

    @property
    def metadata(self):
        return commonDict((x.metadata for x in self.results))


@define
class MetaCaptureDesc:
    group: BasePattern
    select: BasePattern | None = None
    sub_groups: list[MetaCaptureDesc] | dict[str, MetaCaptureDesc] | None = None
    transforms: None = None

    def apply(self, items):
        if self.select is not None:
            items = [x for x in items if self.select.match(x)]
        gathered = gatherByCapture(self.group, items)
        groups = [x for y, x in gathered if y is not NO_MATCH]
        if self.sub_groups is None:
            if len(groups) == 1:
                return groups[0]
            return groups
        ret = []
        for group_items in groups:
            if isinstance(self.sub_groups, dict):
                r = {}
                for x, y in self.sub_groups.items():
                    r[x] = y.apply(items)
            elif isinstance(self.sub_groups, list):
                r = []
                for x in self.sub_groups:
                    r.append(x.apply(items))
            ret.append(r)
        return ret
