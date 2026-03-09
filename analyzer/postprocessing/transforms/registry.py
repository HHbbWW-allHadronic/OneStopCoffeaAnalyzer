from typing import TypeVar
from attrs import define
import functools as ft
from cattrs.strategies import (
    include_subclasses,
    configure_tagged_union,
    configure_union_passthrough,
)

T = TypeVar("T")


@define
class Transform: ...


@define
class TransformHistogram(Transform): ...

@define
class TransformSavedColumns(Transform): ...


def configureConverter(conv):
    base_list_str = conv.get_structure_hook(list[str])
    base_list_int = conv.get_structure_hook(list[int])
    base_list_float = conv.get_structure_hook(list[float])

    @conv.register_structure_hook
    def _(data, t) ->  list[str] | list[int] | list[float]:
        if len(data) == 0:
            return []
        if isinstance(data, str):
            return [data]
        if isinstance(data[0], str):
            return base_list_str(data, list[str])
        elif isinstance(data[0], int):
            return base_list_int(data, list[int])
        else:
            return base_list_float(data, list[float])



    union_strategy = ft.partial(configure_tagged_union, tag_name="name")
    include_subclasses(Transform, conv, union_strategy=union_strategy)
