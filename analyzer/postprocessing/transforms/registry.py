from typing import TypeVar
from attrs import define

T = TypeVar("T")


@define
class Transform: ...


@define
class TransformHistogram(Transform): ...
