from collections import defaultdict
from typing import TypeVar, Generic
from attrs import define

T = TypeVar("T")


@define
class Transform: ...


@define
class TransformHistogram(Transform): ...
