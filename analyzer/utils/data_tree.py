from attrs import define
from typing import Generic,TypeVar

T = TypeVar('T')

@define
class Group(Generic[T]):
    children: dict[str, Group | T]

    def getWithMeta(self, path):
        current_meta = copy.copy(self.metadata)
        return current_meta, current

