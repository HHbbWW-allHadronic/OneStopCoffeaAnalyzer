from typing import Optional
from dataclasses import dataclass, field, fields, replace

@dataclass
class Style:
    color: Optional[str]
    alpha: Optional[float]

    @staticmethod
    def fromDict(data):
        color = data.get("color")
        op = data.get("alpha")
        return Style(color, op)

    def keys(self):
        return (
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __getitem__(self, key):
        return getattr(self, key)

    def toDict(self):
        return dict(
            (field.name, getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        )
