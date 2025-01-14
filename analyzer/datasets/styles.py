from dataclasses import fields
from pydantic import BaseModel


class Style(BaseModel):
    color: str | None
    alpha: float | None

    def keys(self):
        return (
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __getitem__(self, key):
        return getattr(self, key)
