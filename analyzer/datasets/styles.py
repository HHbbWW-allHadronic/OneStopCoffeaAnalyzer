from typing import Optional
from dataclasses import dataclass, field, fields, replace
from pydantic import BaseModel, Field, validator

class Style(BaseModel):
    color: Optional[str]
    alpha: Optional[float]

    def keys(self):
        return (
            field.name
            for field in fields(self)
            if getattr(self, field.name) is not None
        )

    def __getitem__(self, key):
        return getattr(self, key)

