from __future__ import annotations
from attrs import define, field
from analyzer.core.datasets import SampleType


@define
class MetadataSpec:
    era: str
    sampleType: SampleType
    label: str | None = None

    @property
    def name(self):
        return self.label or f"{self.era}_{self.sampleType.value}"


@define(eq=True, frozen=True)
class ADLStatement:
    keyword: str
    expression: str

    def toString(self) -> str:
        return f"  {self.keyword} {self.expression}"


@define
class ADLBlock:
    block_type: str
    name: str
    statements: list[ADLStatement]
    comment: str | None = None
    inherit: str | None = None

    def toString(self) -> str:
        lines = []
        if self.comment:
            lines.append(f"# {self.comment}")

        if self.block_type == "define":
            for stmt in self.statements:
                lines.append(f"{stmt.keyword} {stmt.expression}")
        elif self.block_type == "region_statement":
            pass
        else:
            lines.append(f"{self.block_type} {self.name}")
            if self.inherit:
                lines.append(f"  take {self.inherit}")
            for stmt in self.statements:
                lines.append(stmt.toString())
        return "\n".join(lines)


def buildMetadata(spec: MetadataSpec, eraRepo) -> dict:
    era = eraRepo[spec.era]
    return {
        "era": era,
        "sample_type": spec.sampleType,
        "dataset_name": f"adl_export_{spec.name}",
        "sample_name": f"adl_export_{spec.name}",
    }


class ADLEmitter:
    def __init__(self, title=None, config_path=None, context_name=None):
        self.title = title
        self.config_path = config_path
        self.context_name = context_name
        self.blocks = []

    def addBlock(self, block: ADLBlock):
        self.blocks.append(block)

    def render(self) -> str:
        out = []
        out.append("# Auto-generated ADL from SingleStopCoffea analyzer")
        if self.config_path:
            out.append(f"# Source config: {self.config_path}")
        if self.context_name:
            out.append(f"# Context: {self.context_name}")
        out.append("")

        out.append("info analysis")
        if self.title:
            out.append(f"  title {self.title}")
        out.append("  experiment CMS")
        out.append("")

        out.append("info adl")
        out.append("  adlauthor auto-generated")
        out.append("")
        seen = set()
        objects = []
        for b in self.blocks:
            if b.block_type == "object" and b.name not in seen:
                objects.append(b)
                seen.add(b.name)

        if objects:
            out.append("# --- Object definitions ---")
            out.append("")
            for b in objects:
                out.append(b.toString())
                out.append("")

        # Defines
        defines = []
        seen_defines = set()
        for b in self.blocks:
            if b.block_type == "define":
                for stmt in b.statements:
                    if stmt.expression not in seen_defines:
                        defines.append(stmt)
                        seen_defines.add(stmt.expression)

        if defines:
            out.append("# --- Global definitions ---")
            out.append("")
            for stmt in defines:
                out.append(f"{stmt.keyword} {stmt.expression}")
            out.append("")

        composites = []
        for b in self.blocks:
            if b.block_type == "composite" and b.name not in seen:
                composites.append(b)
                seen.add(b.name)

        if composites:
            out.append("# --- Composite definitions ---")
            out.append("")
            for b in composites:
                out.append(b.toString())
                out.append("")

        regions = [b for b in self.blocks if b.block_type == "region"]
        if regions:
            out.append("# --- Region definitions ---")
            out.append("")
            for b in regions:
                out.append(b.toString())
                out.append("")

        return "\n".join(out)
