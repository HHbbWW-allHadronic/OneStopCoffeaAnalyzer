from analyzer.core.analysis_modules import AnalyzerModule, register_module
from analyzer.core.columns import Column
from attrs import define, field
from .axis import RegularAxis
from .histogram_builder import makeHistogram
from analyzer.core.results import SavedColumns
import correctionlib
import enum


@define
class SaveCols(AnalyzerModule):
    save_name: str
    to_save: dict[str, str]
    remap_hlt_prefix: bool = True

    def run(self, columns, params):
        ret = {}
        era_trigger_names = columns.metadata["era"]["trigger_names"]
        for col in self.to_save:
            real_name = col
            if real_name.startswith("HLTMAP:"):
                real_name = "HLT." + era_trigger_names[real_name[7:]]
            ret[col] = columns[real_name]
        return columns, [SavedColumns(self.save_name, ret)]

    def inputs(self, metadata):
        era_trigger_names = metadata["era"]["trigger_names"]
        ret = []
        for col in self.to_save:
            real_name = col
            if real_name.startswith("HLTMAP:"):
                real_name = "HLT." + era_trigger_names[real_name[7:]]
            ret.append(Column(real_name))
        return ret

    def outputs(self, metadata):
        return []
