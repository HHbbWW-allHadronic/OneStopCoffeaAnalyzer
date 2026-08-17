from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
from attrs import define, field
import correctionlib
import awkward as ak
from typing import Optional

@define
class HCQuarkMaker(AnalyzerModule):
    """
    Select c-tagged jets from a jet collection based on era-specified working points.

    This analyzer identifies c-jets in an event by applying a threshold
    on the c-tagging score as specified by central working point values.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    output_col : Column
        Column where the selected b-jets will be stored.
    working_point : str
        C-tagging working point to use, typically one of ``"L"``, ``"M"``, or ``"T"``.

    Notes
    -----
    - C-tagging thresholds are loaded from the correction file specified
      in ``metadata["era"]["btag_scale_factors"]["file"]``.
    - Desired tagger and path to correction thresholds are specified using 
      the "tagger" and "correction_name" fields in above metadata path.
    """

    input_col: Column
    output_col: Column
    notc_col: Column
    working_point: str                        # Mandatory: No default
    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        taggers, wps = self.getWPs(columns.metadata)
        jets = columns[self.input_col]
        mask1 = jets[taggers["CvB"]] > wps["CvB"][self.working_point]
        mask2 = jets[taggers["CvL"]] > wps["CvL"][self.working_point]
        mask = mask1&mask2
        notmask= ~mask 
        jets["isMedC"] = ak.values_astype(mask, "float64")
        jets["NotMedC"] = ak.values_astype(notmask, "float64")
        bjets = jets[mask]
        notbjets = jets[notmask]
        columns[self.notc_col] = notbjets
        columns[self.output_col] = bjets
        
        return columns, []

    def getWPs(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["c_file"]
        tagger1 = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvB"]
        tagger2 = metadata["era"]["btag_scale_factors"]["c_tagger"]["CvL"]
        taggers = {"CvB": tagger1, "CvL": tagger2}
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__corrections:
            return taggers, self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret ={"CvL": {p: cset[cname].evaluate(p, "CvL") for p in ("L", "M", "T")}, "CvB":  {p: cset[cname].evaluate(p, "CvB") for p in ("L", "M", "T")}} 
        self.__corrections[file_path] = ret
        return taggers, ret

    def preloadForMeta(self, metadata):
        self.getWPs(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        cols = [self.output_col]
        if getattr(self, "notc_col", None):
            cols.append(self.notc_col)
        return cols



@define
class FractionHC(AnalyzerModule):

    
    input_col_num: Column
    input_col_den: Column
    output_col: Column



    
    def run(self, columns, params):
        
        
        num = columns[self.input_col_num]
        den = columns[self.input_col_den]
        fraction = num / den
        columns[self.output_col] = fraction
        return columns, []


        
    def inputs(self, metadata):
        return [self.input_col_num, self.input_col_den]

    def outputs(self, metadata):
        return [self.output_col]


