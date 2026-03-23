from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
from attrs import define, field
import correctionlib
import awkward as ak


@define
class HBQuarkMaker(AnalyzerModule):
    """
    Select b-tagged jets from a jet collection based on era-specified working points.

    This analyzer identifies b-jets in an event by applying a threshold
    on the b-tagging score as specified by central working point values.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    output_col : Column
        Column where the selected b-jets will be stored.
    working_point : str
        B-tagging working point to use, typically one of ``"L"``, ``"M"``, or ``"T"``.
    nontop2b: bool
        Whether to compute a column which has the top two b-tagged jets from
        input_col removed. Default False
    nontop2b_col: Column
        Column where nontop2b will be stored. Must define explicitly if nontop2b is True

    Notes
    -----
    - B-tagging thresholds are loaded from the correction file specified
      in ``metadata["era"]["btag_scale_factors"]["file"]``.
    - Desired tagger and path to correction thresholds are specified using 
      the "tagger" and "correction_name" fields in above metadata path.
    """

    input_col: Column
    output_col: Column
    working_point: str
    nontop2b: bool = False
    nontop2b_col: Column | None = None

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        tagger, wps = self.getWPs(columns.metadata)
        jets = columns[self.input_col]
        mask = jets[tagger] > wps[self.working_point]
        
        if self.nontop2b:
            if self.nontop2b_col is None:
                raise ValueError("nontop2b_col must be provided when nontop2b is True")
            else:
                local_indices = ak.local_index(jets, axis=1)
                # Indices of top two b's, if there are fewer than two then fill -1 (Avoids index out of range). 
                # As no jets have idx -1, they will still be included (nontop2b with 1b present only removes 1b).
                toptwob_indices = ak.fill_none(ak.pad_none(local_indices[mask], 2, axis=1)[:,:2], -1)
                ind_1 = toptwob_indices[:,0]
                ind_2 = toptwob_indices[:,1]
                nontop2b_mask = (local_indices != ind_1) & (local_indices != ind_2)
                nontop2b = jets[nontop2b_mask]
                columns[self.nontop2b_col] = nontop2b

        bjets = jets[mask]
        columns[self.output_col] = bjets
        return columns, []

    def getWPs(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        tagger = metadata["era"]["btag_scale_factors"]["tagger"]
        cname = metadata["era"]["btag_scale_factors"]["correction_name"]

        if file_path in self.__corrections:
            return tagger, self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {p: cset[cname].evaluate(p) for p in ("L", "M", "T")}
        self.__corrections[file_path] = ret
        return tagger, ret

    def preloadForMeta(self, metadata):
        self.getWPs(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        if self.nontop2b:
            return [self.nontop2b_col, self.output_col]
        return [self.output_col]
