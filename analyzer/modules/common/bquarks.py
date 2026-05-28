from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
from attrs import define, field
import correctionlib
from analyzer.core.adl import ADLBlock, ADLStatement


@define
class BQuarkMaker(AnalyzerModule):
    """
    Select b-tagged jets from a jet collection based on DeepJet working points.

    This analyzer identifies b-jets in an event by applying a threshold
    on the DeepJet b-tagging score as specified by central working point values.

    Parameters
    ----------
    input_col : Column
        Column containing the input jet collection.
    output_col : Column
        Column where the selected b-jets will be stored.
    working_point : str
        B-tagging working point to use, typically one of ``"L"``, ``"M"``, or ``"T"``.

    Notes
    -----
    - B-tagging thresholds are loaded from the correction file specified
      in ``metadata["era"]["btag_scale_factors"]["file"]``.
    """

    input_col: Column
    output_col: Column
    working_point: str

    __corrections: dict = field(factory=dict)

    def run(self, columns, params):
        wps = self.getWPs(columns.metadata)
        jets = columns[self.input_col]
        bjets = jets[jets.btagDeepFlavB > wps[self.working_point]]
        columns[self.output_col] = bjets
        return columns, []

    def getWPs(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = {p: cset["deepJet_wp_values"].evaluate(p) for p in ("L", "M", "T")}
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getWPs(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

    def adlExport(self, metadata):
        try:
            wps = self.getWPs(metadata)
            threshold = f"{wps[self.working_point]:.4f}"
        except Exception:
            threshold = f"<deepJet_{self.working_point}_WP>"

        statements = [
            ADLStatement("take", self.input_col.adl_name),
            ADLStatement("select", f"btagDeepFlavB > {threshold}"),
        ]

        return [
            ADLBlock(
                block_type="object",
                name=self.output_col.adl_name,
                statements=statements,
            )
        ]
