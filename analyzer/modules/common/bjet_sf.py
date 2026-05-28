import itertools as it
import awkward as ak
import correctionlib
import correctionlib.convert
from analyzer.core.columns import Column
from attrs import define, field
import correctionlib
from analyzer.core.datasets import SampleType
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    MetadataExpr,
    ParameterSpec,
    ModuleParameterSpec,
    IsSampleType,
)


@define
class BJetShapeSF(AnalyzerModule):
    input_col: Column
    weight_name: str = "b_tag_disc_shape"

    should_run: MetadataExpr = field(factory=lambda: IsSampleType(SampleType.MC))

    __corrections: dict = field(factory=dict)

    def getParameterSpec(self, metadata):
        b_meta = metadata["era"]["btag_scale_factors"]
        systematics = b_meta["systematics"]
        possible_values = it.product(["up", "down"], systematics)
        possible_values = (
            ["central"]
            + [f"{updown}_{name}" for updown, name in possible_values]
            + ["disabled"]
        )

        jes_correlated = b_meta.get("jes_correlated_systematics", [])
        jes_values = list(it.product(["up", "down"], jes_correlated))
        jes_correlated_values = [
            f"{updown}_jes{name}".replace("Regrouped_", "")
            for updown, name in jes_values
        ]
        possible_values += jes_correlated_values

        driven_by = None
        if jes_correlated_values:

            def jesToBtag(jes_val):
                if jes_val == "central":
                    return None
                return jes_val.replace("Regrouped_", "")

            driven_by = {"jes-variation": jesToBtag}

        return ModuleParameterSpec(
            {
                "bjetshapesf-variation": ParameterSpec(
                    default_value="central",
                    possible_values=possible_values,
                    tags={"weight_variation"},
                    driven_by=driven_by,
                ),
            }
        )

    def run(self, columns, params):
        sf_eval = self.getCorrection(columns.metadata)
        systematic = params["bjetshapesf-variation"]
        systematic = systematic.removesuffix("_" + columns.metadata["era"]["name"])
        gj = columns[self.input_col]
        if systematic == "disabled":
            columns["Weights", self.weight_name] = ak.ones_like(ak.firsts(gj.pt))
            return columns, []

        if systematic == "central":
            j = gj
            sf = ak.prod(
                sf_eval.evaluate(
                    "central", j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )
        elif "_cf" in systematic:
            j = gj[gj.hadronFlavour == 4]
            sf = ak.prod(
                sf_eval.evaluate(
                    systematic, j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )
        else:
            j = gj[gj.hadronFlavour != 4]
            sf = ak.prod(
                sf_eval.evaluate(
                    systematic, j.hadronFlavour, abs(j.eta), j.pt, j.btagDeepFlavB
                ),
                axis=1,
            )

        columns["Weights", self.weight_name] = sf
        return columns, []

    def getCorrection(self, metadata):
        file_path = metadata["era"]["btag_scale_factors"]["file"]
        if file_path in self.__corrections:
            return self.__corrections[file_path]
        cset = correctionlib.CorrectionSet.from_file(file_path)
        ret = cset["deepJet_shape"]
        self.__corrections[file_path] = ret
        return ret

    def preloadForMeta(self, metadata):
        self.getCorrection(metadata)

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [Column(("Weights", self.weight_name))]
