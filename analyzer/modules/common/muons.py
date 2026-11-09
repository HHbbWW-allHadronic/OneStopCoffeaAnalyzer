from analyzer.core.analysis_modules import AnalyzerModule, MetadataExpr
from analyzer.core.columns import Column
from attrs import define, field
import enum
from analyzer.core.adl import ADLBlock, ADLStatement


class IdWps(str, enum.Enum):
    loose = "looseId"
    medium = "mediumId"
    tight = "tightId"


class IsoWps(str, enum.Enum):
    very_loose = "very_loose"
    loose = "loose"
    medium = "medium"
    tight = "tight"
    very_tight = "very_tight"
    very_very_tight = "very_very_tight"


cut_mapping = dict(
    very_loose=1, loose=2, medium=3, tight=4, very_tight=5, very_very_tight=6
)


@define
class MuonMaker(AnalyzerModule):
    """
    Select muons based on kinematics, ID, and isolation criteria.

    This analyzer filters muons in an event according to minimum transverse
    momentum, maximum pseudorapidity, a chosen ID working point, and
    optional isolation requirements.

    Parameters
    ----------
    input_col : Column
        Column containing the input muon collection.
    output_col : Column
        Column where the selected muons will be stored.
    id_working_point : IdWps
        Muon ID working point (loose, medium, tight).
    min_pt : float, optional
        Minimum transverse momentum in GeV, by default 10.
    max_abs_eta : float, optional
        Maximum absolute pseudorapidity, by default 2.4.
    max_mini_iso : float, optional
        Maximum mini-isolation, by default 0.1.
    iso_working_point : IsoWps or None, optional
        Optional isolation working point. If provided, muons must meet
        the corresponding isolation requirement.
    """

    input_col: Column
    output_col: Column
    id_working_point: IdWps
    min_pt: float = 10
    max_abs_eta: float = 2.4
    max_mini_iso: float = 0.1
    iso_working_point: IsoWps | None = None

    __corrections: dict = field(factory=dict)

    def lint(self):
        from analyzer.core.linting import LintLevel, LintMessage

        messages = []
        if self.min_pt is not None and self.min_pt < 5:
            messages.append(
                LintMessage(
                    level=LintLevel.WARNING,
                    category="MuonKinematics",
                    message=f"Low muon pt cut: {self.min_pt} GeV.",
                    module_name=self.name(),
                )
            )
        return messages

    def run(self, columns, params):
        muon = columns[self.input_col]
        pass_pt = muon.pt > self.min_pt
        pass_eta = abs(muon.eta) < self.max_abs_eta
        pass_id_wp = muon[self.id_working_point]
        pass_mini_iso = muon.miniPFRelIso_all < self.max_mini_iso
        passed = pass_pt & pass_eta & pass_id_wp & pass_mini_iso
        if self.iso_working_point is not None:
            pass_iso = muon.pfIsoId >= cut_mapping[self.iso_working_point]
            passed = passed & pass_iso

        columns[self.output_col] = muon[passed]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

    def adlExport(self, metadata):
        statements = [
            ADLStatement("take", self.input_col.adl_name),
            ADLStatement("select", f"pt > {self.min_pt}"),
            ADLStatement("select", f"abs(eta) < {self.max_abs_eta}"),
        ]

        statements.append(ADLStatement("select", f"{self.id_working_point} == True"))
        statements.append(
            ADLStatement("select", f"miniPFRelIso_all < {self.max_mini_iso}")
        )

        if self.iso_working_point is not None:
            statements.append(
                ADLStatement(
                    "select", f"pfIsoId >= {cut_mapping[self.iso_working_point]}"
                )
            )

        return [
            ADLBlock(
                block_type="object",
                name=self.output_col.adl_name,
                statements=statements,
            )
        ]
