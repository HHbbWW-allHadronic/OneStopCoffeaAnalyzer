from analyzer.core.analysis import Analysis, DatasetDescription
from analyzer.core.analyzer import Analyzer
from analyzer.core.run_builders import NoSystematics
from analyzer.core.analysis_modules import IsSampleType

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from higgs_to_4l import (
    Hto4LExampleGoodMuonMaker,
    Hto4LExampleGoodElectronMaker,
    Hto4LExampleApplyZZScaleFactor,
    Hto4LExampleHiggsTo4lSelection,
    Hto4LExampleHiggsTo4lReco,
)

from analyzer.modules.common.selection import NObjFilter
from analyzer.modules.common.selection import SelectOnColumns
from analyzer.modules.common.histogram_builder import SimpleHistogram
from analyzer.modules.singlestop.selections import VecDRSelection
from analyzer.core.columns import Column
from analyzer.modules.common.axis import RegularAxis
from analyzer.utils.querying import Pattern


def muonMaker():
    return [
        Hto4LExampleGoodMuonMaker(
            input_col=Column("Muon"),
            output_col=Column("GoodMuon"),
            min_pt=5.0,
            max_abs_eta=2.4,
            max_iso=0.40,
            max_dxy=0.5,
            max_dz=1.0,
            max_sip3d=4.0,
        )
    ]


def electronMaker():
    return [
        Hto4LExampleGoodElectronMaker(
            input_col=Column("Electron"),
            output_col=Column("GoodElectron"),
            min_pt=7.0,
            max_abs_eta=2.5,
            max_iso=0.40,
            max_dxy=0.5,
            max_dz=1.0,
            max_sip3d=4.0,
        )
    ]


def zzScaleFactor():
    return [
        Hto4LExampleApplyZZScaleFactor(
            should_run=IsSampleType(sample_type="MC"),
            weight_name="zz_scale_factor",
        )
    ]


def higgsSetup(channel):
    setup = []
    if channel == "4mu":
        setup.append(
            NObjFilter(
                input_col=Column("GoodMuon"),
                selection_name="nmuons",
                min_count=4,
                max_count=4,
            )
        )
    elif channel == "4e":
        setup.append(
            NObjFilter(
                input_col=Column("GoodElectron"),
                selection_name="nelectrons",
                min_count=4,
                max_count=4,
            )
        )
    elif channel == "2e2mu":
        setup.append(
            NObjFilter(
                input_col=Column("GoodMuon"),
                selection_name="nmuons",
                min_count=2,
                max_count=2,
            )
        )
        setup.append(
            NObjFilter(
                input_col=Column("GoodElectron"),
                selection_name="nelectrons",
                min_count=2,
                max_count=2,
            )
        )
        setup.append(
            VecDRSelection(
                input_col=Column("GoodMuon"), selection_name="mu_dr", min_dr=0.02
            )
        )
        setup.append(
            VecDRSelection(
                input_col=Column("GoodElectron"), selection_name="el_dr", min_dr=0.02
            )
        )

    setup.extend(
        [
            Hto4LExampleHiggsTo4lSelection(channel=channel),
            SelectOnColumns(sel_name="selection"),
        ]
    )

    return setup


def higgsRecoAndSelect(channel):
    return [
        Hto4LExampleHiggsTo4lReco(channel=channel),
        SelectOnColumns(sel_name="higgs_valid_selection"),
    ]


def plotHistograms():
    return [
        SimpleHistogram(
            hist_name="HiggsMass",
            input_cols=[Column("Higgs_mass")],
            axes=[
                RegularAxis(
                    name="Higgs_mass",
                    start=70,
                    stop=180,
                    bins=36,
                    unit="GeV",
                )
            ],
        ),
        SimpleHistogram(
            hist_name="Z1Mass",
            input_cols=[Column("Z1_mass")],
            axes=[RegularAxis(name="Z1_mass", start=40, stop=120, bins=40, unit="GeV")],
        ),
        SimpleHistogram(
            hist_name="Z2Mass",
            input_cols=[Column("Z2_mass")],
            axes=[RegularAxis(name="Z2_mass", start=12, stop=120, bins=40, unit="GeV")],
        ),
    ]


analyzer = Analyzer(default_run_builder=NoSystematics())

analyzer.addPipeline(
    "FourMuon",
    muonMaker()
    + zzScaleFactor()
    + higgsSetup("4mu")
    + higgsRecoAndSelect("4mu")
    + plotHistograms(),
)
analyzer.addPipeline(
    "FourElectron",
    electronMaker()
    + zzScaleFactor()
    + higgsSetup("4e")
    + higgsRecoAndSelect("4e")
    + plotHistograms(),
)
analyzer.addPipeline(
    "TwoElectronTwoMuon",
    muonMaker()
    + electronMaker()
    + zzScaleFactor()
    + higgsSetup("2e2mu")
    + higgsRecoAndSelect("2e2mu")
    + plotHistograms(),
)

ANALYSIS = Analysis(
    analyzer=analyzer,
    location_priorities=[
        ".*FNAL.*",
        ".*US.*",
        ".*(DE|IT|CH|FR).*",
        ".*(T0|T1|T2).*",
        "eos",
    ],
    event_collections=[
        DatasetDescription(
            dataset=Pattern("opendata_SMHiggsToZZTo4L"),
            pipelines=["FourMuon", "FourElectron", "TwoElectronTwoMuon"],
        ),
        DatasetDescription(dataset=Pattern("opendata_ZZTo4mu"), pipelines=["FourMuon"]),
        DatasetDescription(
            dataset=Pattern("opendata_ZZTo4e"), pipelines=["FourElectron"]
        ),
        DatasetDescription(
            dataset=Pattern("opendata_ZZTo2e2mu"), pipelines=["TwoElectronTwoMuon"]
        ),
        DatasetDescription(
            dataset=Pattern("opendata_DoubleMuParked"),
            pipelines=["FourMuon", "TwoElectronTwoMuon"],
        ),
        DatasetDescription(
            dataset=Pattern("opendata_DoubleElectron"), pipelines=["FourElectron"]
        ),
    ],
    extra_dataset_paths=["examples/higgs_to_4l/datasets.yaml"],
    extra_era_paths=["examples/higgs_to_4l/era.yaml"],
)
