def muonMaker():
    return [
        {
            "module_name": "Hto4LExampleGoodMuonMaker",
            "input_col": "Muon",
            "output_col": "GoodMuon",
            "min_pt": 5.0,
            "max_abs_eta": 2.4,
            "max_iso": 0.40,
            "max_dxy": 0.5,
            "max_dz": 1.0,
            "max_sip3d": 4.0,
        }
    ]


def electronMaker():
    return [
        {
            "module_name": "Hto4LExampleGoodElectronMaker",
            "input_col": "Electron",
            "output_col": "GoodElectron",
            "min_pt": 7.0,
            "max_abs_eta": 2.5,
            "max_iso": 0.40,
            "max_dxy": 0.5,
            "max_dz": 1.0,
            "max_sip3d": 4.0,
        }
    ]


def zzScaleFactor():
    return [
        {
            "module_name": "Hto4LExampleApplyZZScaleFactor",
            "should_run": {"sample_type": "MC"},
        }
    ]


def higgsSetup(channel):
    setup = []
    if channel == "4mu":
        setup.extend(
            [
                {
                    "module_name": "NObjFilter",
                    "input_col": "GoodMuon",
                    "selection_name": "nmuons",
                    "min_count": 4,
                    "max_count": 4,
                }
            ]
        )
    elif channel == "4e":
        setup.extend(
            [
                {
                    "module_name": "NObjFilter",
                    "input_col": "GoodElectron",
                    "selection_name": "nelectrons",
                    "min_count": 4,
                    "max_count": 4,
                }
            ]
        )
    elif channel == "2e2mu":
        setup.extend(
            [
                {
                    "module_name": "NObjFilter",
                    "input_col": "GoodMuon",
                    "selection_name": "nmuons",
                    "min_count": 2,
                    "max_count": 2,
                },
                {
                    "module_name": "NObjFilter",
                    "input_col": "GoodElectron",
                    "selection_name": "nelectrons",
                    "min_count": 2,
                    "max_count": 2,
                },
                {
                    "module_name": "VecDRSelection",
                    "input_col": "GoodMuon",
                    "selection_name": "mu_dr",
                    "min_dr": 0.02,
                },
                {
                    "module_name": "VecDRSelection",
                    "input_col": "GoodElectron",
                    "selection_name": "el_dr",
                    "min_dr": 0.02,
                },
            ]
        )

    setup.extend(
        [
            {"module_name": "Hto4LExampleHiggsTo4lSelection", "channel": channel},
            {"module_name": "SelectOnColumns", "sel_name": "selection"},
        ]
    )

    return setup


def higgsRecoAndSelect(channel):
    return [
        {"module_name": "Hto4LExampleHiggsTo4lReco", "channel": channel},
        {"module_name": "SelectOnColumns", "sel_name": "higgs_valid_selection"},
    ]


def plotHistograms():
    return [
        {
            "module_name": "SimpleHistogram",
            "hist_name": "HiggsMass",
            "input_cols": ["Higgs_mass"],
            "axes": [
                {
                    "name": "Higgs_mass",
                    "start": 70,
                    "stop": 180,
                    "bins": 36,
                    "unit": "GeV",
                }
            ],
        },
        {
            "module_name": "SimpleHistogram",
            "hist_name": "Z1Mass",
            "input_cols": ["Z1_mass"],
            "axes": [
                {"name": "Z1_mass", "start": 40, "stop": 120, "bins": 40, "unit": "GeV"}
            ],
        },
        {
            "module_name": "SimpleHistogram",
            "hist_name": "Z2Mass",
            "input_cols": ["Z2_mass"],
            "axes": [
                {"name": "Z2_mass", "start": 12, "stop": 120, "bins": 40, "unit": "GeV"}
            ],
        },
    ]


ANALYSIS = {
    "analyzer": {
        "default_run_builder": {"strategy_name": "NoSystematics"},
        "FourMuon": muonMaker()
        + zzScaleFactor()
        + higgsSetup("4mu")
        + higgsRecoAndSelect("4mu")
        + plotHistograms(),
        "FourElectron": electronMaker()
        + zzScaleFactor()
        + higgsSetup("4e")
        + higgsRecoAndSelect("4e")
        + plotHistograms(),
        "TwoElectronTwoMuon": muonMaker()
        + electronMaker()
        + zzScaleFactor()
        + higgsSetup("2e2mu")
        + higgsRecoAndSelect("2e2mu")
        + plotHistograms(),
    },
    "location_priorities": [
        ".*FNAL.*",
        ".*US.*",
        ".*(DE|IT|CH|FR).*",
        ".*(T0|T1|T2).*",
        "eos",
    ],
    "event_collections": [
        {
            "dataset": "opendata_SMHiggsToZZTo4L",
            "pipelines": ["FourMuon", "FourElectron", "TwoElectronTwoMuon"],
        },
        {"dataset": "opendata_ZZTo4mu", "pipelines": ["FourMuon"]},
        {"dataset": "opendata_ZZTo4e", "pipelines": ["FourElectron"]},
        {"dataset": "opendata_ZZTo2e2mu", "pipelines": ["TwoElectronTwoMuon"]},
        {
            "dataset": "opendata_DoubleMuParked",
            "pipelines": ["FourMuon", "TwoElectronTwoMuon"],
        },
        {"dataset": "opendata_DoubleElectron", "pipelines": ["FourElectron"]},
    ],
    "extra_dataset_paths": ["examples/higgs_to_4l/datasets.yaml"],
    "extra_era_paths": ["examples/higgs_to_4l/era.yaml"],
    "extra_module_paths": ["examples/higgs_to_4l/higgs_to_4l.py"],
}
