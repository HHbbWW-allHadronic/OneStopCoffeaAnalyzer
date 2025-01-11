import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.math_funcs import angleToNPiToPi
from .utils.axes import makeAxis, CommonAxes


@MODULE_REPO.register(ModuleType.Histogram)
def b_quark_kinematics(events, params, histogram_builder, working_point="M"):
    mapping = {"M": "med", "L": "loose", "T": "tight"}
    bjets = events[f"{mapping[working_point]}_bs"]

    histogram_builder.H(
        f"{working_point}_bjet_pt",
        CommonAxes.pt_axis,
        bjets.pt,
    )
    histogram_builder.H(
        f"{working_point}_nb",
        CommonAxes.b_axis,
        ak.num(bjets.pt),
    )
    for i in range(0, 2):
        mask = ak.num(bjets, axis=1) > i
        histogram_builder.H(
            f"{working_point}_b_{i}_pt",
            makeAxis(100, 0, 800, f"$p_T$ of rank {i} {working_point} b jet"),
            bjets[mask][:, i].pt,
            mask=mask,
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(bjets, axis=1) > 1
    top2 = bjets[mask]
    b_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    b_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    b_dr = top2[:, 0].delta_r(top2[:, 1])

    histogram_builder.H(
        f"{working_point}_bb_eta",
        makeAxis(20, 0, 5, f"$\Delta \eta$ between leading 2 {working_point} b jets"),
        b_eta,
        mask=mask,
        description=rf"$\Delta \eta$ between leading 2 {working_point} b jets",
    )
    histogram_builder.H(
        f"{working_point}_bb_phi",
        makeAxis(25, 0, 4, rf"$\Delta \phi$ between leading {working_point} b jets"),
        b_phi,
        mask=mask,
        description=rf"$\Delta \phi$ between leading 2 {working_point} b jets",
    )
    histogram_builder.H(
        f"{working_point}_bdr",
        makeAxis(20, 0, 5, rf"$\Delta R$ between leading 2 {working_point} b jets"),
        b_dr,
        mask=mask,
        description=rf"$\Delta R$ between leading 2 {working_point} $p_T$ b jets",
    )
    inv = top2[:, 0] + top2[:, 1]
    histogram_builder.H(
        f"{working_point}_b_m",
        makeAxis(60, 0, 3000, f"$m$ of leading 2 {working_point} b jets", unit="GeV"),
        inv.mass,
        mask=mask,
    )
    histogram_builder.H(
        f"{working_point}_b_pt",
        makeAxis(20, 0, 1000, f"$p_T$ of leading 2 {working_point} b jets", unit="GeV"),
        inv.pt,
        mask=mask,
    )
