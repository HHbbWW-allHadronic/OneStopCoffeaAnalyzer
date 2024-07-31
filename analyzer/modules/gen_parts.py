import gc
import itertools as it

import awkward as ak
from analyzer.core import analyzerModule
from analyzer.matching import object_matching

from .axes import *
from analyzer.math_funcs import angleToNPiToPi


def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )


def createGoodChildren(gen_particles, children):
    # x = ak.singletons(gen_particles.children)
    good_child_mask = isGoodGenParticle(children)
    while not ak.all(ak.ravel(good_child_mask)):
        good_child_mask = isGoodGenParticle(children)
        maybe_good_children = children[~good_child_mask].children
        maybe_good_children = ak.flatten(maybe_good_children, axis=3)
        children = ak.concatenate(
            [children[good_child_mask], maybe_good_children], axis=2
        )
    return children

    
@analyzerModule("good_gen", categories="main", depends_on=["objects"])
def goodGenParticles(events,analyzer):
    test = createGoodChildren(events.GenPart, events.GenPart.children)

    events["GenPart", "good_children"] = test
    gg = events.GenPart[isGoodGenParticle(events.GenPart)]
    bs = gg.good_children[abs(gg.good_children.pdgId) == 5]
    stop = ak.all(abs(gg[:, 0].pdgId) == 1000006)
    bad = ak.all(abs(gg[:, 1].pdgId) == 1000024)
    x = gg[abs(gg.pdgId) == 1000024][:, 0]
    t = gg[abs(gg.pdgId) == 1000006][:, 0]
    x_children = gg.good_children[abs(gg.pdgId) == 1000024]
    s_children = gg.good_children[abs(gg.pdgId) == 1000006]
    xb = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 5]))
    xd = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 1]))
    xs = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 3]))
    sb = ak.flatten(ak.flatten(s_children[abs(s_children.pdgId) == 5]))

    events["SignalParticles"] = ak.zip(
        dict(
            stop=stop,
            chi=x,
            stop_b=sb,
            chi_b=xb,
            chi_d=xd,
            chi_s=xs,
        )
    )
    
    # events["SignalQuarks"] = ak.concatenate(
    #     [ak.singletons(val) for val in (good_gen[:, i] for i in range(2, 6))], axis=1
    # )
    return events, analyzer


#@analyzerModule("delta_r", ModuleType.MainProducer,require_tags=["signal"], after=["good_gen"])
@analyzerModule("delta_r", depends_on=["good_gen"], categories="main")
def deltaRMatch(events, analyzer):
    matched_jets, matched_quarks, dr, idx_j, idx_q, _ = object_matching(
        events.good_jets, events.SignalQuarks, 0.2, 0.5, True
    )
    # print(f"IndexQ: {idx_q}")
    # print(f"IndexJ: {idx_j}")
    # print(f"MQ: {matched_quarks}")
    # print(f"MJ: {matched_jets}")
    # _, _, _, ridx_q, ridx_j, _ = object_matching(
    #    events.SignalQuarks, events.good_jets, 0.3, 0.5, True
    # )
    events["matched_quarks"] = matched_quarks
    events["matched_jets"] = matched_jets
    events["matched_dr"] = dr
    events["matched_jet_idx"] = idx_j
    return events, analyzer

@analyzerModule("gen_hists", categories="main",depends_on=["good_gen"])
def genHistograms(events, analyzer):
    stop = events.SignalParticles.stop
    stop_b = events.SignalParticles.stop_b
    chi_b = events.SignalParticles.chi_b
    chi = events.SignalParticles.chi
    chi_d = events.SignalParticles.chi_d
    chi_s = events.SignalParticles.chi_s
    mask = ak.num(events.GenPart[isGoodGenParticle(events.GenPart)])<6

        
    analyzer.H("truth_stop_b_pt", pt_axis, stop_b.pt, name="Gen stop-b $p_{T}$")
    analyzer.H("truth_chi_b_pt", pt_axis, chi_b.pt, name="Gen chi-b $p_{T}$")
    analyzer.H("truth_chi_pt", pt_axis, chi.pt, name="Gen chi $p_{T}$")
    analyzer.H("truth_chi_s_pt", pt_axis, chi_s.pt, name="Gen chi-s $p_{T}$")
    analyzer.H("truth_chi_d_pt", pt_axis, chi_d.pt, name="Gen chi-d $p_{T}$")

    analyzer.H("truth_stop_phi", phi_axis, stop.phi, name="Gen stop $\\phi$", mask=mask)
    analyzer.H("truth_stop_b_phi", phi_axis, stop_b.phi, name="Gen stop-b $\\phi$", mask=mask)
    analyzer.H("truth_chi_b_phi", phi_axis, chi_b.phi, name="Gen chi-b $\\phi$", mask=mask)
    analyzer.H("truth_chi_phi", phi_axis, chi.phi, name="Gen chi $\phi$", mask=mask)
    analyzer.H("truth_chi_s_phi", phi_axis, chi_s.phi, name="Gen chi-s $\\phi$", mask=mask)
    analyzer.H("truth_chi_d_phi", phi_axis, chi_d.phi, name="Gen chi-d $\\phi$", mask=mask)

    analyzer.H("truth_stop_eta", eta_axis, stop.eta, name="Gen stop $\\eta$", mask=mask)
    analyzer.H("truth_stop_b_eta", eta_axis, stop_b.eta, name="Gen stop-b $\\eta$", mask=mask)
    analyzer.H("truth_chi_b_eta", eta_axis, chi_b.eta, name="Gen chi-b $\\eta$", mask=mask)
    analyzer.H("truth_chi_eta", eta_axis, chi.eta, name="Gen chi $\\eta$", mask=mask)
    analyzer.H("truth_chi_s_eta", eta_axis, chi_s.eta, name="Gen chi-s $\\eta$", mask=mask)
    analyzer.H("truth_chi_d_eta", eta_axis, chi_d.eta, name="Gen chi-d $\\eta$", mask=mask)

    d_eta = abs(chi.eta-chi_b.eta)
    d_phi = abs(chi.phi-chi_b.phi)
    d_r = chi.delta_r(chi_b)

    d_eta2 = abs(chi.eta-stop_b.eta)
    d_phi2 = abs(chi.phi-stop_b.phi)
    d_r2 = chi.delta_r(stop_b)

    analyzer.H(f"truth_chi_chi_b_d_eta", d_eta_axis, d_eta, name="Gen chi, chi-b $\Delta \eta$", mask=mask)
    analyzer.H(f"truth_chi_chi_b_d_phi", d_phi_axis, d_phi, name="Gen chi, chi-b $\Delta \phi$", mask=mask)
    analyzer.H(f"truth_chi_chi_b_d_r", dr_axis, d_r, name="Gen chi, chi-b $\Delta r$", mask=mask)

    analyzer.H("truth_chi_stop_b_d_eta", d_eta_axis, d_eta2, name="Gen chi, stop-b $\Delta \eta$", mask=mask)
    analyzer.H("truth_chi_stop_b_d_phi", d_phi_axis, d_phi2, name="Gen chi, stop-b $\Delta \phi$", mask=mask)
    analyzer.H("truth_chi_stop_b_d_r", dr_axis, d_r2, name="Gen chi, stop-b $\Delta r$", mask=mask)

    analyzer.H("truth_chi_pt_v_chi_b_pt",  
        [
            makeAxis(
                50, 0, 1000, "Gen $\chi$ $p_{T}$", unit="GeV",
            ),
            makeAxis(
                50, 0, 1000, "Gen $b_{\chi}$ $p_{T}$", unit="GeV",
            ),
        ],
        [chi.pt, chi_b.pt],
        name="truth_chi_pt_v_chi_b_pt",
        mask=mask,
    )

    analyzer.H("truth_chi_pt_v_stop_b_pt", 
        [
            makeAxis(
                50, 0, 1000, "Gen $\chi$ $p_{T}$", unit="GeV",
            ),
            makeAxis(
                50, 0, 1000, "Gen $b_{\\tilde{t}}$ $p_{T}$", unit="GeV",
            ),
        ],
        [chi.pt, stop_b.pt],
        name="truth_chi_pt_v_stop_b_pt",
        mask=mask,
    )

    analyzer.H("truth_chi_eta_v_chi_b_eta", 
        [
            makeAxis(
                20, 0, 5, "Gen $\chi$ $|\eta|$",
            ),
            makeAxis(
                20, 0, 5, "Gen $b_{\chi}$ $|\eta|$",
            ),
        ],
        [abs(chi.eta), abs(chi_b.eta)],
        name="truth_chi_eta_v_chi_b_eta",
        mask=mask,
    )

    analyzer.H("truth_chi_eta_v_stop_b_eta", 
        [
            makeAxis(
                20, 0, 5, "Gen $\chi$ $\eta$",
            ),
            makeAxis(
                20, 0, 5, "Gen $b_{\\tilde{t}}$ $|\eta|$",
            ),
        ],
        [abs(chi.eta), abs(stop_b.eta)],
        name="truth_chi_eta_v_stop_b_eta",
        mask=mask,
    )

    analyzer.H("truth_chi_phi_v_chi_b_phi", 
        [
            makeAxis(
                20, 0, 4, "Gen $\chi$ $\phi$",
            ),
            makeAxis(
                20, 0, 4, "Gen $b_{\chi}$ $\phi$",
            ),
        ],
        [chi.phi, chi_b.phi],
        name="truth_chi_phi_v_chi_b_phi",
        mask=mask,
    )

    analyzer.H("truth_chi_phi_v_stop_b_phi", 
        [
            makeAxis(
                25, 0, 4, "Gen $\chi$ $\phi$",
            ),
            makeAxis(
                25, 0, 4, "Gen $b_{\\tilde{t}}$ $\phi$",
            ),
        ],
        [chi.phi, stop_b.phi],
        name="truth_chi_phi_v_stop_b_phi",
        mask=mask,
    )

    # analyzer.H(
    #     f"medium_bb_eta",
    #     makeAxis(20, 0, 5, "$\\Delta \\eta$ between leading 2 medium b jets"),
    #     mb_eta,
    #     mask=mask,
    #     name=rf"$\Delta \eta$ BB$",
    #     description=rf"$\Delta \eta$ between leading 2 medium b jets",
    # )
    # analyzer.H(
    #     f"medium_bb_phi",
    #     makeAxis(25, 0, 4, "$\\Delta \\phi$ between leading medium b jets"),
    #     mb_phi,
    #     mask=mask,
    #     name=rf"$\Delta \phi$ BB$",
    #     description=rf"$\Delta \phi$ between leading 2 medium b jets",
    # )
    # analyzer.H(
    #     f"medium_bdr",
    #     makeAxis(20, 0, 5, "$\\Delta R$ between leading 2 medium b jets"),
    #     mb_dr,
    #     mask=mask,
    #     name=rf"Medium BJet $\Delta R$",
    #     description=rf"$\Delta R$ between leading 2 medium $p_T$ b jets",
    # )
    # inv = top2[:, 0 + top2[:, 1]
    # analyzer.H(
    #     f"medium_b_m",
    #     makeAxis(60, 0, 3000, f"$m$ of leading 2 medium b jets", unit="GeV"),
    #     inv.mass,
    #     mask=mask,
    #     name=rf"medbmass",
    # )
    # analyzer.H(
    #     f"medium_b_pt",
    #     makeAxis(20, 0, 1000, f"$p_T$ of leading 2 medium b jets", unit="GeV"),
    #     inv.pt,
    #     name=rf"medbpt",
    #     mask=mask,
    # )

    return events, analyzer


# @analyzerModule("delta_r", ModuleType.MainProducer,require_tags=["signal"], after=["good_gen"])
def deltaRMatch(events):
    # ret =  object_matching(events.SignalQuarks, events.good_jets, 0.3, None, False)
    matched_jets, matched_quarks, dr, idx_j, idx_q, _ = object_matching(
        events.good_jets, events.SignalQuarks, 0.2, 0.5, True
    )
    # print(f"IndexQ: {idx_q}")
    # print(f"IndexJ: {idx_j}")
    # print(f"MQ: {matched_quarks}")
    # print(f"MJ: {matched_jets}")
    # _, _, _, ridx_q, ridx_j, _ = object_matching(
    #    events.SignalQuarks, events.good_jets, 0.3, 0.5, True
    # )
    events["matched_quarks"] = matched_quarks
    events["matched_jets"] = matched_jets
    events["matched_dr"] = dr
    events["matched_jet_idx"] = idx_j
    return events, analyzer
