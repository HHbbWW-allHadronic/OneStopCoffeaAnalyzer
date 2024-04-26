import gc
import itertools as it

import awkward as ak
from analyzer.core import analyzerModule
from analyzer.matching import object_matching


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


@analyzerModule("good_gen", categories="main")
def goodGenParticles(events, analyzer):
    good_gen = events.GenPart[isGoodGenParticle(events.GenPart)]
    events["SignalParticles"] = ak.zip(
        dict(
            stop=good_gen[:, 0],
            chi=good_gen[:, 1],
            stop_b=good_gen[:, 2],
            chi_b=good_gen[:, 3],
            chi_d=good_gen[:, 4],
            chi_s=good_gen[:, 5],
        )
    )
    events["SignalQuarks"] = ak.concatenate(
        [ak.singletons(val) for val in (good_gen[:, i] for i in range(2, 6))], axis=1
    )
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
