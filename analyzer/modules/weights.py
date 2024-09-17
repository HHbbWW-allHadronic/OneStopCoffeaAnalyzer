import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

@MODULE_REPO.register(ModuleType.Weight)
def sign_gen_weight(events, params, weight_manager):
    weight_manager.add("pos_neg",  ak.where(events.genWeight > 0, 1.0, -1.0))

@MODULE_REPO.register(ModuleType.Weight)
def gen_weight(events, params, weight_manager):
    weight_manager.add("gen_weight",  events.genWeight)

@MODULE_REPO.register(ModuleType.Weight)
def xsec_weight(events, params, weight_manager):
    weight_manager.add("x_sec",  ak.ones_like(events) * params.x_sec)

