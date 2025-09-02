import operator as op
from analyzer.core import MODULE_REPO, ModuleType


@MODULE_REPO.register(ModuleType.Selection)
def hlt_selection(events, params, selector, hlt_groups: list[list] = None):
    era_info = params.dataset.era
    tn = era_info.trigger_names

    groups = [ft.reduce(op.and_, [tn[name] for name in group]) for group in hlt_groups]
    final_selection = ft.reduce(lambda x, y: x | ((~x) & y), groups)
    selector.add(f"HLT Selection", final_selection)
