from .utils.axes import CommonAxes, makeAxis
from analyzer.core import MODULE_REPO, ModuleType


@MODULE_REPO.register(ModuleType.OtherResults)
def save_columns(events, params, column_names):
    just_cols = events.events[column_names]
    return {"columns" : just_cols}



