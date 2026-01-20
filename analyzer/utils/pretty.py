from rich.progress import track
from analyzer.configuration import CONFIG


def progbar(iterable, **kwargs):
    if CONFIG.general.pretty:
        return track(iterable, **kwargs)
    else:
        return iterable
