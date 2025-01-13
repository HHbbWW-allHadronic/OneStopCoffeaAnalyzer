from rich.progress import Progress
from contextlib import contextmanager
from analyzer.configuration import CONFIG


__ACTIVE_PROGRESS_BAR = None


def progbar(iterable, title=None):
    global __ACTIVE_PROGRESS_BAR
    has_handle = False
    if __ACTIVE_PROGRESS_BAR is None:
        has_handle = True
        __ACTIVE_PROGRESS_BAR = Progress(transient=True, disable=not CONFIG.PRETTY_MODE)
        __ACTIVE_PROGRESS_BAR.start()

    if hasattr(iterable, "__len__"):
        length = len(iterable)
    else:
        length = None

    task = __ACTIVE_PROGRESS_BAR.add_task(title, total=length)

    for val in iterable:
        __ACTIVE_PROGRESS_BAR.advance(task, 1)
        yield val

    __ACTIVE_PROGRESS_BAR.remove_task(task)

    if has_handle:
        __ACTIVE_PROGRESS_BAR.stop()
        del __ACTIVE_PROGRESS_BAR


@contextmanager
def spinner(*args, **kwargs):
    p = Progress(transient=True, disable=not CONFIG.PRETTY_MODE)
    t = p.add_task(*args, **kwargs, total=None, start=None)
    with p:
        yield
