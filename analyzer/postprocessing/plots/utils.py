from pathlib import Path

import mplhep


def addAxesToHist(ax, size=0.1, pad=0.1, position="bottom", extend=False, share=True):
    new_ax = mplhep.append_axes(ax, size, pad, position, extend)
    current_axes = getattr(ax, f"{position}_axes", [])
    if share and position in ("top", "bottom"):
        ax.sharex(new_ax)
    if share and position in ("left", "right"):
        ax.sharey(new_ax)
    setattr(ax, f"{position}_axes", current_axes + [new_ax])
    return new_ax


def saveFig(fig, out, extension=".pdf", metadata=None, **kwargs):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    if extension:
        path = path.with_suffix(extension)
    fig.savefig(path, metadata=metadata, **kwargs)


def fixBadLabels(h):
    for x in h.axes:
        x.label = x.label.replace("textrm", "text")
