from pathlib import Path

import mplhep


def addAxesToHist(ax, size=0.1, pad=0.1, position="bottom", extend=False):
    new_ax = mplhep.append_axes(ax, size, pad, position, extend)
    current_axes = getattr(ax, f"{position}_axes", [])
    setattr(ax, f"{position}_axes", current_axes + [new_ax])
    return new_ax


def saveFig(fig, out, extension=".pdf", metadata=None):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    path = path.with_suffix(extension)
    fig.savefig(path, metadata=metadata)


