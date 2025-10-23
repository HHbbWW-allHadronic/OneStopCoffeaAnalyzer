from pathlib import Path

import matplotlib as mpl
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

def scaleYAxis(ax):
    children = ax.get_children()
    text_children = [x for x in children if isinstance(x, mpl.text.Text)]
    bbs = [t.get_tightbbox() for t in text_children]
    min_b = min(x.y0 for x in bbs)
    max_b = max(x.y1 for x in bbs)
    old_ylim = ax.get_ylim()
    old_ylim_ax = ax.transData.transform(old_ylim)
    new_ax_max_y = old_ylim_ax[1] + (max_b - min_b)
    new_max_y = ax.transData.inverted().transform([0, new_ax_max_y])[1]

    ax.set_ylim((old_ylim[0], new_max_y))
    return ax
    


    


def saveFig(fig, out, extension=".pdf", metadata=None, **kwargs):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    if extension:
        path = path.with_suffix(extension)
    fig.savefig(path, metadata=metadata, **kwargs)


def fixBadLabels(h):
    for x in h.axes:
        x.label = x.label.replace("textrm", "text")
        x.label = x.label.replace(
            "m_{3,\\text{uncompressed}$", "m_{3,\\text{uncompressed}}$"
        )
