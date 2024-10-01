from pathlib import Path


def saveFig(fig, out, extension=".pdf", metadata=None):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    path = path.with_suffix(extension)
    fig.savefig(path, metadata=metadata)


def labelAxis(ax, which, axes, label=None, label_complete=None):
    mapping = dict(x=0, y=1, z=2)
    idx = mapping[which]

    if idx != len(axes):
        this_unit = getattr(axes[idx], "unit", None)
        if not label:
            label = axes[idx].name
            if this_unit:
                label += f" [{this_unit}]"
        getattr(ax, f"set_{which}label")(label)
    else:
        label = label or "Events"
        units = [getattr(x, "unit", None) for x in axes]
        units = [x for x in units if x]
        unit_format = "*".join(units)
        if unit_format:
            label += f" / {unit_format}"
        getattr(ax, f"set_{which}label")(label)


# def autoScale(ax, top_pad=0.3):
#     sc = ax.get_yscale()
#     ax.set_ymargin(0)
#     ax.autoscale_view()
#     lim = ax.get_ylim()
#     delta = np.diff(lim)
#     if sc == "log":
#         top_pad = 10 ** (1 + top_pad)
#         bottom = max(1, lim[0] - delta * 0.05)
#         bottom = lim[0]
#     else:
#         bottom = lim[0] - delta * 0.05
#     top = lim[1] + delta * top_pad
#     ax.set_ylim(bottom, top)


# def autoLim(ax, hist):
#     ax.set_xlim(hist.axes[0][0], hist.axes[0][-1])
