import matplotlib as mpl
import matplotlib.pyplot as plt


def addCmsInfo(ax, pos="in", additional_text=None, color="black") -> mpl.axis.Axis:
    text = r"$\bf{CMS}\ \it{Preliminary}$"
    if additional_text:
        text += additional_text
    if pos == "in":
        ax.text(
            0.02,
            0.98,
            text,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
    elif pos == "out":
        ax.text(
            0.02,
            1.0,
            text,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
    return ax


def addEra(ax, lumi, era, energy="13 TeV") -> mpl.axis.Axis:
    text = f"${lumi}\\ \\mathrm{{fb}}^{{-1}}$ {era} ({energy})"

    ax.text(
        1,
        1,
        text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=20,
    )
    return ax


def addText(ax, x, y, text, **kwargs):
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=20,
        **kwargs,
    )
    return ax
