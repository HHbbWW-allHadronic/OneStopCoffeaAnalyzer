import mplhep

from .common import PlotConfiguration


def addCMSBits(ax, sectors, extra_text=None, text_color=None, plot_configuration=None):
    if plot_configuration is None:
        plot_configuration = PlotConfiguration()
    lumis = set(str(x.sector_params.dataset.lumi) for x in sectors)
    energies = set(str(x.sector_params.dataset.era.energy) for x in sectors)
    lumi_text = (
        plot_configuration.lumi_text
        or f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    )

    mplhep.cms.lumitext(text=lumi_text, ax=ax)
    text = plot_configuration.cms_text
    if extra_text is not None:
        text += "\n" + extra_text
    a,b,c=mplhep.cms.text(text=text, ax=ax, loc=plot_configuration.cms_text_pos)
    if text_color is not None:
        a.set(color=plot_configuration.cms_text_color)
        b.set(color=plot_configuration.cms_text_color)
        c.set(color=plot_configuration.cms_text_color)
        



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
