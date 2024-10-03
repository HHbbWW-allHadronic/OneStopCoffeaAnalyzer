import mplhep


def addCMSBits(ax, sectors):
    lumis = set(str(x.sector_params.dataset.lumi) for x in sectors)
    energies = set(str(x.sector_params.dataset.era.energy) for x in sectors)
    lumi_text = f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
    mplhep.cms.lumitext(text=lumi_text, ax=ax)
    mplhep.cms.text(text="Preliminary", ax=ax, loc=0)

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
