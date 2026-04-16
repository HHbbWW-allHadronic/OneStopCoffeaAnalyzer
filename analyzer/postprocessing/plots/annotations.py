import mplhep

from .common import PlotConfiguration


def _getSampleCategory(all_meta):
    sample_types = set()
    for meta in all_meta:
        st = meta.get("sample_type")
        if st is not None:
            val = st.value if hasattr(st, "value") else str(st)
            sample_types.add(val)
    has_mc = "MC" in sample_types
    has_data = "Data" in sample_types
    return has_mc, has_data


def isSimulationOnly(all_meta):
    has_mc, has_data = _getSampleCategory(all_meta)
    return has_mc and not has_data


def _buildCMSText(cms_text, all_meta):
    has_mc, has_data = _getSampleCategory(all_meta)
    sim_only = has_mc and not has_data
    label = cms_text or ""
    is_private = label.lower().startswith("private work")

    if is_private:
        if has_mc and has_data:
            data_label = "CMS Data/Simulation"
        elif sim_only:
            data_label = "CMS Simulation"
        else:
            data_label = "CMS Data"
        return "", f"Private Work ({data_label})"
    else:
        if sim_only:
            label = f"Simulation {label}" if label else "Simulation"
        return "CMS", label


def addCMSBits(
    ax,
    all_meta,
    extra_text=None,
    text_color=None,
    plot_configuration=None,
):
    if plot_configuration is None:
        plot_configuration = PlotConfiguration()
    info_text = plot_configuration.lumi_text
    if info_text is None:
        lumis = set(str(x["era"]["lumi"]) for x in all_meta)
        energies = set(str(x["era"]["energy"]) for x in all_meta)
        era = set(str(x["era"]["name"]) for x in all_meta)
        era_text = f"{'/'.join(era)}"
        lumi_text = (
            plot_configuration.lumi_text
            or f"{'/'.join(lumis)} fb$^{{-1}}$ ({'/'.join(energies)} TeV)"
        )
        info_text = era_text + ", " + lumi_text

    exp, text = _buildCMSText(plot_configuration.cms_text, all_meta)

    if extra_text is not None:
        text += "\n" + extra_text

    if exp:
        artists = mplhep.label.exp_text(
            text=text,
            exp=exp,
            lumi=info_text,
            ax=ax,
            loc=plot_configuration.cms_text_pos,
            color=text_color or plot_configuration.cms_text_color,
        )
    else:
        loc = plot_configuration.cms_text_pos
        color = text_color or plot_configuration.cms_text_color

        lumi_artist = None
        if info_text is not None:
            lumi_artist = mplhep.label.add_text(
                info_text,
                loc="over right",
                xpad=0,
                ypad=0,
                ax=ax,
            )

        loc_map = {0: "over left", 1: "upper left", 2: "upper left", 3: "over left"}
        text_loc = loc_map.get(loc, "over left")
        label_artist = mplhep.label.add_text(
            text,
            loc=text_loc,
            ax=ax,
            fontstyle="italic",
            color=color,
        )
        artists = (label_artist, None, lumi_artist, None)

    ax._cms_text_artists = artists
    return artists


def removeCMSAnnotations(ax):
    if hasattr(ax, "_cms_text_artists"):
        for artist in ax._cms_text_artists:
            if artist is not None:
                artist.remove()
        del ax._cms_text_artists


def labelAxis(ax, which, axes, label=None, label_complete=None):
    mapping = dict(x=0, y=1, z=2)
    idx = mapping[which]

    if idx != len(axes):
        this_unit = getattr(axes[idx], "unit", None)
        if not label:
            label = axes[idx].label
            if this_unit:
                label += f" [{this_unit}]"

        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))
    else:
        label = label or "Events"
        units = [getattr(x, "unit", None) for x in axes]
        units = [x for x in units if x]
        getattr(ax, f"set_{which}label")(label.replace("textrm", "text"))
