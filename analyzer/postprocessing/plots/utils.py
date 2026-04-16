from pathlib import Path
from matplotlib.axes import Axes

from collections import ChainMap
import copy
import json
import matplotlib as mpl
import mplhep
from .common import PlotConfiguration
from .annotations import addCMSBits, removeCMSAnnotations


INCLUDE_SIDECAR = True


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
    text_children = [
        x for x in children if isinstance(x, mpl.text.Text | mpl.legend.Legend)
    ]
    # breakpoint()
    bbs = [t.get_tightbbox() for t in text_children]
    min_b = min(x.y0 for x in bbs)
    max_b = max(x.y1 for x in bbs)
    old_ylim = ax.get_ylim()
    old_ylim_ax = ax.transData.transform(old_ylim)
    new_ax_max_y = old_ylim_ax[1] + (max_b - min_b)
    new_max_y = ax.transData.inverted().transform([0, new_ax_max_y])[1]

    ax.set_ylim((old_ylim[0], new_max_y))
    return ax


def makeDict(x):
    if isinstance(x, (dict, ChainMap)):
        return {k: makeDict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [makeDict(y) for y in x]
    return x


def saveFig(fig, out, extension=".pdf", metadata=None, **kwargs):
    path = Path(out)
    path.parent.mkdir(exist_ok=True, parents=True)
    if extension:
        path = path.with_suffix(extension)
    fig.savefig(path, **kwargs)
    if INCLUDE_SIDECAR:
        with open(Path(out).with_suffix(".json"), "w") as f:
            json.dump(makeDict(metadata), f)


def saveFigVariants(
    fig,
    ax,
    out,
    all_meta,
    plot_configuration=None,
    metadata=None,
    extra_text=None,
    text_color=None,
    **save_kwargs,
):

    pc = plot_configuration or PlotConfiguration()

    cms_texts = pc.cms_text if isinstance(pc.cms_text, list) else [pc.cms_text or ""]
    suffix_text = len(cms_texts) > 1

    raw_types = (
        pc.image_type if isinstance(pc.image_type, list) else [pc.image_type or ".pdf"]
    )
    extensions = [ext if ext.startswith(".") else f".{ext}" for ext in raw_types]
    suffix_ext = len(extensions) > 1

    base_path = Path(out)
    base_path.parent.mkdir(exist_ok=True, parents=True)

    for variant in cms_texts:
        variant_pc = copy.copy(pc)
        variant_pc.cms_text = variant

        removeCMSAnnotations(ax)
        addCMSBits(
            ax,
            all_meta,
            extra_text=extra_text,
            text_color=text_color,
            plot_configuration=variant_pc,
        )

        text_suffix = f"_{variant.lower().replace(' ', '_')}" if suffix_text else ""
        for ext in extensions:
            variant_path = base_path.with_stem(
                f"{base_path.stem}{text_suffix}"
            ).with_suffix(ext)
            fig.savefig(variant_path, **save_kwargs)

    if INCLUDE_SIDECAR:
        with open(base_path.with_suffix(".json"), "w") as f:
            json.dump(makeDict(metadata), f)


def addLegend(ax: Axes, cfg: PlotConfiguration, **legend_kwargs):
    legend_loc = cfg.legend_loc

    legend = ax.legend(
        loc=legend_loc,
        ncol=cfg.legend_num_cols,
        prop={"family": cfg.legend_font} if cfg.legend_font else None,
        **legend_kwargs,
    )
    frame = legend.get_frame()

    if cfg.legend_fill_color is not None:
        frame.set_facecolor(cfg.legend_fill_color)

    if cfg.legend_fill_alpha is not None:
        frame.set_alpha(cfg.legend_fill_alpha)

    return legend
