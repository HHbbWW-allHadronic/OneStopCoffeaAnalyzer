import itertools as it
import logging
import pickle as pkl
import re
from pathlib import Path

import analyzer.core as ac
import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
from analyzer.datasets import SampleManager
from analyzer.utils import accumulate

from .high_level_plots import plot1D, plot2D, plotPulls, plotRatio
from .mplstyles import loadStyles
from .plottables import PlotObject, createPlotObjects
from .utils import getNormalized


class Plotter:
    def _createLogger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(logging.Formatter(f"[Plotter]: %(message)s"))
        self.logger.addHandler(stream_handler)
        self.logger.info("Creating plotter")


    def __init__(
        self,
        input_data,
        outdir,
        target_lumi=None,
        default_backgrounds=None,
        dataset_dir="datasets",
        coupling="312",
        default_axis_opts=None,
    ):
        loadStyles()
        self._createLogger()

        self.default_backgrounds = default_backgrounds or []
        self.default_axis_opts = default_axis_opts

        self.sample_manager = SampleManager()
        self.sample_manager.loadSamplesFromDirectory(dataset_dir)

        if isinstance(input_data, ac.AnalysisResult):
            results = [input_data]
        else:
            filenames = (
                [input_data] if isinstance(input_data, str) else list(input_data)
            )
            results = [pkl.load(open(f, "rb")) for f in filenames]

        self.target_lumi = (
            target_lumi
            or self.sample_manager.getSet(list(results[0].results.keys())[0]).getLumi()
        )

        self.histos = accumulate(
            [
                f.getMergedHistograms(self.sample_manager, self.target_lumi)
                for f in results
            ]
        )

        self.coupling = coupling

        used_samples = set(it.chain.from_iterable(x.results.keys() for x in results))
        lumis = [
            round(self.sample_manager.getSet(x).getLumi(), 4) for x in used_samples
        ]
        if (
            not target_lumi
            and len(
                set(
                    round(self.sample_manager.getSet(x).getLumi(), 4)
                    for x in used_samples
                )
            )
            > 1
        ):
            raise ValueError(
                "The underlying sampels have different luminosities, and you are not performing scaling"
            )

        if outdir:
            self.outdir = Path(outdir)
            self.outdir.mkdir(exist_ok=True, parents=True)

    def __call__(self, *args, **kwargs):
        self.doPlot(*args, **kwargs)

    def plotPulls(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject.fromHist(
            ho, self.sample_manager[hist_obs].getTitle(), self.sample_manager[hist_obs]
        )
        hppo = PlotObject.fromHist(
            hp,
            self.sample_manager[hist_pred].getTitle(),
            self.sample_manager[hist_pred],
        )
        fig = plotPulls(hppo, hopo, self.coupling, self.target_lumi)
        if self.outdir:
            fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
            plt.close(fig)
            return None
        else:
            return fig

    def plotRatio(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs, ...]
        hp = self.histos[target][hist_pred, ...]
        hopo = PlotObject.fromHist(
            ho, self.sample_manager[hist_obs].getTitle(), self.sample_manager[hist_obs]
        )
        hppo = PlotObject.fromHist(
            hp,
            self.sample_manager[hist_pred].getTitle(),
            self.sample_manager[hist_pred],
        )
        fig = plotPulls(hppo, hopo, self.coupling, self.target_lumi)
        if self.outdir:
            fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
            plt.close(fig)
        else:
            return fig

    def doPlot(
        self,
        hist_name,
        *args,
        add_name=None,
        axis_opts=None,
        **kwargs,
    ):
        all_axis_opts = {**(self.default_axis_opts or {})}
        all_axis_opts.update((axis_opts or {}))
        h = self.histos[hist_name]
        axes_names = {x.name for x in h.axes}
        for n in all_axis_opts.keys():
            if n not in axes_names:
                raise KeyError(f"Name {n} is not an axis in {h}")
        to_split = [x for x, y in all_axis_opts.items() if y is Plotter.Split]
        all_axis_opts = {
            x: y for x, y in all_axis_opts.items() if y is not Plotter.Split
        }
        h = h[all_axis_opts]
        if to_split:
            split_axes = [list(x) for x in h.axes if x.name in to_split]
            split_names = [x.name for x in h.axes if x.name in to_split]
            for combo in it.product(*split_axes):
                if add_name:
                    add_name = add_name + "_"
                else:
                    add_name = ""
                new_name = add_name + "cuts__" + "_".join(str(x) for x in combo)
                f = dict(zip(split_names, (hist.loc(x) for x in combo)))
                to_pass = h[f]
                return self.__doPlot(
                    hist_name, to_pass, *args, **kwargs, add_name=new_name
                )
        else:
            return self.__doPlot(hist_name, h, *args, **kwargs, add_name=add_name)

    def __doPlot(
        self,
        hist_name,
        hist,
        sig_set,
        bkg_set=None,
        scale=None,
        title="",
        add_name="",
        normalize=False,
        sig_style="hist",
        add_label=None,
        top_pad=0.4,
        xlabel_override=None,
    ):
        bkg_set = bkg_set if bkg_set is not None else self.default_backgrounds
        if not scale:
            scale = "linear"

        if add_label is None and add_name:
            add_label = add_name.title()
        self.logger.info(f"Now plotting {hist_name}")
        add_name = add_name + "_" if add_name else ""
        hc = hist[{"dataset": bkg_set + sig_set}]
        if normalize:
            hc = getNormalized(hc, "dataset")
        background_plobjs = createPlotObjects(
            hc, "dataset", self.sample_manager,
            #cat_filter=".*(?!signal).*"
            cat_filter=lambda x: not re.search("signal",x)
        )
        signal_plobjs = createPlotObjects(
            hc, "dataset", self.sample_manager, #cat_filter="signal"
            cat_filter=lambda x: re.search("signal", x)
        )
        if len(hist.axes) == 2:
            print(len(signal_plobjs))
            fig = plot1D(
                signal_plobjs,
                background_plobjs,
                self.target_lumi,
                self.coupling,
                sig_style=sig_style,
                xlabel_override=xlabel_override,
                add_label=add_label,
                top_pad=top_pad,
                scale=scale,
            )
            fig.tight_layout()
            if self.outdir:
                fig.savefig(self.outdir / f"{add_name}{hist_name}.pdf")
                plt.close(fig)
                return None
            else:
                return fig

        elif len(hist.axes) == 3:
            ret = []
            for x in hc.axes[0]:
                realh = hc[{"dataset": x}]
                po = PlotObject.fromHist(realh, x, self.sample_manager[x].style)
                fig = plot2D(
                    po,
                    self.coupling,
                    self.target_lumi,
                    sig_style=sig_style,
                    add_label=add_label,
                    scale=scale,
                )
                fig.tight_layout()
                if self.outdir:
                    fig.savefig(self.outdir / f"{add_name}{hist_name}_{x}.pdf")
                    plt.close(fig)
                else:
                    ret.append(fig)
            return ret
