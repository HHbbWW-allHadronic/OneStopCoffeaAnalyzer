import itertools as it
import logging
import pickle as pkl
import re
import numpy as np
from pathlib import Path

import analyzer.core as ac
import hist
import matplotlib as mpl
import matplotlib.pyplot as plt
import analyzer.datasets as ad
from analyzer.utils import accumulate

from .high_level_plots import plot1D, plot2D, plotPulls, plotRatio
from .mplstyles import loadStyles
from .plottables import PlotObject, createPlotObject, PlotAxis
from .utils import getNormalized, splitHistDict


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
        profile_dir="profiles",
        coupling="312",
        year=2018,
        default_axis_opts=None,
        non_scaled_histos=False,
    ):
        loadStyles()
        self._createLogger()
        self.year = year

        self.default_backgrounds = default_backgrounds or []
        self.default_axis_opts = default_axis_opts

        profile_repo = ad.ProfileRepo()
        profile_repo.loadFromDirectory(profile_dir)
        self.sample_manager = ad.SampleManager()
        self.sample_manager.loadSamplesFromDirectory(dataset_dir, profile_repo)

        if isinstance(input_data, ac.AnalysisResult):
            results = [input_data]
        else:
            filenames = (
                [input_data] if isinstance(input_data, str) else list(input_data)
            )
            results = [pkl.load(open(f, "rb")) for f in filenames]
        self.target_lumi = (
            target_lumi
            or self.sample_manager[list(results[0].results.keys())[0]].getLumi()
        )

        self.histos = accumulate(
            [
                f.getMergedHistograms(self.sample_manager, self.target_lumi)
                for f in results
            ]
        )
        
        if non_scaled_histos:
            self.non_scaled_histos = accumulate(
                [
                    f.getNonScaledHistograms() 
                    for f in results
                ]
            )
            self.non_scaled_histos_labels = accumulate(
                [
                    f.getNonScaledHistogramsLabels()
                    for f in results
                ]
            )
        self.coupling = coupling

        used_samples = set(it.chain.from_iterable(x.results.keys() for x in results))
        lumis = [round(self.sample_manager[x].getLumi(), 4) for x in used_samples]
        if (
            not target_lumi
            and len(
                set(round(self.sample_manager[x].getLumi(), 4) for x in used_samples)
            )
            > 1
        ):
            raise ValueError(
                "The underlying samples have different luminosities, and you are not performing scaling"
            )
        if outdir:
            self.outdir = Path(outdir)
            self.outdir.mkdir(exist_ok=True, parents=True)
        else:
            self.outdir = None

    def __call__(self, *args, **kwargs):
        self.doPlot(*args, **kwargs)

    def plotPulls(self, target, hist_obs, hist_pred):
        ho = self.histos[target][hist_obs]
        hp = self.histos[target][hist_pred]
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
        ho = self.histos[target][hist_obs]
        hp = self.histos[target][hist_pred]
        hopo = PlotObject.fromHist(
            ho, self.sample_manager[hist_obs].title, self.sample_manager[hist_obs].style
        )
        hppo = PlotObject.fromHist(
            hp,
            self.sample_manager[hist_pred].title,
            self.sample_manager[hist_pred].style,
        )
        fig = plotRatio(hppo, hopo, self.coupling, self.target_lumi)
        if self.outdir:
            fig.savefig(self.outdir / f"pull_{hist_obs}_{hist_pred}.pdf")
            plt.close(fig)
        else:
            return fig

    def doPlot(
        self,
        hist_name,
        *args,
        axis_opts=None,
        **kwargs,
    ):
        all_axis_opts = {**(self.default_axis_opts or {})}
        all_axis_opts.update((axis_opts or {}))
        hist_dict = self.histos[hist_name]
        split = splitHistDict(hist_name, hist_dict, all_axis_opts)
        for name, hists in split.items():
            return self.__doPlot(name, hists, *args, **kwargs)

    def __doPlot(
        self,
        hist_name,
        hist_dict,
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
        ratio=False,
        energy='13 TeV',
        control_region=False,
    ):
        bkg_set = bkg_set if bkg_set is not None else self.default_backgrounds
        if not scale:
            scale = "linear"
        if add_label is None and add_name:
            add_label = add_name.title()
        self.logger.info(f"Now plotting {hist_name}")
        add_name = add_name + "_" if add_name else ""
        background_plobjs = {
            n: createPlotObject(n, h, self.sample_manager)
            for n, h in hist_dict.items()
            if n in bkg_set
        }
        signal_plobjs = {
            n: createPlotObject(n, h, self.sample_manager)
            for n, h in hist_dict.items()
            if n in sig_set
        }        
        self.sample_manager.weights_normalized = self.sample_manager.weights.copy()
        if normalize:
            for i,o in enumerate(signal_plobjs.values()):
                self.sample_manager.weights_normalized[i] *= 1/o.sum()
            signal_plobjs = {n: h.normalize() for n, h in signal_plobjs.items()}
            background_plobjs = {n: h.normalize() for n, h in background_plobjs.items()}
        
        r = next(iter(signal_plobjs.values()))
        if len(r.axes) == 1:
            fig = plot1D(
                signal_plobjs.values(),
                background_plobjs.values(),
                self.target_lumi,
                self.coupling,
                self.year,
                sig_style=sig_style,
                xlabel_override=xlabel_override,
                add_label=add_label,
                top_pad=top_pad,
                scale=scale,
                ratio=ratio,
                energy=energy,
                control_region=control_region,
                weights=self.sample_manager.weights_normalized,
            )
            fig.tight_layout()
            if self.outdir:
                fig.savefig(self.outdir / f"{add_name}{hist_name}.pdf")
                plt.close(fig)
                return None
            else:
                return fig

        elif len(r.axes) == 2:
            ret = []
            for obj in signal_plobjs.values():
                fig = plot2D(
                    obj,
                    coupling=self.coupling,
                    lumi=self.target_lumi,
                    era=self.year,
                    sig_style=sig_style,
                    add_label=add_label,
                    scale=scale,
                    energy=energy,
                    control_region=control_region,
                )
                fig.tight_layout()
                if self.outdir:
                    fig.savefig(self.outdir / f"{add_name}{hist_name}_{obj.title}.pdf")
                    plt.close(fig)
                else:
                    ret.append(fig)
            if ratio:
                keys = list(signal_plobjs.keys())
                ob1 = signal_plobjs[keys[0]]
                ob2 = signal_plobjs[keys[1]]

                zscorename = f'({keys[0]}-{keys[1]})/Std[{keys[0]}]'
                varone = ob1.variances()
                one=ob1.values()
                two=ob2.values()
                ratio_histv = np.divide((one-two), np.sqrt(varone), out=np.zeros_like(one), where=varone != 0)
                ob1.update_values(ratio_histv)
                fig = plot2D(
                    ob1,
                    coupling=self.coupling,
                    lumi=self.target_lumi,
                    era=self.year,
                    sig_style=sig_style,
                    add_label=add_label,
                    scale=scale,
                    zscore=ratio,
                    control_region=control_region,
                    energy=energy,
                    zscorename=zscorename,
                )
                if self.outdir:
                    fig.savefig(self.outdir / f"{add_name}{hist_name}_zscore.pdf")
                    plt.close(fig)
                else:
                    ret.append(fig)
                if not self.outdir:
                    return ret
