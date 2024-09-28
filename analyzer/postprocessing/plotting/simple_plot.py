import itertools as it
import logging
import pickle as pkl
import numpy as np
from pathlib import Path

import analyzer.core as ac
import matplotlib.pyplot as plt
import analyzer.datasets as ad
from analyzer.utils import accumulate

from .high_level_plots import plot1D, plot2D, plotPulls, plotRatio
from .mplstyles import loadStyles
from .plottables import PlotObject, createPlotObject
from .utils import splitHistDict


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


        #self.cut_list_for_plot and self.cut_table_dict are actually what get used in plotting
        #self.cut_list_for_plot is a string to be put on the plot itself and contains shortened names for the cuts.
        #Whereas self.cut_table_dict is a dictionary with keys=dataset names (Ex: Data2018) and values of each cut in plain words.
        #self.cut_table_dict is plotted below the plots as a table.
        
        self.cut_list_dict = {}
        for i in results:
            for j in i.results.keys():
                #list(dict.fromkeys(...)) gets rid of duplicates
                self.cut_list_dict[j] =list(dict.fromkeys(i.results[j].cut_list))
       
        cut_map = {"hlt": "", "ht1200": "HT ≥ 1200", "highptjet": "Jet-PT ≥ 300",
         "jets": "4 ≤ N-Jets ≤ 6", "0Lep": "0e, 0μ", "0looseb": "0b",
         "2bjet": "Med-b Jets ≥ 2", "1tightbjet": "Tight-b Jets ≥ 1",
         "b_dr": "b-jet ΔR > 1", "bbpt": "b-jet 1+2 > 200"}

        self.cut_table_dict = {}
        for dataset in self.cut_list_dict:
            cut_map['hlt'] = ' | '.join(self.sample_manager[dataset].profile.hlt)
            self.cut_table_dict[dataset] = [cut_map[i] for i in self.cut_list_dict[dataset]]
        
        self.cut_list_for_plot = [] 
        for dataset in self.cut_list_dict:
           cut_list_for_plot_temp = [f'{dataset}\n'] 
           cut_list_for_plot_temp += self.cut_list_dict[dataset]
           if self.cut_list_for_plot[1:] != cut_list_for_plot_temp[1:]:
               self.cut_list_for_plot += cut_list_for_plot_temp
           else:   
               self.cut_list_for_plot[0] = self.cut_list_for_plot[0][:-1] + "/" + cut_list_for_plot_temp[0]
        temp = self.cut_list_for_plot[0]   
        for i in self.cut_list_for_plot[1:]:
           if '\n' in i:
               temp += i
           else:
               temp += '\t' + i + '\n'
        self.cut_list_for_plot = temp.expandtabs(2)

        self.target_lumi= ( 
            target_lumi
            or self.sample_manager[list(results[0].results.keys())[0]].lumi
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
        lumis = [round(self.sample_manager[x].lumi, 4) for x in used_samples]
        if (
            not target_lumi
            and len(
                set(round(self.sample_manager[x].lumi, 4) for x in used_samples)
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
        cut_table_in_plot=False,
        cut_list_in_plot=False,
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

        r = next(iter(it.chain(signal_plobjs.values(), background_plobjs.values())))
        if not cut_table_in_plot:
            cut_table = None
        if not cut_list_in_plot:
            cut_list = None
        if len(r.axes) == 1:
            if cut_table_in_plot:
                cut_table = self.cut_table_dict
            if cut_list_in_plot:
                cut_list = self.cut_list_for_plot  
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
                cut_table = cut_table,
                cut_list = cut_list,
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
            
            for key,obj in signal_plobjs.items():
                if cut_table_in_plot:
                    cut_table = {key:self.cut_table_dict[key]}
                if cut_list_in_plot:
                    cut_list = self.cut_list_for_plot 
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
                    cut_table=cut_table,
                    cut_list=cut_list,
                )
                fig.tight_layout()
                if self.outdir:
                    fig.savefig(self.outdir / f"{add_name}{hist_name}_{obj.title}.pdf")
                    plt.close(fig)
                else:
                    ret.append(fig)
            if ratio:
                if cut_table_in_plot:
                    cut_table = self.cut_table_dict
                if cut_list_in_plot:
                    cut_list = self.cut_list_for_plot 
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
                    cut_table=cut_table,
                    cut_list=cut_list,
                )
                if self.outdir:
                    fig.savefig(self.outdir / f"{add_name}{hist_name}_zscore.pdf")
                    plt.close(fig)
                else:
                    ret.append(fig)
                if not self.outdir:
                    return ret
