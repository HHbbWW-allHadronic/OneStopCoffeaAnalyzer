import pickle as pkl
import uproot
import matplotlib.pyplot as plt
import hist
import math
import itertools as it
import re
import numpy as np
from pathlib import Path
import json
import sys
import argparse
import re
import gpytorch
from dataclasses import dataclass
import dataclasses
import torch
from typing import Optional, Tuple, List, Union, Dict
from analyzer.plotting.core_plots import (
    loadStyles,
    PlotObject,
    drawPull,
    drawAs1DHist,
    addAxesToHist,
    addTitles1D,
    addPrelim,
    addEra,
    drawAsScatter,
)
from analyzer.plotting.simple_plot import Plotter

loadStyles()


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        init_lengthscale=205,
        init_outscale=10,
    ):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module.base_kernel.lengthscale = init_lengthscale
        self.covar_module.outputscale = init_outscale

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def getLS(self):
        return self.covar_module.base_kernel.lengthscale.item()


def createOptimizer(model, ls_lr=25, os_lr=1e9, mean_lr=500):
    optimizer = torch.optim.Adam(
        [
            {"params": [model.covar_module.base_kernel.raw_lengthscale], "lr": ls_lr},
            {"params": [model.covar_module.raw_outputscale], "lr": os_lr},
            {"params": model.mean_module.parameters(), "lr": mean_lr},
        ]
    )
    return optimizer


def createLikelihood(noise):
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=noise, learn_additional_noise=True
    )
    return likelihood


def trainModel(likelihood, model, train_x, train_y, fix_ls=False):
    if fix_ls:
        optimizer = createOptimizer(model, ls_lr=0)
    else:
        optimizer = createOptimizer(model)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(50):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if not i % 10:
            print(
                "Iter %d - Loss: %.6f   lengthscale: %.3f  scale: %.3f  mean:%0.3f"
                % (
                    i + 1,
                    loss.item(),
                    model.getLS(),
                    model.covar_module.outputscale.item(),
                    model.mean_module.constant.item(),
                )
            )
        optimizer.step()
    return model


def createAndFit(histogram, fixed_ls=None, window=None):
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers.reshape(-1, 1),
        histogram.variances(),
    )
    if window:
        mask = np.ravel((points < window[0]) | (points > window[1]))
        vals, points, varia = vals[mask], points[mask], varia[mask]

    vals, points, varia = (
        torch.from_numpy(vals),
        torch.from_numpy(points),
        torch.from_numpy(varia),
    )
    likelihood = createLikelihood(varia)
    if fixed_ls:
        model = ExactGPModel(points, vals, likelihood, init_lengthscale=fixed_ls)
    else:
        model = ExactGPModel(points, vals, likelihood)
    model.train()
    likelihood.train()
    model = trainModel(likelihood, model, points, vals, fix_ls=fixed_ls is not None)
    model.eval()
    likelihood.eval()
    return model


def chisqr(exp, uncert, pred):
    return np.sum((exp - pred) ** 2 / uncert)


def getPrediction(model, points):
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(points)
        upper, lower = observed_pred.confidence_region()
        variance = observed_pred.variance
        mean, upper, lower, variance = (
            observed_pred.mean.numpy(),
            upper.numpy(),
            lower.numpy(),
            variance.numpy(),
        )
    return mean, upper, lower, variance


def plotGaussianProcess(ax, model, min_bound=1050):
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    mean, upper, lower, _ = getPrediction(model, torch.from_numpy(ls))
    ax.plot(ls.ravel(), mean, label="Mean prediction", color="tab:orange")
    ax.fill_between(
        ls.ravel(),
        lower,
        upper,
        alpha=0.3,
        color="tab:orange",
        label=r"Mean$\pm 2\sigma$",
    )
    return ax


def generatePulls(
    histogram,
    model,
    outdir=Path("diagnostics"),
    min_bound=1050,
    window=None,
    target=None,
    sig_hist=None,
    sig_name=None,
    sig_strength=None,
        add_text=True,
    name="pulls_bkg_only",
):
    bc = histogram.axes[0].centers
    mean_at_obs, upper_at_obs, lower_at_obs, variance_at_obs = getPrediction(
        model, torch.from_numpy(bc.reshape(-1, 1))
    )
    h = np.histogram(bc, weights=mean_at_obs, bins=histogram.axes[0].edges)
    pred = PlotObject((*h, variance_at_obs))
    obs = PlotObject(histogram)

    fig, ax = drawAsScatter(obs, yerr=True)
    plotGaussianProcess(ax, model, min_bound=min_bound)
    ax.tick_params(axis="x", labelbottom=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    if sig_hist:
        drawAs1DHist(ax, PlotObject(sig_hist, "Injected Signal"), yerr=True, fill=None)
    ax.set_yscale("linear")
    ab = ax.bottom_axes[0]
    drawPull(ab, pred, obs, hline_list=[-1, 0, 1])
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    mean_at_pred, upper_at_pred, lower_at_pred, variance_at_pred = getPrediction(model, torch.from_numpy(ls))
    ab.set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")
    if target:
        ax.axvline(target, color="black", linewidth=0.3, linestyle="-.")
        ab.axvline(target, color="black", linewidth=0.3, linestyle="-.")
    if window:
        ax.axvline(window[0], color="red", linewidth=0.3, linestyle="-.")
        ab.axvline(window[0], color="red", linewidth=0.3, linestyle="-.")
        ax.axvline(window[1], color="red", linewidth=0.3, linestyle="-.")
        ab.axvline(window[1], color="red", linewidth=0.3, linestyle="-.")
    addEra(ax, 137)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{312}}''$ ")
    addTitles1D(ax, histogram, top_pad=0.2)
    ax.legend()
    transform = ax.transAxes
    cs = chisqr(histogram.values(), histogram.variances(), mean_at_obs)
    cs = cs/(len(bc)-1)
    scale = model.getLS()
    if add_text:
        text = f"$\chi^2/DOF = {cs:0.2f}$\nLength Scale={round(scale)}"
        if sig_name:
            text += f"\n{sig_name}: r={sig_strength}"
        ax.text(
            0.95,
            0.7,
            text,
            horizontalalignment="right",
            transform=transform,
            fontsize=20,
        )
    outdir.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf")
    return cs

def getMatching(ax, val):
    return next(x for x in ax if val in x)


def ranged(outtype):
    def inner(string):
        if string is None:
            return [None]
        if not "," in string:
            return [outtype(string)]
        else:
            start, end, by = (int(x) for x in string.split(","))
            return [outtype(x) for x in range(start, end, by)]

    return inner


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate template shape output files for fitting"
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("--inject-name", nargs="*", type=str, default=None)
    parser.add_argument("--inject-strength", type=ranged(float), default=None)
    parser.add_argument("--inject-point", type=ranged(float), default=None)
    parser.add_argument("--inject-sigma", type=ranged(float), default=None)
    parser.add_argument("--force-scale", type=ranged(float), default=None)
    parser.add_argument("--window", type=float, nargs=2, default=None)
    parser.add_argument("--add-text", default=False, action='store_true')
    parser.add_argument("-r", "--input-control-region", default=None, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument("-u", "--update", action="store_true", default=False)
    parser.add_argument(
        "-p", "--plot-dir", default=None, type=str, help="If set, save diagnostic plots"
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of the histogram to use for the fit",
    )
    parser.add_argument(
        "-b",
        "--backgrounds",
        required=True,
        type=str,
        nargs="+",
        help="Backgrounds to add",
    )
    parser.add_argument("--only-cr", action="store_true")
    parser.add_argument(
        "-s",
        "--signals",
        required=True,
        type=str,
        nargs="+",
        help="Signals to add to output",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=1,
        help="Rebinning",
    )
    parser.add_argument("-m", "--min-bound", default=1050, type=ranged(float))
    args = parser.parse_args()
    return args


def addSignalToHist(hist, normalize_to, mean, sigma, name="fake_signal"):
    hist = hist.copy(deep=True)
    norm = hist[{"dataset": normalize_to}].sum().value
    gaus = np.random.normal(loc=mean, scale=sigma, size=10000)
    hist.fill(name, gaus, weight=norm / 10000)
    return hist


@dataclass
class GaussianProcessFitResult:
    fit_type: str
    lower_bound: float
    window: tuple[float, float] | None = None
    force_scale: float | None = None
    inject_signal_base: float | None = None
    inject_signal_rate: float | None = None
    inject_signal_params: Tuple[float, float] | None = None
    pull_figure_name: str | None = None

    length_scale: float | None = None
    reduced_chi2: float | None = None

    def getString(self):
        all_parts = []
        if self.inject_signal_base:
            all_parts.append(f"inj_{self.inject_signal_base}")
        else:
            all_parts.append(f"bkg_only")
        if self.lower_bound:
            all_parts.append(f"lb{self.lower_bound}")
        if self.inject_signal_rate:
            all_parts.append(f"r{self.inject_signal_rate}")
        if self.force_scale:
            all_parts.append(f"fs_{self.force_scale}")
        if self.inject_signal_params:
            all_parts.append(
                f"m{self.inject_signal_params[0]}_s{self.inject_signal_params[1]}"
            )
        if self.window:
            all_parts.append(f"w_{self.window[0]}_{self.window[1]}")
        return "__".join(all_parts).replace(".", "p")


def product(*args):
    return it.product(*(x if x is not None else [None] for x in args))


def main():
    args = parseArgs()
    data = pkl.load(open(args.input, "rb"))
    plotdir = Path(args.plot_dir)
    root_output = uproot.recreate(args.output)
    histos = data["histograms"]
    histogram_orig = histos[args.name]
    target = None
    sigh = None
    window = args.window

    respath = plotdir / "results.json"
    if respath.exists() and args.update:
        all_results = json.load(open(plotdir / "results.json", "r"))
    else:
        all_results = []

    for terms in product(
        args.inject_name,
        args.min_bound,
        args.inject_strength,
        args.force_scale,
        args.inject_point,
        args.inject_sigma,
    ):
        histogram = histogram_orig.copy(deep=True)
        signame, mb, rate, scale, inj_point, inj_sigma = terms
        gpresult = GaussianProcessFitResult(signame or "bkg_only", mb)
        gpresult.inject_signal_base = signame
        gpresult.inject_signal_rate = rate
        gpresult.forced_scale = scale
        gpresult.window = window
        bkg_hist = histogram[
            list(
                it.chain(
                    getMatching(histogram.axes[0], bkg) for bkg in args.backgrounds
                )
            ),
            hist.loc(mb) :: hist.rebin(args.rebin),
        ]

        bkg_hist = bkg_hist[sum, ...]
        if signame and rate:
            hname = signame
            if inj_point:
                m, s = inj_point, inj_sigma
                print(f"Using fake signal")
                hname += f"_fake_s{s}"
                gpresult.inject_signal_params = m, s
                histogram = addSignalToHist(histogram, signame, m, s, hname)
            target = int(signame.split("_")[2])
            signame = hname
            print(f"Injecting signal {signame} at rate {rate}")
            sigh = rate * histogram[hname, hist.loc(mb) :: hist.rebin(args.rebin)]
            bkg_hist += sigh
        if args.input_control_region:
            datacr = pkl.load(open(args.input_control_region, "rb"))
            hcr = datacr["histograms"][args.name]
            hcr = hcr[
                getMatching(hcr.axes[0], bkg),
                hist.loc(mb) :: hist.rebin(args.rebin),
            ]
            hcr = hcr[sum, ...]
            cr_model = createAndFit(hcr)
            ls = cr_model.getLS()
            print(f"Found length scale {ls:0.2f}")
            model = createAndFit(bkg_hist, fixed_ls=ls, window=window)
        elif scale:
            print(f"Using manually entered length scale {scale}")
            model = createAndFit(h, args.scale, window=window)
        else:
            model = createAndFit(bkg_hist, window=window)
            print("Optimizing length scale in signal region")
            ls = model.getLS()
            print(f"Found length scale {ls:0.2f}")

        gpresult.length_scale = model.getLS()
        pull_sr_figname = "pull_sr_" + gpresult.getString()
        print(signame)
        print(pull_sr_figname)

        gpresult.pull_figure_name = pull_sr_figname
        cs = generatePulls(
            bkg_hist,
            model,
            plotdir,
            mb,
            target=target,
            sig_hist=sigh,
            sig_name=signame,
            window=window,
            sig_strength=rate,
            name=pull_sr_figname,
            add_text=args.add_text,
        )
        gpresult.reduced_chi2 = cs
        all_results.append(gpresult)
        # root_output[f"SignalChannel/{bkg}/nominal"] = res
    json.dump(
        [
            dataclasses.asdict(x) if isinstance(x, GaussianProcessFitResult) else x
            for x in all_results
        ],
        open(plotdir / "results.json", "w"),
        indent=2,
    )


if __name__ == "__main__":
    main()
