import contextlib
import itertools as it
import math
import pickle as pkl
import shutil
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml

import analyzer.file_utils as futil
import analyzer.plotting as plotting
from analyzer.plotting.mplstyles import loadStyles
import gpytorch
import hist
import matplotlib.pyplot as plt
import torch
from analyzer.plotting.utils import subplots_context
from gpytorch.kernels import ScaleKernel as SK
from rich.progress import Progress, track

from . import models, regression
from .plot_tools import createSlices, simpleGrid

torch.set_default_dtype(torch.float64)


def makeDiagonistPlots(
    pred_mean, pred_var, raw_test, raw_train, raw_hist, dirdata, save_dir
):
    def savePlot(fig, name, data=None):
        data = data or {}
        o = save_dir / name
        dirdata.set(o, data)
        fig.savefig(o)

    all_pulls = (pred_mean - raw_test.outputs) / torch.sqrt(raw_test.variances)
    all_x2 = (pred_mean - raw_test.outputs) ** 2 / raw_test.variances
    x2 = torch.sum(all_x2)

    with subplots_context(layout="tight") as (fig, ax):
        simpleGrid(ax, raw_test.edges, raw_train.inputs, raw_train.outputs)
        ax.set_title("Masked Inputs (Training)")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "training_points.pdf")

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, pred_mean)
        ax.set_title("GPR Mean Prediction")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "gpr_mean.pdf")

    with subplots_context(layout="tight") as (fig, ax):
        simpleGrid(ax, raw_test.edges, raw_test.inputs, raw_test.outputs)
        ax.set_title("Observed Outputs")
        savePlot(fig, "observed_outputs.pdf")

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, raw_test.variances)
        ax.set_title("Observed Variances")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "observed_variances.pdf")

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, pred_var)
        ax.set_title("Pred Variances")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "predicted_variances.pdf")

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(
            ax,
            raw_test.edges,
            raw_test.inputs,
            (raw_test.outputs - pred_mean) / torch.sqrt(raw_test.variances),
        )
        f.set_clim(-5, 5)
        ax.set_title("Pulls")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
        savePlot(fig, "pulls.pdf")

    import uhi

    with subplots_context(layout="tight") as (fig, ax):
        p = plotting.PlotObject.fromHist(
            uhi.numpy_plottable.ensure_plottable_histogram(
                np.histogram(all_pulls[torch.abs(all_pulls) < np.inf], bins=20)
            )
        )
        plotting.drawAs1DHist(ax, p, yerr=False)
        ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
        ax.set_ylabel("Count")
        savePlot(fig, "pulls_hist.pdf")


def makeSlicePlots(
    pred_mean, pred_var, raw_test, raw_hist, window, dim, save_dir, dirdata
):
    pred_mean, _ = regression.pointsToGrid(raw_test.inputs, pred_mean, raw_test.edges)
    pred_var, _ = regression.pointsToGrid(raw_test.inputs, pred_var, raw_test.edges)
    obs_vals, _ = regression.pointsToGrid(
        raw_test.inputs, raw_test.outputs, raw_test.edges
    )
    obs_vars, filled = regression.pointsToGrid(
        raw_test.inputs, raw_test.variances, raw_test.edges
    )

    (save_dir / "slices" / f"along_{dim}").mkdir(parents=True, exist_ok=True)

    for val, f, ax in createSlices(
        pred_mean.hist,
        pred_var.hist,
        obs_vals.hist,
        obs_vars.hist,
        raw_test.edges,
        filled,
        observed_title="CRData",
        window_2d=window,
        dim=dim,
    ):
        plotting.addTitles1D(
            ax, plotting.PlotObject.fromHist(raw_hist[{raw_hist.axes[dim].name: sum}])
        )
        o = (
            save_dir
            / "slices"
            / f"along_{dim}"
            / (f"slice_{round(float(val),3)}".replace(".", "p") + ".pdf")
        )
        f.savefig(o)
        dirdata.set(o, dict(slice_dim=dim, val=float(val)))
        plt.close(f)


def doCompleteRegression(
    hist,
    window,
    base_dir,
    subpath,
    common_data=None,
    rebin=1,
    kernel=None,
    model_maker=None,
):
    common_data = common_data or {}

    torch.set_default_dtype(torch.float64)

    base_dir = Path(base_dir)
    save_dir = base_dir / subpath

    dirdat = futil.DirectoryData(base_dir)
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    (
        (raw_train, raw_test),
        (train, test),
        centers_mask,
        values_mask,
        value_scale,
    ) = regression.preprocessHistograms(hist, window, exclude_less=0.0001)

    with subplots_context() as (fig, ax):
        out = save_dir / "original.pdf"
        plotting.drawAs2DHist(ax, plotting.PlotObject.fromHist(hist))
        fig.savefig(out)
        dirdat.set(out, common_data)

    with subplots_context(1, 2) as (fig, ax):
        out = save_dir / "masked_area.pdf"
        simpleGrid(ax[0], test.edges, test.inputs, test.outputs)
        simpleGrid(ax[1], test.edges, train.inputs, train.outputs)
        fig.savefig(out)
        dirdat.set(out, common_data)

    # model,likelihood = regression.createModel(train, kernel=None)
    # model,likelihood = regression.createModel(train,
    # kernel=SK(
    #                                              gpytorch.kernels.RBFKernel(ard_num_dims=2) +
    #                                              gpytorch.kernels.LinearKernel(ard_num_dims=2,num_dimensions=2)
    #                                          ))
    # model,likelihood = regression.createModel(train, kernel=SK(gpytorch.kernels.MaternKernel(ard_num_dims=2)))
    # model,likelihood = regression.createModel(train, kernel=SK(gpytorch.kernels.PiecewisePolynomialKernel(ard_num_dims=2)))
    # model,likelihood = regression.createModel(train, kernel=gpytorch.kernels.SpectralDeltaKernel(num_dims=2, ard_num_dims=2))
    # model,likelihood = regression.createModel(train, model_maker= ExactProjGPModel )
    # model,likelihood = regression.createModel(train, kernel=SK(models.MatrixRBF()))
    # model,likelihood = regression.createModel(train, kernel=SK(models.PeakedRBF(ard_num_dims=2)))

    # model,likelihood = regression.createModel(train, kernel=gpytorch.kernels.SpectralMixtureKernel(ard_num_dims=2, num_mixtures=8))
    # model,likelihood = regression.createModel(train, kernel=SK(gpytorch.kernels.RQKernel(ard_num_dims=2)))
    model, likelihood = regression.createModel(
        train,
        kernel=kernel,
        # SK(models.GeneralRBF(ard_num_dims=2))
    )
    # model,likelihood = regression.createModel(train, kernel=SK(models.GeneralRQ(ard_num_dims=2)))
    model, likelihood = regression.optimizeHyperparams(
        model, likelihood, train, bar=True, iterations=100
    )

    pred = regression.getPrediction(model, likelihood, test)
    pred_mean = pred.mean * value_scale
    pred_var = pred.variance * value_scale**2
    chi2 = regression.getChi2Blinded(
        raw_test.inputs, pred_mean, raw_test.outputs, raw_test.variances, window
    )
    makeDiagonistPlots(
        pred_mean,
        pred_var,
        raw_test,
        raw_train,
        hist,
        dirdat,
        save_dir,
    )
    makeSlicePlots(
        pred_mean,
        pred_var,
        raw_test,
        hist,
        window,
        0,
        save_dir,
        dirdat,
    )
    makeSlicePlots(pred_mean, pred_var, raw_test, hist, window, 1, save_dir, dirdat)

    p = save_dir / "metadata.yaml"
    with open(p, "w") as f:
        f.write(
            yaml.dump(
                {
                    "window": {"x": list(window[0]), "y": list(window[1])},
                    "chi2_blinded": float(chi2),
                }
            )
        )


def scan(hist):
    x_iter = map(float, torch.arange(1100, 2000, 250))
    x_size_iter = map(float, torch.arange(200, 400, 100))
    y_iter = map(float, torch.arange(0.4, 0.7, 0.15))
    y_size_iter = map(float, torch.arange(0.1, 0.3, 0.1))
    for x, a, y, b in it.product(x_iter, x_size_iter, y_iter, y_size_iter):
        win = [(x, x + a), (y, y + b)]
        path = f"w__{round(win[0][0],3)}_{round(win[0][1],3)}__{round(win[1][0],3)}_{round(win[1][1],3)}".replace(
            ".", "p"
        )
        print(f"Now processing {win}")
        doCompleteRegression(
            hist,
            win,
            "scan",
            path,
            kernel=gpytorch.kernels.ScaleKernel(models.GeneralRBF(ard_num_dims=2)),
        )
        plt.close("all")


def main():
    from analyzer.datasets import SampleManager
    from analyzer.core import AnalysisResult

    res = AnalysisResult.fromFile("results/data_control.pkl")
    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    res.results["CR0b_Data2018"].histograms["h_njet"]
    bkg_name = "CR0b_Data2018"
    hists = res.getMergedHistograms(sample_manager)
    complete_hist = hists["ratio_m14_vs_m24"]
    narrowed = hist
    orig = complete_hist[
        ..., hist.loc(1150) : hist.loc(3000), hist.loc(0.4) : hist.loc(1)
    ]
    narrowed = orig[..., :: hist.rebin(2), :: hist.rebin(2)]
    qcd_hist = narrowed[bkg_name, ...]
    qcd_hist = narrowed[bkg_name, ...]
    scan(qcd_hist)


if __name__ == "__main__":
    loadStyles()
    main()
