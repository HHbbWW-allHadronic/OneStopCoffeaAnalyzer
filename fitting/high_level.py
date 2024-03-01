import contextlib
import math
import pickle as pkl
import shutil
import sys
from collections import namedtuple
from pathlib import Path

import analyzer.file_utils as futil
import analyzer.plotting as plotting
import gpytorch
import hist
import matplotlib.pyplot as plt
import numpy as np
import torch
from analyzer.plotting.utils import subplots_context
from gpytorch.kernels import ScaleKernel as SK
from rich.progress import Progress, track

from . import models, regression
from .plot_tools import createSlices, simpleGrid


def makeDiagonistPlots(
    pred_mean, pred_var, raw_test, raw_train, raw_hist, dirdata, save_dir, common_data
):
    def savePlot(fig, name, data):
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
        savePlot(fig, "training_points.pdf", common_data)

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, pred_mean)
        ax.set_title("GPR Mean Prediction")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "gpr_mean.pdf", common_data)

    with subplots_context(layout="tight") as (fig, ax):
        simpleGrid(ax, raw_test.edges, raw_test.inputs, raw_test.outputs)
        ax.set_title("Observed Outputs")
        savePlot(fig, "observed_outputs.pdf", common_data)

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, raw_test.variances)
        ax.set_title("Observed Variances")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "observed_variances.pdf", common_data)

    with subplots_context(layout="tight") as (fig, ax):
        f = simpleGrid(ax, raw_test.edges, raw_test.inputs, pred_var)
        ax.set_title("Pred Variances")
        plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
        savePlot(fig, "predicted_variances.pdf", common_data)

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
        savePlot(fig, "pulls.pdf", common_data)

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
        savePlot(fig, "pulls_hist.pdf", common_data)


def makeSlicePlots(
    pred_mean, pred_var, raw_test, raw_hist, window, dim, save_dir, dirdata, common_data
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
        dirdata.set(o, dict(**common_data, slice_dim=dim, val=float(val)))
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
        train, kernel=SK(models.GeneralRBF(ard_num_dims=2))
    )
    # model,likelihood = regression.createModel(train, kernel=SK(models.GeneralRQ(ard_num_dims=2)))
    model, likelihood = regression.optimizeHyperparams(
        model, likelihood, train, bar=True, iterations=100
    )

    pred = regression.getPrediction(model, likelihood, test)
    pred_mean = pred.mean * value_scale
    pred_var = pred.variance * value_scale**2
    makeDiagonistPlots(
        pred_mean, pred_var, raw_test, raw_train, hist, dirdat, save_dir, common_data
    )
    makeSlicePlots(
        pred_mean, pred_var, raw_test, hist, window, 0, save_dir, dirdat, common_data
    )
    makeSlicePlots(
        pred_mean, pred_var, raw_test, hist, window, 1, save_dir, dirdat, common_data
    )
