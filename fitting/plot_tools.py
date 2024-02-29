import numpy as np

import analyzer.plotting as plotting
import matplotlib.pyplot as plt
import torch

from .regression import getPrediction, pointsToGrid


def plotGaussianProcess(ax, pobj, mask=None):
    mean = pobj.values
    dev = np.sqrt(pobj.variances)
    points = pobj.axes[0].centers
    if mask is not None:
        mean=mean[mask]
        dev=dev[mask]
        points=points[mask]
    ax.plot(points, mean, label="Mean prediction", color="tab:orange")
    ax.fill_between(
        points,
        mean + dev,
        mean - dev,
        alpha=0.3,
        color="tab:orange",
        label=r"Mean$\pm \sigma$",
    )
    return ax


def generatePulls(ax, observed, model, observed_title="", mask=None):
    edges, data, variances = observed
    mean, model_variance = model

    model_obj = plotting.PlotObject.fromNumpy((mean, edges), model_variance, mask=mask)
    obs_obj = plotting.PlotObject.fromNumpy(
        (data, edges), variances, title=observed_title, mask=mask
    )

    plotting.drawAsScatter(ax, obs_obj, yerr=True)
    plotGaussianProcess(ax, model_obj, mask=mask)

    ax.tick_params(axis="x", labelbottom=False)
    plotting.addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    # if sig_hist:
    #   drawAs1DHist(ax, PlotObject(sig_hist, "Injected Signal"), yerr=True, fill=None)
    ax.set_yscale("linear")
    ab = ax.bottom_axes[0]
    plotting.drawPull(ab, model_obj, obs_obj, hline_list=[-1, 0, 1])

    # ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    # mean_at_pred, upper_at_pred, lower_at_pred, variance_at_pred = getPrediction(
    #    model, torch.from_numpy(ls)
    # )
    # ab.set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")

    return ax


def createSlices(
    pred_mean,
    pred_variance,
    test_mean,
    test_variance,
    bin_edges,
        valid,
    dim=1,
    window_2d=None,
    observed_title="",
):
    num_slices = pred_mean.shape[dim]
    centers = bin_edges[dim][:-1] + torch.diff(bin_edges[dim]) / 2
    for i in range(num_slices):
        val = centers[i]
        if window_2d:
            v = window_2d[dim]
            if val > v[0] and val < v[1]:
                window = window_2d[dim - 1]
            else:
                window = None
        else:
            window = None


        fill_mask = valid.select(dim,i)

        slice_pred_mean = pred_mean.select(dim,i)
        slice_pred_var = pred_variance.select(dim,i)


        slice_obs_mean = test_mean.select(dim,i)
        slice_obs_var = test_variance.select(dim,i)
        fig,ax = plt.subplots()
        generatePulls(
            ax,
            (bin_edges[1 - dim], slice_obs_mean, slice_obs_var),
            (slice_pred_mean, slice_pred_var),
            observed_title=observed_title,
            mask=fill_mask,
        )

        if window:
            ax.axvline(window[0], color="red", linewidth=0.3, linestyle="-.")
            ax.axvline(window[1], color="red", linewidth=0.3, linestyle="-.")

            ax.bottom_axes[0].axvline(
                window[0], color="red", linewidth=0.3, linestyle="-."
            )
            ax.bottom_axes[0].axvline(
                window[1], color="red", linewidth=0.3, linestyle="-."
            )
        plotting.addEra(ax, "59.83")
        plotting.addPrelim(ax)
        plotting.addText(
            ax,
            0.98,
            0.5,
            f"Val={round(float(val),2)}",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.bottom_axes[0].set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")
        ax.legend()
        yield val, fig, ax
