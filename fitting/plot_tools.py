import numpy as np
from .regression import getPrediction
import torch
import matplotlib.pyplot as plt
import analyzer.plotting as plotting


def plotGaussianProcess(ax, pobj):
    mean = pobj.values
    dev = np.sqrt(pobj.variances)
    points = pobj.axes[0].centers
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


def generatePulls(ax, observed, model, observed_title=""):
    edges, data, variances = observed
    mean, model_variance = model

    model_obj = plotting.PlotObject.fromNumpy((mean, edges), model_variance)
    obs_obj = plotting.PlotObject.fromNumpy(
        (data, edges), variances, title=observed_title
    )

    plotting.drawAsScatter(ax, obs_obj, yerr=True)
    plotGaussianProcess(ax, model_obj)

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


def createSlices(model, observed, dim=1, window_2d=None, observed_title=""):
    model_mean, model_variance = model
    slices = torch.unique(observed.inputs[:, dim])
    for val in slices:
        if window_2d:
            v = window_2d[dim]
            if val > v[0] and val < v[1]:
                window = window_2d[dim - 1]
            else:
                window = None
        else:
            window = None
        mask = torch.isclose(observed.inputs[:, dim], val)
        pred_mean = model_mean[mask]
        pred_var = model_variance[mask]
        model_plot_input = (pred_mean, pred_var)
        fig, ax = plt.subplots()
        generatePulls(
            ax,
            (observed.edges[1 - dim], observed.outputs[mask], observed.variances[mask]),
            (pred_mean, pred_var),
            observed_title=observed_title,
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
