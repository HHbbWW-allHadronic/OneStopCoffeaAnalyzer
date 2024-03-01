import contextlib
import math
import pickle as pkl
import sys
from collections import namedtuple

import gpytorch
import hist
import torch
from analyzer.core import AnalysisResult
from analyzer.datasets import SampleManager
from analyzer.file_utils import DirectoryData
from analyzer.plotting.utils import subplots_context
from rich.progress import Progress, track
from torch.masked import as_masked_tensor, masked_tensor

from .models import ExactAnyKernelModel, ExactGPModel, ExactProjGPModel

DataValues = namedtuple("DataValues", "inputs outputs variances edges")


class SimpleTransformer:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def transformData(self, data):
        return (data - self.intercept) / slope

    def iTransformData(self, data):
        return (data * slope) + intercept

    def transformVariances(self, v):
        return v / self.slope**2

    def transformVariances(self, v):
        return v * self.slope**2


def pointsToGrid(points_x, points_y, edges, set_unfilled=None):
    filled = torch.histogramdd(
        points_x, bins=edges, weight=torch.full_like(points_y, True)
    )
    ret = torch.histogramdd(points_x, bins=edges, weight=points_y)
    return ret, filled.hist.bool()


def preprocessHistograms(background_hist, mask_region, exclude_less=None):
    def make_mask(x1, x2, vals=None):
        if mask_region is None:
            a = torch.full_like(x1, False, dtype=torch.bool)
            b = torch.full_like(x2, False, dtype=torch.bool)
        else:
            a = (x1 > mask_region[0][0]) & (x1 < mask_region[0][1])
            b = (x2 > mask_region[1][0]) & (x2 < mask_region[1][1])
        # if vals is not None and exclude_less:
        #    print(a.shape)
        #    print((vals > 0).shape)
        #    val_mask = vals < exclude_less
        #    a = a | val_mask
        #    b = b | val_mask
        return a, b

    edges_x1 = torch.from_numpy(background_hist.axes[0].edges)
    edges_x2 = torch.from_numpy(background_hist.axes[1].edges)
    centers_x1 = torch.diff(edges_x1) / 2 + edges_x1[:-1]
    centers_x2 = torch.diff(edges_x2) / 2 + edges_x2[:-1]

    bin_values = torch.from_numpy(background_hist.values()).T
    bin_vars = torch.from_numpy(background_hist.variances()).T
    centers_grid_x1, centers_grid_x2 = torch.meshgrid(
        centers_x1, centers_x2, indexing="xy"
    )

    if exclude_less:
        values_mask = bin_values < exclude_less
    else:
        values_mask = torch.full_like(bin_values, False, dtype=torch.bool)

    centers_grid = torch.stack((centers_grid_x1, centers_grid_x2), axis=2)
    m1, m2 = make_mask(centers_grid[:, :, 0], centers_grid[:, :, 1], bin_values)
    centers_mask = (m1 | values_mask) & (m2 | values_mask)
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    ma = torch.max(flat_centers.T, axis=1).values
    mi = torch.min(flat_centers.T, axis=1).values
    transformed_centers = (flat_centers - mi) / (ma - mi)
    transformed_edges_x1 = (edges_x1 - mi[0]) / (ma[0] - mi[0])
    transformed_edges_x2 = (edges_x2 - mi[1]) / (ma[1] - mi[1])

    ma = torch.max(flat_bin_values)
    mi = torch.min(flat_bin_values)
    value_scale = ma - mi
    transformed_values = (flat_bin_values) / value_scale
    transformed_vars = flat_bin_vars / value_scale**2
    # transformed_values = flat_bin_values
    # transformed_vars = flat_bin_vars

    transformed_centers_masked = transformed_centers[torch.flatten(~centers_mask)]
    transformed_bin_values_masked = transformed_values[torch.flatten(~centers_mask)]
    transformed_bin_vars_masked = transformed_vars[torch.flatten(~centers_mask)]

    train_x = transformed_centers_masked
    train_y = transformed_bin_values_masked
    train_vars = transformed_bin_vars_masked
    test_x = transformed_centers[torch.flatten(~values_mask)]
    test_y = transformed_values[torch.flatten(~values_mask)]
    test_vars = transformed_vars[torch.flatten(~values_mask)]

    return (
        (
            DataValues(
                flat_centers[torch.flatten(~centers_mask)],
                flat_bin_values[torch.flatten(~centers_mask)],
                flat_bin_vars[torch.flatten(~centers_mask)],
                (edges_x1, edges_x2),
            ),
            DataValues(
                flat_centers[torch.flatten(~values_mask)],
                flat_bin_values[torch.flatten(~values_mask)],
                flat_bin_vars[torch.flatten(~values_mask)],
                (edges_x1, edges_x2),
            ),
        ),
        (
            DataValues(
                train_x,
                train_y,
                train_vars,
                (transformed_edges_x1, transformed_edges_x2),
            ),
            DataValues(
                test_x, test_y, test_vars, (transformed_edges_x1, transformed_edges_x2)
            ),
        ),
        centers_mask,
        values_mask,
        value_scale,
    )


def createModel(train_data, kernel=None, model_maker=None):
    # v = torch.maximum(train_data.variances, torch.tensor(0.00001))
    v = train_data.variances

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=v,
        learn_additional_noise=False,
    )
    if model_maker is None:
        model_maker = ExactAnyKernelModel

    if kernel:
        model = model_maker(
            train_data.inputs, train_data.outputs, likelihood, kernel=kernel
        )
    else:
        model = model_maker(train_data.inputs, train_data.outputs, likelihood)
    return model, likelihood


def optimizeHyperparams(model, likelihood, train_data, iterations=100, bar=True):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    context = Progress() if bar else contextlib.nullcontext()

    with context as progress:
        if bar:
            task1 = progress.add_task("[red]Optimizing...", total=iterations)
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_data.inputs)
            loss = -mll(output, train_data.outputs)
            loss.backward()
            # if not (i + 1) % (iterations / 10) or i == 0:
            #    print(
            #        "Iter {:d} - Loss: {:0.3f}   lengthscale: {:0.4f},{:0.4f}  scale: {:0.3f}  mean:{:0.3f} noise:{:0.3f} ".format(
            #            i + 1,
            #            loss.item(),
            #            model.covar_module.base_kernel.lengthscale.squeeze()[0],
            #            model.covar_module.base_kernel.lengthscale.squeeze()[1],
            #            model.covar_module.outputscale.item(),
            #            model.mean_module.constant.item(),
            #            likelihood.noise.mean(),
            #        )
            #    )
            #    # print(model.proj_mat)
            optimizer.step()
            progress.update(
                task1,
                advance=1,
                description=f"[red]Optimizing(Loss is {round(loss.item(),3)})...",
            )
            progress.refresh()
    return model, likelihood


def getPrediction(model, likelihood, test_data):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations():
        observed_pred = likelihood(model(test_data.inputs), noise=test_data.variances)
    return observed_pred

