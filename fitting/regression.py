import contextlib
import math
import pickle as pkl
import sys
from collections import namedtuple
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

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


class LinearTransform:
    def __init__(self, slope, intercept=None):
        self.slope = torch.atleast_1d(slope)
        if intercept is None:
            self.intercept = torch.zeros_like(self.slope)
        else:
            self.intercept = torch.atleast_1d(intercept)

    def transformData(self, *data):
        if len(data) == 1:
            data = data[0]
            return (data - self.intercept) / self.slope
        else:
            ret = tuple(
                (d - self.intercept[i]) / self.slope[i] for i, d in enumerate(data)
            )
            return ret

    def iTransformData(self, *data):
        if len(data) == 1:
            data = data[0]
            return (data * self.slope) + self.intercept
        else:
            return tuple(
                (d * self.slope[i]) + self.intercept[i] for i, d in enumerate(data)
            )

    def transformVariances(self, v):
        return v / self.slope**2

    def iTransformVariances(self, v):
        return v * self.slope**2

    def __repr__(self):
        return f"LinearTransform({self.slope}, {self.intercept})"


DataValues = namedtuple("DataValues", "X Y V E")


class DataTransformation:
    def __init__(self, transform_x, transform_y):
        self.transform_x = transform_x
        self.transform_y = transform_y

    def transformX(self, edges, X):
        return (
            self.transform_x.transformData(*edges),
            self.transform_x.transformData(X),
        )

    def transformY(self, Y, V):
        return (self.transform_y.transformData(Y), self.transform_y.transformVariances(V))

    def transform(self, dv: DataValues) -> DataValues:
        E, X = self.transformX(dv.E, dv.X)
        Y, V = self.transformY(dv.Y, dv.V)
        return DataValues(X, Y, V, E)

    def iTransformX(self, edges, X):
        return (
            self.transform_x.iTransformData(*edges),
            self.transform_x.iTransformData(X),
        )

    def iTransformY(self, Y, V):
        return (self.transform_y.iTransformData(Y), self.transform_y.iTransformVariances(V))

    def iTransform(self, dv: DataValues) -> DataValues:
        E, X = self.iTransformX(dv.E, dv.X)
        Y, V = self.iTransformY(dv.Y, dv.V)
        return DataValues(X, Y, V, E)

    def __repr__(self):
        return f"DataTransformation({self.transform_x}, {self.transform_y})"


def pointsToGrid(points_x, points_y, edges, set_unfilled=None):
    filled = torch.histogramdd(
        points_x, bins=edges, weight=torch.full_like(points_y, True)
    )
    ret = torch.histogramdd(points_x, bins=edges, weight=points_y)
    return ret, filled.hist.bool()


def getNormalizationTransform(dv) -> DataTransformation:
    X, Y, V, E = dv

    max_x, min_x = torch.max(X, axis=0).values, torch.min(X, axis=0).values
    max_y, min_y = torch.max(Y), torch.min(Y)

    value_scale = max_y - min_y
    input_scale = max_x - min_x

    transform_x = LinearTransform(max_x - min_x, min_x)
    transform_y = LinearTransform(value_scale, min_y)

    return DataTransformation(transform_x, transform_y)


def makeRegressionData(histogram, mask_region, exclude_less=None):
    def make_mask(x1, x2, vals=None):
        if mask_region is None:
            a = torch.full_like(x1, False, dtype=torch.bool)
            b = torch.full_like(x2, False, dtype=torch.bool)
        else:
            a = (x1 > mask_region[0][0]) & (x1 < mask_region[0][1])
            b = (x2 > mask_region[1][0]) & (x2 < mask_region[1][1])
        return a, b

    edges_x1 = torch.from_numpy(histogram.axes[0].edges)
    edges_x2 = torch.from_numpy(histogram.axes[1].edges)

    centers_x1 = torch.diff(edges_x1) / 2 + edges_x1[:-1]
    centers_x2 = torch.diff(edges_x2) / 2 + edges_x2[:-1]

    bin_values = torch.from_numpy(histogram.values()).T
    bin_vars = torch.from_numpy(histogram.variances()).T
    centers_grid_x1, centers_grid_x2 = torch.meshgrid(
        centers_x1, centers_x2, indexing="xy"
    )
    if exclude_less:
        domain_mask = bin_values < exclude_less
    else:
        domain_mask = torch.full_like(bin_values, False, dtype=torch.bool)

    centers_grid = torch.stack((centers_grid_x1, centers_grid_x2), axis=2)
    m1, m2 = make_mask(centers_grid[:, :, 0], centers_grid[:, :, 1], bin_values)
    centers_mask = (m1 | domain_mask) & (m2 | domain_mask)
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    return DataValues(
        flat_centers[torch.flatten(~centers_mask)],
        flat_bin_values[torch.flatten(~centers_mask)],
        flat_bin_vars[torch.flatten(~centers_mask)],
        (edges_x1, edges_x2),
    )


def preprocessHistograms(background_hist, mask_region, exclude_less=None, rebin=1):
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
        value_scale,
    )


def createModel(train_data, kernel=None, model_maker=None, learn_noise=False, **kwargs):
    # v = torch.maximum(train_data.variances, torch.tensor(0.00001))
    v = train_data.V

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=v,
        learn_additional_noise=learn_noise,
    )
    if model_maker is None:
        model_maker = ExactAnyKernelModel

    if kernel:
        model = model_maker(
            train_data.X, train_data.Y, likelihood, kernel=kernel, **kwargs
        )
    else:
        model = model_maker(train_data.X, train_data.Y, likelihood, **kwargs)
    return model, likelihood


def optimizeHyperparams(
    model, likelihood, train_data, iterations=100, bar=True, lr=0.05
):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    context = Progress() if bar else contextlib.nullcontext()

    with context as progress:
        if bar:
            task1 = progress.add_task("[red]Optimizing...", total=iterations)
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_data.X)
            loss = -mll(output, train_data.Y)
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
            if bar:
                progress.update(
                    task1,
                    advance=1,
                    description=f"[red]Optimizing(Loss is {round(loss.item(),3)})...",
                )
                progress.refresh()
            else:
                if i % (iterations // 10) == 0:
                    print(f"Iter {i}: Loss = {loss.item()}")
                    pass
                    # print(f"Covar is {output.covariance_matrix}")

    return model, likelihood


def getPrediction(model, likelihood, test_data):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations():
        observed_pred = likelihood(model(test_data.X), noise=test_data.V)
    return observed_pred


def getChi2Blinded(inputs, pred_mean, test_mean, test_var, window):
    imask_x = (inputs[:, 0] > window[0][0]) & (inputs[:, 0] < window[0][1])
    imask_y = (inputs[:, 1] > window[1][0]) & (inputs[:, 1] < window[1][1])
    mask = imask_x & imask_y

    pred_mean = pred_mean[mask]
    test_mean = test_mean[mask]
    test_var = test_var[mask]
    num = torch.count_nonzero(mask)
    return torch.sum((pred_mean - test_mean) ** 2 / test_var) / num
