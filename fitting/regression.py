import contextlib
import math
import pickle as pkl
import sys
from collections import namedtuple
from typing import (Any, Callable, Dict, Hashable, Iterable, List, Optional,
                    Set, Tuple, Union)

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


def sendToGpu(dv):
    return DataValues(dv.X.cuda(), dv.Y.cuda(), dv.V.cuda(), dv.E)


def getFromGpu(dv):
    return DataValues(dv.X.cpu(), dv.Y.cpu(), dv.V.cpu(), dv.E)


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
        return (
            self.transform_y.transformData(Y),
            self.transform_y.transformVariances(V),
        )

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
        return (
            self.transform_y.iTransformData(Y),
            self.transform_y.iTransformVariances(V),
        )

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
    std_y = Y.std(dim=-1)

    value_scale = std_y
    value_scale = max_y - min_y
    input_scale = max_x - min_x

    transform_x = LinearTransform(max_x - min_x, min_x)
    transform_y = LinearTransform(value_scale, min_y)

    return DataTransformation(transform_x, transform_y)


def rectMasker(mask_region):
    def inner(x1, x2):
        a = (x1 > mask_region[0][0]) & (x1 < mask_region[0][1])
        b = (x2 > mask_region[1][0]) & (x2 < mask_region[1][1])
        return a, b

    return inner


def ellipseMasker(center, a, b):
    def inner(x1, x2):
        axes = torch.tensor([a, b])
        stacked = torch.stack((x1, x2), axis=-1)
        rel = ((stacked - center) ** 2) / axes**2
        mask = (torch.select(rel,-1,0) + torch.select(rel,-1,1)) <= 1.0
        return mask, mask

    return inner


def makeRegressionData(histogram, mask_function=None, exclude_less=None):
    if mask_function is None:
        mask_function = lambda x1, x2: (
            torch.full_like(x1, False, dtype=torch.bool),
            torch.full_like(x2, False, dtype=torch.bool),
        )

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
    m1, m2 = mask_function(centers_grid[:, :, 0], centers_grid[:, :, 1])
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


def createModel(train_data, kernel=None, model_maker=None, learn_noise=False, **kwargs):
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


def getBlindedMask(inputs, pred_mean, test_mean, test_var, window):
    imask_x = (inputs[:, 0] > window[0][0]) & (inputs[:, 0] < window[0][1])
    imask_y = (inputs[:, 1] > window[1][0]) & (inputs[:, 1] < window[1][1])
    mask = imask_x & imask_y
    return mask
    pred_mean = pred_mean[mask]
    test_mean = test_mean[mask]
    test_var = test_var[mask]
    num = torch.count_nonzero(mask)
    return torch.sum((pred_mean - test_mean) ** 2 / test_var) / num
