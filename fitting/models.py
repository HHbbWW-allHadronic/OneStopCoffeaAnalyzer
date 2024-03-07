import math
import sys

import gpytorch
import torch


class GaussianMean(gpytorch.means.Mean):
    def __init__(self, prior=None, init_mean=0.0, init_sigma=1.0, init_scale=1.0):
        super().__init__()
        self.register_parameter(
            name="mean",
            parameter=torch.nn.Parameter(torch.tensor(init_mean, dtype=torch.float64)),
        )
        self.register_parameter(
            name="sigma",
            parameter=torch.nn.Parameter(torch.tensor(init_sigma, dtype=torch.float64)),
        )
        self.register_parameter(
            name="scale",
            parameter=torch.nn.Parameter(torch.tensor(init_scale, dtype=torch.float64)),
        )
        if prior is not None:
            self.register_prior("contant_prior", prior, "mean")
            self.register_prior("constant_prior", prior, "sigma")
            self.register_prior("constant_prior", prior, "scale")

    def forward(self, input):
        inner = (input - self.mean) ** 2 @ (1 / self.sigma**2)
        e = torch.exp(-inner)
        ret = self.scale * e
        return ret


class RotParamMixin:
    def __init__(self, *args, rot_prior=None, rot_constraint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_parameter(
            name="raw_rot",
            parameter=torch.nn.Parameter(torch.tensor(0.0, requires_grad=True)),
        )

        if rot_constraint is None:
            rot_constraint = gpytorch.constraints.Interval(-3.14, 3.14)

        self.register_constraint("raw_rot", rot_constraint)

        if rot_prior is not None:
            self.register_prior(
                "rot_prior", rot_prior, lambda m: m.rot, lambda m, v: m._rot_setter(v)
            )

    @property
    def rot(self):
        return self.raw_rot_constraint.transform(self.raw_rot)

    @rot.setter
    def rot(self, rot):
        self._rot_setter(rot)

    def _rot_setter(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rot)
        self.initialize(raw_rot=self.raw_rot_constraint.inverse_transform(value))

    def getMatrix(self):
        c = torch.cos(self.rot)
        s = torch.sin(self.rot)
        orth_mat = torch.stack((torch.stack([c, -s]), torch.stack([s, c])))
        return orth_mat


class RotMixin(RotParamMixin):
    def forward(self, x1, x2, diag=False, **params):
        diff = torch.unsqueeze(x1, dim=1) - x2
        m = self.getMatrix()
        d = torch.diag(1 / torch.squeeze(self.lengthscale) ** 2)
        real_mat = m.t() @ d @ m
        c = torch.einsum("abi,ij,abj->ab", diff, real_mat, diff)
        covar = self.post_function(c)
        if diag:
            return covar.diagonal()
        else:
            return covar


class GeneralRQ(RotMixin, gpytorch.kernels.RQKernel):
    def post_function(self, dist_mat):
        alpha = self.alpha
        for _ in range(1, len(dist_mat.shape) - len(self.batch_shape)):
            alpha = alpha.unsqueeze(-1)
        return (1 + dist_mat.div(2 * alpha)).pow(-alpha)


class GeneralRBF(RotMixin, gpytorch.kernels.RBFKernel):
    def post_function(self, dist_mat):
        return gpytorch.kernels.rbf_kernel.postprocess_rbf(dist_mat)


class GeneralSpectralMixture(RotParamMixin, gpytorch.kernels.SpectralMixtureKernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        last_dim_is_batch: bool = False,
        **params
    ):
        n, num_dims = x1.shape[-2:]

        if not num_dims == self.ard_num_dims:
            raise RuntimeError(
                "The SpectralMixtureKernel expected the input to have {} dimensionality "
                "(based on the ard_num_dims argument). Got {}.".format(
                    self.ard_num_dims, num_dims
                )
            )

        # Expand x1 and x2 to account for the number of mixtures
        # Should make x1/x2 (... x k x n x d) for k mixtures
        x1_ = x1.unsqueeze(-3)
        x2_ = x2.unsqueeze(-3)

        # Compute distances - scaled by appropriate parameters
        x1_exp = x1_ * self.mixture_scales
        x2_exp = x2_ * self.mixture_scales
        x1_cos = x1_ * self.mixture_means
        x2_cos = x2_ * self.mixture_means

        # Create grids
        x1_exp_, x2_exp_ = self._create_input_grid(x1_exp, x2_exp, diag=diag, **params)
        x1_cos_, x2_cos_ = self._create_input_grid(x1_cos, x2_cos, diag=diag, **params)

        exp_diff = x1_exp_ - x2_exp_
        cos_diff = x1_cos_ - x2_cos_

        m = self.getMatrix()
        real_mat = self.getMatrix()
        exp_val = torch.einsum("cabi,ij,cabj->cab", exp_diff, real_mat, exp_diff)


        #print(self.mixture_means.shape)
        # Compute the exponential and cosine terms
        exp_term = exp_val.mul_(-2 * math.pi**2)
        #exp_term = exp_diff.pow_(2).mul_(-2 * math.pi**2)
        cos_term = cos_diff.mul_(2 * math.pi)
        exp_term = torch.unsqueeze(exp_term,3)
        cos_term = torch.unsqueeze(cos_term.sum(3),3)
        #print(exp_term.shape)
        #print(cos_term.shape)
        res = exp_term.exp_() * cos_term.cos_()

        # Sum over mixtures
        mixture_weights = self.mixture_weights.view(*self.mixture_weights.shape, 1, 1)
        if not diag:
            mixture_weights = mixture_weights.unsqueeze(-2)

        res = (res * mixture_weights).sum(-3 if diag else -4)

        # Product over dimensions
        if last_dim_is_batch:
            # Put feature-dimension in front of data1/data2 dimensions
            res = res.permute(*list(range(0, res.dim() - 3)), -1, -3, -2)
        else:
            res = res.prod(-1)

        return res


class ExactProjGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.register_parameter
        # self.proj_mat = torch.nn.Parameter(torch.tensor([[1,1],[1,0]], dtype=torch.float64))
        self.rot = torch.nn.Parameter(torch.tensor(0.78, dtype=torch.float64))

    def forward(self, x):
        rot_mat = torch.tensor(
            [
                [torch.cos(self.rot), -torch.sin(self.rot)],
                [torch.sin(self.rot), torch.cos(self.rot)],
            ]
        )
        x = x @ rot_mat  # n x d * d x k --> n x k
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactAnyKernelModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean or gpytorch.means.ConstantMean()
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactPeakedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

        self.covar_peak_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.peak = torch.nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64))

    def forward(self, x):
        subbed = x - self.peak
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.covar_peak_module(subbed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PeakedRBF(gpytorch.kernels.RBFKernel):
    is_stationary = False

    def __init__(self, peak_prior=None, peak_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_peak",
            parameter=torch.nn.Parameter(torch.tensor([0.1, 0.1], requires_grad=True)),
        )
        if peak_constraint is None:
            peak_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_peak", peak_constraint)
        if peak_prior is not None:
            self.register_prior(
                "peak_prior",
                peak_prior,
                lambda m: m.peak,
                lambda m, v: m._set_peak(v),
            )

    @property
    def peak(self):
        return self.raw_peak_constraint.transform(self.raw_peak)

    @peak.setter
    def peak(self, value):
        return self._set_peak(value)

    def _set_peak(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_peak)
        self.initialize(raw_peak=self.raw_peak_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        return super().forward(x1 - self.peak, x2 - self.peak, **params)
