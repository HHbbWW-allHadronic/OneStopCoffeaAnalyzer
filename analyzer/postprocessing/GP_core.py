"""Sideband Gaussian-process estimate of a background yield in a blinded window.

Pure numpy/scipy (no OSCA objects) so it can be unit-tested on its own.

Model: bin contents y_i with variances v_i (weighted histogram, v_i = sum w^2) in the control-region
bins are fit with
    y(x) = linear(x) + f(x),   f ~ GP(0, sigma^2 exp(-(x-x')^2 / (2 ell^2))),   noise = v_i
(sigma, ell) maximise the log marginal likelihood. The posterior of y at the signal-region bins gives the
predicted window yield and its uncertainty.

Limits worth knowing:
  * the linear mean is fit by weighted least squares and is NOT marginalised, so its own uncertainty is
    not in the quoted error (the GP term partly covers it);
  * an RBF kernel gives a smooth interpolation: a localised bump inside the window that the sidebands do
    not show cannot be predicted. That is the sculpting failure mode, and the closure test exposes it;
  * signal leaking into the sidebands is NOT modelled here (MC background only for now).
"""
import numpy as np
from scipy.optimize import minimize


def _rbf(x1, x2, sigma, ell):
    d = x1[:, None] - x2[None, :]
    return sigma ** 2 * np.exp(-0.5 * (d / ell) ** 2)


def bin_masks(edges, sr, crs):
    """Bin masks from bin centres. sr = (lo, hi); crs = [(lo, hi), ...]. A bin belongs to a range if its
    centre is in [lo, hi). Raises if a range does not line up with bin edges (a bin would be split)."""
    edges = np.asarray(edges, float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for lo, hi in [sr] + list(crs):
        for x in (lo, hi):
            if np.min(np.abs(edges - x)) > 1e-6 * max(edges[-1] - edges[0], 1.0):
                raise ValueError(f"range edge {x:g} is not a bin edge; rebin so the SR/CR edges line up")
    in_range = lambda lo, hi: (centers >= lo) & (centers < hi)
    sr_mask = in_range(*sr)
    cr_mask = np.zeros_like(sr_mask)
    for lo, hi in crs:
        cr_mask |= in_range(lo, hi)
    if (sr_mask & cr_mask).any():
        raise ValueError("signal region and control regions overlap")
    return centers, sr_mask, cr_mask


def gp_predict(x_cr, y_cr, var_cr, x_sr, ell_bounds=(15.0, 300.0), n_starts=8, seed=0):
    """Fit the GP to the control bins; return dict with the posterior mean/cov at x_sr (in original units)."""
    x_cr, y_cr, var_cr, x_sr = (np.asarray(a, float) for a in (x_cr, y_cr, var_cr, x_sr))
    scale = float(np.max(np.abs(y_cr))) or 1.0
    y = y_cr / scale
    v = var_cr / scale ** 2
    pos = v[v > 0]
    if pos.size == 0:
        raise ValueError("all control-region variances are zero")
    v = np.where(v > 0, v, pos.min())          # empty bins: give them the smallest non-zero variance

    xm = x_cr.mean()
    A = np.vstack([np.ones_like(x_cr), x_cr - xm]).T
    W = 1.0 / v
    coef = np.linalg.solve(A.T @ (W[:, None] * A), A.T @ (W * y))
    r = y - A @ coef

    def nll(theta):
        sigma, ell = np.exp(theta)
        K = _rbf(x_cr, x_cr, sigma, ell) + np.diag(v)
        try:
            L = np.linalg.cholesky(K + 1e-10 * np.eye(len(K)))
        except np.linalg.LinAlgError:
            return 1e30
        a = np.linalg.solve(L.T, np.linalg.solve(L, r))
        return 0.5 * r @ a + np.log(np.diag(L)).sum()

    rng = np.random.default_rng(seed)
    bounds = [(np.log(1e-3), np.log(10.0)), (np.log(ell_bounds[0]), np.log(ell_bounds[1]))]
    best = None
    for _ in range(n_starts):
        t0 = [rng.uniform(*bounds[0]), rng.uniform(*bounds[1])]
        res = minimize(nll, t0, method="L-BFGS-B", bounds=bounds)
        if best is None or res.fun < best.fun:
            best = res
    sigma, ell = np.exp(best.x)

    K = _rbf(x_cr, x_cr, sigma, ell) + np.diag(v)
    Ks = _rbf(x_sr, x_cr, sigma, ell)
    Kss = _rbf(x_sr, x_sr, sigma, ell)
    Kinv_r = np.linalg.solve(K, r)
    A_sr = np.vstack([np.ones_like(x_sr), x_sr - xm]).T
    mean = A_sr @ coef + Ks @ Kinv_r
    cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
    return {"mean": mean * scale, "cov": cov * scale ** 2, "sigma": sigma, "ell": ell,
            "mean_cr_fit": None}


def predict_window(edges, values, variances, sr, crs, **kw):
    """ABCD-free sideband estimate for the window. Returns the predicted yield with its uncertainty, and
    (for MC closure) the observed yield in the window and the pull."""
    values, variances = np.asarray(values, float), np.asarray(variances, float)
    centers, sr_mask, cr_mask = bin_masks(edges, sr, crs)
    out = gp_predict(centers[cr_mask], values[cr_mask], variances[cr_mask], centers[sr_mask], **kw)
    ones = np.ones(sr_mask.sum())
    pred = float(out["mean"].sum())
    pred_err = float(np.sqrt(max(ones @ out["cov"] @ ones, 0.0)))
    obs = float(values[sr_mask].sum())
    obs_err = float(np.sqrt(variances[sr_mask].sum()))
    return {"pred": pred, "pred_err": pred_err, "obs": obs, "obs_err": obs_err,
            "pull": (pred - obs) / np.hypot(pred_err, obs_err),
            "bin_mean": out["mean"], "bin_cov": out["cov"], "centers": centers,
            "sr_mask": sr_mask, "cr_mask": cr_mask, "sigma": out["sigma"], "ell": out["ell"]}


if __name__ == "__main__":
    # toy validation: smooth falling spectrum (closure must hold, pull ~ N(0,1)) vs a bump inside the window
    rng = np.random.default_rng(3)
    edges = np.arange(50.0, 210.0, 10.0)
    SR, CRS = (100.0, 150.0), [(50.0, 100.0), (150.0, 200.0)]

    def toy(bump, n=30000):
        m = rng.exponential(60.0, 6 * n) + 40.0           # falling spectrum
        keep = rng.uniform(size=m.size) < 1.0 / (1.0 + np.exp(-(m - 70.0) / 10.0))   # turn-on
        m = m[keep][:n]
        if bump:
            b = rng.normal(125.0, 12.0, int(bump * n))
            m = np.concatenate([m, b])
        w = rng.lognormal(0.0, 0.7, m.size)
        h, _ = np.histogram(m, bins=edges, weights=w)
        v, _ = np.histogram(m, bins=edges, weights=w ** 2)
        return h, v

    for label, bump in [("smooth", 0.0), ("bump +10% of events", 0.10)]:
        pulls, rel = [], []
        for _ in range(300):
            h, v = toy(bump)
            r = predict_window(edges, h, v, SR, CRS)
            pulls.append(r["pull"]); rel.append((r["pred"] - r["obs"]) / r["obs"])
        pulls = np.array(pulls)
        print(f"{label:22s} pull mean={pulls.mean():+.2f} std={pulls.std():.2f}  "
              f"median (pred-obs)/obs={np.median(rel):+.1%}")