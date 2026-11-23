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
  * control-bin variances are smoothed by default (smooth_variance) to avoid the low bias of a fit whose noise
    follows the observed bin content; the quoted error then also ignores the uncertainty of that smoothing;
  * signal leaking into the sidebands is NOT modelled here (MC background only for now).
"""
import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.stats import f as _fdist, chi2 as _chi2dist


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


def smooth_variance(x, y, var):
    """Replace per-bin variances (sum w^2) by r * yhat(x): r = sum(var)/sum(y) is the pooled weight scale and
    yhat a smooth (unweighted quadratic) fit of the bins. Variances that follow the observed bin content bias a
    weighted fit LOW when bins have few effective events (a downward fluctuation also gets a small variance, so a
    heavier weight). On a toy with ~8 effective events per bin this took the bias from -23% to ~0."""
    x, y, var = (np.asarray(a, float) for a in (x, y, var))
    if y.sum() <= 0 or var.sum() <= 0 or len(x) < 4:
        return var
    r = var.sum() / y.sum()
    yhat = np.polyval(np.polyfit(x, y, 2), x)
    return r * np.clip(yhat, 0.05 * y.mean(), None)


def _exp_mean(x, y, v):
    """Weighted least-squares fit of m(x) = exp(a + b (x - xm)) to the control bins; returns a callable or None
    (None if the bins are not positive enough to start the fit)."""
    xm = x.mean()
    ok = y > 0
    if ok.sum() < 3:
        return None
    b0, a0 = np.polyfit(x[ok] - xm, np.log(y[ok]), 1)
    res = least_squares(lambda p: (y - np.exp(p[0] + p[1] * (x - xm))) / np.sqrt(v), [a0, b0])
    return lambda xx: np.exp(res.x[0] + res.x[1] * (np.asarray(xx, float) - xm))


def gp_predict(x_cr, y_cr, var_cr, x_sr, ell_bounds=(15.0, 300.0), n_starts=8, seed=0, smooth_var=True, mean="exp"):
    """Fit the GP to the control bins; return dict with the posterior mean/cov at x_sr (in original units).
    smooth_var=True uses smooth_variance() for the control-bin noise (recommended for weighted MC).
    mean="exp" (default) uses an exponential m(x) = exp(a + b x) as the prior mean, "linear" a straight line;
    the GP then models what is left over. A falling spectrum is convex, so a straight line between two sidebands
    overshoots inside the window."""
    x_cr, y_cr, var_cr, x_sr = (np.asarray(a, float) for a in (x_cr, y_cr, var_cr, x_sr))
    if smooth_var:
        var_cr = smooth_variance(x_cr, y_cr, var_cr)
    scale = float(np.max(np.abs(y_cr))) or 1.0
    y = y_cr / scale
    v = var_cr / scale ** 2
    pos = v[v > 0]
    if pos.size == 0:
        raise ValueError("all control-region variances are zero")
    v = np.where(v > 0, v, pos.min())          # empty bins: give them the smallest non-zero variance

    xm = x_cr.mean()
    W = 1.0 / v
    mfun = _exp_mean(x_cr, y, v) if mean == "exp" else None
    if mfun is not None:
        m_cr, m_sr = mfun(x_cr), mfun(x_sr)
    else:                                       # straight-line mean (also the fallback for "exp")
        A = np.vstack([np.ones_like(x_cr), x_cr - xm]).T
        coef = np.linalg.solve(A.T @ (W[:, None] * A), A.T @ (W * y))
        m_cr = A @ coef
        m_sr = np.vstack([np.ones_like(x_sr), x_sr - xm]).T @ coef
    r = y - m_cr

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
    post_mean = m_sr + Ks @ Kinv_r
    cov = Kss - Ks @ np.linalg.solve(K, Ks.T)
    return {"mean": post_mean * scale, "cov": cov * scale ** 2, "sigma": sigma, "ell": ell,
            "mean_cr_fit": None}


def predict_window(edges, values, variances, sr, crs, jackknife=True, **kw):
    """Sideband estimate for the window. Returns the predicted yield with its uncertainty, and (for MC closure)
    the observed yield in the window and the pull.

    pred_err combines the GP posterior error with a leave-one-bin-out jackknife over the control bins
    (jackknife=True): a few control bins with huge MC weights can pull the fit, and the jackknife spread shows
    how much the prediction depends on any single bin. pred_err_gp and pred_err_jk are returned separately."""
    values, variances = np.asarray(values, float), np.asarray(variances, float)
    centers, sr_mask, cr_mask = bin_masks(edges, sr, crs)
    x_cr, y_cr, v_cr, x_sr = centers[cr_mask], values[cr_mask], variances[cr_mask], centers[sr_mask]
    out = gp_predict(x_cr, y_cr, v_cr, x_sr, **kw)
    ones = np.ones(sr_mask.sum())
    pred = float(out["mean"].sum())
    err_gp = float(np.sqrt(max(ones @ out["cov"] @ ones, 0.0)))
    err_jk = 0.0
    if jackknife and len(x_cr) > 6:
        kw_j = dict(kw); kw_j["n_starts"] = min(kw.get("n_starts", 8), 3)
        preds = []
        for i in range(len(x_cr)):
            keep = np.arange(len(x_cr)) != i
            try:
                preds.append(gp_predict(x_cr[keep], y_cr[keep], v_cr[keep], x_sr, **kw_j)["mean"].sum())
            except Exception:
                pass
        if len(preds) > 2:
            preds = np.array(preds)
            err_jk = float(np.sqrt((len(preds) - 1) / len(preds) * np.sum((preds - preds.mean()) ** 2)))
    pred_err = float(np.hypot(err_gp, err_jk))
    obs = float(values[sr_mask].sum())
    obs_err = float(np.sqrt(variances[sr_mask].sum()))
    return {"pred": pred, "pred_err": pred_err, "pred_err_gp": err_gp, "pred_err_jk": err_jk,
            "obs": obs, "obs_err": obs_err,
            "pull": (pred - obs) / np.hypot(pred_err, obs_err),
            "bin_mean": out["mean"], "bin_cov": out["cov"], "centers": centers,
            "sr_mask": sr_mask, "cr_mask": cr_mask, "sigma": out["sigma"], "ell": out["ell"]}


def poly_ftest(edges, values, variances, sr, crs, max_degree=4, smooth_var=True, alpha=0.05):
    """Simple parametric fits of the control-region bins plus an F-test on the polynomial order.

    Fits y(x) = sum_k c_k x^k for degree 0..max_degree (degree 1 is the plain a*x + b) by weighted least squares
    (weights 1/v, v the smoothed variance if smooth_var), and an exponential exp(a + b x) for comparison.
    For nested polynomials d -> d+1:
        F = [(chi2_d - chi2_{d+1}) / 1] / [chi2_{d+1} / ndf_{d+1}],   p = P(F_{1, ndf_{d+1}} > F)
    The selected degree is the lowest d whose step to d+1 is NOT significant (p > alpha); if every step is
    significant the highest degree is taken. Every fit is extrapolated to the window: the predicted yield error
    comes from the parameter covariance, scaled by max(1, chi2/ndf) when the fit is poor.
    Caveat: the F-test assumes Gaussian bins with errors known up to a common factor; heavy MC weights break that."""
    values, variances = np.asarray(values, float), np.asarray(variances, float)
    centers, sr_mask, cr_mask = bin_masks(edges, sr, crs)
    x, y, v = centers[cr_mask], values[cr_mask], variances[cr_mask]
    xs_all = centers[sr_mask]
    if smooth_var:
        v = smooth_variance(x, y, v)
    pos = v[v > 0]
    v = np.where(v > 0, v, pos.min() if pos.size else 1.0)
    W = 1.0 / v
    n = len(x)
    xm, half = x.mean(), 0.5 * (x.max() - x.min())
    sc = lambda xx: (np.asarray(xx, float) - xm) / half          # scaled to about [-1, 1] for conditioning
    obs = float(values[sr_mask].sum())
    obs_err = float(np.sqrt(variances[sr_mask].sum()))
    rows = []

    def finish(kind, label, npar, chi2, pred, perr):
        ndf = n - npar
        rows.append({"kind": kind, "label": label, "npar": npar, "chi2": chi2, "ndf": ndf,
                     "chi2_ndf": chi2 / ndf if ndf > 0 else float("nan"),
                     "gof_p": float(_chi2dist.sf(chi2, ndf)) if ndf > 0 else float("nan"),
                     "pred": pred, "pred_err": perr, "pull": (pred - obs) / np.hypot(perr, obs_err),
                     "F": float("nan"), "F_p": float("nan"), "selected": False})

    coefs = {}
    for d in range(0, max_degree + 1):
        if d + 1 >= n:
            break
        A = np.vander(sc(x), d + 1, increasing=True)
        cov = np.linalg.inv(A.T @ (W[:, None] * A))
        c = cov @ (A.T @ (W * y))
        chi2 = float(np.sum(W * (y - A @ c) ** 2))
        ndf = n - (d + 1)
        scale = max(1.0, chi2 / ndf) if ndf > 0 else 1.0
        As = np.vander(sc(xs_all), d + 1, increasing=True)
        ones = np.ones(len(xs_all))
        perr = float(np.sqrt(max(ones @ As @ cov @ As.T @ ones, 0.0) * scale))
        finish("poly", f"poly deg {d}" + (" (ax+b)" if d == 1 else ""), d + 1, chi2, float((As @ c).sum()), perr)
        coefs[d] = c
    polys = [r_ for r_ in rows if r_["kind"] == "poly"]
    for i in range(1, len(polys)):                                   # F-test of step (d-1) -> d, stored on row d
        lo, hi = polys[i - 1], polys[i]
        if hi["ndf"] > 0 and hi["chi2"] > 0:
            F = (lo["chi2"] - hi["chi2"]) / (hi["chi2"] / hi["ndf"])
            hi["F"], hi["F_p"] = float(F), float(_fdist.sf(F, 1, hi["ndf"]))
    sel = len(polys) - 1
    for i in range(len(polys) - 1):                                  # lowest degree whose next step is not significant
        if not (polys[i + 1]["F_p"] <= alpha):
            sel = i
            break
    polys[sel]["selected"] = True
    # exponential, for comparison (not nested in the polynomials, so no F-test)
    try:
        ok = y > 0
        b0, a0 = np.polyfit(sc(x)[ok], np.log(y[ok]), 1, w=y[ok] / np.sqrt(v[ok]))
        res = least_squares(lambda p: (y - np.exp(p[0] + p[1] * sc(x))) * np.sqrt(W), [a0, b0])
        cov = np.linalg.inv(res.jac.T @ res.jac)
        chi2 = float(np.sum(res.fun ** 2)); ndf = n - 2
        scale = max(1.0, chi2 / ndf)
        m_s = np.exp(res.x[0] + res.x[1] * sc(xs_all))
        J = np.vstack([m_s, m_s * sc(xs_all)]).T.sum(axis=0)
        perr = float(np.sqrt(max(J @ cov @ J, 0.0) * scale))
        finish("exp", "exponential", 2, chi2, float(m_s.sum()), perr)
        exp_curve = lambda xx: np.exp(res.x[0] + res.x[1] * sc(xx))
    except Exception:
        exp_curve = None
    curves = {d: (lambda xx, c=c: np.polyval(c[::-1], sc(xx))) for d, c in coefs.items()}
    return {"rows": rows, "selected_degree": sel, "curves": curves, "exp_curve": exp_curve, "alpha": alpha,
            "n_cr": n, "obs": obs, "obs_err": obs_err}


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