"""
jax_optimize.py
---------------
JAX-accelerated Levenberg-Marquardt (LM) optimiser, replacing the levmar/
library used by profit_minimize() in profit.c.

The implementation faithfully reproduces the Levenberg-Marquardt algorithm
used in levmar (Lourakis 2004), adapted for JAX:
  * Uses ``jax.jacfwd`` for Jacobian computation (or user-supplied Jacobian).
  * Core linear solve via ``jax.numpy.linalg.solve`` (Cholesky or LU).
  * Parameter bounds handled through the same bounded/unbounded transform
    as profit_boundtounbound / profit_unboundtobound.
  * ``jax.jit``-compiled inner loop; outer iteration in Python for
    flexible stopping criteria (mirrors the C ``lm_dif`` interface).

Public API
----------
levmar_fit(residual_fn, params0, args, ...)
    Minimise ``sum(residual_fn(params, *args)**2)`` using LM.

levmar_step(J, r, mu, n)
    Compute a single LM parameter update step (pure JAX, jit-able).

bounded_to_unbounded(params, lo, hi, ftypes)
    Map bounded parameters to the real line (profit_boundtounbound).

unbounded_to_bounded(dparams, lo, hi, ftypes)
    Inverse map (profit_unboundtobound).
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable, Any, Sequence
import math

# ---------------------------------------------------------------------------
# Parameter transformation (bounded ↔ unbounded)
# Mirrors profit_boundtounbound / profit_unboundtobound in profit.c
# ---------------------------------------------------------------------------

# Fit type codes (matching parfitenum in profit.h)
PARFIT_FIXED    = 0
PARFIT_UNBOUND  = 1
PARFIT_LINBOUND = 2
PARFIT_LOGBOUND = 3

_CLIP = 50.0   # logit clamp (prevents overflow)


def bounded_to_unbounded(
    params: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    ftypes: Sequence[int],
) -> jnp.ndarray:
    """
    Transform *params* from bounded space to the full real line.

    For each parameter *p*:
      PARFIT_FIXED    → skipped (not included in output)
      PARFIT_UNBOUND  → p / hi[i]
      PARFIT_LINBOUND → logit((p - lo) / (hi - lo))
      PARFIT_LOGBOUND → logit((log(p) - log(lo)) / (log(hi) - log(lo)))

    Parameters
    ----------
    params : jnp.ndarray, shape (nparam,)
    lo, hi : jnp.ndarray, shape (nparam,) – lower/upper bounds
    ftypes : sequence of int, length nparam

    Returns
    -------
    dparams : jnp.ndarray, shape (nfree,) where nfree = #{not FIXED}
    """
    out = []
    for i, ft in enumerate(ftypes):
        p = float(params[i])
        l = float(lo[i])
        h = float(hi[i])
        if ft == PARFIT_FIXED:
            continue
        elif ft == PARFIT_UNBOUND:
            d = p / h if h != 0.0 else p
        elif ft == PARFIT_LINBOUND:
            num = p - l
            den = h - p
            if num > 1e-50:
                d = math.log(num / den) if den > 1e-50 else _CLIP
            else:
                d = -_CLIP
        elif ft == PARFIT_LOGBOUND:
            lp = math.log(max(p, 1e-300))
            ll = math.log(max(l, 1e-300))
            lh = math.log(max(h, 1e-300))
            num = lp - ll
            den = lh - lp
            if num > 1e-50:
                d = math.log(num / den) if den > 1e-50 else _CLIP
            else:
                d = -_CLIP
        else:
            raise ValueError(f"Unknown fit type: {ft}")
        out.append(d)
    return jnp.array(out, dtype=jnp.float64)


def unbounded_to_bounded(
    dparams: jnp.ndarray,
    lo: jnp.ndarray,
    hi: jnp.ndarray,
    ftypes: Sequence[int],
    init: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Inverse of :func:`bounded_to_unbounded`.

    Parameters
    ----------
    dparams : jnp.ndarray, shape (nfree,)
    lo, hi  : jnp.ndarray, shape (nparam,)
    ftypes  : sequence of int, length nparam
    init    : jnp.ndarray, shape (nparam,) – initial values for FIXED params

    Returns
    -------
    params : jnp.ndarray, shape (nparam,)
    """
    out = []
    f = 0
    for i, ft in enumerate(ftypes):
        l = float(lo[i])
        h = float(hi[i])
        if ft == PARFIT_FIXED:
            out.append(float(init[i]) if init is not None else 0.0)
            continue
        d = float(dparams[f])
        d = max(-_CLIP, min(_CLIP, d))
        if ft == PARFIT_UNBOUND:
            p = d * h
        elif ft == PARFIT_LINBOUND:
            p = (h - l) / (1.0 + math.exp(-d)) + l
        elif ft == PARFIT_LOGBOUND:
            p = math.exp(math.log(max(h / l, 1e-300)) / (1.0 + math.exp(-d))) * l
        else:
            raise ValueError(f"Unknown fit type: {ft}")
        out.append(p)
        f += 1
    return jnp.array(out, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Core LM step
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=())
def levmar_step(
    J: jnp.ndarray,
    r: jnp.ndarray,
    mu: float,
) -> jnp.ndarray:
    """
    Compute the Levenberg-Marquardt parameter update Δp.

    Solves: (J^T J + μ I) Δp = J^T r

    Parameters
    ----------
    J  : jnp.ndarray, shape (m, n) – Jacobian (∂residuals/∂params)
    r  : jnp.ndarray, shape (m,)   – residuals
    mu : float                      – damping parameter

    Returns
    -------
    delta_p : jnp.ndarray, shape (n,)
    """
    JtJ = J.T @ J                              # (n, n)
    Jtr = J.T @ r                              # (n,)
    A = JtJ + mu * jnp.eye(J.shape[1], dtype=J.dtype)
    delta_p = jnp.linalg.solve(A, Jtr)        # (n,)  solves A Δp = Jtr
    return delta_p


# ---------------------------------------------------------------------------
# LM main loop
# ---------------------------------------------------------------------------

def levmar_fit(
    residual_fn: Callable,
    params0: jnp.ndarray,
    args: tuple = (),
    lo: jnp.ndarray | None = None,
    hi: jnp.ndarray | None = None,
    ftypes: Sequence[int] | None = None,
    max_iter: int = 200,
    tau: float = 1e-3,
    eps1: float = 1e-12,
    eps2: float = 1e-12,
    eps3: float = 1e-12,
    verbose: bool = False,
) -> dict:
    """
    Levenberg-Marquardt minimiser  (replaces profit_minimize + levmar).

    Minimises ``½ ‖residual_fn(params, *args)‖²`` using LM.

    If *lo*, *hi*, and *ftypes* are provided, optimisation is performed in
    the unbounded parameter space and results are mapped back.

    Parameters
    ----------
    residual_fn : callable(params, *args) → jnp.ndarray shape (m,)
    params0     : jnp.ndarray, shape (n,) – initial parameters
    args        : extra arguments passed to residual_fn
    lo, hi      : lower/upper bounds per parameter (for bounded optimisation)
    ftypes      : fit types per parameter (PARFIT_* constants)
    max_iter    : maximum iterations
    tau         : initial damping factor: μ₀ = τ * max(diag(JᵀJ))
    eps1..3     : convergence thresholds (gradient, step, cost)
    verbose     : print iteration info

    Returns
    -------
    result : dict with keys
        'params'  : jnp.ndarray, shape (n,)  – optimal parameters
        'covar'   : jnp.ndarray, shape (n,n) – covariance matrix estimate
        'chi2'    : float  – final sum of squared residuals
        'niter'   : int    – number of iterations performed
        'reason'  : str    – stopping reason
    """
    use_bounds = (lo is not None and hi is not None and ftypes is not None)

    # Work in unbounded space if bounds are provided
    if use_bounds:
        lo_arr = jnp.array(lo, dtype=jnp.float64)
        hi_arr = jnp.array(hi, dtype=jnp.float64)
        p = bounded_to_unbounded(params0.astype(jnp.float64), lo_arr, hi_arr, ftypes)
        init_arr = params0.astype(jnp.float32)

        def _res(dp):
            bp = unbounded_to_bounded(dp, lo_arr, hi_arr, ftypes, init_arr)
            return residual_fn(bp, *args).astype(jnp.float64)
    else:
        p = params0.astype(jnp.float64)

        def _res(dp):
            return residual_fn(dp.astype(jnp.float32), *args).astype(jnp.float64)

    # Jacobian via forward-mode AD
    jac_fn = jax.jacfwd(_res)

    r = _res(p)
    J = jac_fn(p)                   # (m, n)
    JtJ = J.T @ J
    mu = float(tau * jnp.max(jnp.diag(JtJ)))
    nu = 2.0
    chi2 = float(jnp.dot(r, r))

    reason = "max_iter reached"

    for k in range(max_iter):
        # Gradient check
        Jtr = J.T @ r
        if float(jnp.max(jnp.abs(Jtr))) <= eps1:
            reason = "small gradient"
            break

        # Solve for step
        delta_p = levmar_step(J, r, mu)

        step_norm = float(jnp.linalg.norm(delta_p))
        p_norm = float(jnp.linalg.norm(p))

        if step_norm <= eps2 * (p_norm + eps2):
            reason = "small step"
            break

        p_new = p + delta_p
        r_new = _res(p_new)
        chi2_new = float(jnp.dot(r_new, r_new))

        # Gain ratio
        rho_denom = float(delta_p @ (mu * delta_p + Jtr))
        rho = (chi2 - chi2_new) / rho_denom if abs(rho_denom) > 1e-300 else 0.0

        if verbose:
            print(f"  LM iter {k}: chi2={chi2:.6g}  mu={mu:.3g}  rho={rho:.3g}")

        if rho > 0:
            p = p_new
            r = r_new
            J = jac_fn(p)
            JtJ = J.T @ J
            chi2 = chi2_new
            mu *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            nu = 2.0
            if chi2 <= eps3:
                reason = "small cost"
                break
        else:
            mu *= nu
            nu *= 2.0

    # Convert back to bounded space
    if use_bounds:
        params_out = unbounded_to_bounded(p, lo_arr, hi_arr, ftypes, init_arr)
        params_out = params_out.astype(jnp.float32)
    else:
        params_out = p.astype(jnp.float32)

    # Covariance estimate: (JᵀJ)⁻¹ * chi2/(m-n)
    n = len(p)
    m = len(r)
    try:
        JtJ_final = (jac_fn(p)).T @ jac_fn(p)
        dof = max(m - n, 1)
        covar = jnp.linalg.pinv(JtJ_final) * (chi2 / dof)
    except Exception:
        covar = jnp.full((n, n), jnp.nan)

    return {
        'params': params_out,
        'covar': covar.astype(jnp.float32),
        'chi2': float(chi2),
        'niter': k + 1,
        'reason': reason,
    }


# ---------------------------------------------------------------------------
# Batch fitting with vmap (fit many objects simultaneously)
# ---------------------------------------------------------------------------

def levmar_fit_batch(
    residual_fn: Callable,
    params_batch: jnp.ndarray,
    args_batch: tuple,
    **kwargs,
) -> list:
    """
    Fit multiple objects in sequence.  (True vmap would require static
    shapes for each object; use this when objects have different sizes.)

    Parameters
    ----------
    residual_fn  : callable(params, *args_single) → residuals
    params_batch : jnp.ndarray, shape (N, nparam)
    args_batch   : tuple of jnp.ndarray, each shape (N, ...) – per-object args

    Returns
    -------
    results : list of N result dicts from :func:`levmar_fit`
    """
    N = params_batch.shape[0]
    results = []
    for i in range(N):
        args_i = tuple(a[i] for a in args_batch)
        res = levmar_fit(residual_fn, params_batch[i], args=args_i, **kwargs)
        results.append(res)
    return results


# ---------------------------------------------------------------------------
# Utility: propagate_covar  (matches propagate_covar() in profit.c)
# ---------------------------------------------------------------------------

@jax.jit
def propagate_covar(
    covar_in: jnp.ndarray,
    jac: jnp.ndarray,
) -> jnp.ndarray:
    """
    Propagate a covariance matrix through a Jacobian:
        covar_out = jac @ covar_in @ jac^T

    Parameters
    ----------
    covar_in : jnp.ndarray, shape (n, n)
    jac      : jnp.ndarray, shape (m, n)

    Returns
    -------
    covar_out : jnp.ndarray, shape (m, m)
    """
    return jac @ covar_in @ jac.T
