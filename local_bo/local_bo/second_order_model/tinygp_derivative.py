import jax, jax.numpy as jnp
from tinygp import kernels, transforms, GaussianProcess
from typing import Callable, Tuple, Dict, Union

__all__ = ["fit_derivative_gp"]

# -----------------------------------------------------------------------------
# Derivative‑aware kernel -------------------------------------------------------
# -----------------------------------------------------------------------------
class DerivativeKernel(kernels.Kernel):
    """Extend a latent kernel to handle value + directional‑derivative data."""

    kernel: kernels.Kernel  # latent k(x,x′)

    # mixed Hessian  ∂²k/∂x∂x′
    @staticmethod
    def _mixed_hess(k, x1, x2):
        g   = jax.grad(k, argnums=0)
        H   = jax.jacrev(g, argnums=1)(x1, x2)
        return H

    # full covariance ---------------------------------------------------------
    def evaluate(self, Z1, Z2):
        x1, d1 = Z1
        x2, d2 = Z2
        k = self.kernel.evaluate

        ff = k(x1, x2)
        fd = jnp.dot(jax.grad(k, argnums=1)(x1, x2), d2)
        df = jnp.dot(d1, jax.grad(k, argnums=0)(x1, x2))
        dd = jnp.dot(d1, jnp.dot(self._mixed_hess(k, x1, x2), d2))

        return jax.lax.cond(
            jnp.any(d1),
            lambda _: jax.lax.cond(jnp.any(d2), lambda _ : dd, lambda _ : df, None),
            lambda _: jax.lax.cond(jnp.any(d2), lambda _ : fd, lambda _ : ff, None),
            None,
        )

    # variance of a single observation ---------------------------------------
    def evaluate_diag(self, Z):
        x, d = Z
        k = self.kernel.evaluate
        var_val = k(x, x) + 1e-6
        H       = self._mixed_hess(k, x, x)
        var_der = jnp.dot(d, jnp.dot(H, d)) + 1e-6
        return jax.lax.cond(jnp.any(d), lambda _ : var_der, lambda _ : var_val, None)

# -----------------------------------------------------------------------------
# Utility to stack data --------------------------------------------------------
# -----------------------------------------------------------------------------

def _stack_data(X_f, y_f, X_g, Y_g):
    if len(X_f):
        X_all = jnp.concatenate([X_f, jnp.repeat(X_g, X_g.shape[1], 0)])
        dirs  = jnp.concatenate([
            jnp.zeros_like(X_f),
            jnp.tile(jnp.eye(X_g.shape[1]), (len(X_g), 1))
        ])
        y_all = jnp.concatenate([y_f, Y_g.flatten()])
    else:
        X_all = jnp.repeat(X_g, X_g.shape[1], 0)
        dirs  = jnp.tile(jnp.eye(X_g.shape[1]), (len(X_g), 1))
        y_all = Y_g.flatten()
    return X_all, dirs, y_all

# -----------------------------------------------------------------------------
# Public API -------------------------------------------------------------------
# -----------------------------------------------------------------------------

def fit_derivative_gp(
    X_f: jnp.ndarray,
    y_f: jnp.ndarray,
    X_g: jnp.ndarray,
    Y_g: jnp.ndarray,
    *,
    log_amp: float = 1.0,
    log_scale: Union[jnp.ndarray, float] = 1.0,
    log_noise: float = -9.2,
    optimise: bool = False,
):
    """Fit an anisotropic RBF GP to value + gradient data.

    * ``log_scale`` may be a scalar (isotropic) or a vector (one length‐scale
      per input dimension).
    * Returns ``(params_dict, predict_mean, predict_var)``.
    """

    X_all, dirs_all, y_all = _stack_data(X_f, y_f, X_g, Y_g)

    # -------------------------------------------------------------------------
    # Helper to build a GP given *log* params
    # -------------------------------------------------------------------------
    def make_gp(p):
        A      = jnp.exp(-p["log_scale"])           # 1 / ℓ_j  (broadcasts)
        latent = kernels.ExpSquared()
        base   = transforms.Linear(A, latent)
        base   = jnp.exp(2.0 * p["log_amp"]) * base
        kern   = DerivativeKernel(base)
        return GaussianProcess(kern, (X_all, dirs_all), diag=jnp.exp(2.0*p["log_noise"]))#+ 1e-6)

    params = dict(log_amp=jnp.asarray(log_amp),
                  log_scale=jnp.asarray(log_scale),
                  log_noise=jnp.asarray(log_noise))

    # -------------------------------------------------------------------------
    # Optimise in unconstrained space if requested
    # -------------------------------------------------------------------------
    if optimise:
        import jaxopt

        θ0 = jnp.concatenate([
            params["log_amp"].reshape(1),
            jnp.atleast_1d(params["log_scale"]),
            params["log_noise"].reshape(1),
        ])

        def θ_to_p(θ):
            d = len(jnp.atleast_1d(params["log_scale"]))
            return dict(
                log_amp   = θ[0],
                log_scale = θ[1:1+d],          # always 1-D, even if d==1
                log_noise = θ[-1],
            )

        loss = lambda θ: -make_gp(θ_to_p(θ)).log_probability(y_all)

        import jaxopt
        solver    = jaxopt.BFGS(fun=loss, tol=1e-6, maxiter=250)
        θ_opt     = solver.run(θ0).params        # 1-D DeviceArray
        params    = θ_to_p(θ_opt)

    gp = make_gp(params)

    # -------------------------------------------------------------------------
    # Prediction closures
    # -------------------------------------------------------------------------
    def predict_mean(Xq, dirs_q):
        return gp.condition(y_all, (Xq, dirs_q)).gp

    return params, predict_mean
