import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .model import forward_model

def loss(theta, d_obs, lam=0.0, theta_prior=None, prior_cov=None):
    """
    Loss function from Eq. 10:
        L(θ) = ||f(θ) - d_obs||² + λ·R(θ)

    Parameters
    ----------
    theta : ndarray
        Current parameter estimate.
    d_obs : ndarray
        Observed cosmological quantities.
    lam : float
        Regularisation strength (λ).
    theta_prior : ndarray or None
        Prior mean (for a quadratic penalty). If None, no prior term.
    prior_cov : ndarray or None
        Prior covariance matrix (inverse used for quadratic term). Ignored if
        ``theta_prior`` is None.

    Returns
    -------
    float
        Scalar loss value.
    """
    resid = forward_model(theta) - d_obs
    data_term = np.dot(resid, resid)

    if theta_prior is None:
        return data_term

    diff = theta - theta_prior
    if prior_cov is None:
        prior_term = np.dot(diff, diff)
    else:
        inv_cov = np.linalg.inv(prior_cov)
        prior_term = diff @ inv_cov @ diff

    return data_term + lam * prior_term

def run_cosmo_tuning(obs_file,
                     lambda_reg=0.01,
                     lr=1e-3,
                     n_iter=5000,
                     seed=None):
    """
    Gradient‑flow optimiser for the cosmological loss.

    Parameters
    ----------
    obs_file : str
        CSV file containing the observed cosmological quantities (one column per
        observable, header row required).
    lambda_reg : float, default 0.01
        Regularisation weight λ.
    lr : float, default 1e-3
        Learning rate for the gradient step.
    n_iter : int, default 5000
        Maximum number of optimisation iterations.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{'theta': final_parameters, 'loss_history': list_of_losses}``
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------
    # Load observations
    # --------------------------------------------------------------
    d_obs = pd.read_csv(obs_file).values.squeeze()
    n_params = d_obs.shape[0]

    # Initialise parameters (simple Gaussian draw)
    theta = rng.normal(loc=0.0, scale=1.0, size=n_params)

    # Prior (use the Planck central values as a dummy prior)
    theta_prior = np.zeros_like(theta)  # you could replace with real Planck means
    prior_cov = np.eye(n_params)       # identity = weak prior

    loss_hist = []

    for i in range(n_iter):
        # Numerical gradient (finite differences) – cheap for the demo
        eps = 1e-6
        grad = np.zeros_like(theta)
        base_loss = loss(theta, d_obs, lam=lambda_reg,
                         theta_prior=theta_prior, prior_cov=prior_cov)

        for j in range(len(theta)):
            theta_eps = theta.copy()
            theta_eps[j] += eps
            grad[j] = (loss(theta_eps, d_obs, lam=lambda_reg,
                           theta_prior=theta_prior, prior_cov=prior_cov) - base_loss) / eps

        # Gradient descent step
        theta -= lr * grad
        loss_hist.append(base_loss)

        # Simple stopping criterion
        if i > 0 and abs(loss_hist[-2] - loss_hist[-1]) < 1e-8:
            break

    # --------------------------------------------------------------
    # Plot loss trajectory (saved automatically)
    # --------------------------------------------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Cosmological tuning – loss vs. iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosmo_tuning_loss.png")
    plt.close()

    return {"theta": theta, "loss_history": loss_hist}
