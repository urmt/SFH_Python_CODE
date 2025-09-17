import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .model import forward_model
import autograd.numpy as anp
from autograd import grad


def loss(theta, d_obs, lam=0.0, theta_prior=None, prior_cov=None):
    """
    Loss function from Eq. 10: L(θ) = ||f(θ) - d_obs||² + λ·R(θ).

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
        theta_prior is None.

    Returns
    -------
    float
        Scalar loss value.
    """
    resid = forward_model(theta) - d_obs
    data_term = anp.dot(resid, resid)

    if theta_prior is None:
        return data_term

    diff = theta - theta_prior
    if prior_cov is None:
        prior_term = anp.dot(diff, diff)
    else:
        inv_cov = anp.linalg.inv(prior_cov)
        prior_term = diff @ inv_cov @ diff

    return data_term + lam * prior_term


def run_cosmo_tuning(obs_file,
                     lambda_reg=0.01,
                     lr=1e-3,
                     n_iter=5000,
                     seed=None):
    """
    Gradient-flow optimizer for the cosmological loss, supporting SFH Protocol 3.

    Parameters
    ----------
    obs_file : str
        CSV file containing observed cosmological quantities (one column per
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
        {'theta': final_parameters, 'loss_history': list_of_losses,
         'converged': bool}
    """
    rng = np.random.default_rng(seed)

    # Load observations
    d_obs = pd.read_csv(obs_file).values.squeeze()
    n_params = d_obs.shape[0]

    # Initialise parameters with SFH-inspired prior (e.g., Planck-like)
    theta = rng.normal(loc=0.3, scale=0.1, size=n_params)  # Ω_m ~ 0.3
    theta_prior = np.full(n_params, 0.3)  # Dummy Planck mean
    prior_cov = 0.01 * np.eye(n_params)  # Weak prior variance

    # Use autograd for exact gradient
    loss_grad = grad(lambda theta: loss(theta, d_obs, lam=lambda_reg,
                                        theta_prior=theta_prior, prior_cov=prior_cov))

    loss_hist = []
    theta_best = theta.copy()
    best_loss = np.inf

    for i in range(n_iter):
        grad_val = loss_grad(theta)
        theta -= lr * grad_val
        current_loss = loss(theta, d_obs, lam=lambda_reg,
                           theta_prior=theta_prior, prior_cov=prior_cov)
        loss_hist.append(current_loss)

        if current_loss < best_loss:
            best_loss = current_loss
            theta_best = theta.copy()

        # Adaptive stopping and learning rate
        if i > 10 and abs(np.mean(loss_hist[-10:-1]) - current_loss) < 1e-8:
            break
        if i % 100 == 0 and i > 0:
            lr = max(lr * 0.95, 1e-6)  # Reduce lr if stuck

    # Plot loss trajectory
    plt.figure(figsize=(6, 4))
    plt.plot(loss_hist, label="Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Cosmological Tuning – Loss vs. Iteration (SFH Protocol 3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cosmo_tuning_loss.png")
    plt.close()

    return {
        "theta": theta_best,
        "loss_history": loss_hist,
        "converged": i < n_iter - 1
    }


if __name__ == "__main__":
    # Example usage with synthetic data
    import os
    obs_data = pd.DataFrame(np.random.normal(0.3, 0.1, 3))
    obs_data.to_csv("data/obs_data.csv", index=False)
    result = run_cosmo_tuning("data/obs_data.csv", lambda_reg=0.01, lr=1e-3, n_iter=2000)
    print("Optimization Result:", result)
    os.remove("data/obs_data.csv")
