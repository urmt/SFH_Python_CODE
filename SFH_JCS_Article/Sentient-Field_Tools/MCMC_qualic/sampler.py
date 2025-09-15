import numpy as np
import pandas as pd
from scipy.stats import norm

def _log_likelihood(theta, q_obs, sigma=0.1):
    """
    Simple Gaussian likelihood for the qualic variable Q.

    Parameters
    ----------
    theta : array‑like, shape (2,)
        Model parameters (mu, sigma_q) where mu is the mean Q and sigma_q the
        intrinsic spread.
    q_obs : array‑like
        Observed Q (or a proxy such as γ‑band power).
    sigma : float, optional
        Observation noise (default 0.1).

    Returns
    -------
    float
        Log‑likelihood value.
    """
    mu, sigma_q = theta
    var = sigma_q ** 2 + sigma ** 2
    return np.sum(norm.logpdf(q_obs, loc=mu, scale=np.sqrt(var)))

def _log_prior(theta):
    """Weakly‑informative Normal(0,10) priors on both parameters."""
    mu, sigma_q = theta
    if sigma_q <= 0:
        return -np.inf
    return norm.logpdf(mu, 0, 10) + norm.logpdf(sigma_q, 0, 10)

def run_mcmc(q_obs,
             n_iter=20000,
             burn_in=5000,
             step_sd=0.2,
             seed=None):
    """
    Run a vanilla Metropolis‑Hastings sampler.

    Parameters
    ----------
    q_obs : array‑like
        Observed qualic values (e.g., γ‑band power).
    n_iter : int, default 20000
        Total number of MCMC iterations.
    burn_in : int, default 5000
        Number of initial samples to discard.
    step_sd : float, default 0.2
        Standard deviation of the symmetric Gaussian proposal.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    pandas.DataFrame
        Posterior draws after burn‑in with columns ``mu`` and ``sigma_q``.
    """
    rng = np.random.default_rng(seed)

    # initialise chain at reasonable values
    cur_theta = np.array([np.mean(q_obs), np.std(q_obs)])
    cur_logpost = _log_likelihood(cur_theta, q_obs) + _log_prior(cur_theta)

    samples = []

    for i in range(n_iter):
        prop_theta = cur_theta + rng.normal(scale=step_sd, size=2)
        prop_logpost = _log_likelihood(prop_theta, q_obs) + _log_prior(prop_theta)

        # acceptance probability
        log_alpha = prop_logpost - cur_logpost
        if np.log(rng.random()) < log_alpha:
            cur_theta, cur_logpost = prop_theta, prop_logpost

        if i >= burn_in:
            samples.append(cur_theta.copy())

    df = pd.DataFrame(samples, columns=["mu", "sigma_q"])
    return df
