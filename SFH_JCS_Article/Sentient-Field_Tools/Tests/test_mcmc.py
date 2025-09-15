import numpy as np
import pandas as pd
import pytest

from mcmc_qualic.sampler import run_mcmc


@pytest.fixture
def synthetic_q():
    """
    Generate a tiny synthetic qualic dataset.
    True parameters: mu = 1.5, sigma_q = 0.3
    Observation noise sigma = 0.1 (hard‑coded in the sampler).
    """
    rng = np.random.default_rng(123)
    true_mu = 1.5
    true_sigma_q = 0.3
    n = 50
    # intrinsic variation + observation noise
    q = rng.normal(true_mu, np.sqrt(true_sigma_q**2 + 0.1**2), size=n)
    return q


def test_mcmc_returns_dataframe(synthetic_q):
    """The sampler must return a pandas DataFrame with the correct columns."""
    df = run_mcmc(synthetic_q, n_iter=5000, burn_in=1000, step_sd=0.15, seed=42)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) == {"mu", "sigma_q"}
    assert len(df) == 4000  # 5000‑1000 burn‑in


def test_posterior_covers_true_values(synthetic_q):
    """
    With a modest number of iterations the posterior mean should be
    reasonably close to the true generating parameters.
    """
    df = run_mcmc(synthetic_q, n_iter=8000, burn_in=2000, step_sd=0.15, seed=42)
    mu_est = df["mu"].mean()
    sigma_est = df["sigma_q"].mean()

    # tolerance is generous because the toy data are noisy
    assert abs(mu_est - 1.5) < 0.25
    assert abs(sigma_est - 0.3) < 0.20
