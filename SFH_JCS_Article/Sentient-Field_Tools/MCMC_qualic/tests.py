import os
import numpy as np
import pandas as pd
from mcmc_qualic import run_mcmc


def test_run_mcmc_basic():
    """Test basic MCMC run with synthetic data."""
    q_obs = np.random.normal(2.0, 0.5, 100)
    result = run_mcmc(q_obs, n_iter=1000, burn_in=200, n_chains=2)
    assert "samples" in result, "Missing samples in result"
    assert "diagnostics" in result, "Missing diagnostics in result"
    samples_df = result["samples"]
    assert not samples_df.empty, "Samples DataFrame should not be empty"
    assert len(samples_df.columns) == 2, "Should have mu and sigma_q columns"
    assert np.mean(result["diagnostics"]["accept_rate"]) > 0, "Acceptance rate should be positive"


def test_run_mcmc_edge_cases():
    """Test edge cases like empty data and invalid parameters."""
    # Empty data
    with np.testing.assert_raises(ValueError):
        run_mcmc(np.array([]))

    # Negative burn-in
    with np.testing.assert_raises(ValueError):
        run_mcmc(np.random.normal(2.0, 0.5, 10), burn_in=-1)


def test_diagnostics_convergence():
    """Test diagnostic values for reasonable convergence."""
    q_obs = np.random.normal(2.0, 0.5, 100)
    result = run_mcmc(q_obs, n_iter=2000, burn_in=500, n_chains=4)
    diagnostics = result["diagnostics"]
    assert not np.isnan(diagnostics["R_hat_mu"]), "R-hat should be computable"
    assert diagnostics["ESS_mu"] > 0, "Effective sample size should be positive"
    assert abs(diagnostics["R_hat_mu"] - 1) < 0.2, "R-hat should be close to 1 for convergence"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
