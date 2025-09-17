import os
import pandas as pd
import numpy as np
from cosmology import run_cosmo_tuning, loss
from model import forward_model


def test_loss_function():
    """Test the loss function with and without prior terms."""
    theta = np.array([0.3, 0.7])
    d_obs = np.array([0.31, 0.69])
    # Without prior
    l_no_prior = loss(theta, d_obs, lam=0.0)
    assert l_no_prior >= 0, "Loss should be non-negative"
    assert np.isclose(l_no_prior, np.sum((forward_model(theta) - d_obs) ** 2), atol=1e-5), "Data term mismatch"

    # With prior
    theta_prior = np.array([0.3, 0.7])
    prior_cov = np.eye(2)
    l_with_prior = loss(theta, d_obs, lam=0.01, theta_prior=theta_prior, prior_cov=prior_cov)
    assert l_with_prior > l_no_prior, "Prior term should increase loss"


def test_run_cosmo_tuning_basic():
    """Test basic optimization with synthetic data."""
    obs_data = pd.DataFrame(np.random.normal(0.3, 0.1, 3))
    obs_data.to_csv("data/test_obs.csv", index=False)
    result = run_cosmo_tuning("data/test_obs.csv", lambda_reg=0.01, lr=1e-3, n_iter=100)
    os.remove("data/test_obs.csv")

    assert "theta" in result, "Missing theta in result"
    assert "loss_history" in result, "Missing loss_history in result"
    assert "converged" in result, "Missing converged in result"
    assert len(result["loss_history"]) == 100, "Incorrect number of iterations"
    assert result["loss_history"][-1] < result["loss_history"][0], "Loss should decrease"


def test_run_cosmo_tuning_edge_cases():
    """Test edge cases like empty file and invalid lr."""
    # Empty file
    obs_data = pd.DataFrame()
    obs_data.to_csv("data/empty_obs.csv", index=False)
    with np.testing.assert_raises(ValueError):
        run_cosmo_tuning("data/empty_obs.csv")
    os.remove("data/empty_obs.csv")

    # Invalid learning rate
    with np.testing.assert_raises(ValueError):
        run_cosmo_tuning("data/test_obs.csv", lr=-1e-3)  # lr should be positive


def test_forward_model():
    """Test the forward model consistency."""
    theta = np.array([0.3, 0.7, 0.1])
    result1 = forward_model(theta)
    result2 = forward_model(theta)  # Should be reproducible due to fixed seed
    np.testing.assert_array_almost_equal(result1, result2, decimal=5)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
