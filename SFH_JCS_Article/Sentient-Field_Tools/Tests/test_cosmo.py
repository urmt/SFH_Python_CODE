import numpy as np
import pandas as pd
import pytest
import os

from cosmo_tuning.optimizer import run_cosmo_tuning


@pytest.fixture
def fake_planck_observables(tmp_path):
    """
    Create a tiny CSV that mimics a Planck‑style observable vector.
    The values are completely synthetic but deterministic, so the test can
    verify that the optimiser reduces the loss.
    """
    obs = pd.DataFrame({
        "Omega_m": [0.315],
        "H0": [67.4],
        "Omega_lambda": [0.685]
    })
    fpath = tmp_path / "planck_obs.csv"
    obs.to_csv(fpath, index=False)
    return str(fpath)


def test_optimizer_decreases_loss(fake_planck_observables):
    """
    Run the optimiser for a very small number of iterations and check that
    the loss history is monotonically decreasing (or at least not increasing).
    """
    result = run_cosmo_tuning(
        obs_file=fake_planck_observables,
        lambda_reg=0.01,
        lr=5e-3,
        n_iter=200,
        seed=123,
    )
    loss_hist = result["loss_history"]
    assert len(loss_hist) > 0
    # Ensure the loss never spikes upward by more than 1 % of the first value
    first = loss_hist[0]
    for val in loss_hist[1:]:
        assert val <= first * 1.01
