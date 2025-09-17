import numpy as np


def forward_model(theta):
    """
    Linear surrogate for the SFH cosmological forward model, approximating qualic
    field dynamics (Protocol 3). In a full implementation, this would incorporate
    non-linear physics and Hardy-Ramanujan partitions (Eq. 2).

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector (e.g., Ω_m, H0, Ω_Λ, …).

    Returns
    -------
    ndarray, shape (n_obs,)
        Predicted cosmological observables.
    """
    rng = np.random.default_rng(42)
    n_obs = len(theta)  # For demo, match input dimension
    A = rng.normal(size=(n_obs, len(theta)))  # Pseudo-random mapping
    # Apply SFH-inspired non-linearity (simplified partition effect)
    scaled_theta = theta * np.exp(-np.sum(theta**2) / n_params)  # Dampen large deviations
    return A @ scaled_theta
