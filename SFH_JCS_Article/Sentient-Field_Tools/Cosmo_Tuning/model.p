import numpy as np

def forward_model(theta):
    """
    Very simple linear surrogate for the full SFH forward model.
    In a real implementation this would call a physics‑based simulator;
    here we use a linear mapping that is sufficient for the demo.

    Parameters
    ----------
    theta : ndarray, shape (n_params,)
        Parameter vector (e.g., Ω_m, H0, Ω_Λ, …).

    Returns
    -------
    ndarray, shape (n_obs,)
        Predicted cosmological observables.
    """
    # A is a fixed matrix that maps parameters to observables.
    # We create a pseudo‑random but reproducible matrix.
    rng = np.random.default_rng(42)
    n_obs = len(theta)  # for the demo we keep dimensions equal
    A = rng.normal(size=(n_obs, len(theta)))
    return A @ theta
