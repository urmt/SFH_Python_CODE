"""optimizer – Gradient-Flow Optimization for SFH Cosmological Parameters.

Version: 1.0.0
This module implements a gradient-flow optimizer for cosmological parameters
within the Sentience-Field Hypothesis (SFH) Protocol 3. It minimizes the loss
function L(θ) = ||f(θ) - d_obs||² + λ·R(θ), aligning with SFH's qualic field
optimization (Equation 4). The forward model approximates cosmological
observables, supporting empirical validation of qualic energy distributions.

Returns a dict with 'theta', 'loss_history', and 'converged'.
"""

from .optimizer import run_cosmo_tuning
