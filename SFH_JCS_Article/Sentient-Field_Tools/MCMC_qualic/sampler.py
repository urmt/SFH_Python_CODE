import numpy as np
import pandas as pd
from scipy.stats import norm


def _log_likelihood(theta, q_obs, sigma=0.1):
    """
    Simple Gaussian likelihood for the qualic variable Q, supporting SFH Protocol 2.

    Parameters
    ----------
    theta : array-like, shape (2,)
        Model parameters (mu, sigma_q) where mu is the mean Q and sigma_q the
        intrinsic spread.
    q_obs : array-like
        Observed Q (e.g., γ-band power from neural data).
    sigma : float, optional
        Observation noise (default 0.1).

    Returns
    -------
    float
        Log-likelihood value.
    """
    mu, sigma_q = theta
    var = sigma_q ** 2 + sigma ** 2
    return np.sum(norm.logpdf(q_obs, loc=mu, scale=np.sqrt(var)))


def _log_prior(theta):
    """Weakly-informative Normal(0,10) priors on both parameters for SFH robustness."""
    mu, sigma_q = theta
    if sigma_q <= 0:
        return -np.inf
    return norm.logpdf(mu, 0, 10) + norm.logpdf(sigma_q, 0, 10)


def _coherence_proxy(q_obs):
    """SFH-specific proxy for coherence C(q): inverse variance of observations."""
    var_obs = np.var(q_obs)
    return 1 / var_obs if var_obs > 0 else 1e6  # High coherence if low variance


def _effective_sample_size(chain):
    """Simple ESS calculation using autocorrelation time for 1D chain."""
    chain = np.asarray(chain).flatten()  # Ensure 1D
    n = len(chain)
    if n < 2:
        return 0.0
    mean_chain = np.mean(chain)
    centered = chain - mean_chain
    acf = np.correlate(centered, centered, mode='full')
    acf = acf[acf.size // 2:] / (np.var(chain) * n)
    cutoff = min(100, n // 2)
    positive_acf = acf[1:cutoff][acf[1:cutoff] > 0]
    tau = 1 + 2 * np.sum(positive_acf)
    return n / tau if tau > 0 else n


def _gelman_rubin_param(chains_param):
    """R-hat for a single parameter across chains."""
    chains_param = np.array(chains_param)  # (n_chains, n_samples)
    if chains_param.shape[0] < 2:
        return np.nan
    n_chains, n_iter = chains_param.shape
    chain_means = np.mean(chains_param, axis=1)
    between_var = n_iter / (n_chains - 1) * np.var(chain_means)
    within_vars = np.var(chains_param, axis=1)
    within_var = np.mean(within_vars)
    var_est = (1 - 1 / n_chains) * within_var + between_var / n_iter
    psi_within = within_var * (n_iter - 1) / n_iter
    r_hat = np.sqrt(var_est / psi_within)
    return r_hat


def run_mcmc(q_obs,
             n_iter=20000,
             burn_in=5000,
             target_accept=0.3,
             n_chains=4,
             seed=None):
    """
    Enhanced Metropolis-Hastings sampler with adaptive step size, multiple chains,
    coherence proxy initialization, and diagnostics for SFH qualic modeling.

    Parameters
    ----------
    q_obs : array-like
        Observed qualic values (e.g., γ-band power from Protocol 2).
    n_iter : int, default 20000
        Total iterations per chain.
    burn_in : int, default 5000
        Burn-in per chain.
    target_accept : float, default 0.3
        Target acceptance rate for adaptive tuning.
    n_chains : int, default 4
        Number of parallel chains for diagnostics.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict
        Contains 'samples' (pd.MultiIndex DataFrame), 'diagnostics' (dict with ESS, R-hat).
    """
    if len(q_obs) == 0:
        raise ValueError("q_obs must be non-empty.")
    
    rng = np.random.default_rng(seed)
    c_proxy = _coherence_proxy(q_obs)
    init_mu = np.mean(q_obs) * (1 + 0.1 * c_proxy / (1 + c_proxy))  # Bias toward coherence
    init_sigma = np.std(q_obs) if len(q_obs) > 1 else 0.1
    init_theta = np.array([init_mu, max(init_sigma, 0.01)])  # Ensure positive sigma
    
    all_samples = []
    all_chains_mu = []
    all_chains_sigma = []
    total_accepts = 0
    total_steps = 0
    
    for chain_id in range(n_chains):
        cur_theta = init_theta + rng.normal(scale=0.01, size=2) * chain_id  # Slight offset
        cur_logpost = _log_likelihood(cur_theta, q_obs) + _log_prior(cur_theta)
        samples = []
        accepts = 0
        step_sd = 0.2
        
        for i in range(n_iter):
            prop_theta = cur_theta + rng.normal(scale=step_sd, size=2)
            prop_logpost = _log_likelihood(prop_theta, q_obs) + _log_prior(prop_theta)
            log_alpha = prop_logpost - cur_logpost
            accept = np.log(rng.random()) < log_alpha
            if accept:
                cur_theta = prop_theta
                cur_logpost = prop_logpost
                accepts += 1
            
            total_accepts += int(accept)
            total_steps += 1
            
            if i >= burn_in:
                samples.append(cur_theta.copy())
            
            # Adaptive tuning every 100 steps
            if (i + 1) % 100 == 0:
                local_rate = accepts / 100
                if local_rate < target_accept * 0.8:
                    step_sd = max(step_sd * 0.95, 0.01)  # Prevent too small
                elif local_rate > target_accept * 1.2:
                    step_sd *= 1.05
                accepts = 0
        
        all_samples.append(samples)
        all_chains_mu.append([s[0] for s in samples])
        all_chains_sigma.append([s[1] for s in samples])
    
    # Combine samples into MultiIndex DataFrame
    combined_df = pd.concat([pd.DataFrame(chain, columns=["mu", "sigma_q"])
                             for chain in all_samples],
                            keys=range(n_chains), names=["chain", None])
    
    # Diagnostics
    ess_mu = [_effective_sample_size(chain) for chain in all_chains_mu]
    ess_sigma = [_effective_sample_size(chain) for chain in all_chains_sigma]
    r_hat_mu = _gelman_rubin_param(all_chains_mu)
    r_hat_sigma = _gelman_rubin_param(all_chains_sigma)
    accept_rate = total_accepts / total_steps if total_steps > 0 else 0.0
    
    diagnostics = {
        "ESS_mu": np.mean(ess_mu),
        "ESS_sigma": np.mean(ess_sigma),
        "R_hat_mu": r_hat_mu,
        "R_hat_sigma": r_hat_sigma,
        "accept_rate": accept_rate,
        "n_effective": len(combined_df)
    }
    
    # Warn if diagnostics indicate poor convergence
    if not np.isnan(r_hat_mu) and abs(r_hat_mu - 1) > 0.1:
        print("Warning: R-hat for mu > 1.1, consider longer chains.")
    if np.mean(ess_mu) < len(combined_df) / 10:
        print("Warning: Low ESS for mu, increase n_iter.")
    
    return {"samples": combined_df, "diagnostics": diagnostics}
