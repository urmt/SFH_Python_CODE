import matplotlib.pyplot as plt
import arviz as az


def plot_trace(result):
    """
    Plot trace and posterior histograms using ArviZ from MCMC results.

    Parameters
    ----------
    result : dict
        Output of `run_mcmc` with 'samples' (pd.MultiIndex DataFrame) key.
    """
    if 'samples' not in result:
        raise ValueError("Result must contain 'samples' key.")
    data = az.convert_to_inference_data(result['samples'])
    az.plot_trace(data)
    plt.show()


def gelman_rubin(result):
    """
    Compute the Gelman-Rubin R̂ statistic from MCMC diagnostics.

    Parameters
    ----------
    result : dict
        Output of `run_mcmc` with 'diagnostics' key containing R-hat values.

    Returns
    -------
    dict
        Mapping from parameter name to R̂.
    """
    if 'diagnostics' not in result:
        raise ValueError("Result must contain 'diagnostics' key.")
    diagnostics = result['diagnostics']
    return {
        "R_hat_mu": diagnostics["R_hat_mu"],
        "R_hat_sigma": diagnostics["R_hat_sigma"]
    }
