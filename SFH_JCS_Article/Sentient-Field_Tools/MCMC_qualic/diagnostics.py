import arviz as az
import matplotlib.pyplot as plt

def plot_trace(df):
    """
    Plot trace and posterior histograms using ArviZ.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of ``run_mcmc`` (columns ``mu`` and ``sigma_q``).
    """
    data = az.convert_to_inference_data(df)
    az.plot_trace(data)
    plt.show()

def gelman_rubin(chains):
    """
    Compute the Gelman‑Rubin R̂ statistic for a list of chains.

    Parameters
    ----------
    chains : list of pandas.DataFrame
        Each element is the posterior from an independent chain.

    Returns
    -------
    dict
        Mapping from parameter name to R̂.
    """
    combined = az.concat(chains, dim="chain")
    rhat = az.rhat(combined)
    return {var: float(val) for var, val in rhat.items()}
