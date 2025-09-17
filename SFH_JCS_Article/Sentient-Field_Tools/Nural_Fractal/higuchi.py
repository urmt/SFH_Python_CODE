import mne
import numpy as np
import pandas as pd
import nolds
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind


def _load_eeg(eeg_path):
    """
    Load an EEG file (EDF, BDF, or FIF) with MNE and apply a 1-45 Hz band-pass filter.

    Parameters
    ----------
    eeg_path : str
        Path to the EEG recording file.

    Returns
    -------
    mne.io.Raw
        Filtered raw EEG object.
    """
    raw = mne.io.read_raw(eeg_path, preload=True, verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    return raw


def _compute_coherence(seg):
    """
    Compute phase coherence (Kuramoto order parameter) as a proxy for C(q).

    Parameters
    ----------
    seg : ndarray
        EEG segment of shape (n_channels, n_samples).

    Returns
    -------
    float
        Mean phase coherence across channels.
    """
    from scipy.signal import hilbert
    analytic = hilbert(seg, axis=1)
    phases = np.angle(analytic)
    order_param = np.abs(np.mean(np.exp(1j * phases), axis=1))
    return float(np.mean(order_param))


def compute_fractal(eeg_path,
                    kmax=12,
                    epoch_len_sec=2.0,
                    out_csv=None,
                    insight_epochs=None):
    """
    Compute Higuchi fractal dimension and phase coherence for each channel and epoch,
    supporting SFH Protocol 2 (neural fractal-dimension shifts during insight).

    Parameters
    ----------
    eeg_path : str
        Path to the EEG recording file.
    kmax : int, default 12
        Maximum k parameter for Higuchi’s algorithm (must be > 1).
    epoch_len_sec : float, default 2.0
        Length of non-overlapping epochs (seconds, must be positive).
    out_csv : str or None
        If provided, write a CSV with columns 'epoch', 'channel', 'fractal_dim', 'coherence'.
    insight_epochs : list of int or None
        Epoch indices (0-based) corresponding to insight trials for statistical testing.

    Returns
    -------
    dict
        Contains 'df' (pandas.DataFrame with fractal_dim and coherence),
        'stats' (dict with t-test p-value if insight_epochs provided).
    """
    if kmax <= 1:
        raise ValueError("kmax must be greater than 1 for Higuchi's algorithm.")
    if epoch_len_sec <= 0:
        raise ValueError("epoch_len_sec must be positive.")

    raw = _load_eeg(eeg_path)
    sfreq = raw.info["sfreq"]
    epoch_samples = int(epoch_len_sec * sfreq)

    data = raw.get_data()  # shape (n_channels, n_times)
    n_chan, n_time = data.shape
    n_epochs = n_time // epoch_samples
    if n_epochs * epoch_samples != n_time:
        print("Warning: Data truncated to fit epochs.")

    rows = []
    for ep in range(n_epochs):
        start = ep * epoch_samples
        stop = start + epoch_samples
        seg = data[:, start:stop]

        coherence = _compute_coherence(seg)
        for ch_idx in range(n_chan):
            ts = seg[ch_idx, :]
            fd = nolds.higuchi_fd(ts, kmax=kmax)
            rows.append({
                "epoch": ep,
                "channel": raw.ch_names[ch_idx],
                "fractal_dim": fd,
                "coherence": coherence
            })

    df = pd.DataFrame(rows)

    stats = {}
    if insight_epochs is not None:
        insight_fd = df[df["epoch"].isin(insight_epochs)]["fractal_dim"]
        control_fd = df[~df["epoch"].isin(insight_epochs)]["fractal_dim"]
        if len(insight_fd) > 0 and len(control_fd) > 0:
            t_stat, p_value = ttest_ind(insight_fd, control_fd)
            stats = {"t_stat": float(t_stat), "p_value": float(p_value)}
            if p_value < 0.01:
                print("Significant FD shift (p < 0.01) detected, supporting SFH Protocol 2.")

    if out_csv:
        df.to_csv(out_csv, index=False)

    return {"df": df, "stats": stats}


def plot_fractal_results(result):
    """
    Plot fractal dimension and coherence distributions from compute_fractal results.

    Parameters
    ----------
    result : dict
        Output from compute_fractal with 'df' key.
    """
    if "df" not in result:
        raise ValueError("Result must contain 'df' key.")
    df = result["df"]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(df["fractal_dim"], bins=20, color="skyblue", edgecolor="black")
    plt.title("Higuchi Fractal Dimension Distribution")
    plt.xlabel("Fractal Dimension")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(df["coherence"], bins=20, color="lightgreen", edgecolor="black")
    plt.title("Phase Coherence Distribution")
    plt.xlabel("Coherence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage with synthetic data
    import os
    from mne.datasets import sample

    sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
    result = compute_fractal(sample_path, kmax=10, epoch_len_sec=2.0, out_csv="data/fractal_results.csv")
    print("Fractal Analysis Result:", result)
    plot_fractal_results(result)
