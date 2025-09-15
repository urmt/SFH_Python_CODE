import mne
import numpy as np
import pandas as pd
import nolds
import matplotlib.pyplot as plt

def _load_eeg(eeg_path):
    """
    Load an EEG file (EDF, BDF, or FIF) with MNE.
    Returns the raw object after a basic band‑pass filter (1‑45 Hz).
    """
    raw = mne.io.read_raw(eeg_path, preload=True, verbose=False)
    raw.filter(l_freq=1.0, h_freq=45.0, verbose=False)
    return raw

def compute_fractal(eeg_path,
                   kmax=12,
                   epoch_len_sec=2.0,
                   out_csv=None):
    """
    Compute Higuchi fractal dimension for each channel and each epoch.

    Parameters
    ----------
    eeg_path : str
        Path to the EEG recording file.
    kmax : int, default 12
        Maximum k parameter for Higuchi’s algorithm.
    epoch_len_sec : float, default 2.0
        Length of non‑overlapping epochs (seconds).
    out_csv : str or None
        If provided, write a CSV with columns
        ``epoch, channel, fractal_dim``.

    Returns
    -------
    pandas.DataFrame
        One row per (epoch, channel) pair.
    """
    raw = _load_eeg(eeg_path)
    sfreq = raw.info["sfreq"]
    epoch_samples = int(epoch_len_sec * sfreq)

    data = raw.get_data()               # shape (n_channels, n_times)
    n_chan, n_time = data.shape
    n_epochs = n_time // epoch_samples

    rows = []
    for ep in range(n_epochs):
        start = ep * epoch_samples
        stop = start + epoch_samples
        seg = data[:, start:stop]

        for ch_idx in range(n_chan):
            ts = seg[ch_idx, :]
            fd = nolds.higuchi_fd(ts, kmax=kmax)
            rows.append({"epoch": ep, "channel": raw.ch_names[ch_idx], "fractal_dim": fd})

    df = pd.DataFrame(rows)

    if out_csv:
        df.to_csv(out_csv, index=False)

    return df
