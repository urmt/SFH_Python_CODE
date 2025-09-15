import numpy as np
import pandas as pd
import pytest
import os
import mne

from neural_fractal.higuchi import compute_fractal


@pytest.fixture
def synthetic_eeg(tmp_path):
    """
    Build a tiny synthetic EEG recording (2 channels, 10 seconds at 250 Hz)
    consisting of a mixture of sine waves and pink noise.  The file is saved
    as an EDF so the loader in `neural_fractal` can read it.
    """
    sfreq = 250.0
    t = np.arange(0, 10, 1 / sfreq)
    # Channel 1: 10 Hz sinusoid + pink noise
    ch1 = np.sin(2 * np.pi * 10 * t) + np.random.RandomState(0).normal(scale=0.5, size=len(t))
    # Channel 2: 20 Hz sinusoid + pink noise
    ch2 = np.sin(2 * np.pi * 20 * t) + np.random.RandomState(1).normal(scale=0.5, size=len(t))

    info = mne.create_info(ch_names=["Cz", "Pz"], sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.vstack([ch1, ch2]), info)

    fpath = tmp_path / "synthetic.edf"
    raw.export(str(fpath), fmt="edf")  # requires pyEDFlib (installed via mne)
    return str(fpath)


def test_fractal_dimensions_are_reasonable(synthetic_eeg, tmp_path):
    """
    The Higuchi FD for a pure sinusoid is ≈ 1.0, while for noisy signals it
    moves toward 2.0.  Our synthetic data contain both, so we expect values
    between 1.0 and 1.8.
    """
    out_csv = tmp_path / "fd_results.csv"
    df = compute_fractal(str(synthetic_eeg), kmax=12, epoch_len_sec=2.0, out_csv=str(out_csv))

    # Check that the DataFrame has the expected columns
    assert set(df.columns) == {"epoch", "channel", "fractal_dim"}

    # All fractal dimensions should lie in a plausible range
    assert df["fractal_dim"].between(1.0, 1.9).all()

    # Verify that the CSV was written correctly
    df_check = pd.read_csv(str(out_csv))
    pd.testing.assert_frame_equal(df, df_check)
