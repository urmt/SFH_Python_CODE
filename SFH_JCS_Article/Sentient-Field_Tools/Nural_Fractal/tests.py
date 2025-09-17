import os
import numpy as np
from mne.datasets import sample
from higuchi import compute_fractal, plot_fractal_results, _load_eeg


def test_load_eeg():
    """Test EEG loading and filtering."""
    sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
    raw = _load_eeg(sample_path)
    assert raw.info["sfreq"] > 0, "Sampling frequency should be positive"
    assert raw.get_data().shape[1] > 0, "Data should be loaded"
    assert raw.filter(l_freq=1.0, h_freq=45.0) is None, "Filter should apply silently"


def test_compute_fractal_basic():
    """Test fractal dimension computation with sample data."""
    sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
    result = compute_fractal(sample_path, kmax=10, epoch_len_sec=2.0)
    assert "df" in result, "Result should contain 'df' key"
    df = result["df"]
    assert not df.empty, "DataFrame should not be empty"
    assert "fractal_dim" in df.columns, "Missing fractal_dim column"
    assert "coherence" in df.columns, "Missing coherence column"
    assert df["fractal_dim"].min() >= 1.0, "FD should be >= 1.0"
    assert df["fractal_dim"].max() <= 2.0, "FD should be <= 2.0"


def test_compute_fractal_insight():
    """Test statistical testing with simulated insight epochs."""
    sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
    result = compute_fractal(sample_path, kmax=10, epoch_len_sec=2.0, insight_epochs=[0, 1])
    assert "stats" in result, "Result should contain 'stats' key"
    stats = result["stats"]
    assert "p_value" in stats, "Missing p_value in stats"
    assert isinstance(stats["p_value"], float), "p_value should be a float"


def test_compute_fractal_invalid_input():
    """Test error handling for invalid inputs."""
    with np.testing.assert_raises(ValueError):
        compute_fractal("nonexistent.fif", kmax=1)  # kmax <= 1
    with np.testing.assert_raises(ValueError):
        compute_fractal("nonexistent.fif", epoch_len_sec=0)  # epoch_len_sec <= 0


def test_plot_fractal_results():
    """Test plot generation with sample data."""
    sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
    result = compute_fractal(sample_path, kmax=10, epoch_len_sec=2.0)
    plot_fractal_results(result)
    plt.close()  # Avoid display in test environment


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
