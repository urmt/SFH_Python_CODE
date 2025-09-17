markdown# higuchi: Fractal Dimension Analysis for SFH Protocol 2

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-green.svg)
![Last Updated](https://img.shields.io/badge/last_updated-Sep_17_2025-orange.svg)

Welcome to **higuchi**, a Python module for computing the Higuchi fractal dimension (FD) and phase coherence from EEG data, tailored for the **Sentience-Field Hypothesis (SFH)** Protocol 2. This tool supports SFH's neuroscience application (Section 3.2 of *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality* by Mark Rowe Traver, IngramSpark, 2025, ISBN: 978-0-123456-78-9), analyzing fractal shifts during insight as a proxy for qualic fertility \(F(q)\) and coherence \(C(q)\).

## Overview

The Higuchi FD measures signal complexity, reflecting SFH's prediction of increased neural complexity during "aha" moments. Phase coherence (Kuramoto order parameter) complements this by assessing synchronization, together supporting the optimization functional \(J(q) = \alpha C(q) + \beta F(q)\). This module is optimized for EEG analysis, providing statistical validation of insight-related shifts.

## Installation

To install the `higuchi` module, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/urmt/SFH_Python_CODE.git
   cd SFH_Python_CODE/SFH_JCS_Article/Sentient-Field_Tools/neuroscience

Install Dependencies:
Ensure you have Python 3.11 and the required packages:
bashpip install mne numpy pandas nolds scipy matplotlib

Install the Module:
Install locally for use in your projects:
bashpip install .


Usage
Basic Example
The following example computes fractal dimensions and coherence for a sample EEG dataset, with optional insight testing:
pythonfrom mne.datasets import sample
from higuchi import compute_fractal, plot_fractal_results

# Load sample EEG data
sample_path = sample.data_path() + "/MEG/sample/sample_audvis_raw.fif"
result = compute_fractal(sample_path, kmax=10, epoch_len_sec=2.0, insight_epochs=[0, 1], out_csv="data/fractal_results.csv")
print("Fractal Analysis Result:", result)

# Visualize results
plot_fractal_results(result)
Key Functions

compute_fractal(eeg_path, kmax=12, epoch_len_sec=2.0, out_csv=None, insight_epochs=None):

Returns a dictionary with 'df' (DataFrame with FD and coherence) and 'stats' (t-test results if insight_epochs provided).


plot_fractal_results(result):

Plots distributions of FD and coherence.



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Description").
Push to the branch (git push origin feature-name).
Open a Pull Request with a clear description.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by The Sentience-Field Hypothesis by Mark Rowe Traver (2025).
Built with support from the open-source community, including MNE, NumPy, and nolds.
Thanks to collaborators advancing SFH's neuroscience research.

Contact
For questions or collaboration, contact Mark Rowe Traver at mark.traver@aurora.edu.
