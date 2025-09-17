markdown# mcmc_qualic: Metropolis-Hastings Sampler for the Sentience-Field Hypothesis

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-green.svg)
![Last Updated](https://img.shields.io/badge/last_updated-Sep_17_2025-orange.svg)

Welcome to **mcmc_qualic**, a Python package implementing a Metropolis-Hastings Markov Chain Monte Carlo (MCMC) sampler tailored for statistical modeling of qualic energy within the framework of the **Sentience-Field Hypothesis (SFH)**. This tool supports empirical validation of SFH, particularly for Protocol 2 (neural fractal-dimension shifts during insight), and is designed for researchers in physics, neuroscience, and consciousness studies.

## Overview

The Sentience-Field Hypothesis, proposed by Mark Rowe Traver in *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality* (IngramSpark, 2025, ISBN: 978-0-123456-78-9), posits that consciousness is a fundamental, non-local field generating discrete qualic partitions optimized via a coherence-fertility functional. This package provides a robust MCMC implementation to estimate qualic energy parameters (mean \(\mu\) and intrinsic spread \(\sigma_q\)) from observed data, such as \(\gamma\)-band power from EEG, aligning with SFH's interdisciplinary applications.

Key features include:
- Adaptive step size tuning for efficient sampling.
- Multiple chain support with convergence diagnostics (ESS and R-hat).
- A coherence proxy to bias initialization toward SFH's theoretical constructs.
- Open-source, reproducible code hosted at [https://github.com/urmt/SFH_Python_CODE](https://github.com/urmt/SFH_Python_CODE).

## Installation

To install `mcmc_qualic`, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/urmt/SFH_Python_CODE.git
   cd SFH_Python_CODE/SFH_JCS_Article/Sentient-Field_Tools/mcmc_qualic

Install Dependencies:
Ensure you have Python 3.11 and the required packages:
bashpip install numpy pandas scipy
For advanced diagnostics and visualization (optional), install:
bashpip install arviz matplotlib

Install the Package:
Install locally for use in your projects:
bashpip install .


Usage
Basic Example
The following example demonstrates how to use mcmc_qualic to sample from synthetic qualic data, mimicking Protocol 2's neural insight analysis:
pythonimport numpy as np
from mcmc_qualic import run_mcmc

# Simulate synthetic gamma-band power data
np.random.seed(42)
q_obs = np.random.normal(2.0, 0.5, 100)

# Run MCMC with default settings
result = run_mcmc(q_obs, n_iter=20000, burn_in=5000, n_chains=4)

# Access samples and diagnostics
samples = result['samples']
diagnostics = result['diagnostics']
print("Diagnostics:", diagnostics)

# Visualize with diagnostics module (optional)
from mcmc_qualic.diagnostics import plot_trace
plot_trace(result)
Key Functions

run_mcmc(q_obs, n_iter=20000, burn_in=5000, target_accept=0.3, n_chains=4, seed=None):

Returns a dictionary with 'samples' (MultiIndex DataFrame) and 'diagnostics' (ESS, R-hat, etc.).


diagnostics.plot_trace(result): Plots trace and histograms using ArviZ.
diagnostics.gelman_rubin(result): Computes Gelman-Rubin R-hat statistic.

Contributing
Contributions to enhance mcmc_qualic are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Description").
Push to the branch (git push origin feature-name).
Open a Pull Request with a clear description.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by the theoretical work in The Sentience-Field Hypothesis by Mark Rowe Traver (2025).
Built with support from the open-source community, including NumPy, SciPy, and ArviZ.
Special thanks to reviewers and collaborators advancing SFH research.

Contact
For questions or collaboration, contact Mark Rowe Traver at mark.traver@aurora.edu.
