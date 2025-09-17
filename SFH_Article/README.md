
# Sentient‑Field Tools  

A lightweight, open‑source Python toolbox that implements the quantitative models described in **The Sentient‑Field Hypothesis** (Traver 2025).  
The repository contains four independent modules, each corresponding to a script mentioned in the book:

| Module | Purpose | Main entry point | Typical input | Typical output |
|--------|---------|------------------|---------------|----------------|
| **mcmc_qualic** | Metropolis–Hastings sampling of the qualic‑field posterior (Eq. 8). | `mcmc_qualic.sampler.run_mcmc(...)` | Vector of observed γ‑band power (or any proxy for the qualic variable **Q**). | CSV/HDF5 of posterior draws, trace plots, Gelman‑Rubin diagnostics. |
| **eco_coherence** | Compute the eco‑coherence statistic for directed food‑web networks (Eq. 9). | `eco_coherence.core.compute_ecocoherence(edge_file, weight_col='biomass')` | Edge list CSV (source, target, optional weight). | Scalar eco‑coherence value **Cₑ𝚌ₒ** and optional per‑node contribution series. |
| **cosmo_tuning** | Gradient‑flow optimisation of the cosmological tuning loss (Eq. 10) and comparison with Planck priors. | `cosmo_tuning.optimizer.run_cosmo_tuning(obs_file, lambda_reg=0.01, lr=1e‑3)` | CSV of cosmological observables (Ωₘ, H₀, Ω_Λ, …). | Optimised parameter vector, loss trajectory, diagnostic plots. |
| **neural_fractal** | Higuchi fractal‑dimension analysis of EEG recordings (Protocol 2). | `neural_fractal.higuchi.compute_fractal(eeg_file, kmax=12)` | EDF/MAT/CSV EEG file (≥ 250 Hz). | DataFrame of fractal dimension **D** per channel/epoch, summary statistics, optional topographic map. |

All modules are **pure‑Python** (no compiled extensions) and depend only on widely‑used scientific packages listed in `requirements.txt`.  
Each module ships with:

* a short **demo Jupyter notebook** (`demo.ipynb`) that downloads a public dataset, runs the analysis, and produces the key figure;
* a **unit‑test** in the `tests/` directory (run with `pytest`);
* a **command‑line interface** (`python -m <module_name> …`) for quick batch processing.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your‑username/sentient-field-tools.git
cd sentient-field-tools

# 2. (Recommended) create a fresh virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the package and its dependencies
pip install -r requirements.txt
pip install -e .

Quick usage examples
1. Eco‑coherence (Florida Bay food web)
python -m eco_coherence path/to/florida_bay_edges.csv --weight-col biomass
Copy
Or from Python:

from eco_coherence.core import compute_ecocoherence
C = compute_ecocoherence('data/florida_bay_edges.csv', weight_col='biomass')
print(f'Eco‑coherence = {C:.3f}')
Copy
2. Qualic‑field MCMC (γ‑band power from PhysioNet)
from mcmc_qualic.sampler import run_mcmc
samples = run_mcmc(q_observed, n_iter=20000, burn_in=5000, step_sd=0.2)
samples.to_csv('results/qualic_posterior.csv')
Copy
3. Cosmological tuning
python -m cosmo_tuning data/planck_observables.csv --lambda-reg 0.01 --lr 1e-3
Copy
4. Neural fractal dimension
python -m neural_fractal data/subj01.edf --kmax 12 --out results/subj01_fractal.csv
Copy
Documentation & Reproducibility
Jupyter notebooks (*/demo.ipynb) illustrate end‑to‑end analyses with publicly available data.

Zenodo archive – every tagged release on GitHub is automatically minted with a DOI (e.g., 10.5281/zenodo.XXXXXXX). Cite the software as:

Traver, M. R. (2025). Sentient‑Field Tools – Python implementations of the qualic‑field models (Version 0.1.0) [Software]. Zenodo. https://doi.org/10.xxxx/zenodo.xxxxxx

OSF pre‑registration – the full analysis plan (mixed‑effects model, power calculation, etc.) is registered on OSF (doi:10.17605/OSF.IO/XYZ123). The repository URL is added to the OSF record for full transparency.

Contributing
Fork the repository.
Create a feature branch (git checkout -b my‑feature).
Add/modify code and tests.
Run the full test suite: pytest -v.
Submit a Pull Request.
Please adhere to the NumPy docstring style and keep the public API stable (functions listed in the table above).

License
The code is released under the MIT License.
Data files that are part of the demos are subject to the licenses of the original providers (PhysioNet, EcoBase, Planck Collaboration, BCI Competition IV).

Contact
For bugs, feature requests, or questions about the scientific models, open an issue on GitHub or email the maintainer at m.traver@proton.me.

Happy coding! 🚀

