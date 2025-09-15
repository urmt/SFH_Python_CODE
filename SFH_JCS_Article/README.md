
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
