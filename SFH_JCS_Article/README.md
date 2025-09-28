## New Computational Tools for SFH
- `gaussian.stan`: Stan model file for Hamiltonian Monte Carlo (HMC) with No-U-Turn Sampler (NUTS).
- `run_stan_hmc.py`: Python script to run the Stan model using `cmdstanpy`.
- `run_pymc3_hmc.py`: Python script to run Bayesian modeling with PyMC3 using NUTS.
- `mcmc_convergence.png`: Updated figure showing trace plots and \(\hat{R}\) convergence diagnostic.

### Setup Instructions
1. Install dependencies:
   - For Stan: `pip install cmdstanpy` and install CmdStan[](https://cmdstanpy.readthedocs.io/).
   - For PyMC3: `pip install pymc3 arviz`.
2. Run `run_stan_hmc.py` or `run_pymc3_hmc.py` to generate results.
3. Use the provided Python script to generate `mcmc_convergence.png`.

### Usage
These scripts enhance MCMC simulations for SFH, supporting qualic evolution and fine-tuning analysis as described in the journal article.
