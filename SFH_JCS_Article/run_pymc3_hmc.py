import numpy as np
import pymc3 as pm
import arviz as az

# Simulate synthetic data
np.random.seed(42)
N = 100
y = np.random.normal(loc=0, scale=1, size=N)

# Define and fit model with PyMC3
with pm.Model() as model:
    mu = pm.Normal('mu', mu=0, sd=10)
    sigma = pm.HalfCauchy('sigma', beta=5)
    likelihood = pm.Normal('y', mu=mu, sd=sigma, observed=y)
    trace = pm.sample(chains=4, tune=500, draws=1500, random_seed=42)

# Save trace to file
az.to_netcdf(trace, 'pymc3_hmc_trace.nc')
print("PyMC3 HMC run completed. Trace saved to 'pymc3_hmc_trace.nc'.")
