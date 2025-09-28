import numpy as np
import cmdstanpy

# Simulate synthetic data
np.random.seed(42)
N = 100
y = np.random.normal(loc=0, scale=1, size=N)

# Compile and run Stan model
model = cmdstanpy.CmdStanModel(stan_file='gaussian.stan')
data = {'N': N, 'y': y.tolist()}
fit = model.sample(data=data, chains=4, parallel_chains=4, iter_warmup=500, iter_sampling=1500)

# Save summary to file
fit.summary().to_csv('stan_hmc_summary.csv')
print("Stan HMC run completed. Summary saved to 'stan_hmc_summary.csv'.")
