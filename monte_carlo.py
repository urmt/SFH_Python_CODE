"""
Comprehensive Monte Carlo Sampler for SFH Analysis
=================================================

Advanced Monte Carlo methods for parameter space exploration, uncertainty
quantification, and stochastic sampling in Structural-Functional Harmony research.

This module provides:
- Multi-dimensional parameter space sampling
- Advanced sampling strategies (LHS, Sobol, Halton sequences)
- Adaptive sampling based on fitness landscapes
- Bayesian optimization for parameter tuning
- Markov Chain Monte Carlo (MCMC) for posterior sampling
- Variance reduction techniques
- Convergence diagnostics and quality metrics

Mathematical Foundation:
Monte Carlo methods use random sampling to solve mathematical problems
that might be deterministic in principle. For SFH analysis, we sample
parameter spaces to explore coherence-fertility relationships and
optimize partition-based fitness functions.
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern
import logging
from typing import Dict, List, Tuple, Callable, Optional, Any, Union
import time
from collections import defaultdict
import warnings
from dataclasses import dataclass
import random

# Suppress sklearn warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

logger = logging.getLogger(__name__)

@dataclass
class SamplingConfig:
    """Configuration for Monte Carlo sampling."""
    method: str = "uniform"  # uniform, lhs, sobol, halton, adaptive
    n_samples: int = 1000
    random_state: Optional[int] = None
    convergence_threshold: float = 0.01
    max_iterations: int = 10000
    batch_size: int = 100

@dataclass
class MCMCConfig:
    """Configuration for MCMC sampling."""
    n_chains: int = 4
    n_samples: int = 5000
    n_warmup: int = 1000
    step_size: float = 0.1
    target_acceptance: float = 0.65

class MonteCarloSampler:
    """
    Comprehensive Monte Carlo sampler for parameter space exploration.

    Supports multiple sampling strategies, adaptive methods, and Bayesian
    optimization for efficient exploration of high-dimensional parameter spaces.
    """

    def __init__(self, random_state: Optional[int] = None):
        """
        Initialize Monte Carlo sampler.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        if random_state is not None:
            np.random.seed(random_state)
            random.seed(random_state)

        self.sample_history = []
        self.fitness_history = []
        self.convergence_metrics = {}
        self.sampling_stats = defaultdict(list)

        # Initialize samplers
        self._sobol_generator = None
        self._halton_generators = {}

        logger.info(f"MonteCarloSampler initialized with random_state={random_state}")

    def sample_parameter_space(self, parameter_ranges: Dict[str, List[float]],
                             n_samples: int,
                             method: str = "uniform") -> Dict[str, List[float]]:
        """
        Sample parameter space using specified method.

        Args:
            parameter_ranges: Dict mapping parameter names to [min, max] ranges
            n_samples: Number of samples to generate
            method: Sampling method ('uniform', 'lhs', 'sobol', 'halton')

        Returns:
            Dictionary with parameter names and sampled values
        """
        logger.info(f"Sampling {n_samples} points using {method} method")

        if method == "uniform":
            return self._uniform_sampling(parameter_ranges, n_samples)
        elif method == "lhs":
            return self._latin_hypercube_sampling(parameter_ranges, n_samples)
        elif method == "sobol":
            return self._sobol_sampling(parameter_ranges, n_samples)
        elif method == "halton":
            return self._halton_sampling(parameter_ranges, n_samples)
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _uniform_sampling(self, parameter_ranges: Dict[str, List[float]],
                         n_samples: int) -> Dict[str, List[float]]:
        """Uniform random sampling."""
        samples = {}

        for param_name, (min_val, max_val) in parameter_ranges.items():
            samples[param_name] = np.random.uniform(min_val, max_val, n_samples).tolist()

        return samples

    def _latin_hypercube_sampling(self, parameter_ranges: Dict[str, List[float]],
                                 n_samples: int) -> Dict[str, List[float]]:
        """
        Latin Hypercube Sampling for better space coverage.

        LHS ensures that each parameter is sampled uniformly across its range
        while maintaining good coverage of the multi-dimensional space.
        """
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        # Generate LHS samples in [0,1]^d
        lhs_samples = np.zeros((n_samples, n_params))

        for i in range(n_params):
            # Create permuted indices
            indices = np.random.permutation(n_samples)
            # Generate stratified samples
            uniform_samples = np.random.uniform(0, 1, n_samples)
            lhs_samples[:, i] = (indices + uniform_samples) / n_samples

        # Transform to actual parameter ranges
        samples = {}
        for i, param_name in enumerate(param_names):
            min_val, max_val = parameter_ranges[param_name]
            samples[param_name] = (min_val + lhs_samples[:, i] * (max_val - min_val)).tolist()

        return samples

    def _sobol_sampling(self, parameter_ranges: Dict[str, List[float]],
                       n_samples: int) -> Dict[str, List[float]]:
        """
        Sobol sequence sampling for quasi-random low-discrepancy sampling.

        Sobol sequences provide better uniformity than pseudo-random sequences
        for multi-dimensional integration and parameter space exploration.
        """
        try:
            from scipy.stats import qmc

            param_names = list(parameter_ranges.keys())
            n_params = len(param_names)

            # Generate Sobol sequence
            sampler = qmc.Sobol(d=n_params, scramble=True, seed=self.random_state)
            sobol_samples = sampler.random(n_samples)

            # Transform to actual parameter ranges
            samples = {}
            for i, param_name in enumerate(param_names):
                min_val, max_val = parameter_ranges[param_name]
                samples[param_name] = (min_val + sobol_samples[:, i] * (max_val - min_val)).tolist()

            return samples

        except ImportError:
            logger.warning("SciPy QMC not available, falling back to LHS")
            return self._latin_hypercube_sampling(parameter_ranges, n_samples)

    def _halton_sampling(self, parameter_ranges: Dict[str, List[float]],
                        n_samples: int) -> Dict[str, List[float]]:
        """
        Halton sequence sampling using different prime bases.

        Halton sequences use different prime number bases for each dimension
        to generate low-discrepancy quasi-random sequences.
        """
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        if n_params > len(primes):
            logger.warning(f"Too many parameters ({n_params}) for Halton sequence, using first {len(primes)} primes")
            n_params = len(primes)
            param_names = param_names[:n_params]

        def halton_sequence(n: int, base: int) -> np.ndarray:
            """Generate Halton sequence for given base."""
            sequence = np.zeros(n)
            for i in range(n):
                result = 0.0
                f = 1.0 / base
                index = i + 1
                while index > 0:
                    result += f * (index % base)
                    index //= base
                    f /= base
                sequence[i] = result
            return sequence

        # Generate Halton samples
        samples = {}
        for i, param_name in enumerate(param_names):
            halton_vals = halton_sequence(n_samples, primes[i])
            min_val, max_val = parameter_ranges[param_name]
            samples[param_name] = (min_val + halton_vals * (max_val - min_val)).tolist()

        return samples

    def adaptive_sampling(self, parameter_ranges: Dict[str, List[float]],
                         fitness_function: Callable,
                         config: SamplingConfig) -> Tuple[Dict[str, List[float]], List[float]]:
        """
        Adaptive sampling that focuses on promising regions.

        Uses Gaussian Process regression to model the fitness landscape
        and adaptive sampling to explore high-fitness regions more intensively.

        Args:
            parameter_ranges: Parameter ranges to explore
            fitness_function: Function to evaluate fitness of parameter sets
            config: Sampling configuration

        Returns:
            Tuple of (samples, fitness_values)
        """
        logger.info(f"Starting adaptive sampling with {config.n_samples} target samples")

        param_names = list(parameter_ranges.keys())
        n_dims = len(param_names)

        # Initial samples using LHS
        initial_samples = max(10, n_dims * 2)  # At least 10 or 2*dim samples
        initial_dict = self._latin_hypercube_sampling(parameter_ranges, initial_samples)

        # Convert to array format for GP
        X = np.array([initial_dict[name] for name in param_names]).T
        y = np.array([fitness_function(self._dict_to_params(X[i], param_names, parameter_ranges))
                     for i in range(len(X))])

        # Initialize Gaussian Process
        kernel = RBF(length_scale=1.0) + Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

        all_samples = X.tolist()
        all_fitness = y.tolist()

        # Adaptive sampling loop
        for iteration in range((config.n_samples - initial_samples) // config.batch_size):
            # Fit GP to current data
            gp.fit(X, y)

            # Generate candidate points
            candidates = self._generate_candidates(parameter_ranges, config.batch_size * 10)
            candidate_array = np.array([candidates[name] for name in param_names]).T

            # Predict with uncertainty
            mu, sigma = gp.predict(candidate_array, return_std=True)

            # Acquisition function: Upper Confidence Bound
            beta = 2.0  # Exploration-exploitation trade-off
            acquisition = mu + beta * sigma

            # Select best candidates
            best_indices = np.argsort(acquisition)[-config.batch_size:]
            new_X = candidate_array[best_indices]

            # Evaluate new points
            new_y = np.array([fitness_function(self._dict_to_params(new_X[i], param_names, parameter_ranges))
                             for i in range(len(new_X))])

            # Update data
            X = np.vstack([X, new_X])
            y = np.concatenate([y, new_y])
            all_samples.extend(new_X.tolist())
            all_fitness.extend(new_y.tolist())

            # Check convergence
            if len(all_fitness) > 50:
                recent_improvement = np.std(all_fitness[-20:]) / np.mean(np.abs(all_fitness[-20:]))
                if recent_improvement < config.convergence_threshold:
                    logger.info(f"Converged after {iteration + 1} iterations")
                    break

        # Convert back to dict format
        result_samples = {}
        for i, param_name in enumerate(param_names):
            result_samples[param_name] = [sample[i] for sample in all_samples]

        return result_samples, all_fitness

    def _generate_candidates(self, parameter_ranges: Dict[str, List[float]],
                           n_candidates: int) -> Dict[str, List[float]]:
        """Generate candidate points for adaptive sampling."""
        return self._latin_hypercube_sampling(parameter_ranges, n_candidates)

    def _dict_to_params(self, sample: np.ndarray, param_names: List[str],
                       parameter_ranges: Dict[str, List[float]]) -> Dict[str, float]:
        """Convert array sample to parameter dictionary."""
        return {name: float(sample[i]) for i, name in enumerate(param_names)}

    def mcmc_sampling(self, parameter_ranges: Dict[str, List[float]],
                     log_likelihood: Callable,
                     config: MCMCConfig) -> Dict[str, np.ndarray]:
        """
        Markov Chain Monte Carlo sampling for posterior distributions.

        Implements Metropolis-Hastings algorithm with adaptive step size.

        Args:
            parameter_ranges: Parameter ranges (used as uniform priors)
            log_likelihood: Log-likelihood function
            config: MCMC configuration

        Returns:
            Dictionary with parameter chains
        """
        logger.info(f"Starting MCMC with {config.n_chains} chains, {config.n_samples} samples each")

        param_names = list(parameter_ranges.keys())
        n_params = len(param_names)

        # Initialize chains
        chains = {}
        for param in param_names:
            chains[param] = np.zeros((config.n_chains, config.n_samples))

        acceptance_rates = np.zeros(config.n_chains)

        for chain_idx in range(config.n_chains):
            logger.debug(f"Running chain {chain_idx + 1}/{config.n_chains}")

            # Initialize chain at random point
            current_params = {}
            for param in param_names:
                min_val, max_val = parameter_ranges[param]
                current_params[param] = np.random.uniform(min_val, max_val)

            current_log_likelihood = log_likelihood(current_params)
            step_sizes = {param: config.step_size * (parameter_ranges[param][1] - parameter_ranges[param][0])
                         for param in param_names}

            n_accepted = 0

            for sample_idx in range(config.n_samples):
                # Propose new state
                proposed_params = {}
                for param in param_names:
                    proposed_params[param] = np.random.normal(current_params[param], step_sizes[param])
                    # Reflect if outside bounds
                    min_val, max_val = parameter_ranges[param]
                    if proposed_params[param] < min_val:
                        proposed_params[param] = 2 * min_val - proposed_params[param]
                    elif proposed_params[param] > max_val:
                        proposed_params[param] = 2 * max_val - proposed_params[param]

                # Evaluate proposal
                try:
                    proposed_log_likelihood = log_likelihood(proposed_params)

                    # Accept/reject
                    log_alpha = proposed_log_likelihood - current_log_likelihood
                    if log_alpha > 0 or np.random.random() < np.exp(log_alpha):
                        current_params = proposed_params
                        current_log_likelihood = proposed_log_likelihood
                        n_accepted += 1

                except Exception as e:
                    logger.debug(f"Error evaluating proposal: {e}")
                    # Reject proposal
                    pass

                # Store current state
                for param in param_names:
                    chains[param][chain_idx, sample_idx] = current_params[param]

                # Adaptive step size during warmup
                if sample_idx < config.n_warmup and sample_idx % 100 == 0 and sample_idx > 0:
                    acceptance_rate = n_accepted / (sample_idx + 1)
                    if acceptance_rate < config.target_acceptance - 0.1:
                        for param in param_names:
                            step_sizes[param] *= 0.9
                    elif acceptance_rate > config.target_acceptance + 0.1:
                        for param in param_names:
                            step_sizes[param] *= 1.1

            acceptance_rates[chain_idx] = n_accepted / config.n_samples
            logger.debug(f"Chain {chain_idx + 1} acceptance rate: {acceptance_rates[chain_idx]:.3f}")

        logger.info(f"MCMC completed. Average acceptance rate: {np.mean(acceptance_rates):.3f}")
        return chains

    def importance_sampling(self, parameter_ranges: Dict[str, List[float]],
                           target_distribution: Callable,
                           proposal_distribution: Optional[Callable] = None,
                           n_samples: int = 10000) -> Tuple[Dict[str, List[float]], List[float]]:
        """
        Importance sampling for efficient sampling from complex distributions.

        Args:
            parameter_ranges: Parameter ranges
            target_distribution: Target distribution (unnormalized)
            proposal_distribution: Proposal distribution (default: uniform)
            n_samples: Number of samples

        Returns:
            Tuple of (samples, weights)
        """
        logger.info(f"Importance sampling with {n_samples} samples")

        # Default proposal: uniform distribution
        if proposal_distribution is None:
            samples = self._uniform_sampling(parameter_ranges, n_samples)
            proposal_density = np.ones(n_samples)  # Uniform density
        else:
            samples = proposal_distribution(parameter_ranges, n_samples)
            proposal_density = np.array([1.0] * n_samples)  # Placeholder

        # Calculate importance weights
        param_names = list(parameter_ranges.keys())
        weights = []

        for i in range(n_samples):
            param_dict = {name: samples[name][i] for name in param_names}
            target_val = target_distribution(param_dict)
            weight = target_val / proposal_density[i] if proposal_density[i] > 0 else 0
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]

        return samples, weights

    def bootstrap_sampling(self, original_samples: Dict[str, List[float]],
                          n_bootstrap: int = 1000) -> List[Dict[str, List[float]]]:
        """
        Bootstrap resampling for uncertainty quantification.

        Args:
            original_samples: Original sample set
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of bootstrap sample sets
        """
        param_names = list(original_samples.keys())
        n_original = len(original_samples[param_names[0]])

        bootstrap_samples = []

        for _ in range(n_bootstrap):
            # Random indices with replacement
            indices = np.random.choice(n_original, n_original, replace=True)

            bootstrap_set = {}
            for param in param_names:
                bootstrap_set[param] = [original_samples[param][i] for i in indices]

            bootstrap_samples.append(bootstrap_set)

        return bootstrap_samples

    def calculate_sample_statistics(self, samples: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate comprehensive statistics for sample sets.

        Args:
            samples: Sample dictionary

        Returns:
            Statistics for each parameter
        """
        stats = {}

        for param_name, values in samples.items():
            param_stats = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1),
                'var': np.var(values, ddof=1),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75),
                'iqr': np.percentile(values, 75) - np.percentile(values, 25),
                'skewness': stats.skew(values),
                'kurtosis': stats.kurtosis(values),
                'effective_sample_size': self._calculate_ess(values)
            }
            stats[param_name] = param_stats

        return stats

    def _calculate_ess(self, samples: List[float]) -> float:
        """Calculate effective sample size for MCMC diagnostics."""
        if len(samples) < 10:
            return len(samples)

        # Simple autocorrelation-based ESS estimate
        autocorr = np.correlate(samples, samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # Find first negative value or cutoff at 0.1
        cutoff = 1
        for i in range(1, min(len(autocorr), len(samples)//4)):
            if autocorr[i] < 0.1:
                cutoff = i
                break

        # Integrated autocorrelation time
        tau_int = 1 + 2 * np.sum(autocorr[1:cutoff])
        ess = len(samples) / (2 * tau_int + 1)

        return max(1, ess)

    def convergence_diagnostics(self, chains: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate MCMC convergence diagnostics.

        Args:
            chains: MCMC chains for each parameter

        Returns:
            Convergence metrics (R-hat, effective sample size, etc.)
        """
        diagnostics = {}

        for param_name, param_chains in chains.items():
            n_chains, n_samples = param_chains.shape

            if n_chains < 2:
                diagnostics[param_name] = {'r_hat': 1.0, 'ess': n_samples}
                continue

            # Calculate R-hat (Gelman-Rubin statistic)
            chain_means = np.mean(param_chains, axis=1)
            chain_vars = np.var(param_chains, axis=1, ddof=1)

            overall_mean = np.mean(chain_means)
            between_var = n_samples * np.var(chain_means, ddof=1)
            within_var = np.mean(chain_vars)

            marginal_var = ((n_samples - 1) * within_var + between_var) / n_samples
            r_hat = np.sqrt(marginal_var / within_var) if within_var > 0 else 1.0

            # Effective sample size
            all_samples = param_chains.flatten()
            ess = self._calculate_ess(all_samples.tolist())

            diagnostics[param_name] = {
                'r_hat': r_hat,
                'ess': ess,
                'ess_per_chain': ess / n_chains
            }

        return diagnostics

    def get_sampling_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of sampling performance."""
        return {
            'total_samples_generated': len(self.sample_history),
            'sampling_methods_used': list(set(self.sampling_stats.keys())),
            'convergence_metrics': self.convergence_metrics.copy(),
            'performance_stats': dict(self.sampling_stats)
        }
