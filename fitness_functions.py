"""
Fitness Calculator Module for SFH Master Framework
==================================================

This module implements comprehensive fitness calculations for Structural-Functional Harmony analysis,
including coherence metrics, fertility measures, and combined SFH scoring.

Author: SFH Research Team
Version: 1.0.0
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize_scalar
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class FitnessCalculator:
    """
    Advanced fitness calculator for Structural-Functional Harmony analysis.

    This class implements sophisticated metrics for evaluating the harmony between
    structural mathematical properties (like integer partitions) and functional
    parameters in complex systems.
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FitnessCalculator with configuration parameters.

        Args:
            config: Configuration dictionary containing fitness parameters and weights
        """
        self.config = config or {}

        # Coherence component weights
        self.coherence_weights = self.config.get('coherence_weights', {
            'structural': 0.4,
            'functional': 0.4,
            'harmony': 0.2
        })

        # Fertility component weights
        self.fertility_weights = self.config.get('fertility_weights', {
            'diversity': 0.5,
            'complexity': 0.3,
            'adaptability': 0.2
        })

        # SFH combination parameters
        self.sfh_combination_method = self.config.get('sfh_combination', 'harmonic_mean')
        self.normalize_metrics = self.config.get('normalize_metrics', True)

        # Thresholds for quality assessment
        self.quality_thresholds = self.config.get('quality_thresholds', {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.2
        })

        # Cache for expensive calculations
        self._calculation_cache = {}
        self.cache_enabled = self.config.get('enable_cache', True)

        logger.info(f"✅ FitnessCalculator initialized")
        logger.info(f"   Coherence weights: {self.coherence_weights}")
        logger.info(f"   Fertility weights: {self.fertility_weights}")
        logger.info(f"   Combination method: {self.sfh_combination_method}")

    def _parse_fitness_args(self, args, kwargs, method_name: str):
        """
        Flexible argument parser for fitness calculation methods.

        Handles multiple calling patterns that your framework might use.
        """
        n = None
        partition_data = None
        mc_sample = None
        weight = 1.0

        # Try to extract from kwargs first
        n = kwargs.get('n', None)
        partition_data = kwargs.get('partition_data', None)
        mc_sample = kwargs.get('mc_sample', None)
        weight = kwargs.get('weight', kwargs.get('resonance_factor', 1.0))  # Handle different weight param names

        # Fill in missing values from args
        if len(args) >= 1 and n is None:
            n_arg = args[0]
            # Handle case where n is passed as a list
            if isinstance(n_arg, (list, tuple)) and len(n_arg) > 0:
                n = n_arg[0]
            else:
                n = n_arg
        if len(args) >= 2 and partition_data is None:
            partition_data = args[1]
        if len(args) >= 3 and mc_sample is None:
            mc_sample = args[2]
        if len(args) >= 4:
            weight = args[3]

        # Provide defaults for missing required parameters
        if partition_data is None:
            logger.warning(f"{method_name}: partition_data missing, using default empty dict")
            partition_data = {}
        if mc_sample is None:
            logger.warning(f"{method_name}: mc_sample missing, using default [0.5, 0.5, 0.5]")
            mc_sample = np.array([0.5, 0.5, 0.5])
        if n is None:
            logger.warning(f"{method_name}: n missing, using default value 10")
            n = 10

        # Ensure proper types
        try:
            n = int(n)
            weight = float(weight)
            if not isinstance(partition_data, dict):
                partition_data = {}
            if not isinstance(mc_sample, np.ndarray):
                mc_sample = np.array(mc_sample) if hasattr(mc_sample, '__iter__') else np.array([0.5, 0.5, 0.5])
        except Exception as e:
            logger.error(f"{method_name}: Error converting parameter types: {e}")
            # Use safe defaults
            n = 10
            weight = 1.0
            partition_data = {}
            mc_sample = np.array([0.5, 0.5, 0.5])

        logger.debug(f"{method_name} parsed args: n={n}, weight={weight}, "
                    f"partition_keys={list(partition_data.keys())}, mc_sample_shape={mc_sample.shape}")

        return n, partition_data, mc_sample, weight

    def _cache_key(self, n: int, mc_sample: np.ndarray) -> str:
        """Generate cache key for calculations."""
        return f"{n}_{hash(tuple(mc_sample))}"

    def calculate_coherence(self, *args, **kwargs) -> float:
        """
        Calculate coherence metric based on structural-functional alignment.

        Flexible method that can handle multiple calling patterns:
        - calculate_coherence(n, partition_data, mc_sample, weight=1.0)
        - calculate_coherence(n=n, partition_data=partition_data, mc_sample=mc_sample)
        - Other variations your framework might use

        Returns:
            float: Coherence score between 0 and 1
        """
        try:
            # Parse arguments flexibly
            n, partition_data, mc_sample, weight = self._parse_fitness_args(args, kwargs, 'coherence')
            return self._calculate_coherence_core(n, partition_data, mc_sample, weight)
        except Exception as e:
            logger.error(f"❌ Error parsing coherence arguments: {e}")
            logger.error(f"   Args: {args}")
            logger.error(f"   Kwargs: {kwargs}")
            return 0.0

    def _calculate_coherence_core(self, n: int, partition_data: Dict, mc_sample: np.ndarray, weight: float) -> float:
        """Core coherence calculation logic."""
        try:
            # Check cache first
            cache_key = self._cache_key(n, mc_sample) + "_coherence"
            if self.cache_enabled and cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key] * weight

            # Extract parameters from Monte Carlo sample
            alpha, beta, gamma = mc_sample[:3] if len(mc_sample) >= 3 else [0, 0, 0]

            # Get partition information with safety checks
            exact_partitions = max(1, partition_data.get('exact', 1))
            hr_approximation = max(1, partition_data.get('hr_approx', 1))
            error = min(1.0, max(0.0, partition_data.get('error', 1.0)))

            # 1. Structural coherence: Partition approximation quality
            structural_coherence = max(0, 1 - error)

            # 2. Functional coherence: Parameter harmony and balance
            # Balance between alpha and beta (structural vs functional emphasis)
            param_sum = max(alpha + beta, 1e-10)
            param_balance = 1 - abs(alpha - beta) / param_sum

            # Gamma as stability/harmony factor
            gamma_stability = min(1.0, gamma / max(param_sum, 1e-10))

            # Parameter distribution quality (not too extreme)
            param_variance = np.var([alpha, beta, gamma])
            param_stability = 1 / (1 + param_variance)

            functional_coherence = (param_balance + gamma_stability + param_stability) / 3

            # 3. Harmony coherence: Integration of structural and functional
            # Partition complexity measure
            partition_complexity = np.log(exact_partitions) / np.log(max(n, 2))

            # Parameter complexity measure
            param_complexity = np.sqrt(alpha**2 + beta**2 + gamma**2) / np.sqrt(3)

            # Resonance between partition and parameter complexity
            complexity_resonance = 1 / (1 + abs(partition_complexity - param_complexity))

            # Scale-dependent harmony
            scale_factor = np.log(max(n, 2))
            scale_harmony = min(1.0, (alpha + beta + gamma) / scale_factor)

            harmony_coherence = (complexity_resonance + scale_harmony) / 2

            # Weighted combination of coherence components
            coherence = (
                self.coherence_weights['structural'] * structural_coherence +
                self.coherence_weights['functional'] * functional_coherence +
                self.coherence_weights['harmony'] * harmony_coherence
            )

            # Apply weight factor to final result if provided
            coherence = coherence * weight

            # Normalize and bound the result
            coherence = max(0, min(1, coherence))

            # Cache the result (before weight is applied)
            if self.cache_enabled:
                self._calculation_cache[cache_key] = coherence / weight if weight != 0 else coherence

            logger.debug(f"Coherence for n={n}: structural={structural_coherence:.4f}, "
                        f"functional={functional_coherence:.4f}, harmony={harmony_coherence:.4f}, "
                        f"total={coherence:.4f}")

            return coherence

        except Exception as e:
            logger.error(f"❌ Error calculating coherence for n={n}: {e}")
            return 0.0

    def calculate_fertility(self, *args, **kwargs) -> float:
        """
        Calculate fertility metric based on generative and adaptive potential.

        Flexible method that can handle multiple calling patterns.

        Returns:
            float: Fertility score between 0 and 1
        """
        try:
            # Parse arguments flexibly
            n, partition_data, mc_sample, weight = self._parse_fitness_args(args, kwargs, 'fertility')
            return self._calculate_fertility_core(n, partition_data, mc_sample, weight)
        except Exception as e:
            logger.error(f"❌ Error parsing fertility arguments: {e}")
            logger.error(f"   Args: {args}")
            logger.error(f"   Kwargs: {kwargs}")
            return 0.0

    def _calculate_fertility_core(self, n: int, partition_data: Dict, mc_sample: np.ndarray, weight: float) -> float:
        """Core fertility calculation logic."""
        try:
            # Check cache first
            cache_key = self._cache_key(n, mc_sample) + "_fertility"
            if self.cache_enabled and cache_key in self._calculation_cache:
                return self._calculation_cache[cache_key] * weight

            alpha, beta, gamma = mc_sample[:3] if len(mc_sample) >= 3 else [0, 0, 0]
            exact_partitions = max(1, partition_data.get('exact', 1))

            # 1. Diversity: Partition richness and variety
            # Logarithmic scaling of partition count
            max_theoretical_diversity = np.log(exact_partitions) / np.log(max(n, 2))
            diversity = min(1.0, max_theoretical_diversity)

            # Parameter diversity contribution
            param_entropy = -sum([p * np.log(max(p, 1e-10)) for p in [alpha, beta, gamma]
                                if p > 0]) / np.log(3)
            diversity_enhanced = (diversity + param_entropy) / 2

            # 2. Complexity: Multi-scale parameter interactions
            # Linear parameter interactions
            linear_complexity = (alpha + beta + gamma) / 3

            # Non-linear parameter interactions
            interaction_terms = [
                alpha * beta, beta * gamma, alpha * gamma,
                alpha * beta * gamma
            ]
            nonlinear_complexity = np.mean(interaction_terms) if any(interaction_terms) else 0

            # Combine linear and non-linear complexity
            total_complexity = (linear_complexity + nonlinear_complexity) / 2
            complexity = min(1.0, total_complexity)

            # 3. Adaptability: Scale and condition responsiveness
            # Scale adaptability
            scale_factor = np.log(max(n, 2))
            scale_adaptability = min(1.0, (alpha + beta + gamma) / scale_factor)

            # Parameter flexibility (not too rigid, not too chaotic)
            param_std = np.std([alpha, beta, gamma])
            param_mean = np.mean([alpha, beta, gamma])
            flexibility = 1 - abs(param_std - param_mean/2) / max(param_mean, 1e-10)
            flexibility = max(0, min(1, flexibility))

            # Robustness to perturbations
            param_stability = 1 / (1 + np.var([alpha, beta, gamma]))

            adaptability = (scale_adaptability + flexibility + param_stability) / 3

            # Weighted combination of fertility components
            fertility = (
                self.fertility_weights['diversity'] * diversity_enhanced +
                self.fertility_weights['complexity'] * complexity +
                self.fertility_weights['adaptability'] * adaptability
            )

            # Apply weight factor to final result if provided
            fertility = fertility * weight

            # Normalize and bound the result
            fertility = max(0, min(1, fertility))

            # Cache the result (before weight is applied)
            if self.cache_enabled:
                self._calculation_cache[cache_key] = fertility / weight if weight != 0 else fertility

            logger.debug(f"Fertility for n={n}: diversity={diversity_enhanced:.4f}, "
                        f"complexity={complexity:.4f}, adaptability={adaptability:.4f}, "
                        f"total={fertility:.4f}")

            return fertility

        except Exception as e:
            logger.error(f"❌ Error calculating fertility for n={n}: {e}")
            return 0.0

    def calculate_sfh_metric(self, *args, **kwargs) -> Dict[str, float]:
        """
        Calculate complete SFH (Structural-Functional Harmony) metric.

        Flexible method that can handle multiple calling patterns.

        Returns:
            Dict containing coherence, fertility, combined SFH scores, and metadata
        """
        try:
            # Parse arguments flexibly
            n, partition_data, mc_sample, weight = self._parse_fitness_args(args, kwargs, 'sfh_metric')
        except Exception as e:
            logger.error(f"❌ Error parsing SFH metric arguments: {e}")
            logger.error(f"   Args: {args}")
            logger.error(f"   Kwargs: {kwargs}")
            return self._create_error_result(0, np.array([0, 0, 0]), str(e))

        try:
            # Calculate individual metrics using parsed arguments
            coherence = self._calculate_coherence_core(n, partition_data, mc_sample, weight)
            fertility = self._calculate_fertility_core(n, partition_data, mc_sample, weight)

            # Calculate combined SFH metric using specified method
            if self.sfh_combination_method == 'harmonic_mean':
                # Harmonic mean emphasizes balance (penalizes extreme imbalances)
                if coherence > 0 and fertility > 0:
                    sfh_combined = 2 * (coherence * fertility) / (coherence + fertility)
                else:
                    sfh_combined = 0.0
            elif self.sfh_combination_method == 'geometric_mean':
                # Geometric mean for multiplicative effects
                sfh_combined = np.sqrt(coherence * fertility)
            elif self.sfh_combination_method == 'arithmetic_mean':
                # Simple average
                sfh_combined = (coherence + fertility) / 2
            elif self.sfh_combination_method == 'weighted_mean':
                # Weighted combination (coherence slightly favored for stability)
                weights = self.config.get('sfh_weights', {'coherence': 0.6, 'fertility': 0.4})
                sfh_combined = (weights['coherence'] * coherence +
                               weights['fertility'] * fertility)
            else:
                # Default to harmonic mean
                if coherence > 0 and fertility > 0:
                    sfh_combined = 2 * (coherence * fertility) / (coherence + fertility)
                else:
                    sfh_combined = 0.0

            # Quality assessment
            quality_level = self._assess_quality(sfh_combined)

            # Additional derived metrics
            balance_ratio = min(coherence, fertility) / max(coherence, fertility, 1e-10)
            synergy_factor = sfh_combined - max(coherence, fertility)  # How much synergy adds

            # Construct comprehensive results
            results = {
                # Primary metrics
                'coherence': coherence,
                'fertility': fertility,
                'sfh_combined': sfh_combined,

                # Metadata
                'n': n,
                'parameters': mc_sample.tolist(),
                'quality_level': quality_level,

                # Derived metrics
                'balance_ratio': balance_ratio,
                'synergy_factor': synergy_factor,
                'combination_method': self.sfh_combination_method,

                # Partition context
                'exact_partitions': partition_data.get('exact', 0),
                'partition_error': partition_data.get('error', 1.0)
            }

            logger.info(f"🧬 SFH metrics for n={n}: coherence={coherence:.4f}, "
                       f"fertility={fertility:.4f}, combined={sfh_combined:.4f} ({quality_level})")

            return results

        except Exception as e:
            logger.error(f"❌ Error calculating SFH metrics for n={n}: {e}")
            return self._create_error_result(n, mc_sample, str(e))

    def calculate_fitness(self, *args, **kwargs) -> float:
        """
        General fitness calculation method (alias for combined fitness).

        Returns:
            float: Fitness score between 0 and 1
        """
        return self.calculate_combined_fitness(*args, **kwargs)

    def fitness(self, *args, **kwargs) -> float:
        """
        Short alias for fitness calculation.

        Returns:
            float: Fitness score between 0 and 1
        """
        return self.calculate_combined_fitness(*args, **kwargs)

    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Comprehensive evaluation method (alias for SFH metric calculation).

        Returns:
            Dict: Full evaluation results including all metrics
        """
        return self.calculate_sfh_metric(*args, **kwargs)

    def calculate_combined_fitness(self, *args, **kwargs) -> float:
        """
        Calculate combined fitness metric (alias for SFH combined score).

        This method provides compatibility with frameworks expecting a single
        combined fitness value rather than the full SFH metric dictionary.

        Returns:
            float: Combined fitness score between 0 and 1
        """
        try:
            # Get the full SFH metric
            sfh_result = self.calculate_sfh_metric(*args, **kwargs)

            # Return just the combined score
            combined_score = sfh_result.get('sfh_combined', 0.0)

            logger.debug(f"Combined fitness calculated: {combined_score:.4f}")
            return combined_score

        except Exception as e:
            logger.error(f"❌ Error calculating combined fitness: {e}")
            logger.error(f"   Args: {args}")
            logger.error(f"   Kwargs: {kwargs}")
            return 0.0

    def _assess_quality(self, sfh_score: float) -> str:
        """Assess quality level based on SFH score."""
        if sfh_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif sfh_score >= self.quality_thresholds['good']:
            return 'good'
        elif sfh_score >= self.quality_thresholds['fair']:
            return 'fair'
        elif sfh_score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'very_poor'

    def _create_error_result(self, n: int, mc_sample: np.ndarray, error_msg: str) -> Dict[str, float]:
        """Create standardized error result."""
        return {
            'coherence': 0.0,
            'fertility': 0.0,
            'sfh_combined': 0.0,
            'n': n,
            'parameters': mc_sample.tolist(),
            'quality_level': 'error',
            'balance_ratio': 0.0,
            'synergy_factor': 0.0,
            'error': error_msg
        }

    def batch_calculate_sfh(self, n_values: List[int], partition_results: Dict,
                           mc_samples: np.ndarray) -> List[Dict[str, Any]]:
        """
        Calculate SFH metrics for multiple n values and parameter combinations.

        Args:
            n_values: List of integer values to analyze
            partition_results: Dictionary mapping n values to partition data
            mc_samples: Array of Monte Carlo samples

        Returns:
            List of SFH metric dictionaries
        """
        results = []
        total_combinations = len(n_values) * len(mc_samples)

        logger.info(f"🔄 Starting batch SFH calculation: {total_combinations} combinations")

        for i, n in enumerate(n_values):
            partition_data = partition_results.get(n, {})

            for j, mc_sample in enumerate(mc_samples):
                try:
                    sfh_result = self.calculate_sfh_metric(n, partition_data, mc_sample)
                    sfh_result['batch_index'] = len(results)
                    sfh_result['n_index'] = i
                    sfh_result['sample_index'] = j
                    results.append(sfh_result)

                except Exception as e:
                    logger.error(f"❌ Error in batch calculation for n={n}, sample {j}: {e}")
                    results.append(self._create_error_result(n, mc_sample, str(e)))

        logger.info(f"✅ Batch calculation completed: {len(results)} results")
        return results

    def analyze_fitness_landscape(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive analysis of the fitness landscape from SFH calculations.

        Args:
            results: List of SFH metric dictionaries from batch calculations

        Returns:
            Dict containing detailed landscape analysis
        """
        if not results:
            logger.warning("⚠️ No results provided for landscape analysis")
            return {'error': 'No results to analyze', 'n_samples': 0}

        try:
            # Extract metric arrays
            coherence_values = np.array([r.get('coherence', 0) for r in results])
            fertility_values = np.array([r.get('fertility', 0) for r in results])
            sfh_values = np.array([r.get('sfh_combined', 0) for r in results])
            n_values = np.array([r.get('n', 0) for r in results])

            # Basic statistics
            landscape_analysis = {
                'coherence_stats': self._calculate_detailed_stats(coherence_values),
                'fertility_stats': self._calculate_detailed_stats(fertility_values),
                'sfh_stats': self._calculate_detailed_stats(sfh_values),
                'n_samples': len(results)
            }

            # Correlation analysis
            try:
                correlations = {
                    'coherence_fertility': np.corrcoef(coherence_values, fertility_values)[0, 1],
                    'coherence_sfh': np.corrcoef(coherence_values, sfh_values)[0, 1],
                    'fertility_sfh': np.corrcoef(fertility_values, sfh_values)[0, 1],
                    'n_coherence': np.corrcoef(n_values, coherence_values)[0, 1],
                    'n_fertility': np.corrcoef(n_values, fertility_values)[0, 1],
                    'n_sfh': np.corrcoef(n_values, sfh_values)[0, 1]
                }
                landscape_analysis['correlations'] = correlations
            except Exception as e:
                logger.warning(f"⚠️ Could not calculate correlations: {e}")
                landscape_analysis['correlations'] = {}

            # Optimal points analysis
            optimal_indices = {
                'best_coherence': np.argmax(coherence_values),
                'best_fertility': np.argmax(fertility_values),
                'best_sfh': np.argmax(sfh_values),
                'most_balanced': np.argmin(np.abs(coherence_values - fertility_values))
            }

            landscape_analysis['optimal_points'] = {}
            for point_type, idx in optimal_indices.items():
                landscape_analysis['optimal_points'][point_type] = {
                    'result': results[idx],
                    'index': int(idx)
                }

            # Quality distribution
            quality_levels = [r.get('quality_level', 'unknown') for r in results]
            quality_counts = {}
            for level in ['excellent', 'good', 'fair', 'poor', 'very_poor', 'error', 'unknown']:
                quality_counts[level] = quality_levels.count(level)

            landscape_analysis['quality_distribution'] = quality_counts

            # Parameter sensitivity analysis (if parameter data available)
            try:
                parameters = np.array([r.get('parameters', [0, 0, 0])[:3] for r in results])
                if parameters.shape[1] >= 3:
                    param_sensitivity = {
                        'alpha_sfh_corr': np.corrcoef(parameters[:, 0], sfh_values)[0, 1],
                        'beta_sfh_corr': np.corrcoef(parameters[:, 1], sfh_values)[0, 1],
                        'gamma_sfh_corr': np.corrcoef(parameters[:, 2], sfh_values)[0, 1]
                    }
                    landscape_analysis['parameter_sensitivity'] = param_sensitivity
            except Exception as e:
                logger.debug(f"Parameter sensitivity analysis failed: {e}")

            # Landscape characteristics
            landscape_analysis['landscape_characteristics'] = {
                'sfh_range': float(np.max(sfh_values) - np.min(sfh_values)),
                'sfh_variance': float(np.var(sfh_values)),
                'high_performance_ratio': float(np.mean(sfh_values > 0.7)),
                'coherence_fertility_balance': float(np.mean(np.abs(coherence_values - fertility_values)))
            }

            logger.info(f"🔍 Fitness landscape analyzed: {len(results)} samples")
            logger.info(f"   Best SFH: {np.max(sfh_values):.4f}")
            logger.info(f"   Mean SFH: {np.mean(sfh_values):.4f}")
            logger.info(f"   High performance ratio: {landscape_analysis['landscape_characteristics']['high_performance_ratio']:.3f}")

            return landscape_analysis

        except Exception as e:
            logger.error(f"❌ Error in fitness landscape analysis: {e}")
            return {'error': f'Analysis failed: {str(e)}', 'n_samples': len(results)}

    def _calculate_detailed_stats(self, values: np.ndarray) -> Dict[str, float]:
        """Calculate detailed statistics for a value array."""
        if len(values) == 0:
            return {'error': 'No values provided'}

        try:
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75)),
                'skewness': float(stats.skew(values)) if len(values) > 2 else 0.0,
                'kurtosis': float(stats.kurtosis(values)) if len(values) > 3 else 0.0
            }
        except Exception as e:
            logger.warning(f"⚠️ Error calculating detailed stats: {e}")
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }

    def get_top_performers(self, results: List[Dict], n_top: int = 10,
                          metric: str = 'sfh_combined') -> List[Dict]:
        """
        Get the top performing results based on specified metric.

        Args:
            results: List of SFH calculation results
            n_top: Number of top performers to return
            metric: Metric to use for ranking ('sfh_combined', 'coherence', 'fertility')

        Returns:
            List of top performing results, sorted by metric value
        """
        if not results:
            return []

        try:
            # Sort by specified metric
            sorted_results = sorted(results,
                                  key=lambda x: x.get(metric, 0),
                                  reverse=True)

            top_performers = sorted_results[:n_top]

            logger.info(f"🏆 Top {len(top_performers)} performers by {metric}:")
            for i, result in enumerate(top_performers[:5]):  # Log top 5
                logger.info(f"   {i+1}. n={result.get('n', '?')}, "
                           f"{metric}={result.get(metric, 0):.4f}")

            return top_performers

        except Exception as e:
            logger.error(f"❌ Error getting top performers: {e}")
            return []

    def clear_cache(self):
        """Clear the calculation cache."""
        self._calculation_cache.clear()
        logger.info("🧹 Calculation cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the calculation cache."""
        return {
            'cache_size': len(self._calculation_cache),
            'cache_enabled': self.cache_enabled
        }
