#!/usr/bin/env python3
"""
Advanced Statistical Analysis System for SFH Partition Research
Provides comprehensive statistical tools for mathematical partition analysis
"""

import math
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy import stats
from scipy.stats import kstest, normaltest, shapiro, anderson, jarque_bera
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """Advanced statistical analyzer for partition theory research."""

    def __init__(self, random_seed=None, confidence_level=0.95):
        """Initialize statistical analyzer with configuration."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def comprehensive_distribution_analysis(self, data: List[float],
                                          distribution_name: str = "unknown") -> Dict[str, Any]:
        """Perform comprehensive statistical analysis of a distribution."""
        if len(data) < 3:
            return {"error": "Insufficient data for statistical analysis"}

        data_array = np.array(data, dtype=float)
        n = len(data_array)

        analysis = {
            "distribution_name": distribution_name,
            "sample_size": n,
            "descriptive_statistics": self._calculate_descriptive_statistics(data_array),
            "normality_tests": self._perform_normality_tests(data_array),
            "distribution_fitting": self._fit_theoretical_distributions(data_array),
            "confidence_intervals": self._calculate_confidence_intervals(data_array),
            "outlier_analysis": self._detect_outliers(data_array),
            "goodness_of_fit": self._goodness_of_fit_tests(data_array)
        }

        return analysis

    def _calculate_descriptive_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive descriptive statistics."""
        try:
            stats_dict = {
                "mean": float(np.mean(data)),
                "median": float(np.median(data)),
                "mode": float(stats.mode(data, keepdims=True)[0][0]) if len(data) > 1 else float(data[0]),
                "std_dev": float(np.std(data, ddof=1)),
                "variance": float(np.var(data, ddof=1)),
                "skewness": float(stats.skew(data)),
                "kurtosis": float(stats.kurtosis(data)),
                "range": float(np.max(data) - np.min(data)),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "q1": float(np.percentile(data, 25)),
                "q3": float(np.percentile(data, 75)),
                "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
                "cv": float(np.std(data, ddof=1) / np.mean(data)) if np.mean(data) != 0 else 0,
                "standard_error": float(np.std(data, ddof=1) / np.sqrt(len(data)))
            }

            # Add percentiles
            percentiles = [5, 10, 90, 95, 99]
            for p in percentiles:
                stats_dict[f"p{p}"] = float(np.percentile(data, p))

            return stats_dict

        except Exception as e:
            return {"error": f"Error calculating descriptive statistics: {e}"}

    def _perform_normality_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform multiple normality tests."""
        if len(data) < 8:
            return {"error": "Insufficient data for normality tests"}

        normality_results = {}

        try:
            # Shapiro-Wilk test (best for n < 5000)
            if len(data) <= 5000:
                shapiro_stat, shapiro_p = shapiro(data)
                normality_results["shapiro_wilk"] = {
                    "statistic": float(shapiro_stat),
                    "p_value": float(shapiro_p),
                    "is_normal": shapiro_p > self.alpha,
                    "interpretation": "Normal" if shapiro_p > self.alpha else "Non-normal"
                }

            # D'Agostino and Pearson's test
            if len(data) >= 20:
                dagostino_stat, dagostino_p = normaltest(data)
                normality_results["dagostino_pearson"] = {
                    "statistic": float(dagostino_stat),
                    "p_value": float(dagostino_p),
                    "is_normal": dagostino_p > self.alpha,
                    "interpretation": "Normal" if dagostino_p > self.alpha else "Non-normal"
                }

            # Anderson-Darling test
            anderson_result = anderson(data, dist='norm')
            critical_values = anderson_result.critical_values
            significance_levels = anderson_result.significance_levels

            # Find appropriate critical value for our confidence level
            alpha_percent = self.alpha * 100
            critical_idx = np.argmin(np.abs(significance_levels - alpha_percent))
            critical_value = critical_values[critical_idx]

            normality_results["anderson_darling"] = {
                "statistic": float(anderson_result.statistic),
                "critical_value": float(critical_value),
                "significance_level": float(significance_levels[critical_idx]),
                "is_normal": anderson_result.statistic < critical_value,
                "interpretation": "Normal" if anderson_result.statistic < critical_value else "Non-normal"
            }

            # Jarque-Bera test
            if len(data) >= 2000:
                jb_stat, jb_p = jarque_bera(data)
                normality_results["jarque_bera"] = {
                    "statistic": float(jb_stat),
                    "p_value": float(jb_p),
                    "is_normal": jb_p > self.alpha,
                    "interpretation": "Normal" if jb_p > self.alpha else "Non-normal"
                }

        except Exception as e:
            normality_results["error"] = f"Error in normality tests: {e}"

        return normality_results

    def _fit_theoretical_distributions(self, data: np.ndarray) -> Dict[str, Any]:
        """Fit data to various theoretical distributions and compare."""
        distributions_to_fit = [
            ('normal', stats.norm),
            ('exponential', stats.expon),
            ('gamma', stats.gamma),
            ('lognormal', stats.lognorm),
            ('weibull', stats.weibull_min),
            ('beta', stats.beta),
            ('uniform', stats.uniform)
        ]

        fitting_results = {}

        for name, distribution in distributions_to_fit:
            try:
                # Fit parameters
                params = distribution.fit(data)

                # Calculate log-likelihood
                log_likelihood = np.sum(distribution.logpdf(data, *params))

                # Calculate AIC and BIC
                k = len(params)  # number of parameters
                n = len(data)
                aic = 2 * k - 2 * log_likelihood
                bic = k * np.log(n) - 2 * log_likelihood

                # Kolmogorov-Smirnov test
                ks_stat, ks_p = kstest(data, lambda x: distribution.cdf(x, *params))

                fitting_results[name] = {
                    "parameters": [float(p) for p in params],
                    "log_likelihood": float(log_likelihood),
                    "aic": float(aic),
                    "bic": float(bic),
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "good_fit": ks_p > self.alpha
                }

            except Exception as e:
                fitting_results[name] = {"error": f"Failed to fit {name}: {e}"}

        # Find best fit based on AIC
        valid_fits = {k: v for k, v in fitting_results.items() if 'aic' in v}
        if valid_fits:
            best_fit = min(valid_fits.keys(), key=lambda x: valid_fits[x]['aic'])
            fitting_results["best_fit"] = best_fit

        return fitting_results

    def _calculate_confidence_intervals(self, data: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for various parameters."""
        n = len(data)
        mean = np.mean(data)
        std_err = np.std(data, ddof=1) / np.sqrt(n)

        # t-distribution critical value
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)

        intervals = {
            "mean": (float(mean - t_critical * std_err),
                    float(mean + t_critical * std_err)),
            "median": tuple(map(float, self._bootstrap_ci(data, np.median))),
            "std_dev": tuple(map(float, self._bootstrap_ci(data, lambda x: np.std(x, ddof=1))))
        }

        return intervals

    def _bootstrap_ci(self, data: np.ndarray, statistic_func, n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for any statistic."""
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_stats.append(statistic_func(bootstrap_sample))

        bootstrap_stats = np.array(bootstrap_stats)
        lower = np.percentile(bootstrap_stats, 100 * self.alpha / 2)
        upper = np.percentile(bootstrap_stats, 100 * (1 - self.alpha / 2))

        return (lower, upper)

    def _detect_outliers(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        outlier_results = {}

        # IQR method
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        iqr_outliers = data[(data < lower_bound) | (data > upper_bound)]
        outlier_results["iqr_method"] = {
            "outliers": iqr_outliers.tolist(),
            "count": len(iqr_outliers),
            "proportion": len(iqr_outliers) / len(data),
            "bounds": (float(lower_bound), float(upper_bound))
        }

        # Z-score method (modified for robustness)
        median = np.median(data)
        mad = np.median(np.abs(data - median))  # Median Absolute Deviation
        modified_z_scores = 0.6745 * (data - median) / mad if mad != 0 else np.zeros_like(data)

        z_outliers = data[np.abs(modified_z_scores) > 3.5]
        outlier_results["modified_z_score"] = {
            "outliers": z_outliers.tolist(),
            "count": len(z_outliers),
            "proportion": len(z_outliers) / len(data),
            "threshold": 3.5
        }

        return outlier_results

    def _goodness_of_fit_tests(self, data: np.ndarray) -> Dict[str, Any]:
        """Perform goodness-of-fit tests."""
        gof_results = {}

        try:
            # Chi-square goodness of fit test for normality
            # Create bins for chi-square test
            n_bins = max(5, min(20, int(np.sqrt(len(data)))))
            observed, bin_edges = np.histogram(data, bins=n_bins)

            # Expected frequencies under normal distribution
            mean, std = np.mean(data), np.std(data, ddof=1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            expected = len(data) * np.diff(stats.norm.cdf(bin_edges, mean, std))

            # Remove bins with expected frequency < 5
            valid_bins = expected >= 5
            if np.sum(valid_bins) >= 3:
                observed_valid = observed[valid_bins]
                expected_valid = expected[valid_bins]

                chi2_stat = np.sum((observed_valid - expected_valid)**2 / expected_valid)
                df = len(observed_valid) - 1 - 2  # -2 for estimated mean and std
                chi2_p = 1 - stats.chi2.cdf(chi2_stat, df) if df > 0 else np.nan

                gof_results["chi_square_normal"] = {
                    "statistic": float(chi2_stat),
                    "p_value": float(chi2_p) if not np.isnan(chi2_p) else None,
                    "degrees_of_freedom": int(df),
                    "good_fit": chi2_p > self.alpha if not np.isnan(chi2_p) else None
                }

        except Exception as e:
            gof_results["chi_square_normal"] = {"error": f"Chi-square test failed: {e}"}

        return gof_results

    def compare_distributions(self, data1: List[float], data2: List[float],
                            names: Tuple[str, str] = ("Sample 1", "Sample 2")) -> Dict[str, Any]:
        """Compare two distributions using various statistical tests."""
        if len(data1) < 3 or len(data2) < 3:
            return {"error": "Insufficient data for comparison"}

        data1_array = np.array(data1, dtype=float)
        data2_array = np.array(data2, dtype=float)

        comparison = {
            "sample_names": names,
            "sample_sizes": (len(data1), len(data2)),
            "descriptive_comparison": self._compare_descriptives(data1_array, data2_array),
            "statistical_tests": self._perform_comparison_tests(data1_array, data2_array),
            "effect_size": self._calculate_effect_sizes(data1_array, data2_array)
        }

        return comparison

    def _compare_descriptives(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Compare descriptive statistics between two samples."""
        stats1 = self._calculate_descriptive_statistics(data1)
        stats2 = self._calculate_descriptive_statistics(data2)

        comparison = {}
        for key in stats1:
            if key != "error" and isinstance(stats1[key], (int, float)):
                comparison[key] = {
                    "sample_1": stats1[key],
                    "sample_2": stats2.get(key, np.nan),
                    "difference": stats1[key] - stats2.get(key, np.nan),
                    "ratio": stats1[key] / stats2.get(key, np.nan) if stats2.get(key, 0) != 0 else np.nan
                }

        return comparison

    def _perform_comparison_tests(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, Any]:
        """Perform statistical tests comparing two distributions."""
        tests = {}

        try:
            # Mann-Whitney U test (non-parametric)
            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            tests["mann_whitney_u"] = {
                "statistic": float(u_stat),
                "p_value": float(u_p),
                "significant": u_p < self.alpha,
                "interpretation": "Significantly different" if u_p < self.alpha else "Not significantly different"
            }

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(data1, data2)
            tests["kolmogorov_smirnov"] = {
                "statistic": float(ks_stat),
                "p_value": float(ks_p),
                "significant": ks_p < self.alpha,
                "interpretation": "Different distributions" if ks_p < self.alpha else "Same distribution"
            }

            # Welch's t-test (assumes unequal variances)
            t_stat, t_p = stats.ttest_ind(data1, data2, equal_var=False)
            tests["welch_t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(t_p),
                "significant": t_p < self.alpha,
                "interpretation": "Different means" if t_p < self.alpha else "Same means"
            }

            # Levene's test for equal variances
            levene_stat, levene_p = stats.levene(data1, data2)
            tests["levene_variance"] = {
                "statistic": float(levene_stat),
                "p_value": float(levene_p),
                "significant": levene_p < self.alpha,
                "interpretation": "Different variances" if levene_p < self.alpha else "Equal variances"
            }

        except Exception as e:
            tests["error"] = f"Error in comparison tests: {e}"

        return tests

    def _calculate_effect_sizes(self, data1: np.ndarray, data2: np.ndarray) -> Dict[str, float]:
        """Calculate effect sizes for the comparison."""
        effect_sizes = {}

        try:
            # Cohen's d
            mean1, mean2 = np.mean(data1), np.mean(data2)
            std1, std2 = np.std(data1, ddof=1), np.std(data2, ddof=1)
            n1, n2 = len(data1), len(data2)

            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

            effect_sizes["cohens_d"] = float(cohens_d)

            # Glass's delta
            glass_delta = (mean1 - mean2) / std2 if std2 != 0 else 0
            effect_sizes["glass_delta"] = float(glass_delta)

            # Hedges' g (bias-corrected Cohen's d)
            correction_factor = 1 - (3 / (4*(n1+n2-2) - 1))
            hedges_g = cohens_d * correction_factor
            effect_sizes["hedges_g"] = float(hedges_g)

        except Exception as e:
            effect_sizes["error"] = f"Error calculating effect sizes: {e}"

        return effect_sizes

    def advanced_regression_analysis(self, x_data: List[float], y_data: List[float],
                                   model_type: str = "linear") -> Dict[str, Any]:
        """Perform advanced regression analysis."""
        if len(x_data) != len(y_data) or len(x_data) < 3:
            return {"error": "Invalid or insufficient data for regression"}

        x_array = np.array(x_data, dtype=float)
        y_array = np.array(y_data, dtype=float)

        regression_results = {
            "model_type": model_type,
            "sample_size": len(x_data),
            "linear_regression": self._linear_regression(x_array, y_array),
            "polynomial_regression": self._polynomial_regression(x_array, y_array),
            "model_diagnostics": self._regression_diagnostics(x_array, y_array)
        }

        return regression_results

    def _linear_regression(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive linear regression analysis."""
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

            # Calculate additional statistics
            n = len(x)
            y_pred = slope * x + intercept
            residuals = y - y_pred

            # R-squared and adjusted R-squared
            r_squared = r_value ** 2
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2)

            # Standard error of the estimate
            mse = np.sum(residuals ** 2) / (n - 2)
            se_estimate = np.sqrt(mse)

            # Confidence intervals for slope and intercept
            t_critical = stats.t.ppf(1 - self.alpha/2, n-2)

            slope_ci = (slope - t_critical * std_err, slope + t_critical * std_err)

            # Standard error of intercept
            x_mean = np.mean(x)
            se_intercept = se_estimate * np.sqrt(1/n + x_mean**2 / np.sum((x - x_mean)**2))
            intercept_ci = (intercept - t_critical * se_intercept,
                          intercept + t_critical * se_intercept)

            return {
                "slope": float(slope),
                "intercept": float(intercept),
                "r_value": float(r_value),
                "r_squared": float(r_squared),
                "adjusted_r_squared": float(adj_r_squared),
                "p_value": float(p_value),
                "standard_error": float(std_err),
                "se_estimate": float(se_estimate),
                "slope_ci": tuple(map(float, slope_ci)),
                "intercept_ci": tuple(map(float, intercept_ci)),
                "significant": p_value < self.alpha
            }

        except Exception as e:
            return {"error": f"Linear regression failed: {e}"}

    def _polynomial_regression(self, x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> Dict[str, Any]:
        """Fit polynomial models and select best degree."""
        polynomial_results = {}

        for degree in range(1, min(max_degree + 1, len(x) - 1)):
            try:
                coeffs = np.polyfit(x, y, degree)
                poly_func = np.poly1d(coeffs)
                y_pred = poly_func(x)

                # Calculate R-squared
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)

                # Calculate AIC
                n = len(x)
                mse = ss_res / (n - degree - 1)
                aic = n * np.log(mse) + 2 * (degree + 1)

                polynomial_results[f"degree_{degree}"] = {
                    "coefficients": coeffs.tolist(),
                    "r_squared": float(r_squared),
                    "aic": float(aic),
                    "mse": float(mse)
                }

            except Exception as e:
                polynomial_results[f"degree_{degree}"] = {"error": f"Failed: {e}"}

        # Select best model based on AIC
        valid_models = {k: v for k, v in polynomial_results.items() if 'aic' in v}
        if valid_models:
            best_model = min(valid_models.keys(), key=lambda x: valid_models[x]['aic'])
            polynomial_results["best_model"] = best_model

        return polynomial_results

    def _regression_diagnostics(self, x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform regression diagnostics."""
        try:
            slope, intercept, _, _, _ = stats.linregress(x, y)
            y_pred = slope * x + intercept
            residuals = y - y_pred

            diagnostics = {
                "residual_analysis": {
                    "mean_residual": float(np.mean(residuals)),
                    "std_residual": float(np.std(residuals)),
                    "durbin_watson": self._durbin_watson_test(residuals)
                },
                "influence_measures": self._calculate_influence_measures(x, y, residuals),
                "assumption_tests": self._test_regression_assumptions(x, residuals)
            }

            return diagnostics

        except Exception as e:
            return {"error": f"Regression diagnostics failed: {e}"}

    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Calculate Durbin-Watson statistic for autocorrelation."""
        diff_residuals = np.diff(residuals)
        return float(np.sum(diff_residuals**2) / np.sum(residuals**2))

    def _calculate_influence_measures(self, x: np.ndarray, y: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Calculate influence measures for regression."""
        n = len(x)

        # Leverage values
        x_centered = x - np.mean(x)
        leverage = 1/n + x_centered**2 / np.sum(x_centered**2)

        # Cook's distance (simplified)
        mse = np.sum(residuals**2) / (n - 2)
        cooks_d = (residuals**2 / (2 * mse)) * (leverage / (1 - leverage)**2)

        return {
            "leverage": leverage.tolist(),
            "cooks_distance": cooks_d.tolist(),
            "high_leverage_points": int(np.sum(leverage > 2*2/n)),  # 2p/n threshold
            "influential_points": int(np.sum(cooks_d > 4/n))  # 4/n threshold
        }

    def _test_regression_assumptions(self, x: np.ndarray, residuals: np.ndarray) -> Dict[str, Any]:
        """Test key regression assumptions."""
        assumptions = {}

        try:
            # Test for normality of residuals
            _, normality_p = shapiro(residuals) if len(residuals) <= 5000 else normaltest(residuals)
            assumptions["residual_normality"] = {
                "p_value": float(normality_p),
                "assumption_met": normality_p > self.alpha
            }

            # Test for homoscedasticity (constant variance)
            # Breusch-Pagan test approximation
            abs_residuals = np.abs(residuals)
            bp_slope, _, bp_r, bp_p, _ = stats.linregress(x, abs_residuals)
            assumptions["homoscedasticity"] = {
                "p_value": float(bp_p),
                "assumption_met": bp_p > self.alpha
            }

        except Exception as e:
            assumptions["error"] = f"Assumption testing failed: {e}"

        return assumptions

    # Legacy compatibility methods with enhanced functionality
    def calculate_z_scores(self, data: List[float]) -> Dict[str, Any]:
        """Enhanced z-score calculation with additional statistics."""
        if len(data) < 2:
            return {"z_scores": [], "mean": 0, "std_dev": 0}

        data_array = np.array(data, dtype=float)

        # Use more robust statistics
        mean = np.mean(data_array)
        std_dev = np.std(data_array, ddof=1)  # Sample standard deviation

        if std_dev == 0:
            z_scores = [0.0] * len(data_array)
        else:
            z_scores = [(x - mean) / std_dev for x in data_array]

        # Add outlier detection
        outliers = [i for i, z in enumerate(z_scores) if abs(z) > 3]

        return {
            "z_scores": z_scores,
            "mean": float(mean),
            "std_dev": float(std_dev),
            "outlier_indices": outliers,
            "outlier_count": len(outliers)
        }

    def kolmogorov_smirnov_test(self, sample1: List[float], sample2: List[float]) -> Dict[str, Any]:
        """Enhanced KS test with proper statistical computation."""
        if len(sample1) < 3 or len(sample2) < 3:
            return {"error": "Insufficient data for KS test"}

        try:
            ks_statistic, p_value = stats.ks_2samp(sample1, sample2)

            return {
                "ks_statistic": float(ks_statistic),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
                "interpretation": "Different distributions" if p_value < self.alpha else "Same distribution",
                "effect_size": "Large" if ks_statistic > 0.5 else "Medium" if ks_statistic > 0.3 else "Small"
            }

        except Exception as e:
            return {"error": f"KS test failed: {e}"}

    def bootstrap_confidence_interval(self, data: List[float], n_bootstrap: int = 1000,
                                    statistic_func=None, **kwargs) -> Dict[str, Any]:
        """Enhanced bootstrap with multiple statistics and better error handling."""
        if len(data) == 0:
            return {"error": "No data provided"}

        if statistic_func is None:
            statistic_func = np.mean

        try:
            data_array = np.array(data, dtype=float)

            bootstrap_stats = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
                bootstrap_stats.append(statistic_func(bootstrap_sample))

            bootstrap_stats = np.array(bootstrap_stats)

            # Calculate confidence interval
            lower_percentile = 100 * self.alpha / 2
            upper_percentile = 100 * (1 - self.alpha / 2)

            ci_lower = np.percentile(bootstrap_stats, lower_percentile)
            ci_upper = np.percentile(bootstrap_stats, upper_percentile)

            # Calculate bootstrap statistics
            bootstrap_mean = np.mean(bootstrap_stats)
            bootstrap_std = np.std(bootstrap_stats, ddof=1)
            original_stat = statistic_func(data_array)

            # Bias calculation
            bias = bootstrap_mean - original_stat
            bias_corrected_stat = original_stat - bias

            return {
                "confidence_interval": (float(ci_lower), float(ci_upper)),
                "confidence_level": self.confidence_level,
                "n_bootstrap": n_bootstrap,
                "original_statistic": float(original_stat),
                "bootstrap_mean": float(bootstrap_mean),
                "bootstrap_std": float(bootstrap_std),
                "bias": float(bias),
                "bias_corrected_statistic": float(bias_corrected_stat),
                "bootstrap_distribution": bootstrap_stats.tolist()
            }

        except Exception as e:
            return {"error": f"Bootstrap failed: {e}"}

    def time_series_analysis(self, data: List[float], timestamps: Optional[List] = None) -> Dict[str, Any]:
        """Perform basic time series analysis for partition sequences."""
        if len(data) < 4:
            return {"error": "Insufficient data for time series analysis"}

        data_array = np.array(data, dtype=float)
        n = len(data_array)

        # Generate timestamps if not provided
        if timestamps is None:
            timestamps = list(range(n))

        time_series_results = {
            "series_length": n,
            "trend_analysis": self._analyze_trend(data_array),
            "autocorrelation": self._calculate_autocorrelation(data_array),
            "stationarity_tests": self._test_stationarity(data_array),
            "seasonal_decomposition": self._basic_seasonal_analysis(data_array),
            "change_point_detection": self._detect_change_points(data_array)
        }

        return time_series_results

    def _analyze_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in time series data."""
        try:
            n = len(data)
            x = np.arange(n)

            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, data)

            # Mann-Kendall trend test (simplified)
            def mann_kendall_test(data):
                n = len(data)
                s = 0
                for i in range(n-1):
                    for j in range(i+1, n):
                        if data[j] > data[i]:
                            s += 1
                        elif data[j] < data[i]:
                            s -= 1

                # Variance calculation (simplified)
                var_s = n * (n - 1) * (2 * n + 5) / 18

                if s > 0:
                    z = (s - 1) / np.sqrt(var_s)
                elif s < 0:
                    z = (s + 1) / np.sqrt(var_s)
                else:
                    z = 0

                p_value = 2 * (1 - stats.norm.cdf(abs(z)))
                return s, z, p_value

            mk_s, mk_z, mk_p = mann_kendall_test(data)

            return {
                "linear_slope": float(slope),
                "linear_intercept": float(intercept),
                "linear_r_squared": float(r_value**2),
                "linear_p_value": float(p_value),
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "no trend",
                "mann_kendall_statistic": int(mk_s),
                "mann_kendall_z": float(mk_z),
                "mann_kendall_p_value": float(mk_p),
                "significant_trend": mk_p < self.alpha
            }

        except Exception as e:
            return {"error": f"Trend analysis failed: {e}"}

    def _calculate_autocorrelation(self, data: np.ndarray, max_lags: int = None) -> Dict[str, Any]:
        """Calculate autocorrelation function."""
        try:
            n = len(data)
            if max_lags is None:
                max_lags = min(20, n // 4)

            # Center the data
            data_centered = data - np.mean(data)

            autocorrelations = []
            for lag in range(max_lags + 1):
                if lag == 0:
                    autocorr = 1.0
                else:
                    if lag >= n:
                        autocorr = 0.0
                    else:
                        numerator = np.sum(data_centered[:-lag] * data_centered[lag:])
                        denominator = np.sum(data_centered**2)
                        autocorr = numerator / denominator if denominator != 0 else 0.0

                autocorrelations.append(autocorr)

            # Ljung-Box test for serial correlation
            def ljung_box_test(autocorrs, n, h=min(10, max_lags)):
                lb_stat = n * (n + 2) * np.sum([autocorrs[i]**2 / (n - i) for i in range(1, h+1)])
                p_value = 1 - stats.chi2.cdf(lb_stat, h)
                return lb_stat, p_value

            lb_stat, lb_p = ljung_box_test(autocorrelations, n)

            return {
                "autocorrelations": autocorrelations,
                "lags": list(range(max_lags + 1)),
                "ljung_box_statistic": float(lb_stat),
                "ljung_box_p_value": float(lb_p),
                "serial_correlation_detected": lb_p < self.alpha
            }

        except Exception as e:
            return {"error": f"Autocorrelation analysis failed: {e}"}

    def _test_stationarity(self, data: np.ndarray) -> Dict[str, Any]:
        """Test for stationarity using multiple methods."""
        stationarity_results = {}

        try:
            # Augmented Dickey-Fuller test (simplified version)
            def adf_test_simplified(data):
                n = len(data)
                # Simple version: test if first difference has unit root
                diff_data = np.diff(data)

                # Regression: diff_data[t] = alpha + beta * data[t-1] + error
                y = diff_data[1:]
                x = data[:-2]

                if len(x) > 0 and len(y) > 0:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

                    # Simple test statistic (not exact ADF, but indicative)
                    t_stat = slope / std_err if std_err != 0 else 0

                    return {
                        "test_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "is_stationary": p_value < self.alpha
                    }
                else:
                    return {"error": "Insufficient data for ADF test"}

            stationarity_results["adf_test"] = adf_test_simplified(data)

            # KPSS test (simplified)
            def kpss_test_simplified(data):
                n = len(data)

                # Calculate cumulative sum of deviations from mean
                y = np.cumsum(data - np.mean(data))

                # Calculate test statistic
                s_squared = np.sum((data - np.mean(data))**2) / n
                kpss_stat = np.sum(y**2) / (n**2 * s_squared) if s_squared != 0 else 0

                # Critical values (approximate)
                critical_values = {"1%": 0.739, "5%": 0.463, "10%": 0.347}

                return {
                    "test_statistic": float(kpss_stat),
                    "critical_values": critical_values,
                    "is_stationary": kpss_stat < critical_values["5%"]
                }

            stationarity_results["kpss_test"] = kpss_test_simplified(data)

        except Exception as e:
            stationarity_results["error"] = f"Stationarity testing failed: {e}"

        return stationarity_results

    def _basic_seasonal_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Basic seasonal pattern analysis."""
        try:
            n = len(data)

            # Test for potential seasonal periods
            seasonal_results = {}

            # Test common seasonal periods
            potential_periods = [2, 3, 4, 6, 12] if n >= 24 else [2, 3, 4] if n >= 12 else [2]
            potential_periods = [p for p in potential_periods if p < n/2]

            for period in potential_periods:
                if n >= 2 * period:
                    # Simple seasonal strength measure
                    seasonal_means = []
                    for i in range(period):
                        seasonal_values = [data[j] for j in range(i, n, period)]
                        if seasonal_values:
                            seasonal_means.append(np.mean(seasonal_values))

                    if len(seasonal_means) == period:
                        overall_mean = np.mean(data)
                        seasonal_variance = np.var(seasonal_means, ddof=1) if len(seasonal_means) > 1 else 0
                        total_variance = np.var(data, ddof=1)

                        seasonal_strength = seasonal_variance / total_variance if total_variance != 0 else 0

                        seasonal_results[f"period_{period}"] = {
                            "seasonal_means": seasonal_means,
                            "seasonal_strength": float(seasonal_strength),
                            "has_seasonality": seasonal_strength > 0.1
                        }

            return seasonal_results

        except Exception as e:
            return {"error": f"Seasonal analysis failed: {e}"}

    def _detect_change_points(self, data: np.ndarray) -> Dict[str, Any]:
        """Simple change point detection."""
        try:
            n = len(data)
            change_points = []

            # CUSUM-based change point detection (simplified)
            mean_data = np.mean(data)
            std_data = np.std(data, ddof=1)

            if std_data == 0:
                return {"change_points": [], "method": "CUSUM", "note": "No variation in data"}

            # Cumulative sum of standardized deviations
            cusum_pos = np.zeros(n)
            cusum_neg = np.zeros(n)

            h = 3 * std_data  # Threshold

            for i in range(1, n):
                cusum_pos[i] = max(0, cusum_pos[i-1] + (data[i] - mean_data) - 0.5 * std_data)
                cusum_neg[i] = max(0, cusum_neg[i-1] - (data[i] - mean_data) - 0.5 * std_data)

                if cusum_pos[i] > h or cusum_neg[i] > h:
                    change_points.append(i)
                    cusum_pos[i] = 0  # Reset
                    cusum_neg[i] = 0

            return {
                "change_points": change_points,
                "n_change_points": len(change_points),
                "method": "CUSUM",
                "threshold": float(h)
            }

        except Exception as e:
            return {"error": f"Change point detection failed: {e}"}

    def partition_specific_analysis(self, partition_data: List[int],
                                   analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """Specialized analysis for integer partition data."""
        if not partition_data or not all(isinstance(x, int) and x > 0 for x in partition_data):
            return {"error": "Invalid partition data - must be positive integers"}

        partition_results = {
            "data_type": "integer_partitions",
            "analysis_type": analysis_type,
            "basic_properties": self._analyze_partition_properties(partition_data),
            "multiplicative_structure": self._analyze_multiplicative_patterns(partition_data),
            "growth_analysis": self._analyze_partition_growth(partition_data),
            "distribution_analysis": self.comprehensive_distribution_analysis(
                [float(x) for x in partition_data], "partition_sequence"
            )
        }

        return partition_results

    def _analyze_partition_properties(self, data: List[int]) -> Dict[str, Any]:
        """Analyze basic properties of partition sequences."""
        try:
            n = len(data)

            # Basic statistics
            total_sum = sum(data)
            max_part = max(data)
            min_part = min(data)
            unique_parts = len(set(data))

            # Multiplicative properties
            def gcd_of_list(numbers):
                from math import gcd
                result = numbers[0]
                for i in range(1, len(numbers)):
                    result = gcd(result, numbers[i])
                return result

            overall_gcd = gcd_of_list(data) if len(data) > 1 else data[0]

            # Growth rates
            if n > 1:
                differences = [data[i+1] - data[i] for i in range(n-1)]
                ratios = [data[i+1] / data[i] for i in range(n-1) if data[i] != 0]
            else:
                differences = []
                ratios = []

            return {
                "sequence_length": n,
                "total_sum": total_sum,
                "max_partition": max_part,
                "min_partition": min_part,
                "unique_partitions": unique_parts,
                "diversity_ratio": unique_parts / n,
                "overall_gcd": overall_gcd,
                "differences": differences[:10],  # First 10 differences
                "ratios": ratios[:10],  # First 10 ratios
                "mean_difference": float(np.mean(differences)) if differences else 0,
                "mean_ratio": float(np.mean(ratios)) if ratios else 0
            }

        except Exception as e:
            return {"error": f"Partition property analysis failed: {e}"}

    def _analyze_multiplicative_patterns(self, data: List[int]) -> Dict[str, Any]:
        """Analyze multiplicative patterns in partition data."""
        try:
            # Prime factorization patterns
            def prime_factors(n):
                factors = []
                d = 2
                while d * d <= n:
                    while n % d == 0:
                        factors.append(d)
                        n //= d
                    d += 1
                if n > 1:
                    factors.append(n)
                return factors

            # Analyze first few terms for patterns
            sample_size = min(20, len(data))
            factorization_data = {}

            for i, value in enumerate(data[:sample_size]):
                factors = prime_factors(value)
                factorization_data[f"term_{i+1}"] = {
                    "value": value,
                    "prime_factors": factors,
                    "number_of_factors": len(factors),
                    "number_of_distinct_factors": len(set(factors))
                }

            # Look for patterns in factor counts
            factor_counts = [len(prime_factors(x)) for x in data[:sample_size]]
            distinct_factor_counts = [len(set(prime_factors(x))) for x in data[:sample_size]]

            return {
                "sample_size": sample_size,
                "factorization_sample": factorization_data,
                "factor_count_sequence": factor_counts,
                "distinct_factor_count_sequence": distinct_factor_counts,
                "mean_factor_count": float(np.mean(factor_counts)),
                "mean_distinct_factor_count": float(np.mean(distinct_factor_counts))
            }

        except Exception as e:
            return {"error": f"Multiplicative pattern analysis failed: {e}"}

    def _analyze_partition_growth(self, data: List[int]) -> Dict[str, Any]:
        """Analyze growth patterns in partition sequences."""
        try:
            n = len(data)
            if n < 3:
                return {"error": "Insufficient data for growth analysis"}

            # Convert to float for analysis
            float_data = [float(x) for x in data]
            indices = list(range(1, n+1))

            # Test different growth models
            growth_models = {}

            # Linear growth
            linear_result = self._linear_regression(np.array(indices), np.array(float_data))
            growth_models["linear"] = linear_result

            # Exponential growth (log-linear)
            try:
                log_data = [np.log(x) if x > 0 else 0 for x in float_data]
                if all(x > 0 for x in float_data):  # All positive
                    exp_result = self._linear_regression(np.array(indices), np.array(log_data))
                    growth_models["exponential"] = {
                        "log_linear_fit": exp_result,
                        "growth_rate": float(np.exp(exp_result.get("slope", 0)) - 1) if "slope" in exp_result else None
                    }
            except:
                growth_models["exponential"] = {"error": "Cannot fit exponential model"}

            # Power law growth (log-log)
            try:
                log_indices = [np.log(i) for i in indices]
                log_data = [np.log(x) if x > 0 else 0 for x in float_data]
                if all(x > 0 for x in float_data):
                    power_result = self._linear_regression(np.array(log_indices), np.array(log_data))
                    growth_models["power_law"] = {
                        "log_log_fit": power_result,
                        "power_exponent": float(power_result.get("slope", 0)) if "slope" in power_result else None
                    }
            except:
                growth_models["power_law"] = {"error": "Cannot fit power law model"}

            # Determine best fit
            valid_models = {}
            for model_name, model_data in growth_models.items():
                if isinstance(model_data, dict) and "r_squared" in str(model_data):
                    # Extract R-squared from nested structure
                    if model_name == "linear":
                        r_squared = model_data.get("r_squared", 0)
                    else:
                        nested_fit = model_data.get("log_linear_fit") or model_data.get("log_log_fit", {})
                        r_squared = nested_fit.get("r_squared", 0) if isinstance(nested_fit, dict) else 0

                    valid_models[model_name] = r_squared

            best_model = max(valid_models.keys(), key=lambda x: valid_models[x]) if valid_models else "none"

            return {
                "growth_models": growth_models,
                "best_fit_model": best_model,
                "model_comparison": valid_models
            }

        except Exception as e:
            return {"error": f"Growth analysis failed: {e}"}


# Convenience functions for quick analysis
def quick_analysis(data: List[float], name: str = "dataset") -> Dict[str, Any]:
    """Perform quick comprehensive analysis of a dataset."""
    analyzer = StatisticalAnalyzer()
    return analyzer.comprehensive_distribution_analysis(data, name)


def compare_sequences(seq1: List[float], seq2: List[float],
                     names: Tuple[str, str] = ("Sequence 1", "Sequence 2")) -> Dict[str, Any]:
    """Quick comparison of two sequences."""
    analyzer = StatisticalAnalyzer()
    return analyzer.compare_distributions(seq1, seq2, names)


def analyze_partition_sequence(partitions: List[int], name: str = "partition_sequence") -> Dict[str, Any]:
    """Quick analysis of integer partition sequence."""
    analyzer = StatisticalAnalyzer()
    return analyzer.partition_specific_analysis(partitions)


# Example usage and testing
if __name__ == "__main__":
    # Example usage
    analyzer = StatisticalAnalyzer(random_seed=42)

    # Test with sample partition data
    sample_partitions = [1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231]

    print("Testing Statistical Analyzer for SFH Partition Research")
    print("=" * 60)

    # Comprehensive analysis
    result = analyzer.partition_specific_analysis(sample_partitions)

    if "error" not in result:
        print(f"✓ Analysis completed successfully")
        print(f"  Sample size: {result['basic_properties']['sequence_length']}")
        print(f"  Max partition: {result['basic_properties']['max_partition']}")
        print(f"  Growth model: {result['growth_analysis'].get('best_fit_model', 'unknown')}")
    else:
        print(f"✗ Analysis failed: {result['error']}")

    print("\nStatistical Analyzer ready for SFH partition research!")
