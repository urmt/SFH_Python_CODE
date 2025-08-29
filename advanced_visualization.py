#!/usr/bin/env python3
"""
Advanced Visualization System for SFH Partition Research
Generates comprehensive, publication-ready visualizations from statistical analysis results
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
import os
from datetime import datetime
import json

# Set publication-quality defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

# Publication settings
PUBLICATION_DPI = 300
FIGURE_SIZE_SINGLE = (10, 8)
FIGURE_SIZE_DOUBLE = (16, 10)
FIGURE_SIZE_WIDE = (20, 8)
FIGURE_SIZE_TALL = (12, 16)

class AdvancedVisualizer:
    """Advanced visualization system for partition analysis results."""

    def __init__(self, output_dir: str = "visualizations", theme: str = "publication"):
        """Initialize the visualization system."""
        self.output_dir = output_dir
        self.theme = theme
        self.color_palette = self._setup_color_palette()
        self.figure_count = 0

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Set style based on theme
        self._setup_plotting_style()

    def _setup_color_palette(self) -> Dict[str, str]:
        """Setup publication-quality color palette."""
        return {
            'primary': '#2E86C1',      # Professional blue
            'secondary': '#E74C3C',    # Red for emphasis
            'accent': '#28B463',       # Green for positive
            'warning': '#F39C12',      # Orange for warnings
            'neutral': '#5D6D7E',      # Gray for neutral
            'background': '#FDFEFE',   # Off-white background
            'grid': '#D5DBDB',         # Light gray for grids
            'text': '#2C3E50'          # Dark blue-gray for text
        }

    def _setup_plotting_style(self):
        """Setup publication-quality plotting style."""
        plt.rcParams.update({
            'figure.facecolor': self.color_palette['background'],
            'axes.facecolor': self.color_palette['background'],
            'axes.edgecolor': self.color_palette['text'],
            'axes.labelcolor': self.color_palette['text'],
            'text.color': self.color_palette['text'],
            'xtick.color': self.color_palette['text'],
            'ytick.color': self.color_palette['text'],
            'grid.color': self.color_palette['grid'],
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
            'font.size': 12,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'lines.linewidth': 2,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'axes.grid': True,
            'grid.alpha': 0.3
        })

    def create_comprehensive_report(self, analysis_results: Dict[str, Any],
                                  data_name: str = "partition_sequence") -> Dict[str, str]:
        """Generate comprehensive visualization report from analysis results."""

        report_files = {}

        print(f"Generating comprehensive visualization report for {data_name}...")

        # 1. Distribution Analysis Visualizations
        if 'descriptive_statistics' in analysis_results:
            report_files.update(self._create_distribution_visualizations(
                analysis_results, data_name))

        # 2. Statistical Test Visualizations
        if 'normality_tests' in analysis_results:
            report_files.update(self._create_statistical_test_visualizations(
                analysis_results, data_name))

        # 3. Distribution Fitting Visualizations
        if 'distribution_fitting' in analysis_results:
            report_files.update(self._create_distribution_fitting_visualizations(
                analysis_results, data_name))

        # 4. Regression Analysis Visualizations
        if 'linear_regression' in analysis_results or 'polynomial_regression' in analysis_results:
            report_files.update(self._create_regression_visualizations(
                analysis_results, data_name))

        # 5. Time Series Visualizations (if applicable)
        if 'trend_analysis' in analysis_results:
            report_files.update(self._create_time_series_visualizations(
                analysis_results, data_name))

        # 6. Partition-Specific Visualizations
        if 'basic_properties' in analysis_results:
            report_files.update(self._create_partition_visualizations(
                analysis_results, data_name))

        # 7. Summary Dashboard
        dashboard_file = self._create_summary_dashboard(analysis_results, data_name)
        if dashboard_file:
            report_files['summary_dashboard'] = dashboard_file

        print(f"Generated {len(report_files)} visualization files")
        return report_files

    def _create_distribution_visualizations(self, results: Dict[str, Any],
                                          data_name: str) -> Dict[str, str]:
        """Create distribution analysis visualizations."""
        files = {}

        # Extract data if available (reconstruct from statistics if needed)
        stats = results.get('descriptive_statistics', {})

        if not stats or 'error' in stats:
            return files

        # 1. Distribution Summary Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE_DOUBLE)
        fig.suptitle(f'Distribution Analysis: {data_name}', fontsize=18, fontweight='bold')

        # Box plot (using quartiles)
        if all(k in stats for k in ['q1', 'median', 'q3', 'min', 'max']):
            box_data = [stats['q1'], stats['median'], stats['q3']]
            whiskers = [stats['min'], stats['max']]

            ax1.boxplot([box_data], patch_artist=True,
                       boxprops=dict(facecolor=self.color_palette['primary'], alpha=0.7),
                       medianprops=dict(color=self.color_palette['secondary'], linewidth=2))
            ax1.set_title('Distribution Box Plot')
            ax1.set_ylabel('Values')
            ax1.grid(True, alpha=0.3)

        # Statistics bar plot
        stat_names = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
        stat_values = [stats.get('mean', 0), stats.get('median', 0),
                      stats.get('std_dev', 0), stats.get('skewness', 0),
                      stats.get('kurtosis', 0)]

        colors = [self.color_palette['primary'], self.color_palette['accent'],
                 self.color_palette['warning'], self.color_palette['secondary'],
                 self.color_palette['neutral']]

        bars = ax2.bar(stat_names, stat_values, color=colors, alpha=0.8)
        ax2.set_title('Key Statistics')
        ax2.set_ylabel('Value')
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, stat_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        # Percentile plot
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentile_values = [stats.get(f'p{p}', stats.get('median', 0)) for p in [5, 10]]
        percentile_values.extend([stats.get('q1', 0), stats.get('median', 0),
                                stats.get('q3', 0)])
        percentile_values.extend([stats.get(f'p{p}', stats.get('median', 0)) for p in [90, 95]])

        ax3.plot(percentiles, percentile_values, 'o-', color=self.color_palette['primary'],
                linewidth=2, markersize=6)
        ax3.set_title('Percentile Distribution')
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Value')
        ax3.grid(True, alpha=0.3)

        # Outlier analysis (if available)
        if 'outlier_analysis' in results:
            outlier_data = results['outlier_analysis']
            methods = []
            counts = []
            proportions = []

            for method, data in outlier_data.items():
                if isinstance(data, dict) and 'count' in data:
                    methods.append(method.replace('_', ' ').title())
                    counts.append(data['count'])
                    proportions.append(data.get('proportion', 0) * 100)

            if methods:
                x_pos = np.arange(len(methods))
                bars = ax4.bar(x_pos, proportions, color=self.color_palette['warning'], alpha=0.8)
                ax4.set_title('Outlier Detection Results')
                ax4.set_xlabel('Detection Method')
                ax4.set_ylabel('Outlier Percentage (%)')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels(methods, rotation=45)

                # Add count labels
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'n={count}', ha='center', va='bottom')

        plt.tight_layout()
        filename = f"{self.output_dir}/distribution_analysis_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['distribution_analysis'] = filename

        return files

    def _create_statistical_test_visualizations(self, results: Dict[str, Any],
                                              data_name: str) -> Dict[str, str]:
        """Create statistical test result visualizations."""
        files = {}

        normality_tests = results.get('normality_tests', {})
        if not normality_tests or 'error' in normality_tests:
            return files

        # Normality Tests Dashboard
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE_DOUBLE)
        fig.suptitle(f'Normality Test Results: {data_name}', fontsize=18, fontweight='bold')

        # Test results summary
        test_names = []
        test_stats = []
        p_values = []
        interpretations = []

        for test_name, test_data in normality_tests.items():
            if isinstance(test_data, dict) and 'statistic' in test_data:
                test_names.append(test_name.replace('_', ' ').title())
                test_stats.append(test_data['statistic'])
                p_values.append(test_data.get('p_value', 1.0))
                interpretations.append(test_data.get('interpretation', 'Unknown'))

        if test_names:
            # P-values comparison
            colors = [self.color_palette['accent'] if p > 0.05 else self.color_palette['secondary']
                     for p in p_values]
            bars = ax1.bar(range(len(test_names)), p_values, color=colors, alpha=0.8)
            ax1.axhline(y=0.05, color=self.color_palette['warning'], linestyle='--',
                       linewidth=2, label='α = 0.05')
            ax1.set_title('Normality Test P-Values')
            ax1.set_ylabel('P-Value')
            ax1.set_xticks(range(len(test_names)))
            ax1.set_xticklabels(test_names, rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Add p-value labels
            for bar, p_val in zip(bars, p_values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{p_val:.4f}', ha='center', va='bottom')

            # Test statistics
            ax2.bar(range(len(test_names)), test_stats,
                   color=self.color_palette['primary'], alpha=0.8)
            ax2.set_title('Test Statistics')
            ax2.set_ylabel('Statistic Value')
            ax2.set_xticks(range(len(test_names)))
            ax2.set_xticklabels(test_names, rotation=45)
            ax2.grid(True, alpha=0.3)

        # Normality conclusion pie chart
        normal_count = sum(1 for interp in interpretations if 'normal' in interp.lower())
        non_normal_count = len(interpretations) - normal_count

        if normal_count + non_normal_count > 0:
            sizes = [normal_count, non_normal_count]
            labels = ['Normal', 'Non-Normal']
            colors_pie = [self.color_palette['accent'], self.color_palette['secondary']]

            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, colors=colors_pie,
                                              autopct='%1.0f%%', startangle=90)
            ax3.set_title('Test Consensus')

            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        # Confidence intervals (if available)
        if 'confidence_intervals' in results:
            ci_data = results['confidence_intervals']
            params = []
            lower_bounds = []
            upper_bounds = []
            means = []

            for param, (lower, upper) in ci_data.items():
                if isinstance(lower, (int, float)) and isinstance(upper, (int, float)):
                    params.append(param.replace('_', ' ').title())
                    lower_bounds.append(lower)
                    upper_bounds.append(upper)
                    means.append((lower + upper) / 2)

            if params:
                y_pos = np.arange(len(params))
                errors = [[m - l for m, l in zip(means, lower_bounds)],
                         [u - m for m, u in zip(means, upper_bounds)]]

                ax4.errorbar(means, y_pos, xerr=errors, fmt='o',
                           color=self.color_palette['primary'], capsize=5, capthick=2)
                ax4.set_title('95% Confidence Intervals')
                ax4.set_xlabel('Parameter Value')
                ax4.set_yticks(y_pos)
                ax4.set_yticklabels(params)
                ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        filename = f"{self.output_dir}/statistical_tests_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['statistical_tests'] = filename

        return files

    def _create_distribution_fitting_visualizations(self, results: Dict[str, Any],
                                                  data_name: str) -> Dict[str, str]:
        """Create distribution fitting visualizations."""
        files = {}

        fitting_results = results.get('distribution_fitting', {})
        if not fitting_results:
            return files

        # Distribution Fitting Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=FIGURE_SIZE_DOUBLE)
        fig.suptitle(f'Distribution Fitting Analysis: {data_name}',
                    fontsize=18, fontweight='bold')

        # Extract fitting data
        distributions = []
        aic_values = []
        bic_values = []
        ks_stats = []
        ks_pvals = []
        good_fits = []

        for dist_name, dist_data in fitting_results.items():
            if isinstance(dist_data, dict) and 'aic' in dist_data and dist_name != 'best_fit':
                distributions.append(dist_name.replace('_', ' ').title())
                aic_values.append(dist_data['aic'])
                bic_values.append(dist_data['bic'])
                ks_stats.append(dist_data['ks_statistic'])
                ks_pvals.append(dist_data['ks_p_value'])
                good_fits.append(dist_data.get('good_fit', False))

        if distributions:
            # AIC comparison
            colors_aic = [self.color_palette['accent'] if fit else self.color_palette['neutral']
                         for fit in good_fits]
            bars1 = ax1.bar(range(len(distributions)), aic_values, color=colors_aic, alpha=0.8)
            ax1.set_title('AIC Comparison (Lower is Better)')
            ax1.set_ylabel('AIC Value')
            ax1.set_xticks(range(len(distributions)))
            ax1.set_xticklabels(distributions, rotation=45)
            ax1.grid(True, alpha=0.3)

            # Highlight best fit
            if 'best_fit' in fitting_results:
                best_fit = fitting_results['best_fit'].replace('_', ' ').title()
                if best_fit in distributions:
                    best_idx = distributions.index(best_fit)
                    bars1[best_idx].set_color(self.color_palette['primary'])
                    bars1[best_idx].set_edgecolor(self.color_palette['secondary'])
                    bars1[best_idx].set_linewidth(3)

            # KS test p-values
            colors_ks = [self.color_palette['accent'] if p > 0.05 else self.color_palette['secondary']
                        for p in ks_pvals]
            ax2.bar(range(len(distributions)), ks_pvals, color=colors_ks, alpha=0.8)
            ax2.axhline(y=0.05, color=self.color_palette['warning'], linestyle='--',
                       linewidth=2, label='α = 0.05')
            ax2.set_title('Kolmogorov-Smirnov Test P-Values')
            ax2.set_ylabel('P-Value')
            ax2.set_xticks(range(len(distributions)))
            ax2.set_xticklabels(distributions, rotation=45)
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # Model comparison scatter plot
            ax3.scatter(aic_values, ks_pvals, c=ks_stats, cmap='viridis',
                       s=100, alpha=0.8, edgecolors='black')
            ax3.set_xlabel('AIC Value')
            ax3.set_ylabel('KS Test P-Value')
            ax3.set_title('Model Quality Comparison')
            ax3.axhline(y=0.05, color=self.color_palette['secondary'], linestyle='--', alpha=0.7)
            ax3.grid(True, alpha=0.3)

            # Add distribution labels
            for i, (dist, aic, pval) in enumerate(zip(distributions, aic_values, ks_pvals)):
                ax3.annotate(dist, (aic, pval), xytext=(5, 5),
                           textcoords='offset points', fontsize=8)

            # Fit quality summary
            fit_counts = [sum(good_fits), len(good_fits) - sum(good_fits)]
            fit_labels = ['Good Fit', 'Poor Fit']
            fit_colors = [self.color_palette['accent'], self.color_palette['secondary']]

            wedges, texts, autotexts = ax4.pie(fit_counts, labels=fit_labels,
                                              colors=fit_colors, autopct='%1.0f%%',
                                              startangle=90)
            ax4.set_title('Overall Fit Quality')

            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')

        plt.tight_layout()
        filename = f"{self.output_dir}/distribution_fitting_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['distribution_fitting'] = filename

        return files

    def _create_regression_visualizations(self, results: Dict[str, Any],
                                        data_name: str) -> Dict[str, str]:
        """Create regression analysis visualizations."""
        files = {}

        # Check if we have regression results
        has_linear = 'linear_regression' in results
        has_poly = 'polynomial_regression' in results
        has_diagnostics = 'model_diagnostics' in results

        if not (has_linear or has_poly):
            return files

        # Regression Analysis Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Regression Analysis: {data_name}', fontsize=20, fontweight='bold')

        # Linear regression results
        if has_linear:
            linear_results = results['linear_regression']

            if 'error' not in linear_results:
                # Model summary
                stats_names = ['R²', 'Adj R²', 'Slope', 'Intercept', 'P-Value']
                stats_values = [
                    linear_results.get('r_squared', 0),
                    linear_results.get('adjusted_r_squared', 0),
                    linear_results.get('slope', 0),
                    linear_results.get('intercept', 0),
                    linear_results.get('p_value', 1)
                ]

                colors = [self.color_palette['accent'] if linear_results.get('significant', False)
                         else self.color_palette['neutral'] for _ in stats_names]
                colors[4] = self.color_palette['secondary'] if stats_values[4] < 0.05 else self.color_palette['neutral']

                bars = axes[0, 0].bar(stats_names, stats_values, color=colors, alpha=0.8)
                axes[0, 0].set_title('Linear Regression Summary')
                axes[0, 0].set_ylabel('Value')
                plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)

                # Add value labels
                for bar, value in zip(bars, stats_values):
                    height = bar.get_height()
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2.,
                                   height + abs(height)*0.01,
                                   f'{value:.4f}', ha='center', va='bottom')

        # Polynomial regression comparison
        if has_poly:
            poly_results = results['polynomial_regression']

            degrees = []
            r_squareds = []
            aics = []

            for key, data in poly_results.items():
                if key.startswith('degree_') and isinstance(data, dict) and 'r_squared' in data:
                    degree = int(key.split('_')[1])
                    degrees.append(degree)
                    r_squareds.append(data['r_squared'])
                    aics.append(data['aic'])

            if degrees:
                # R-squared by degree
                axes[0, 1].plot(degrees, r_squareds, 'o-', color=self.color_palette['primary'],
                               linewidth=2, markersize=8)
                axes[0, 1].set_title('R² by Polynomial Degree')
                axes[0, 1].set_xlabel('Polynomial Degree')
                axes[0, 1].set_ylabel('R-squared')
                axes[0, 1].grid(True, alpha=0.3)

                # Highlight best model
                if 'best_model' in poly_results:
                    best_degree = int(poly_results['best_model'].split('_')[1])
                    best_idx = degrees.index(best_degree)
                    axes[0, 1].scatter([best_degree], [r_squareds[best_idx]],
                                     color=self.color_palette['secondary'], s=150,
                                     marker='*', zorder=5, label='Best Model')
                    axes[0, 1].legend()

                # AIC by degree
                axes[0, 2].plot(degrees, aics, 's-', color=self.color_palette['warning'],
                               linewidth=2, markersize=8)
                axes[0, 2].set_title('AIC by Polynomial Degree')
                axes[0, 2].set_xlabel('Polynomial Degree')
                axes[0, 2].set_ylabel('AIC (Lower is Better)')
                axes[0, 2].grid(True, alpha=0.3)

        # Regression diagnostics
        if has_diagnostics:
            diagnostics = results['model_diagnostics']

            # Residual analysis
            if 'residual_analysis' in diagnostics:
                residual_data = diagnostics['residual_analysis']

                # Durbin-Watson test result
                dw_stat = residual_data.get('durbin_watson', 2.0)
                dw_interpretation = 'No Autocorr.' if 1.5 < dw_stat < 2.5 else 'Autocorr. Present'

                axes[1, 0].bar(['Durbin-Watson'], [dw_stat],
                              color=self.color_palette['accent'] if 1.5 < dw_stat < 2.5
                              else self.color_palette['secondary'], alpha=0.8)
                axes[1, 0].axhline(y=2.0, color=self.color_palette['neutral'],
                                  linestyle='--', label='Ideal = 2.0')
                axes[1, 0].set_title('Autocorrelation Test')
                axes[1, 0].set_ylabel('Durbin-Watson Statistic')
                axes[1, 0].legend()
                axes[1, 0].text(0, dw_stat + 0.1, f'{dw_interpretation}',
                               ha='center', fontweight='bold')

            # Influence measures
            if 'influence_measures' in diagnostics:
                influence_data = diagnostics['influence_measures']

                high_leverage = influence_data.get('high_leverage_points', 0)
                influential = influence_data.get('influential_points', 0)

                categories = ['High Leverage', 'Influential Points']
                counts = [high_leverage, influential]

                bars = axes[1, 1].bar(categories, counts,
                                     color=[self.color_palette['warning'],
                                           self.color_palette['secondary']], alpha=0.8)
                axes[1, 1].set_title('Influential Observations')
                axes[1, 1].set_ylabel('Count')

                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                    str(int(count)), ha='center', va='bottom')

            # Assumption tests
            if 'assumption_tests' in diagnostics:
                assumption_data = diagnostics['assumption_tests']

                tests = []
                results = []
                p_vals = []

                for test_name, test_result in assumption_data.items():
                    if isinstance(test_result, dict) and 'assumption_met' in test_result:
                        tests.append(test_name.replace('_', ' ').title())
                        results.append(test_result['assumption_met'])
                        p_vals.append(test_result.get('p_value', 1.0))

                if tests:
                    colors = [self.color_palette['accent'] if met else self.color_palette['secondary']
                             for met in results]

                    bars = axes[1, 2].bar(tests, p_vals, color=colors, alpha=0.8)
                    axes[1, 2].axhline(y=0.05, color=self.color_palette['warning'],
                                      linestyle='--', linewidth=2, label='α = 0.05')
                    axes[1, 2].set_title('Regression Assumptions')
                    axes[1, 2].set_ylabel('P-Value')
                    axes[1, 2].legend()
                    plt.setp(axes[1, 2].xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        filename = f"{self.output_dir}/regression_analysis_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['regression_analysis'] = filename

        return files

    def _create_time_series_visualizations(self, results: Dict[str, Any],
                                         data_name: str) -> Dict[str, str]:
        """Create time series analysis visualizations."""
        files = {}

        # Time Series Analysis Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Time Series Analysis: {data_name}', fontsize=20, fontweight='bold')

        # Trend analysis
        if 'trend_analysis' in results:
            trend_data = results['trend_analysis']

            # Linear trend visualization
            slope = trend_data.get('linear_slope', 0)
            r_squared = trend_data.get('linear_r_squared', 0)
            trend_direction = trend_data.get('trend_direction', 'no trend')

            # Trend strength gauge
            trend_strength = abs(slope)
            max_strength = max(trend_strength * 2, 0.1)  # Scale for visualization

            colors = [self.color_palette['accent'] if 'increasing' in trend_direction
                     else self.color_palette['secondary'] if 'decreasing' in trend_direction
                     else self.color_palette['neutral']]

            axes[0, 0].bar(['Trend Strength'], [trend_strength], color=colors[0], alpha=0.8)
            axes[0, 0].set_title(f'Trend Analysis\n({trend_direction.title()})')
            axes[0, 0].set_ylabel('Absolute Slope')
            axes[0, 0].text(0, trend_strength + max_strength*0.05,
                           f'R² = {r_squared:.4f}', ha='center', fontweight='bold')

            # Mann-Kendall test results
            mk_stat = trend_data.get('mann_kendall_statistic', 0)
            mk_p = trend_data.get('mann_kendall_p_value', 1.0)
            significant_trend = trend_data.get('significant_trend', False)

            axes[0, 1].bar(['Mann-Kendall'], [mk_p],
                          color=self.color_palette['accent'] if significant_trend
                          else self.color_palette['secondary'], alpha=0.8)
            axes[0, 1].axhline(y=0.05, color=self.color_palette['warning'],
                              linestyle='--', linewidth=2, label='α = 0.05')
            axes[0, 1].set_title('Trend Significance Test')
            axes[0, 1].set_ylabel('P-Value')
            axes[0, 1].legend()
            axes[0, 1].text(0, mk_p + 0.02, f'S = {mk_stat}', ha='center', fontweight='bold')

        # Autocorrelation analysis
        if 'autocorrelation' in results:
            autocorr_data = results['autocorrelation']

            autocorrelations = autocorr_data.get('autocorrelations', [])
            lags = autocorr_data.get('lags', [])

            if len(autocorrelations) > 1 and len(lags) > 1:
                axes[0, 2].stem(lags[1:], autocorrelations[1:], basefmt=' ')
                axes[0, 2].axhline(y=0, color=self.color_palette['neutral'], linestyle='-', alpha=0.5)
                axes[0, 2].set_title('Autocorrelation Function')
                axes[0, 2].set_xlabel('Lag')
                axes[0, 2].set_ylabel('Autocorrelation')
                axes[0, 2].grid(True, alpha=0.3)

                # Add significance bounds (approximate)
                n = len(autocorrelations)
                bound = 1.96 / np.sqrt(n) if n > 0 else 0.1
                axes[0, 2].axhline(y=bound, color=self.color_palette['secondary'],
                                  linestyle='--', alpha=0.7, label='95% bounds')
                axes[0, 2].axhline(y=-bound, color=self.color_palette['secondary'],
                                  linestyle='--', alpha=0.7)
                axes[0, 2].legend()

            # Ljung-Box test result
            lb_stat = autocorr_data.get('ljung_box_statistic', 0)
            lb_p = autocorr_data.get('ljung_box_p_value', 1.0)
            serial_corr = autocorr_data.get('serial_correlation_detected', False)

            axes[1, 0].bar(['Ljung-Box Test'], [lb_p],
                          color=self.color_palette['secondary'] if serial_corr
                          else self.color_palette['accent'], alpha=0.8)
            axes[1, 0].axhline(y=0.05, color=self.color_palette['warning'],
                              linestyle='--', linewidth=2, label='α = 0.05')
            axes[1, 0].set_title('Serial Correlation Test')
            axes[1, 0].set_ylabel('P-Value')
            axes[1, 0].legend()
            axes[1, 0].text(0, lb_p + 0.02, f'LB = {lb_stat:.2f}',
                           ha='center', fontweight='bold')

        # Stationarity tests
        if 'stationarity_tests' in results:
            stationarity_data = results['stationarity_tests']

            test_names = []
            test_results = []
            interpretations = []

            for test_name, test_result in stationarity_data.items():
                if isinstance(test_result, dict) and 'is_stationary' in test_result:
                    test_names.append(test_name.replace('_', ' ').title())
                    test_results.append(test_result['is_stationary'])
                    interpretations.append('Stationary' if test_result['is_stationary']
                                         else 'Non-Stationary')

            if test_names:
                colors = [self.color_palette['accent'] if stat else self.color_palette['secondary']
                         for stat in test_results]

                y_pos = np.arange(len(test_names))
                axes[1, 1].barh(y_pos, [1 if stat else 0 for stat in test_results],
                               color=colors, alpha=0.8)
                axes[1, 1].set_title('Stationarity Tests')
                axes[1, 1].set_xlabel('Stationarity Result')
                axes[1, 1].set_yticks(y_pos)
                axes[1, 1].set_yticklabels(test_names)
                axes[1, 1].set_xlim(0, 1.2)

                # Add result labels
                for i, (result, interp) in enumerate(zip(test_results, interpretations)):
                    axes[1, 1].text(0.6, i, interp, ha='center', va='center',
                                   fontweight='bold', color='white')

        # Change point detection
        if 'change_point_detection' in results:
            change_data = results['change_point_detection']

            n_changes = change_data.get('n_change_points', 0)
            method = change_data.get('method', 'Unknown')

            # Visualize change point frequency
            axes[1, 2].bar(['Change Points'], [n_changes],
                          color=self.color_palette['warning'] if n_changes > 0
                          else self.color_palette['accent'], alpha=0.8)
            axes[1, 2].set_title(f'Change Point Detection\n({method})')
            axes[1, 2].set_ylabel('Number of Change Points')

            if n_changes > 0:
                axes[1, 2].text(0, n_changes + 0.1, f'{n_changes} detected',
                               ha='center', va='bottom', fontweight='bold')
            else:
                axes[1, 2].text(0, 0.1, 'No change points',
                               ha='center', va='bottom', fontweight='bold')

        # Seasonal analysis
        if 'seasonal_decomposition' in results:
            seasonal_data = results['seasonal_decomposition']

            # Find strongest seasonal pattern
            max_strength = 0
            best_period = 0

            for key, data in seasonal_data.items():
                if key.startswith('period_') and isinstance(data, dict):
                    strength = data.get('seasonal_strength', 0)
                    if strength > max_strength:
                        max_strength = strength
                        best_period = int(key.split('_')[1])

            # Override one of the existing plots if we have seasonal data
            if max_strength > 0:
                # Replace the change point plot with seasonal analysis
                axes[1, 2].clear()

                periods = []
                strengths = []
                has_seasonality = []

                for key, data in seasonal_data.items():
                    if key.startswith('period_') and isinstance(data, dict):
                        period = int(key.split('_')[1])
                        strength = data.get('seasonal_strength', 0)
                        periods.append(period)
                        strengths.append(strength)
                        has_seasonality.append(data.get('has_seasonality', False))

                colors = [self.color_palette['accent'] if has_season
                         else self.color_palette['neutral'] for has_season in has_seasonality]

                bars = axes[1, 2].bar(range(len(periods)), strengths, color=colors, alpha=0.8)
                axes[1, 2].set_title('Seasonal Pattern Analysis')
                axes[1, 2].set_xlabel('Period')
                axes[1, 2].set_ylabel('Seasonal Strength')
                axes[1, 2].set_xticks(range(len(periods)))
                axes[1, 2].set_xticklabels(periods)

                # Highlight strongest pattern
                if best_period > 0:
                    best_idx = periods.index(best_period)
                    bars[best_idx].set_edgecolor(self.color_palette['secondary'])
                    bars[best_idx].set_linewidth(3)

        plt.tight_layout()
        filename = f"{self.output_dir}/time_series_analysis_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['time_series_analysis'] = filename

        return files

    def _create_partition_visualizations(self, results: Dict[str, Any],
                                       data_name: str) -> Dict[str, str]:
        """Create partition-specific visualizations."""
        files = {}

        # Partition Analysis Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Partition Sequence Analysis: {data_name}',
                    fontsize=20, fontweight='bold')

        # Basic properties
        if 'basic_properties' in results:
            props = results['basic_properties']

            # Partition statistics
            stat_names = ['Total Sum', 'Max Partition', 'Unique Parts', 'Overall GCD']
            stat_values = [
                props.get('total_sum', 0),
                props.get('max_partition', 0),
                props.get('unique_partitions', 0),
                props.get('overall_gcd', 1)
            ]

            # Normalize for visualization (log scale for large values)
            normalized_values = []
            for val in stat_values:
                if val > 0:
                    normalized_values.append(np.log10(val) if val > 1 else val)
                else:
                    normalized_values.append(0)

            bars = axes[0, 0].bar(range(len(stat_names)), normalized_values,
                                 color=self.color_palette['primary'], alpha=0.8)
            axes[0, 0].set_title('Partition Properties (Log Scale)')
            axes[0, 0].set_xticks(range(len(stat_names)))
            axes[0, 0].set_xticklabels(stat_names, rotation=45)
            axes[0, 0].set_ylabel('Log₁₀(Value)')

            # Add actual value labels
            for bar, orig_val in zip(bars, stat_values):
                height = bar.get_height()
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.05,
                               str(orig_val), ha='center', va='bottom', fontweight='bold')

            # Diversity analysis
            diversity_ratio = props.get('diversity_ratio', 0)
            sequence_length = props.get('sequence_length', 1)

            # Pie chart for diversity
            unique_parts = props.get('unique_partitions', 0)
            repeated_parts = sequence_length - unique_parts

            if unique_parts > 0 or repeated_parts > 0:
                sizes = [unique_parts, repeated_parts] if repeated_parts > 0 else [unique_parts]
                labels = ['Unique', 'Repeated'] if repeated_parts > 0 else ['All Unique']
                colors_diversity = [self.color_palette['accent'], self.color_palette['neutral']][:len(sizes)]

                wedges, texts, autotexts = axes[0, 1].pie(sizes, labels=labels, colors=colors_diversity,
                                                         autopct='%1.1f%%', startangle=90)
                axes[0, 1].set_title('Partition Diversity')

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

        # Growth analysis
        if 'growth_analysis' in results:
            growth_data = results['growth_analysis']

            # Model comparison
            model_names = []
            r_squared_values = []

            growth_models = growth_data.get('growth_models', {})
            for model_name, model_data in growth_models.items():
                if isinstance(model_data, dict):
                    # Extract R-squared from different model types
                    r_sq = 0
                    if 'r_squared' in model_data:
                        r_sq = model_data['r_squared']
                    elif 'log_linear_fit' in model_data and isinstance(model_data['log_linear_fit'], dict):
                        r_sq = model_data['log_linear_fit'].get('r_squared', 0)
                    elif 'log_log_fit' in model_data and isinstance(model_data['log_log_fit'], dict):
                        r_sq = model_data['log_log_fit'].get('r_squared', 0)

                    if r_sq > 0:
                        model_names.append(model_name.replace('_', ' ').title())
                        r_squared_values.append(r_sq)

            if model_names:
                best_model = growth_data.get('best_fit_model', '').replace('_', ' ').title()
                colors = [self.color_palette['primary'] if name == best_model
                         else self.color_palette['accent'] for name in model_names]

                bars = axes[0, 2].bar(range(len(model_names)), r_squared_values,
                                     color=colors, alpha=0.8)
                axes[0, 2].set_title('Growth Model Comparison')
                axes[0, 2].set_xlabel('Model Type')
                axes[0, 2].set_ylabel('R-squared')
                axes[0, 2].set_xticks(range(len(model_names)))
                axes[0, 2].set_xticklabels(model_names, rotation=45)

                # Highlight best model
                for i, (bar, name, r_sq) in enumerate(zip(bars, model_names, r_squared_values)):
                    if name == best_model:
                        bar.set_edgecolor(self.color_palette['secondary'])
                        bar.set_linewidth(3)

                    # Add R² labels
                    height = bar.get_height()
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{r_sq:.3f}', ha='center', va='bottom')

        # Multiplicative structure analysis
        if 'multiplicative_structure' in results:
            mult_data = results['multiplicative_structure']

            # Factor count analysis
            factor_counts = mult_data.get('factor_count_sequence', [])
            distinct_factor_counts = mult_data.get('distinct_factor_count_sequence', [])

            if factor_counts and distinct_factor_counts:
                indices = list(range(1, len(factor_counts) + 1))

                axes[1, 0].plot(indices, factor_counts, 'o-',
                               color=self.color_palette['primary'], linewidth=2,
                               markersize=6, label='Total Factors')
                axes[1, 0].plot(indices, distinct_factor_counts, 's-',
                               color=self.color_palette['secondary'], linewidth=2,
                               markersize=6, label='Distinct Factors')
                axes[1, 0].set_title('Prime Factor Analysis')
                axes[1, 0].set_xlabel('Term Index')
                axes[1, 0].set_ylabel('Factor Count')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)

            # Average factor statistics
            mean_factors = mult_data.get('mean_factor_count', 0)
            mean_distinct = mult_data.get('mean_distinct_factor_count', 0)

            axes[1, 1].bar(['Mean Factors', 'Mean Distinct'],
                          [mean_factors, mean_distinct],
                          color=[self.color_palette['primary'], self.color_palette['secondary']],
                          alpha=0.8)
            axes[1, 1].set_title('Average Factor Statistics')
            axes[1, 1].set_ylabel('Average Count')

            # Add value labels
            for i, val in enumerate([mean_factors, mean_distinct]):
                axes[1, 1].text(i, val + 0.05, f'{val:.2f}',
                               ha='center', va='bottom', fontweight='bold')

        # Partition differences and ratios
        if 'basic_properties' in results:
            props = results['basic_properties']

            differences = props.get('differences', [])[:10]  # First 10
            ratios = props.get('ratios', [])[:10]  # First 10

            if differences or ratios:
                # Create subplot for differences and ratios
                if differences:
                    indices_diff = list(range(1, len(differences) + 1))
                    axes[1, 2].bar(indices_diff, differences,
                                  color=self.color_palette['warning'], alpha=0.8,
                                  width=0.4, label='Differences')

                if ratios:
                    # Plot ratios on secondary y-axis
                    ax2 = axes[1, 2].twinx()
                    indices_ratio = list(range(1, len(ratios) + 1))
                    ax2.plot(indices_ratio, ratios, 'ro-',
                            color=self.color_palette['secondary'], linewidth=2,
                            markersize=4, label='Ratios')
                    ax2.set_ylabel('Ratio', color=self.color_palette['secondary'])
                    ax2.tick_params(axis='y', labelcolor=self.color_palette['secondary'])

                axes[1, 2].set_title('First Differences & Ratios')
                axes[1, 2].set_xlabel('Term Index')
                axes[1, 2].set_ylabel('Difference', color=self.color_palette['warning'])
                axes[1, 2].tick_params(axis='y', labelcolor=self.color_palette['warning'])

                # Add legends
                lines1, labels1 = axes[1, 2].get_legend_handles_labels()
                if ratios:
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    axes[1, 2].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
                else:
                    axes[1, 2].legend()

        plt.tight_layout()
        filename = f"{self.output_dir}/partition_analysis_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['partition_analysis'] = filename

        return files

    def _create_summary_dashboard(self, results: Dict[str, Any],
                                data_name: str) -> Optional[str]:
        """Create comprehensive summary dashboard."""

        # Create a large summary figure
        fig = plt.figure(figsize=(24, 16))
        fig.suptitle(f'Comprehensive Analysis Summary: {data_name}',
                    fontsize=24, fontweight='bold', y=0.98)

        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3,
                             left=0.05, right=0.95, top=0.92, bottom=0.05)

        # 1. Key Statistics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        if 'descriptive_statistics' in results:
            stats = results['descriptive_statistics']
            key_stats = ['mean', 'std_dev', 'skewness', 'kurtosis']
            values = [stats.get(stat, 0) for stat in key_stats]
            labels = ['Mean', 'Std Dev', 'Skewness', 'Kurtosis']

            bars = ax1.bar(labels, values, color=self.color_palette['primary'], alpha=0.8)
            ax1.set_title('Key Statistics', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        # 2. Normality Assessment (top center-left)
        ax2 = fig.add_subplot(gs[0, 1])
        if 'normality_tests' in results:
            normality_data = results['normality_tests']
            normal_count = 0
            total_tests = 0

            for test_name, test_result in normality_data.items():
                if isinstance(test_result, dict) and 'is_normal' in test_result:
                    total_tests += 1
                    if test_result['is_normal']:
                        normal_count += 1

            if total_tests > 0:
                normal_pct = (normal_count / total_tests) * 100
                non_normal_pct = 100 - normal_pct

                sizes = [normal_pct, non_normal_pct]
                labels = ['Normal', 'Non-Normal']
                colors = [self.color_palette['accent'], self.color_palette['secondary']]

                wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors,
                                                  autopct='%1.0f%%', startangle=90)
                ax2.set_title('Normality Consensus', fontweight='bold')

                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')

        # 3. Distribution Fitting (top center-right)
        ax3 = fig.add_subplot(gs[0, 2])
        if 'distribution_fitting' in results:
            fitting_data = results['distribution_fitting']
            good_fits = 0
            total_fits = 0

            for dist_name, dist_result in fitting_data.items():
                if isinstance(dist_result, dict) and 'good_fit' in dist_result and dist_name != 'best_fit':
                    total_fits += 1
                    if dist_result['good_fit']:
                        good_fits += 1

            if total_fits > 0:
                fit_pct = (good_fits / total_fits) * 100
                poor_fit_pct = 100 - fit_pct

                ax3.bar(['Good Fits', 'Poor Fits'], [good_fits, total_fits - good_fits],
                       color=[self.color_palette['accent'], self.color_palette['secondary']],
                       alpha=0.8)
                ax3.set_title('Distribution Fitting Results', fontweight='bold')
                ax3.set_ylabel('Count')

                # Add percentage labels
                ax3.text(0, good_fits + 0.1, f'{fit_pct:.0f}%',
                        ha='center', va='bottom', fontweight='bold')
                ax3.text(1, (total_fits - good_fits) + 0.1, f'{poor_fit_pct:.0f}%',
                        ha='center', va='bottom', fontweight='bold')

        # 4. Outlier Analysis (top right)
        ax4 = fig.add_subplot(gs[0, 3])
        if 'outlier_analysis' in results:
            outlier_data = results['outlier_analysis']

            methods = []
            proportions = []

            for method, result in outlier_data.items():
                if isinstance(result, dict) and 'proportion' in result:
                    methods.append(method.replace('_', ' ').title())
                    proportions.append(result['proportion'] * 100)

            if methods:
                bars = ax4.bar(methods, proportions,
                              color=self.color_palette['warning'], alpha=0.8)
                ax4.set_title('Outlier Detection', fontweight='bold')
                ax4.set_ylabel('Outlier Percentage (%)')
                plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

                for bar, prop in zip(bars, proportions):
                    height = bar.get_height()
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                            f'{prop:.1f}%', ha='center', va='bottom', fontsize=9)

        # 5-8. Time series analysis (middle row)
        if 'trend_analysis' in results:
            # Trend strength
            ax5 = fig.add_subplot(gs[1, 0])
            trend_data = results['trend_analysis']

            slope = abs(trend_data.get('linear_slope', 0))
            r_squared = trend_data.get('linear_r_squared', 0)
            significant = trend_data.get('significant_trend', False)

            color = self.color_palette['accent'] if significant else self.color_palette['neutral']
            ax5.bar(['Trend Strength'], [slope], color=color, alpha=0.8)
            ax5.set_title('Trend Analysis', fontweight='bold')
            ax5.set_ylabel('|Slope|')
            ax5.text(0, slope + slope*0.05, f'R²={r_squared:.3f}',
                    ha='center', va='bottom', fontweight='bold')

        if 'autocorrelation' in results:
            # Autocorrelation summary
            ax6 = fig.add_subplot(gs[1, 1])
            autocorr_data = results['autocorrelation']

            serial_correlation = autocorr_data.get('serial_correlation_detected', False)
            ljung_box_p = autocorr_data.get('ljung_box_p_value', 1.0)

            color = self.color_palette['secondary'] if serial_correlation else self.color_palette['accent']
            ax6.bar(['Serial Correlation'], [1 if serial_correlation else 0],
                   color=color, alpha=0.8)
            ax6.set_title('Autocorrelation Test', fontweight='bold')
            ax6.set_ylabel('Detected')
            ax6.set_ylim(0, 1.2)
            ax6.text(0, 0.6, 'Present' if serial_correlation else 'Absent',
                    ha='center', va='center', fontweight='bold', color='white')

        # 9-12. Regression analysis (if available)
        if 'linear_regression' in results:
            ax9 = fig.add_subplot(gs[1, 2])
            linear_data = results['linear_regression']

            if 'error' not in linear_data:
                r_squared = linear_data.get('r_squared', 0)
                significant = linear_data.get('significant', False)

                color = self.color_palette['accent'] if significant else self.color_palette['neutral']
                ax9.bar(['Linear Fit'], [r_squared], color=color, alpha=0.8)
                ax9.set_title('Linear Regression', fontweight='bold')
                ax9.set_ylabel('R-squared')
                ax9.set_ylim(0, 1)
                ax9.text(0, r_squared + 0.02, f'{r_squared:.3f}',
                        ha='center', va='bottom', fontweight='bold')

        # 13-16. Partition-specific analysis (bottom rows)
        if 'basic_properties' in results:
            # Growth pattern
            ax13 = fig.add_subplot(gs[2, :2])  # Span 2 columns

            props = results['basic_properties']
            ratios = props.get('ratios', [])[:20]  # First 20 ratios

            if ratios:
                indices = list(range(1, len(ratios) + 1))
                ax13.plot(indices, ratios, 'o-', color=self.color_palette['primary'],
                         linewidth=2, markersize=4)
                ax13.set_title('Growth Pattern (First 20 Ratios)', fontweight='bold')
                ax13.set_xlabel('Term Index')
                ax13.set_ylabel('Ratio to Previous Term')
                ax13.grid(True, alpha=0.3)

                # Add mean line
                mean_ratio = np.mean(ratios)
                ax13.axhline(y=mean_ratio, color=self.color_palette['secondary'],
                           linestyle='--', linewidth=2, label=f'Mean = {mean_ratio:.3f}')
                ax13.legend()

        # Multiplicative structure summary
        if 'multiplicative_structure' in results:
            ax14 = fig.add_subplot(gs[2, 2:])  # Span 2 columns

            mult_data = results['multiplicative_structure']
            mean_factors = mult_data.get('mean_factor_count', 0)
            mean_distinct = mult_data.get('mean_distinct_factor_count', 0)

            categories = ['Average\nTotal Factors', 'Average\nDistinct Factors']
            values = [mean_factors, mean_distinct]

            bars = ax14.bar(categories, values,
                           color=[self.color_palette['primary'], self.color_palette['secondary']],
                           alpha=0.8)
            ax14.set_title('Prime Factorization Summary', fontweight='bold')
            ax14.set_ylabel('Average Count')

            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax14.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                         f'{val:.2f}', ha='center', va='bottom', fontweight='bold')

        # Summary statistics table (bottom)
        ax_table = fig.add_subplot(gs[3, :])
        ax_table.axis('tight')
        ax_table.axis('off')

        # Create summary table
        table_data = []

        if 'descriptive_statistics' in results:
            stats = results['descriptive_statistics']
            table_data.append([
                'Sample Size', str(results.get('sample_size', 'N/A')),
                'Mean', f"{stats.get('mean', 0):.4f}",
                'Std Dev', f"{stats.get('std_dev', 0):.4f}"
            ])
            table_data.append([
                'Median', f"{stats.get('median', 0):.4f}",
                'Skewness', f"{stats.get('skewness', 0):.4f}",
                'Kurtosis', f"{stats.get('kurtosis', 0):.4f}"
            ])

        # Add test results
        normality_consensus = "Mixed"
        if 'normality_tests' in results:
            normality_data = results['normality_tests']
            normal_votes = sum(1 for test_result in normality_data.values()
                              if isinstance(test_result, dict) and test_result.get('is_normal', False))
            total_votes = sum(1 for test_result in normality_data.values()
                             if isinstance(test_result, dict) and 'is_normal' in test_result)

            if total_votes > 0:
                if normal_votes == total_votes:
                    normality_consensus = "Normal"
                elif normal_votes == 0:
                    normality_consensus = "Non-Normal"
                else:
                    normality_consensus = f"Mixed ({normal_votes}/{total_votes})"

        best_distribution = "Unknown"
        if 'distribution_fitting' in results:
            fitting_data = results['distribution_fitting']
            best_distribution = fitting_data.get('best_fit', 'Unknown').replace('_', ' ').title()

        trend_direction = "No Trend"
        if 'trend_analysis' in results:
            trend_data = results['trend_analysis']
            trend_direction = trend_data.get('trend_direction', 'no trend').title()

        table_data.append([
            'Normality', normality_consensus,
            'Best Distribution', best_distribution,
            'Trend', trend_direction
        ])

        if table_data:
            table = ax_table.table(cellText=table_data,
                                  cellLoc='center',
                                  loc='center',
                                  bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1.2, 2)

            # Style the table
            for i in range(len(table_data)):
                for j in range(6):
                    cell = table[(i, j)]
                    if j % 2 == 0:  # Headers
                        cell.set_facecolor(self.color_palette['primary'])
                        cell.set_text_props(weight='bold', color='white')
                    else:  # Values
                        cell.set_facecolor(self.color_palette['background'])
                        cell.set_text_props(color=self.color_palette['text'])

        # Add timestamp and analysis info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=10,
                style='italic', color=self.color_palette['neutral'])
        fig.text(0.98, 0.02, f'Analysis: {data_name}', fontsize=10,
                style='italic', color=self.color_palette['neutral'], ha='right')

        filename = f"{self.output_dir}/summary_dashboard_{data_name}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()

        return filename

    def create_comparative_analysis(self, results1: Dict[str, Any], results2: Dict[str, Any],
                                   names: Tuple[str, str] = ("Dataset 1", "Dataset 2")) -> Dict[str, str]:
        """Create comparative analysis visualizations between two datasets."""

        files = {}

        # Comparative Analysis Dashboard
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Comparative Analysis: {names[0]} vs {names[1]}',
                    fontsize=20, fontweight='bold')

        # Extract comparable statistics
        stats1 = results1.get('descriptive_statistics', {})
        stats2 = results2.get('descriptive_statistics', {})

        if stats1 and stats2 and 'error' not in stats1 and 'error' not in stats2:
            # Statistical comparison
            stat_names = ['Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis']
            stats1_values = [stats1.get('mean', 0), stats1.get('median', 0),
                           stats1.get('std_dev', 0), stats1.get('skewness', 0),
                           stats1.get('kurtosis', 0)]
            stats2_values = [stats2.get('mean', 0), stats2.get('median', 0),
                           stats2.get('std_dev', 0), stats2.get('skewness', 0),
                           stats2.get('kurtosis', 0)]

            x = np.arange(len(stat_names))
            width = 0.35

            bars1 = axes[0, 0].bar(x - width/2, stats1_values, width,
                                  label=names[0], color=self.color_palette['primary'], alpha=0.8)
            bars2 = axes[0, 0].bar(x + width/2, stats2_values, width,
                                  label=names[1], color=self.color_palette['secondary'], alpha=0.8)

            axes[0, 0].set_title('Statistical Comparison')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(stat_names, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Normality comparison
        norm1 = results1.get('normality_tests', {})
        norm2 = results2.get('normality_tests', {})

        if norm1 and norm2:
            # Count normal vs non-normal for each dataset
            def count_normality_votes(norm_data):
                normal_votes = 0
                total_votes = 0
                for test_result in norm_data.values():
                    if isinstance(test_result, dict) and 'is_normal' in test_result:
                        total_votes += 1
                        if test_result['is_normal']:
                            normal_votes += 1
                return normal_votes, total_votes

            normal1, total1 = count_normality_votes(norm1)
            normal2, total2 = count_normality_votes(norm2)

            if total1 > 0 and total2 > 0:
                norm_pct1 = (normal1 / total1) * 100
                norm_pct2 = (normal2 / total2) * 100

                categories = [names[0], names[1]]
                normal_pcts = [norm_pct1, norm_pct2]

                bars = axes[0, 1].bar(categories, normal_pcts,
                                     color=[self.color_palette['accent'], self.color_palette['warning']],
                                     alpha=0.8)
                axes[0, 1].set_title('Normality Assessment')
                axes[0, 1].set_ylabel('% Tests Indicating Normal')
                axes[0, 1].set_ylim(0, 100)

                for bar, pct in zip(bars, normal_pcts):
                    height = bar.get_height()
                    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 2,
                                   f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')

        # Distribution fitting comparison
        fit1 = results1.get('distribution_fitting', {})
        fit2 = results2.get('distribution_fitting', {})

        if fit1 and fit2:
            best1 = fit1.get('best_fit', 'Unknown').replace('_', ' ').title()
            best2 = fit2.get('best_fit', 'Unknown').replace('_', ' ').title()

            # Count good fits for each
            good_fits1 = sum(1 for dist_result in fit1.values()
                           if isinstance(dist_result, dict) and dist_result.get('good_fit', False))
            total_fits1 = sum(1 for dist_result in fit1.values()
                            if isinstance(dist_result, dict) and 'good_fit' in dist_result)

            good_fits2 = sum(1 for dist_result in fit2.values()
                           if isinstance(dist_result, dict) and dist_result.get('good_fit', False))
            total_fits2 = sum(1 for dist_result in fit2.values()
                            if isinstance(dist_result, dict) and 'good_fit' in dist_result)

            if total_fits1 > 0 and total_fits2 > 0:
                fit_pct1 = (good_fits1 / total_fits1) * 100
                fit_pct2 = (good_fits2 / total_fits2) * 100

                categories = [names[0], names[1]]
                fit_pcts = [fit_pct1, fit_pct2]

                bars = axes[0, 2].bar(categories, fit_pcts,
                                     color=[self.color_palette['primary'], self.color_palette['secondary']],
                                     alpha=0.8)
                axes[0, 2].set_title('Distribution Fitting Success')
                axes[0, 2].set_ylabel('% Good Fits')
                axes[0, 2].set_ylim(0, 100)

                for bar, pct in zip(bars, fit_pcts):
                    height = bar.get_height()
                    axes[0, 2].text(bar.get_x() + bar.get_width()/2., height + 2,
                                   f'{pct:.0f}%', ha='center', va='bottom', fontweight='bold')

                # Add best distribution labels
                axes[0, 2].text(0, -10, f'Best: {best1}', ha='center', va='top',
                               fontsize=10, style='italic')
                axes[0, 2].text(1, -10, f'Best: {best2}', ha='center', va='top',
                               fontsize=10, style='italic')

        # Trend analysis comparison
        trend1 = results1.get('trend_analysis', {})
        trend2 = results2.get('trend_analysis', {})

        if trend1 and trend2:
            slope1 = abs(trend1.get('linear_slope', 0))
            slope2 = abs(trend2.get('linear_slope', 0))
            r2_1 = trend1.get('linear_r_squared', 0)
            r2_2 = trend2.get('linear_r_squared', 0)

            # Trend strength comparison
            categories = [names[0], names[1]]
            slopes = [slope1, slope2]
            r_squareds = [r2_1, r2_2]

            x = np.arange(len(categories))
            width = 0.35

            bars1 = axes[1, 0].bar(x - width/2, slopes, width, label='|Slope|',
                                  color=self.color_palette['primary'], alpha=0.8)
            bars2 = axes[1, 0].bar(x + width/2, r_squareds, width, label='R²',
                                  color=self.color_palette['accent'], alpha=0.8)

            axes[1, 0].set_title('Trend Analysis Comparison')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(categories)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Outlier comparison
        outlier1 = results1.get('outlier_analysis', {})
        outlier2 = results2.get('outlier_analysis', {})

        if outlier1 and outlier2:
            # Get outlier proportions for common methods
            methods = set(outlier1.keys()) & set(outlier2.keys())

            if methods:
                method_names = []
                prop1_values = []
                prop2_values = []

                for method in methods:
                    if (isinstance(outlier1[method], dict) and 'proportion' in outlier1[method] and
                        isinstance(outlier2[method], dict) and 'proportion' in outlier2[method]):
                        method_names.append(method.replace('_', ' ').title())
                        prop1_values.append(outlier1[method]['proportion'] * 100)
                        prop2_values.append(outlier2[method]['proportion'] * 100)

                if method_names:
                    x = np.arange(len(method_names))
                    width = 0.35

                    bars1 = axes[1, 1].bar(x - width/2, prop1_values, width, label=names[0],
                                          color=self.color_palette['primary'], alpha=0.8)
                    bars2 = axes[1, 1].bar(x + width/2, prop2_values, width, label=names[1],
                                          color=self.color_palette['secondary'], alpha=0.8)

                    axes[1, 1].set_title('Outlier Detection Comparison')
                    axes[1, 1].set_ylabel('Outlier Percentage (%)')
                    axes[1, 1].set_xticks(x)
                    axes[1, 1].set_xticklabels(method_names, rotation=45)
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)

        # Overall quality assessment
        quality_scores = []
        quality_names = []

        # Calculate simple quality scores based on available metrics
        for i, (results, name) in enumerate([(results1, names[0]), (results2, names[1])]):
            score = 0
            components = 0

            # Normality score
            if 'normality_tests' in results:
                norm_data = results['normality_tests']
                normal_votes, total_votes = count_normality_votes(norm_data)
                if total_votes > 0:
                    score += (normal_votes / total_votes) * 25  # Up to 25 points
                    components += 1

            # Fitting score
            if 'distribution_fitting' in results:
                fit_data = results['distribution_fitting']
                good_fits = sum(1 for dist_result in fit_data.values()
                              if isinstance(dist_result, dict) and dist_result.get('good_fit', False))
                total_fits = sum(1 for dist_result in fit_data.values()
                               if isinstance(dist_result, dict) and 'good_fit' in dist_result)
                if total_fits > 0:
                    score += (good_fits / total_fits) * 25  # Up to 25 points
                    components += 1

            # Trend clarity score
            if 'trend_analysis' in results:
                trend_data = results['trend_analysis']
                r_squared = trend_data.get('linear_r_squared', 0)
                score += r_squared * 25  # Up to 25 points
                components += 1

            # Low outlier bonus
            if 'outlier_analysis' in results:
                outlier_data = results['outlier_analysis']
                avg_outlier_prop = 0
                outlier_methods = 0
                for method_result in outlier_data.values():
                    if isinstance(method_result, dict) and 'proportion' in method_result:
                        avg_outlier_prop += method_result['proportion']
                        outlier_methods += 1

                if outlier_methods > 0:
                    avg_outlier_prop /= outlier_methods
                    # Bonus for low outlier rates (inverted score)
                    score += (1 - min(avg_outlier_prop, 1)) * 25  # Up to 25 points
                    components += 1

            if components > 0:
                score = score / components * (components / 4)  # Normalize to 0-100 scale
                quality_scores.append(score)
                quality_names.append(name)

        if len(quality_scores) == 2:
            bars = axes[1, 2].bar(quality_names, quality_scores,
                                 color=[self.color_palette['primary'], self.color_palette['secondary']],
                                 alpha=0.8)
            axes[1, 2].set_title('Overall Quality Assessment')
            axes[1, 2].set_ylabel('Quality Score (0-100)')
            axes[1, 2].set_ylim(0, 100)

            for bar, score in zip(bars, quality_scores):
                height = bar.get_height()
                axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 2,
                               f'{score:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        filename = f"{self.output_dir}/comparative_analysis_{names[0]}_vs_{names[1]}.png"
        plt.savefig(filename, dpi=PUBLICATION_DPI, bbox_inches='tight')
        plt.close()
        files['comparative_analysis'] = filename

        return files

    def create_publication_report(self, analysis_results: Dict[str, Any],
                                data_name: str = "partition_sequence") -> str:
        """Generate a comprehensive publication-ready report."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{self.output_dir}/publication_report_{data_name}_{timestamp}.md"

        with open(report_filename, 'w') as f:
            f.write(f"# Statistical Analysis Report: {data_name}\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")

            if 'descriptive_statistics' in analysis_results:
                stats = analysis_results['descriptive_statistics']
                f.write(f"- **Sample Size:** {analysis_results.get('sample_size', 'N/A')}\n")
                f.write(f"- **Mean:** {stats.get('mean', 'N/A'):.4f}\n")
                f.write(f"- **Standard Deviation:** {stats.get('std_dev', 'N/A'):.4f}\n")
                f.write(f"- **Skewness:** {stats.get('skewness', 'N/A'):.4f}\n")
                f.write(f"- **Kurtosis:** {stats.get('kurtosis', 'N/A'):.4f}\n\n")

            # Normality Assessment
            if 'normality_tests' in analysis_results:
                f.write("## Normality Assessment\n\n")
                normality_data = analysis_results['normality_tests']

                normal_count = 0
                total_count = 0

                for test_name, test_result in normality_data.items():
                    if isinstance(test_result, dict) and 'interpretation' in test_result:
                        f.write(f"- **{test_name.replace('_', ' ').title()}:** {test_result['interpretation']} ")
                        f.write(f"(p = {test_result.get('p_value', 'N/A'):.4f})\n")

                        total_count += 1
                        if test_result.get('is_normal', False):
                            normal_count += 1

                f.write(f"\n**Consensus:** {normal_count}/{total_count} tests indicate normality.\n\n")

            # Distribution Fitting
            if 'distribution_fitting' in analysis_results:
                f.write("## Distribution Fitting Results\n\n")
                fitting_data = analysis_results['distribution_fitting']

                if 'best_fit' in fitting_data:
                    best_fit = fitting_data['best_fit'].replace('_', ' ').title()
                    f.write(f"**Best Fitting Distribution:** {best_fit}\n\n")

                f.write("| Distribution | AIC | BIC | KS p-value | Good Fit |\n")
                f.write("|--------------|-----|-----|------------|----------|\n")

                for dist_name, dist_result in fitting_data.items():
                    if isinstance(dist_result, dict) and 'aic' in dist_result and dist_name != 'best_fit':
                        f.write(f"| {dist_name.replace('_', ' ').title()} | ")
                        f.write(f"{dist_result.get('aic', 'N/A'):.2f} | ")
                        f.write(f"{dist_result.get('bic', 'N/A'):.2f} | ")
                        f.write(f"{dist_result.get('ks_p_value', 'N/A'):.4f} | ")
                        f.write(f"{'Yes' if dist_result.get('good_fit', False) else 'No'} |\n")

                f.write("\n")

            # Add other sections as needed...
            f.write("## Methodology\n\n")
            f.write("This analysis was conducted using advanced statistical methods including:\n")
            f.write("- Multiple normality tests (Shapiro-Wilk, D'Agostino-Pearson, Anderson-Darling)\n")
            f.write("- Comprehensive distribution fitting with AIC/BIC model selection\n")
            f.write("- Robust outlier detection using multiple methods\n")
            f.write("- Time series analysis for sequential data\n")
            f.write("- Bootstrap confidence interval estimation\n\n")

            f.write("All visualizations were generated at publication quality (300 DPI) ")
            f.write("using professional color palettes and typography.\n\n")

            f.write("---\n")
            f.write("*This report was automatically generated by the Advanced Statistical Analysis System.*\n")

        return report_filename


# Convenience functions
def quick_visualize(analysis_results: Dict[str, Any], data_name: str = "dataset",
                   output_dir: str = "visualizations") -> Dict[str, str]:
    """Quick comprehensive visualization of analysis results."""
    visualizer = AdvancedVisualizer(output_dir=output_dir)
    return visualizer.create_comprehensive_report(analysis_results, data_name)


def compare_analyses(results1: Dict[str, Any], results2: Dict[str, Any],
                    names: Tuple[str, str] = ("Dataset 1", "Dataset 2"),
                    output_dir: str = "visualizations") -> Dict[str, str]:
    """Quick comparative visualization of two analyses."""
    visualizer = AdvancedVisualizer(output_dir=output_dir)
    return visualizer.create_comparative_analysis(results1, results2, names)


def generate_publication_package(analysis_results: Dict[str, Any],
                               data_name: str = "dataset",
                               output_dir: str = "publication_package") -> Dict[str, Any]:
    """Generate complete publication package with all visualizations and report."""
    visualizer = AdvancedVisualizer(output_dir=output_dir)

    # Generate all visualizations
    viz_files = visualizer.create_comprehensive_report(analysis_results, data_name)

    # Generate publication report
    report_file = visualizer.create_publication_report(analysis_results, data_name)

    return {
        "visualizations": viz_files,
        "report": report_file,
        "output_directory": output_dir
    }


# Example usage
if __name__ == "__main__":
    print("Advanced Visualization System for SFH Partition Research")
    print("=" * 60)
    print("Ready to generate publication-quality visualizations!")
    print("\nKey Features:")
    print("✓ Comprehensive statistical visualizations")
    print("✓ Publication-ready quality (300 DPI)")
    print("✓ Professional color schemes and typography")
    print("✓ Automated report generation")
    print("✓ Comparative analysis capabilities")
    print("✓ Partition-specific visualizations")
    print("✓ Time series analysis plots")
    print("✓ Summary dashboards")
    print("\nUse quick_visualize() or generate_publication_package() to get started!")
