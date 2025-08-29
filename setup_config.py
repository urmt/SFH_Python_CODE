"""
SFH Configuration Setup Script
=============================
Run this script to create the complete config/ directory structure
with all necessary parameter files, seeds, and environment specifications.
"""

import json
import os
from pathlib import Path

def create_config_structure():
    """Create the complete config directory structure."""

    # Create main directories
    directories = [
        "config",
        "config/defaults",
        "config/experiments",
        "results",
        "logs",
        "cache",
        "plots",
        "exports"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create __init__.py files
    init_files = [
        "config/__init__.py",
        "config/defaults/__init__.py",
        "config/experiments/__init__.py"
    ]

    config_init_content = '''"""
SFH Configuration Package
=========================
Configuration management for Sentience Field Hypothesis research framework.
"""

from .parameters import ParameterManager, load_parameters, save_parameters
from .random_seeds import SeedManager, get_reproducible_seed
from .environment import EnvironmentManager, setup_environment

__version__ = "1.0.0"
__all__ = [
    'ParameterManager', 'load_parameters', 'save_parameters',
    'SeedManager', 'get_reproducible_seed',
    'EnvironmentManager', 'setup_environment'
]
'''

    defaults_init_content = '''"""Default configuration files for SFH framework."""'''
    experiments_init_content = '''"""Experiment-specific configuration files."""'''

    init_contents = [config_init_content, defaults_init_content, experiments_init_content]

    for init_file, content in zip(init_files, init_contents):
        with open(init_file, 'w') as f:
            f.write(content)
        print(f"Created: {init_file}")

def create_json_configs():
    """Create all JSON configuration files."""

    # Base parameters
    base_parameters = {
        "field_parameters": {
            "field_strength": 1.0,
            "coupling_constant": 0.1,
            "resonance_frequency": 1.0,
            "decay_rate": 0.01,
            "field_dimension": 3
        },
        "computation_parameters": {
            "max_n": 1000,
            "precision": 50,
            "convergence_threshold": 1e-6,
            "max_iterations": 10000
        },
        "monte_carlo_parameters": {
            "n_samples": 10000,
            "n_chains": 4,
            "burn_in": 1000,
            "thinning": 1,
            "proposal_scale": 0.1
        },
        "analysis_parameters": {
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
            "significance_level": 0.05
        }
    }

    # Monte Carlo configuration
    monte_carlo_config = {
        "sampling_method": "metropolis_hastings",
        "parameter_bounds": {
            "field_strength": [0.1, 10.0],
            "coupling_constant": [0.001, 1.0],
            "resonance_frequency": [0.1, 10.0],
            "decay_rate": [0.001, 0.1]
        },
        "proposal_distributions": {
            "field_strength": {"type": "normal", "scale": 0.1},
            "coupling_constant": {"type": "lognormal", "scale": 0.05},
            "resonance_frequency": {"type": "normal", "scale": 0.05},
            "decay_rate": {"type": "normal", "scale": 0.001}
        },
        "convergence_criteria": {
            "r_hat_threshold": 1.1,
            "effective_sample_size": 400,
            "max_chains": 8
        },
        "adaptive_sampling": {
            "enabled": True,
            "adaptation_period": 100,
            "target_acceptance": 0.44
        }
    }

    # Visualization configuration
    visualization_config = {
        "plot_settings": {
            "figure_size": [10, 8],
            "dpi": 300,
            "font_size": 12,
            "line_width": 2,
            "marker_size": 6,
            "grid": True,
            "legend": True
        },
        "color_schemes": {
            "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"],
            "publication": ["#000000", "#666666", "#999999", "#cccccc"],
            "colorblind": ["#0173b2", "#de8f05", "#029e73", "#cc78bc", "#ca9161"]
        },
        "plot_types": {
            "partition_comparison": {
                "x_label": "n",
                "y_label": "p(n)",
                "title": "Partition Function Comparison",
                "log_scale": True,
                "show_error_bars": True
            },
            "fitness_landscape": {
                "x_label": "Parameter 1",
                "y_label": "Parameter 2",
                "title": "SFH Fitness Landscape",
                "colormap": "viridis",
                "contour_levels": 20
            },
            "convergence_plot": {
                "x_label": "Iteration",
                "y_label": "Fitness Score",
                "title": "Monte Carlo Convergence",
                "show_chains": True
            },
            "statistical_summary": {
                "plot_types": ["histogram", "qq_plot", "residuals"],
                "show_statistics": True
            }
        },
        "export_formats": ["png", "pdf", "svg"],
        "save_high_res": True,
        "output_directory": "plots"
    }

    # Analysis configuration
    analysis_config = {
        "statistical_tests": {
            "kolmogorov_smirnov": {
                "alpha": 0.05,
                "alternative": "two-sided"
            },
            "anderson_darling": {
                "significance_level": 0.05
            },
            "shapiro_wilk": {
                "alpha": 0.05
            },
            "bootstrap": {
                "n_bootstrap": 1000,
                "confidence_intervals": [0.90, 0.95, 0.99],
                "method": "percentile",
                "bias_correction": True
            }
        },
        "goodness_of_fit": {
            "metrics": ["rmse", "mae", "r_squared", "aic", "bic", "log_likelihood"],
            "cross_validation": {
                "k_folds": 5,
                "shuffle": True,
                "stratify": False,
                "random_state": 42
            }
        },
        "outlier_detection": {
            "method": "modified_z_score",
            "threshold": 3.5,
            "remove_outliers": False,
            "flag_outliers": True
        },
        "uncertainty_quantification": {
            "method": "bootstrap",
            "percentiles": [2.5, 25, 50, 75, 97.5],
            "propagate_errors": True
        }
    }

    # Experiment configurations
    experiment_001 = {
        "name": "baseline_validation",
        "description": "Validate SFH against known partition values",
        "field_parameters": {
            "field_strength": 1.5,
            "coupling_constant": 0.12,
            "resonance_frequency": 1.1,
            "decay_rate": 0.015,
            "field_dimension": 3
        },
        "computation_parameters": {
            "max_n": 500,
            "precision": 100,
            "convergence_threshold": 1e-8
        },
        "monte_carlo_parameters": {
            "n_samples": 5000,
            "n_chains": 6,
            "burn_in": 500
        },
        "random_seed": 12345,
        "expected_outcomes": {
            "min_fitness": 0.7,
            "convergence_iterations": 1000
        },
        "tags": ["baseline", "validation", "small_scale"]
    }

    experiment_002 = {
        "name": "parameter_sweep",
        "description": "Systematic parameter space exploration",
        "parameter_ranges": {
            "field_strength": [0.5, 5.0],
            "coupling_constant": [0.01, 0.5],
            "resonance_frequency": [0.5, 2.0],
            "decay_rate": [0.005, 0.05]
        },
        "monte_carlo_parameters": {
            "n_samples": 20000,
            "n_chains": 8,
            "burn_in": 2000,
            "thinning": 2
        },
        "computation_parameters": {
            "max_n": 1000,
            "precision": 75
        },
        "random_seed": 54321,
        "analysis": {
            "correlation_analysis": True,
            "sensitivity_analysis": True,
            "uncertainty_quantification": True,
            "fitness_landscape": True
        },
        "tags": ["parameter_sweep", "exploration", "large_scale"]
    }

    experiment_template = {
        "name": "experiment_name",
        "description": "Brief description of experiment goals",
        "field_parameters": {
            "field_strength": 1.0,
            "coupling_constant": 0.1,
            "resonance_frequency": 1.0,
            "decay_rate": 0.01,
            "field_dimension": 3
        },
        "computation_parameters": {
            "max_n": 1000,
            "precision": 50,
            "convergence_threshold": 1e-6,
            "max_iterations": 10000
        },
        "monte_carlo_parameters": {
            "n_samples": 10000,
            "n_chains": 4,
            "burn_in": 1000,
            "thinning": 1
        },
        "analysis_parameters": {
            "bootstrap_samples": 1000,
            "confidence_level": 0.95,
            "significance_level": 0.05
        },
        "random_seed": None,
        "notes": "Additional experiment notes",
        "expected_outcomes": {},
        "tags": []
    }

    # Save all JSON files
    json_files = [
        ("config/defaults/base_parameters.json", base_parameters),
        ("config/defaults/monte_carlo_config.json", monte_carlo_config),
        ("config/defaults/visualization_config.json", visualization_config),
        ("config/defaults/analysis_config.json", analysis_config),
        ("config/experiments/experiment_001.json", experiment_001),
        ("config/experiments/experiment_002.json", experiment_002),
        ("config/experiments/template.json", experiment_template)
    ]

    for filepath, data in json_files:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Created: {filepath}")

def create_random_seeds_file():
    """Create the random_seeds.py file in config/."""

    seeds_content = '''"""
SFH Random Seed Management
==========================
Reproducible random number generation for all SFH components.
"""

import numpy as np
import random
import hashlib
import json
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path

class SeedManager:
    """Manages random seeds for reproducible experiments."""

    def __init__(self, master_seed: int = None):
        """Initialize with master seed."""
        if master_seed is None:
            # Generate from current time but make it reproducible
            master_seed = int(datetime.now().strftime("%Y%m%d%H"))

        self.master_seed = master_seed
        self.component_seeds = {}

        # Set global seeds
        random.seed(master_seed)
        np.random.seed(master_seed)

        # Generate component-specific seeds
        self._generate_component_seeds()

    def _generate_component_seeds(self):
        """Generate seeds for each SFH component."""
        components = [
            'partition_calc',
            'monte_carlo',
            'fitness_functions',
            'statistical_analysis',
            'visualization',
            'bootstrap',
            'cross_validation',
            'parameter_sampling'
        ]

        # Use hash-based seed generation for consistency
        for component in components:
            hash_input = f"{self.master_seed}_{component}"
            hash_obj = hashlib.md5(hash_input.encode())
            seed = int(hash_obj.hexdigest()[:8], 16)
            self.component_seeds[component] = seed

    def get_seed(self, component: str) -> int:
        """Get seed for specific component."""
        return self.component_seeds.get(component, self.master_seed)

    def set_component_seed(self, component: str):
        """Set numpy/random seeds for specific component."""
        seed = self.get_seed(component)
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def get_experiment_seeds(self, n_experiments: int) -> List[int]:
        """Generate seeds for multiple experiments."""
        seeds = []
        for i in range(n_experiments):
            experiment_seed = self.master_seed + i * 1000
            seeds.append(experiment_seed)
        return seeds

    def save_seed_state(self, filename: str = "config/current_seeds.json"):
        """Save current seed state."""
        seed_data = {
            'master_seed': self.master_seed,
            'component_seeds': self.component_seeds,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(seed_data, f, indent=2)

    def load_seed_state(self, filename: str = "config/current_seeds.json"):
        """Load seed state from file."""
        with open(filename, 'r') as f:
            seed_data = json.load(f)

        self.master_seed = seed_data['master_seed']
        self.component_seeds = seed_data['component_seeds']

        # Reset global seeds
        random.seed(self.master_seed)
        np.random.seed(self.master_seed)

def get_reproducible_seed(component: str = None, master_seed: int = None) -> int:
    """Get reproducible seed for component."""
    manager = SeedManager(master_seed)
    if component:
        return manager.get_seed(component)
    return manager.master_seed

# Predefined seed sets for common experiments
EXPERIMENT_SEEDS = {
    'baseline_validation': 12345,
    'parameter_sweep': 54321,
    'sensitivity_analysis': 98765,
    'cross_validation': 13579,
    'bootstrap_analysis': 24680,
    'convergence_test': 11111,
    'robustness_test': 22222,
    'publication_results': 33333
}

if __name__ == "__main__":
    # Test seed manager
    print("Testing SFH Seed Manager...")
    manager = SeedManager(42)

    print(f"Master seed: {manager.master_seed}")
    print("Component seeds:")
    for component, seed in manager.component_seeds.items():
        print(f"  {component}: {seed}")

    # Save seed state
    manager.save_seed_state()
    print("\\nSeed state saved to config/current_seeds.json")
'''

    with open("config/random_seeds.py", 'w') as f:
        f.write(seeds_content)
    print("Created: config/random_seeds.py")

def create_environment_file():
    """Create the environment.py file in config/."""

    env_content = '''"""
SFH Environment Management
==========================
Environment setup, dependency checking, and system configuration.
"""

import os
import sys
import platform
import subprocess
import warnings
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class EnvironmentManager:
    """Manages SFH runtime environment."""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.python_info = self._get_python_info()
        self.dependencies = self._check_dependencies()

    def _get_system_info(self) -> Dict[str, str]:
        """Get system information."""
        return {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'architecture': platform.architecture()[0]
        }

    def _get_python_info(self) -> Dict[str, str]:
        """Get Python environment information."""
        return {
            'version': platform.python_version(),
            'implementation': platform.python_implementation(),
            'executable': sys.executable,
            'prefix': sys.prefix,
            'path': sys.path[0] if sys.path else ""
        }

    def _check_dependencies(self) -> Dict[str, Dict[str, str]]:
        """Check required dependencies."""
        required_packages = [
            'numpy',
            'scipy',
            'matplotlib',
            'pandas',
            'sympy',
            'pytest',
            'tqdm',
            'seaborn',
            'scikit-learn'
        ]

        dependencies = {}
        for package in required_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                dependencies[package] = {'status': 'installed', 'version': version}
            except ImportError:
                dependencies[package] = {'status': 'missing', 'version': 'N/A'}

        return dependencies

    def check_environment(self) -> Dict[str, bool]:
        """Comprehensive environment check."""
        checks = {}

        # Python version check (>=3.8)
        py_version = tuple(map(int, platform.python_version().split('.')))
        checks['python_version'] = py_version >= (3, 8)

        # Required packages
        missing_packages = [pkg for pkg, info in self.dependencies.items()
                          if info['status'] == 'missing']
        checks['dependencies'] = len(missing_packages) == 0

        # Memory check (rough estimate)
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            checks['memory'] = available_memory_gb >= 2.0  # Need at least 2GB
        except ImportError:
            checks['memory'] = True  # Assume OK if can't check

        # Disk space check
        try:
            statvfs = os.statvfs('.')
            available_space_gb = statvfs.f_frsize * statvfs.f_bavail / (1024**3)
            checks['disk_space'] = available_space_gb >= 1.0  # Need at least 1GB
        except (OSError, AttributeError):
            checks['disk_space'] = True  # Assume OK if can't check

        return checks

    def setup_directories(self):
        """Create necessary directories."""
        directories = [
            'results',
            'logs',
            'cache',
            'plots',
            'exports'
        ]

        for directory in directories:
            Path(directory).mkdir(exist_ok=True)

    def get_environment_report(self) -> str:
        """Generate environment report."""
        report = []
        report.append("SFH Environment Report")
        report.append("=" * 50)

        # System info
        report.append("\\nSystem Information:")
        for key, value in self.system_info.items():
            report.append(f"  {key}: {value}")

        # Python info
        report.append("\\nPython Information:")
        for key, value in self.python_info.items():
            report.append(f"  {key}: {value}")

        # Dependencies
        report.append("\\nDependencies:")
        for pkg, info in self.dependencies.items():
            status = info['status']
            version = info['version']
            report.append(f"  {pkg}: {status} (v{version})")

        # Environment checks
        report.append("\\nEnvironment Checks:")
        checks = self.check_environment()
        for check, passed in checks.items():
            status = "PASS" if passed else "FAIL"
            report.append(f"  {check}: {status}")

        return "\\n".join(report)

    def save_environment_report(self, filename: str = "logs/environment_report.txt"):
        """Save environment report to file."""
        Path("logs").mkdir(exist_ok=True)
        with open(filename, 'w') as f:
            f.write(self.get_environment_report())

def setup_environment() -> EnvironmentManager:
    """Setup SFH environment."""
    env_manager = EnvironmentManager()
    env_manager.setup_directories()

    # Check environment
    checks = env_manager.check_environment()
    failed_checks = [check for check, passed in checks.items() if not passed]

    if failed_checks:
        warnings.warn(f"Environment checks failed: {failed_checks}")

    # Save environment report
    env_manager.save_environment_report()

    return env_manager

if __name__ == "__main__":
    # Test environment
    print("Testing SFH Environment Manager...")
    env_manager = setup_environment()
    print("\\n" + env_manager.get_environment_report())
'''

    with open("config/environment.py", 'w') as f:
        f.write(env_content)
    print("Created: config/environment.py")

def create_parameters_file():
    """Create the parameters.py file in config/."""

    params_content = '''"""
SFH Parameter Management
========================
Centralized parameter loading, validation, and management.
"""

import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import warnings

class ParameterManager:
    """Manages SFH parameters with validation and defaults."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.defaults_dir = self.config_dir / "defaults"
        self.experiments_dir = self.config_dir / "experiments"

        # Ensure directories exist
        self.config_dir.mkdir(exist_ok=True)
        self.defaults_dir.mkdir(exist_ok=True)
        self.experiments_dir.mkdir(exist_ok=True)

        # Parameter schemas for validation
        self.parameter_schemas = {
            'field_parameters': {
                'field_strength': {'type': float, 'range': (0.001, 100.0), 'default': 1.0},
                'coupling_constant': {'type': float, 'range': (0.001, 10.0), 'default': 0.1},
                'resonance_frequency': {'type': float, 'range': (0.1, 100.0), 'default': 1.0},
                'decay_rate': {'type': float, 'range': (0.001, 1.0), 'default': 0.01},
                'field_dimension': {'type': int, 'range': (2, 10), 'default': 3}
            },
            'computation_parameters': {
                'max_n': {'type': int, 'range': (10, 10000), 'default': 1000},
                'precision': {'type': int, 'range': (10, 1000), 'default': 50},
                'convergence_threshold': {'type': float, 'range': (1e-10, 1e-3), 'default': 1e-6},
                'max_iterations': {'type': int, 'range': (100, 100000), 'default': 10000}
            },
            'monte_carlo_parameters': {
                'n_samples': {'type': int, 'range': (100, 1000000), 'default': 10000},
                'n_chains': {'type': int, 'range': (1, 100), 'default': 4},
                'burn_in': {'type': int, 'range': (10, 10000), 'default': 1000},
                'thinning': {'type': int, 'range': (1, 100), 'default': 1}
            },
            'analysis_parameters': {
                'bootstrap_samples': {'type': int, 'range': (100, 100000), 'default': 1000},
                'confidence_level': {'type': float, 'range': (0.8, 0.99), 'default': 0.95},
                'significance_level': {'type': float, 'range': (0.01, 0.2), 'default': 0.05}
            }
        }

    def validate_parameters(self, params: Dict[str, Any], schema_name: str = None) -> Dict[str, Any]:
        """Validate parameters against schema."""
        if schema_name and schema_name in self.parameter_schemas:
            schema = self.parameter_schemas[schema_name]
            validated = {}

            for param, spec in schema.items():
                value = params.get(param, spec['default'])

                # Type checking
                if not isinstance(value, spec['type']):
                    try:
                        value = spec['type'](value)
                    except (ValueError, TypeError):
                        warnings.warn(f"Invalid type for {param}, using default")
                        value = spec['default']

                # Range checking
                if 'range' in spec:
                    min_val, max_val = spec['range']
                    if not (min_val <= value <= max_val):
                        warnings.warn(f"{param} out of range, clipping to valid range")
                        value = max(min_val, min(max_val, value))

                validated[param] = value

            return validated

        return params.copy()

    def load_defaults(self) -> Dict[str, Any]:
        """Load default parameter set."""
        defaults_file = self.defaults_dir / "base_parameters.json"

        if defaults_file.exists():
            with open(defaults_file, 'r') as f:
                return json.load(f)
        else:
            # Generate defaults from schemas
            defaults = {}
            for schema_name, schema in self.parameter_schemas.items():
                defaults[schema_name] = {
                    param: spec['default']
                    for param, spec in schema.items()
                }
            return defaults

    def load_experiment_config(self, experiment_name: str) -> Dict[str, Any]:
        """Load specific experiment configuration."""
        exp_file = self.experiments_dir / f"{experiment_name}.json"

        if exp_file.exists():
            with open(exp_file, 'r') as f:
                return json.load(f)
        else:
            warnings.warn(f"Experiment {experiment_name} not found, using defaults")
            return self.load_defaults()

    def save_experiment_config(self, experiment_name: str, params: Dict[str, Any]):
        """Save experiment configuration."""
        exp_file = self.experiments_dir / f"{experiment_name}.json"

        with open(exp_file, 'w') as f:
            json.dump(params, f, indent=2)

    def get_parameter_ranges(self, schema_name: str) -> Dict[str, tuple]:
        """Get valid parameter ranges for optimization."""
        if schema_name not in self.parameter_schemas:
            return {}

        ranges = {}
        for param, spec in self.parameter_schemas[schema_name].items():
            if 'range' in spec:
                ranges[param] = spec['range']

        return ranges

    def list_experiments(self) -> List[str]:
        """List available experiment configurations."""
        if not self.experiments_dir.exists():
            return []

        experiments = []
        for file in self.experiments_dir.glob("*.json"):
            if file.name != "template.json":
                experiments.append(file.stem)

        return sorted(experiments)

    def create_experiment_from_template(self, experiment_name: str,
                                      modifications: Dict[str, Any] = None):
        """Create new experiment from template."""
        template_file = self.experiments_dir / "template.json"

        if template_file.exists():
            with open(template_file, 'r') as f:
                template = json.load(f)
        else:
            template = self.load_defaults()

        # Apply modifications
        if modifications:
            template.update(modifications)

        # Set experiment name
        template['name'] = experiment_name

        # Save new experiment
        self.save_experiment_config(experiment_name, template)

def load_parameters(config_name: str = "base_parameters", experiment: str = None) -> Dict[str, Any]:
    """Load parameters from config file or experiment."""
    manager = ParameterManager()

    if experiment:
        return manager.load_experiment_config(experiment)
    else:
        return manager.load_defaults()

def save_parameters(params: Dict[str, Any], filename: str):
    """Save parameters to config file."""
    config_dir = Path("config/experiments")
    config_dir.mkdir(parents=True, exist_ok=True)

    filepath = config_dir / filename
    if not filepath.suffix:
        filepath = filepath.with_suffix('.json')

    with open(filepath, 'w') as f:
        json.dump(params, f, indent=2)

if __name__ == "__main__":
    # Test parameter manager
    print("Testing SFH Parameter Manager...")
    manager = ParameterManager()

    # Load defaults
    defaults = manager.load_defaults()
    print("\\nDefault parameters loaded successfully")

    # List experiments
    experiments = manager.list_experiments()
    print(f"\\nAvailable experiments: {experiments}")

    # Test validation
    test_params = {
        'field_strength': 2.5,
        'coupling_constant': 0.15,
        'max_n': 500
    }

    validated = manager.validate_parameters(test_params, 'field_parameters')
    print(f"\\nValidated parameters: {validated}")
'''

    with open("config/parameters.py", 'w') as f:
        f.write(params_content)
    print("Created: config/parameters.py")

def main():
    """Main setup function."""
    print("Setting up SFH Configuration Framework...")
    print("=" * 50)

    # Create directory structure
    create_config_structure()
    print()

    # Create Python modules
    create_parameters_file()
    create_random_seeds_file()
    create_environment_file()
    print()

    # Create JSON configurations
    create_json_configs()
    print()

    # Create seed state file
    from config.random_seeds import SeedManager
    seed_manager = SeedManager(12345)  # Default master seed
    seed_manager.save_seed_state()
    print("Created: config/current_seeds.json")

    print("\n" + "=" * 50)
    print("SFH Configuration setup complete!")
    print("\nNext steps:")
    print("1. Run this script: python setup_config.py")
    print("2. Check environment: python -c 'from config.environment import setup_environment; setup_environment()'")
    print("3. Ready to create master script!")

if __name__ == "__main__":
    main()
