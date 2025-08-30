# SFH Python Framework

**Research Code for the Sentience-Field Hypothesis (SFH)**

This repository provides the official Python framework used in *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality* (SFH).  
It contains the mathematical models, simulation workflows, and visualization tools that reproduce the figures and results presented in the book.  

The framework is designed to be **modular, reproducible, and extendable**, supporting both research exploration and community collaboration.

---

## ✨ Research Framework

The repository is structured around a reproducible workflow:

- **`setup_config.py`**  
  Initializes a full experiment environment with configuration files, result directories, and reproducibility settings.  
  It creates JSON-based configs for parameters, Monte Carlo runs, statistical analysis, and visualization defaults.  
  It also auto-generates helper modules (`parameters.py`, `random_seeds.py`, `environment.py`).  

- **`sfh_master_framework.py`**  
  Main orchestration script. Runs partition analysis, checks mathematical identities (Euler, pentagonal theorem), performs statistical analysis, and generates publication-ready plots.

- **Supporting modules**  
  - `partition_calc.py` → Hardy–Ramanujan partitions, Euler checks, forbidden configurations.  
  - `statistical_analysis.py` → probability distributions, coherence–fertility balance metrics.  
  - `advanced_visualization.py` → figures, histograms, growth curves, phase-transition plots.

This modular approach ensures experiments can be repeated with identical parameters and compared across different research runs.

---

## 🚀 Usage Guide

### 1. Clone the repository
```bash
git clone https://github.com/urmt/SFH_Python_CODE.git
cd SFH_Python_CODE
```

### 2. Initialize the framework
Run once to set up configs, results folders, and reproducibility modules:
```bash
python setup_config.py
```

### 3. Run the main analysis
```bash
python sfh_master_framework.py
```

This will:
- Generate partition results and statistical analysis.
- Check Euler and pentagonal theorem consistency.
- Save plots and data into the `plots/` and `results/` folders.

### 4. Customize experiments
Edit the JSON configs in `config/` (e.g. `parameters.json`, `monte_carlo.json`) to change run parameters.  
Re-run the framework to generate new results.

---

## 📊 Reproducibility & Results

- **All results in the SFH book can be reproduced** with this framework.  
- Plots are stored in `/plots` and numerical outputs in `/results`.  
- Run logs are automatically created to ensure transparency of parameter choices.  

This ensures that the SFH theory can be **independently tested, verified, and extended** by the community.

---

## 🤝 Contributing

Contributions are welcome!  
You may:
- Submit pull requests with code improvements or additional modules.
- Open issues to report bugs or suggest enhancements.
- Request repository management access if you’d like to co-maintain the project.

The goal of this repository is to remain **open, collaborative, and community-driven**, reflecting the exploratory spirit of SFH.

---

## 📖 Citation

If you use this repository in academic work, please cite both the book and the repo:

**Book**  
Traver, M.R. (2025). *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality.*

**Code Repository**  
Traver, M.R. (2025). *SFH Python Framework: Research Code for the Sentience-Field Hypothesis.* GitHub.  
https://github.com/urmt/SFH_Python_CODE

---
