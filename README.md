# SFH Python Framework

**Research Code for *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality***  

This repository contains the official Python framework that supports the scientific work in the book *The Sentience-Field Hypothesis (SFH)*.  
It provides reproducible mathematical experiments, statistical analyses, and visualizations that correspond to the results, models, and figures presented in the manuscript.  

The framework is designed to meet academic reproducibility standards (e.g., for arXiv submission) while remaining accessible to developers and researchers who want to explore, extend, or critique the SFH model.

---

## ✨ Overview

The Sentience-Field Hypothesis proposes that consciousness emerges as a fundamental property of reality, encoded in quantized field structures.  
This framework provides the computational backbone for that hypothesis, including:

- **Partition mathematics** (Hardy–Ramanujan formula, Euler identities, pentagonal theorem checks).  
- **Stochastic & statistical analyses** (Monte Carlo experiments, coherence–fertility optimization, forbidden configuration detection).  
- **Advanced visualization** (plots of distributions, phase transitions, partition growth).  

By running the included scripts, researchers can **reproduce the figures and tables** in the SFH book and extend the analysis with their own data.  

---

## 🧮 Research Framework

Run `setup_config.py` once to generate a structured research environment:  

```
config/               → JSON parameter sets for experiments  
results/              → Numerical outputs from runs  
plots/                → Generated visualizations (PNG, PDF)  
logs/                 → Metadata logs (timestamps, seeds, environment info)  
```

It also auto-generates reusable Python modules:  

- `parameters.py` → imports JSON configs into experiments  
- `random_seeds.py` → ensures reproducibility across runs  
- `environment.py` → documents environment + dependencies  

This structure guarantees that **all results are reproducible** and can be replicated by others.  

---

## 🚀 Usage Guide

### 1. Clone the repo
```bash
git clone https://github.com/urmt/SFH_Python_CODE.git
cd SFH_Python_CODE
```

### 2. Setup experiment configs
```bash
python setup_config.py
```

### 3. Run the framework
```bash
python sfh_master_framework.py
```

### 4. Inspect results
- Numerical results → `results/`  
- Visualizations → `plots/`  
- Run logs → `logs/`  

---

## 📖 Reproducibility & Results

The figures and analyses in the *SFH book* are generated directly from this framework.  
Researchers can:  

- Modify `config/*.json` files to change experimental parameters.  
- Rerun `sfh_master_framework.py` to reproduce existing figures or generate new ones.  
- Extend the framework with additional statistical or visualization modules.  

---

## 🤝 Contributing

This project is open for collaboration.  

- **Commits & Pull Requests** are welcome.  
- Researchers are encouraged to fork the repo, run new experiments, and submit improvements.  
- If you wish to **co-manage the repository**, please open an issue or request access.  

The goal is to make SFH an **open scientific platform** where critical discussion, verification, and refinement are possible.  

---

## 📚 Citation

If you use this framework in academic work, please cite both the book and this repository:  

- Traver, M.R. (2025). *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality.*  
- SFH Python Framework (2025). GitHub repository: https://github.com/urmt/SFH_Python_CODE  

---
