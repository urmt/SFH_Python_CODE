# SFH Simulation: Fundamental Constants and Cosmic Coherence

## Overview

This repository contains the Python script `sfh_simulation_v6-Enhanced-Combo.py`, a comprehensive simulation for exploring the impact of fundamental physical constants on the "Coherence" and "Fertility" of a hypothetical universe. The model explores a multi-dimensional parameter space to identify optimal configurations and assess the uniqueness of our own universe's constants.

This is a single-file program that runs the entire simulation, generates data files, and produces key plots for analysis.

## Key Concepts

* **Coherence:** A measure of a universe's internal consistency and stability, particularly with respect to atomic structure and gravitational behavior.
* **Fertility:** A measure of a universe's capacity to form complex structures and elements necessary for life, specifically focusing on stellar nucleosynthesis.
* **Pareto Frontier:** The set of optimal trade-offs between Coherence and Fertility, representing the best possible outcomes where no single score can be improved without sacrificing the other.
* **Partial Rank Correlation Coefficient (PRCC):** A statistical method used to determine the sensitivity of the model outputs (Coherence and Fertility) to each input parameter (the fundamental constants).

## Files

* `sfh_simulation_v6-Enhanced-Combo.py`: The main simulation script. This file is a self-contained program that executes the entire workflow.
* `samples_v6.csv`: A CSV file containing the raw output of the Monte Carlo simulation. Each row represents a single random sample of the fundamental constants and the resulting Coherence and Fertility scores.
* `pareto_v6.csv`: A CSV file containing the data points that form the Pareto frontier. These are the most optimal samples from the simulation.
* `weight_sweep_v6.csv`: A CSV file containing the results of the weight sweep analysis, which determines the optimal combined score for various weightings of Coherence and Fertility.

## How to Run

### Prerequisites

You need Python 3 and the following libraries installed:

* `numpy`
* `pandas`
* `matplotlib`
* `scipy`

You can install them using pip:

```bash
pip install numpy pandas matplotlib scipy
```

### Execution

Simply run the Python script from your terminal:

```bash
python sfh_simulation_v6-Enhanced-Combo.py
```

The script will automatically perform the following actions:

1.  Run the Monte Carlo simulation to generate `N` samples.
2.  Calculate the Coherence and Fertility scores for each sample.
3.  Identify the Pareto frontier.
4.  Perform a weight sweep analysis.
5.  Generate and save the `samples_v6.csv`, `pareto_v6.csv`, and `weight_sweep_v6.csv` data files.
6.  Generate and save three plots (`sfh_plots_v6_histograms.png`, `sfh_plots_v6_2d_hist.png`, `weight_sweep_v6.png`) in the same directory.
7.  Generate a JSON report (`sfh_report_v6.json`) summarizing the statistical tests, PRCC values, and universe scores.

## Outputs

After running the script, your directory will contain the following new files:

* `samples_v6.csv`
* `pareto_v6.csv`
* `weight_sweep_v6.csv`
* `sfh_plots_v6_histograms.png`: Histograms showing the distribution of Coherence and Fertility scores.
* `sfh_plots_v6_2d_hist.png`: A 2D histogram visualizing the relationship between Coherence and Fertility, with the Pareto frontier and our universe's position highlighted.
* `weight_sweep_v6.png`: A plot of the optimal combined score as a function of the weight given to Coherence.
* `sfh_report_v6.json`: A detailed report in JSON format.
