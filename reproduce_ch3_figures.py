"""
reproduce_ch3_figures.py
Reproduces the figures from Chapter 3 of:
'The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality'

Outputs images with the exact filenames referenced in the manuscript.
"""

import os
import matplotlib.pyplot as plt
import partition_calc as pc
import statistical_analysis as sa
import advanced_visualization as av

# Ensure directories exist
os.makedirs("figures", exist_ok=True)

def run_ch3_analysis():
    print("=== Reproducing SFH Book Chapter 3 Figures ===")

    # Step 1: Partition calculations (N=20,000 Monte Carlo universes)
    calc = pc.PartitionCalculator(max_n=100)
    results = calc.run_partition_analysis(num_samples=20000)
    print("✅ Partition analysis completed with 20,000 simulated universes")

    # Step 2: Statistical analysis
    analyzer = sa.StatisticalAnalyzer(results)
    summary = analyzer.basic_summary()
    print("✅ Statistical summary generated")
    print("Summary:", summary)

    # === FIGURE 3.1 ===
    # Histograms of Coherence & Fertility distributions
    fig1 = av.plot_coherence_fertility_histograms(results, num_samples=20000)
    fig1.savefig("sfh_plots_v6_histograms.png", dpi=300)
    plt.close(fig1)
    print("✅ Figure 3.1 saved as sfh_plots_v6_histograms.png")

    # === FIGURE 3.2 ===
    # Universe position on discrete Pareto Frontier
    fig2 = av.plot_pareto_frontier(results)
    fig2.savefig("figures/enhanced_universe_comparison.png", dpi=300)
    plt.close(fig2)
    print("✅ Figure 3.2 saved as figures/enhanced_universe_comparison.png")

    # === FIGURE 3.3 ===
    # Weight sweep including partition multiplicity
    fig3 = av.plot_weight_sweep(results)
    fig3.savefig("weight_sweep_v6.png", dpi=300)
    plt.close(fig3)
    print("✅ Figure 3.3 saved as weight_sweep_v6.png")

    print("🎉 All Chapter 3 figures reproduced successfully.")

if __name__ == "__main__":
    run_ch3_analysis()
