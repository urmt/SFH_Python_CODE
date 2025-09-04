#!/usr/bin/env python3
"""
Field Dynamics Simulator (SFH Project)
--------------------------------------
Core simulation of sentient field behavior and coherence emergence
based on discrete partition dynamics.

Author: SFH Research Team
Repository: https://github.com/urmt/SFH_Python_CODE
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path


class FieldDynamicsSimulator:
    def __init__(self, n_steps=500, n_entities=200, seed=None, output_dir="output"):
        self.n_steps = n_steps
        self.n_entities = n_entities
        self.rng = np.random.default_rng(seed)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Initialize coherence & fertility states randomly in [0,1]
        self.coherence = self.rng.random(n_entities)
        self.fertility = self.rng.random(n_entities)

        # Store trajectory
        self.history = {"coherence": [], "fertility": [], "partition_balance": []}

    def partition_balance(self):
        """
        Partition-inspired balance metric:
        Measures how evenly the population splits into 'high' vs 'low' states.
        """
        high_c = np.sum(self.coherence > 0.5)
        low_c = self.n_entities - high_c
        balance = abs(high_c - low_c) / self.n_entities
        return 1.0 - balance  # closer to 1 = more balanced

    def step(self):
        """
        Update coherence & fertility using partition-inspired dynamics.
        """
        noise = 0.05 * self.rng.standard_normal(self.n_entities)

        # Qualic optimization rule:
        # - Coherence rises if fertility balanced
        # - Fertility rises if coherence stable
        delta_c = 0.1 * (self.fertility - 0.5) \
                  - 0.05 * (self.coherence - 0.5) \
                  + 0.05 * self.partition_balance() \
                  + noise

        delta_f = 0.1 * (self.coherence - 0.5) \
                  - 0.05 * (self.fertility - 0.5) \
                  + 0.05 * self.partition_balance() \
                  + noise

        self.coherence = np.clip(self.coherence + delta_c, 0, 1)
        self.fertility = np.clip(self.fertility + delta_f, 0, 1)

        # Store averages per step
        self.history["coherence"].append(self.coherence.mean())
        self.history["fertility"].append(self.fertility.mean())
        self.history["partition_balance"].append(self.partition_balance())

    def run(self):
        for _ in range(self.n_steps):
            self.step()
        return self.history

    def save_results(self):
        """
        Save trajectory data as JSON for reproducibility.
        """
        json_path = self.output_dir / "field_dynamics_results.json"
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"✅ Results saved to {json_path}")

    def plot_results(self):
        """
        Generate plots of coherence, fertility, and partition balance.
        """
        steps = np.arange(len(self.history["coherence"]))

        plt.figure(figsize=(10, 6))
        plt.plot(steps, self.history["coherence"], label="Coherence", linewidth=2)
        plt.plot(steps, self.history["fertility"], label="Fertility", linewidth=2)
        plt.plot(steps, self.history["partition_balance"], label="Partition Balance", linewidth=2, linestyle="--")
        plt.xlabel("Time Step")
        plt.ylabel("Average Value")
        plt.title("Sentient Field Dynamics Simulation")
        plt.legend()
        plt.grid(True, alpha=0.3)

        out_path = self.output_dir / "field_dynamics_evolution.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"✅ Plot saved to {out_path}")


if __name__ == "__main__":
    sim = FieldDynamicsSimulator(n_steps=500, n_entities=300, seed=42)
    history = sim.run()
    sim.save_results()
    sim.plot_results()
    print("🎉 Simulation complete.")
