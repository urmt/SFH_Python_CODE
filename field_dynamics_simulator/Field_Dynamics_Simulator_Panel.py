#!/usr/bin/env python3
"""
Field Dynamics Simulator (SFH Project) - Panel Version
-------------------------------------------------------
Runs the SFH Field Dynamics Simulator with interactive sliders
and generates a standalone HTML dashboard.

Author: SFH Research Team
Repository: https://github.com/urmt/SFH_Python_CODE
"""

import numpy as np
import panel as pn
import matplotlib.pyplot as plt
from io import BytesIO
import base64

pn.extension(sizing_mode="stretch_width")

class FieldDynamicsSimulator:
    def __init__(self, n_steps=500, n_entities=200, seed=None, noise=0.05, alpha=0.1, beta=0.05):
        self.n_steps = n_steps
        self.n_entities = n_entities
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        self.alpha = alpha  # fertility → coherence coupling
        self.beta = beta    # coherence → fertility coupling

        # Initialize states
        self.coherence = self.rng.random(n_entities)
        self.fertility = self.rng.random(n_entities)
        self.history = {"coherence": [], "fertility": []}

    def step(self):
        noise = self.noise * self.rng.standard_normal(self.n_entities)
        delta_c = self.alpha * (self.fertility - 0.5) - self.beta * (self.coherence - 0.5) + noise
        delta_f = self.alpha * (self.coherence - 0.5) - self.beta * (self.fertility - 0.5) + noise
        self.coherence = np.clip(self.coherence + delta_c, 0, 1)
        self.fertility = np.clip(self.fertility + delta_f, 0, 1)
        self.history["coherence"].append(self.coherence.mean())
        self.history["fertility"].append(self.fertility.mean())

    def run(self):
        self.history = {"coherence": [], "fertility": []}
        for _ in range(self.n_steps):
            self.step()
        return self.history

    def plot_results(self):
        steps = np.arange(len(self.history["coherence"]))
        plt.figure(figsize=(10,6))
        plt.plot(steps, self.history["coherence"], label="Coherence", linewidth=2)
        plt.plot(steps, self.history["fertility"], label="Fertility", linewidth=2)
        plt.xlabel("Time Step")
        plt.ylabel("Average Value")
        plt.title("SFH Field Dynamics Simulation")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Save to base64 PNG for Panel
        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)
        return f"<img src='data:image/png;base64,{base64.b64encode(buf.read()).decode()}'/>"


# --- Panel UI ---

# Sliders
n_steps_slider = pn.widgets.IntSlider(name="Steps", value=500, start=100, end=2000, step=100)
n_entities_slider = pn.widgets.IntSlider(name="Entities", value=200, start=50, end=1000, step=50)
noise_slider = pn.widgets.FloatSlider(name="Noise", value=0.05, start=0.0, end=0.2, step=0.01)
alpha_slider = pn.widgets.FloatSlider(name="Alpha (fertility → coherence)", value=0.1, start=0.0, end=0.3, step=0.01)
beta_slider = pn.widgets.FloatSlider(name="Beta (coherence → fertility)", value=0.05, start=0.0, end=0.3, step=0.01)
seed_slider = pn.widgets.IntSlider(name="Random Seed", value=42, start=0, end=100)

# Simulation function
def run_sim(n_steps, n_entities, noise, alpha, beta, seed):
    sim = FieldDynamicsSimulator(n_steps=n_steps, n_entities=n_entities,
                                 noise=noise, alpha=alpha, beta=beta, seed=seed)
    sim.run()
    return sim.plot_results()

# Bind function to widgets
dynamic_plot = pn.bind(run_sim,
                       n_steps=n_steps_slider,
                       n_entities=n_entities_slider,
                       noise=noise_slider,
                       alpha=alpha_slider,
                       beta=beta_slider,
                       seed=seed_slider)

# Layout
dashboard = pn.Column(
    "# 🌌 SFH Field Dynamics Simulator",
    "Adjust parameters and observe real-time coherence & fertility dynamics.",
    pn.Row(n_steps_slider, n_entities_slider),
    pn.Row(noise_slider, alpha_slider, beta_slider, seed_slider),
    dynamic_plot
)

# To serve interactively: 
#   panel serve Field_Dynamics_Simulator_Panel.py --show
#
# To export standalone HTML:
#   python Field_Dynamics_Simulator_Panel.py

if __name__ == "__main__":
    dashboard.save("Field_Dynamics_Simulator.html", embed=True)
    print("✅ Dashboard saved as Field_Dynamics_Simulator.html")

