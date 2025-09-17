markdown# ecological: Eco-Coherence Analysis for the Sentience-Field Hypothesis

![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.11-green.svg)
![Last Updated](https://img.shields.io/badge/last_updated-Sep_17_2025-orange.svg)

Welcome to **ecological**, a Python module for computing eco-coherence (\(C_{\text{eco}}\)) statistics within the framework of the **Sentience-Field Hypothesis (SFH)**. This tool supports SFH's ecological application (Section 4.1 of the manuscript), linking ecosystem stability to qualic coherence and fertility, as outlined in *The Sentience-Field Hypothesis: Consciousness as the Fabric of Reality* by Mark Rowe Traver (IngramSpark, 2025, ISBN: 978-0-123456-78-9).

## Overview

The eco-coherence metric \(C_{\text{eco}} = \frac{1}{N(N-1)} \sum_{i \neq j} \phi_{ij}\) quantifies the stability of ecological networks, where \(\phi_{ij}\) is the normalized interaction strength between species. This module integrates with SFH by providing a proxy for qualic coherence \(C(q)\) and a fertility measure (Shannon entropy), enabling analysis of how disturbances (e.g., deforestation) impact global qualic fertility \(F(q)\). The code is optimized for resilience studies, such as those of Amazon rainforests (\(C_{\text{eco}} \approx 1.4\)).

## Installation

To install the `ecological` module, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/urmt/SFH_Python_CODE.git
   cd SFH_Python_CODE/SFH_JCS_Article/Sentient-Field_Tools/ecological

Install Dependencies:
Ensure you have Python 3.11 and the required packages:
bashpip install numpy pandas networkx

Install the Module:
Install locally for use in your projects:
bashpip install .


Usage
Basic Example
The following example computes eco-coherence for a synthetic ecological network:
pythonimport pandas as pd
from ecological import compute_ecocoherence, validate_ecocoherence

# Create a sample edge file
sample_data = pd.DataFrame({
    "source": ["A", "B", "C", "D"],
    "target": ["B", "C", "D", "A"],
    "weight": [1.0, 0.5, 1.5, 0.8]
})
sample_data.to_csv("data/sample_edges.csv", index=False)

# Compute eco-coherence
result = compute_ecocoherence("data/sample_edges.csv", weight_col="weight")
print("Eco-Coherence Result:", result)

# Validate against SFH expectations
is_valid = validate_ecocoherence(result)
print("Validation:", is_valid)
Key Functions

compute_ecocoherence(edge_file, weight_col="weight"):

Returns a dictionary with 'C_eco' (eco-coherence), 'N' (node count), and 'fertility_proxy' (Shannon entropy).


validate_ecocoherence(result):

Checks if (C_{\text{eco}}) falls within SFH's expected range (0.5 to 2.0).



Contributing
Contributions are welcome! Please:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Description").
Push to the branch (git push origin feature-name).
Open a Pull Request with a clear description.

License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

Inspired by The Sentience-Field Hypothesis by M.R. Traver (2025).
Built with support from the open-source community, including NumPy, pandas, and NetworkX.
Thanks to collaborators advancing SFH's ecological applications.

Contact
For questions or collaboration, contact Mark R. Traver at thesfh@proton.me

