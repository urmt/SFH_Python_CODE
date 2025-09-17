"""eco_coherence – Eco-coherence metric for directed food-web networks.

Version: 1.0.0
This module implements the eco-coherence statistic Cₑ𝚌ₒ as defined in Equation 9
of the Sentience-Field Hypothesis (SFH) by Mark Rowe Traver (2025), supporting
ecological applications (Section 4.1). It computes coherence and a fertility proxy
from food-web edge data, aligning with SFH's qualic field dynamics.

Returns a dict with 'C_eco' (float), 'N' (int), and 'fertility_proxy' (float).
"""

from .core import compute_ecocoherence
