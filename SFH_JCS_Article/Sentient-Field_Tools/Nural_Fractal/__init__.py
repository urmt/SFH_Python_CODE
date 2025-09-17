"""higuchi – Fractal Dimension Analysis for EEG in SFH Protocol 2.

Version: 1.0.0
This module computes the Higuchi fractal dimension (FD) and phase coherence for EEG
data, supporting the Sentience-Field Hypothesis (SFH) Protocol 2. It quantifies
neural complexity (FD as a proxy for qualic fertility F(q)) and synchronization
(coherence as a proxy for qualic coherence C(q)), as per SFH's neural application
(Section 3.2). The implementation aids in detecting fractal shifts during insight.

Returns a dict with 'df' (pandas.DataFrame) and 'stats' (dict with t-test results).
"""

from .higuchi import compute_fractal, plot_fractal_results
