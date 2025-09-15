import pandas as pd
import numpy as np
import networkx as nx

def compute_ecocoherence(edge_file, weight_col="weight"):
    """
    Compute the eco‑coherence statistic defined in Eq. 9.

    Parameters
    ----------
    edge_file : str
        Path to a CSV containing at least two columns: ``source`` and ``target``.
        An optional third column (named by ``weight_col``) holds the trophic weight
        (e.g., biomass). If absent, a uniform weight of 1 is assumed.
    weight_col : str, default "weight"
        Column name that stores the trophic weight.

    Returns
    -------
    float
        Eco‑coherence value Cₑ𝚌ₒ.
    """
    # ------------------------------------------------------------------
    # 1️⃣ Load edge list
    # ------------------------------------------------------------------
    df = pd.read_csv(edge_file)
    required = {"source", "target"}
    if not required.issubset(df.columns):
        raise ValueError("Edge file must contain 'source' and 'target' columns.")
    if weight_col not in df.columns:
        df[weight_col] = 1.0

    # ------------------------------------------------------------------
    # 2️⃣ Build directed graph
    # ------------------------------------------------------------------
    G = nx.from_pandas_edgelist(df,
                                 source="source",
                                 target="target",
                                 edge_attr=weight_col,
                                 create_using=nx.DiGraph())

    # ------------------------------------------------------------------
    # 3️⃣ Compute node degrees (total degree = in + out)
    # ------------------------------------------------------------------
    deg = dict(G.degree())
    k_vals = np.array(list(deg.values()))
    k_mean = k_vals.mean()

    # ------------------------------------------------------------------
    # 4️⃣ Gather trophic weights per node (sum of outgoing edge weights)
    # ------------------------------------------------------------------
    w = {}
    for node in G.nodes():
        out_edges = G.out_edges(node, data=True)
        w[node] = sum(d[weight_col] for _, _, d in out_edges) or 1.0  # avoid zero

    w_arr = np.array([w[n] for n in G.nodes()])

    # ------------------------------------------------------------------
    # 5️⃣ Apply Eq. 9
    # ------------------------------------------------------------------
    C = np.sum(w_arr * np.log(k_vals / k_mean))
    return float(C)
