#!/usr/bin/env python3
# sfh_simulation_v6.py -- Optimized SFH Model with Enhanced Plotting
import json, math, warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, shapiro, kstest, poisson, norm
from numpy.linalg import lstsq

RNG = np.random.default_rng(2025)

# Baselines (consistent across versions)
alpha_0 = 1.0 / 137.035999084
c_0 = 299792458.0
m_e_0 = 9.1093837015e-31
m_p_0 = 1.67262192369e-27
mu_0 = m_p_0 / m_e_0
alpha_s_0 = 0.1181
G_0 = 6.67430e-11
G_F_0 = 1.1663787e-5

# Ranges (widened slightly for exploration, per v3)
ALPHA_RANGE = (alpha_0 * 0.8, alpha_0 * 1.2)
MU_RANGE = (mu_0 * 0.85, mu_0 * 1.15)
ALPHAS_RANGE = (alpha_s_0 * 0.8, alpha_s_0 * 1.2)
G_RANGE = (G_0 * 0.2, G_0 * 5.0)
GF_RANGE = (G_F_0 * 0.6, G_F_0 * 1.4)

@dataclass
class Params:
    alpha: float; mu: float; alpha_s: float; G: float; G_F: float

def bounded(x): return np.clip(x, 0.0, 1.0)

# Upgraded proxies from v3, with noise & literature-tuned sensitivities
def atomic_score(params: Params, noise=0.02):
    E1 = 0.5 * params.alpha**2 * m_e_0 * c_0**2
    E1_0 = 0.5 * alpha_0**2 * m_e_0 * c_0**2
    a0 = 1.0 / (params.alpha * m_e_0)
    a0_0 = 1.0 / (alpha_0 * m_e_0)
    e_ratio = E1 / E1_0
    a_ratio = a0 / a0_0
    sE = math.exp(-0.5 * (math.log(e_ratio) / 0.1)**2)  # Tighter sigma for binding
    sA = math.exp(-0.5 * (math.log(a_ratio) / 0.15)**2)
    score = 0.6 * sE + 0.4 * sA + RNG.normal(0, noise)
    return bounded(score)

def triple_alpha_score(params: Params, noise=0.02):
    s_alpha = 30.0
    s_alphas = 10.0
    da = (params.alpha - alpha_0) / alpha_0
    das = (params.alpha_s - alpha_s_0) / alpha_s_0
    frac_shift = s_alpha * da + s_alphas * das
    score = math.exp(-0.5 * (frac_shift / 0.01)**2) + RNG.normal(0, noise)
    return bounded(score)

def deuteron_score(params: Params, noise=0.02):
    k1 = 10.0
    k2 = 5.0
    da_s = (params.alpha_s - alpha_s_0) / alpha_s_0
    dmu = (params.mu - mu_0) / mu_0
    frac = k1 * da_s + k2 * dmu
    score = math.exp(-0.5 * (frac / 0.03)**2) + RNG.normal(0, noise)
    return bounded(score)

def pp_score(params: Params, noise=0.02):
    proxy = params.alpha * math.sqrt(params.mu)
    proxy0 = alpha_0 * math.sqrt(mu_0)
    score = math.exp(-0.5 * (math.log(proxy / proxy0) / 0.05)**2) + RNG.normal(0, noise)
    return bounded(score)

def bbn_score(params: Params, noise=0.02):
    devG = (params.G - G_0) / G_0
    devGF = (params.G_F - G_F_0) / G_F_0
    dev = math.sqrt((devG / 0.1)**2 + (devGF / 0.05)**2)
    score = math.exp(-0.5 * dev**2) + RNG.normal(0, noise)
    return bounded(score)

def grav_coh(params: Params, noise=0.02):
    rel = (params.G - G_0) / G_0
    score = math.exp(- (rel / 0.3)**2) + RNG.normal(0, noise)
    return bounded(score)

def coherence_score(params: Params, noise=0.02):
    atom = atomic_score(params, noise)
    grav = grav_coh(params, noise)
    ta = triple_alpha_score(params, noise)
    return bounded(0.65 * atom + 0.25 * grav + 0.10 * ta)

def fertility_score(params: Params, noise=0.02):
    ta = triple_alpha_score(params, noise)
    dd = deuteron_score(params, noise)
    pp = pp_score(params, noise)
    bb = bbn_score(params, noise)
    return bounded(0.50 * ta + 0.20 * dd + 0.20 * pp + 0.10 * bb)

# MC sampling
def sample_mc(N=20000, noise=0.02):
    alphas = RNG.uniform(*ALPHA_RANGE, N)
    mus = RNG.uniform(*MU_RANGE, N)
    alpha_ss = RNG.uniform(*ALPHAS_RANGE, N)
    Gs = RNG.uniform(*G_RANGE, N)
    GFs = RNG.uniform(*GF_RANGE, N)
    rows = []
    for a, m, as_, g, gf in zip(alphas, mus, alpha_ss, Gs, GFs):
        p = Params(a, m, as_, g, gf)
        coh = coherence_score(p, noise)
        fert = fertility_score(p, noise)
        rows.append({'alpha': a, 'mu': m, 'alpha_s': as_, 'G': g, 'G_F': gf, 'coherence': coh, 'fertility': fert})
    return pd.DataFrame(rows)

# Pareto frontier with bootstrap CIs
def pareto_frontier(df):
    pts = df[['coherence', 'fertility']].values
    is_pareto = np.ones(len(pts), dtype=bool)
    for i, p in enumerate(pts):
        if not is_pareto[i]: continue
        dominated = np.all(pts >= p, axis=1) & np.any(pts > p, axis=1)
        is_pareto[dominated] = False
    return df[is_pareto].sort_values('coherence', ascending=False).reset_index(drop=True)

def bootstrap_pareto(df, n_boot=1000):
    paretos = [pareto_frontier(df.sample(frac=1, replace=True)) for _ in range(n_boot)]
    coh_bins = np.linspace(0, 1, 20)
    means, lows, highs = [], [], []
    for low, high in zip(coh_bins[:-1], coh_bins[1:]):
        ferts = [p[(p['coherence'] >= low) & (p['coherence'] < high)]['fertility'].mean() for p in paretos if not p[(p['coherence'] >= low) & (p['coherence'] < high)].empty]
        if ferts:
            m = np.nanmean(ferts)
            ci_l, ci_h = np.percentile(ferts, [2.5, 97.5])
            means.append(m); lows.append(ci_l); highs.append(ci_h)
        else:
            means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
    mid_bins = (coh_bins[:-1] + coh_bins[1:]) / 2
    return mid_bins, np.array(means), np.array(lows), np.array(highs)

# PRCC
def prcc(df, target, params):
    ranked = df[params + [target]].rank().to_numpy()
    X, y = ranked[:, :-1], ranked[:, -1]
    prccs = {}
    for i, col in enumerate(params):
        others = [j for j in range(X.shape[1]) if j != i]
        A = np.column_stack([np.ones(len(X)), X[:, others]])
        beta = lstsq(A, X[:, i], rcond=None)[0]
        r_x = X[:, i] - A @ beta
        B = np.column_stack([np.ones(len(X)), X[:, others]])
        beta_y = lstsq(B, y, rcond=None)[0]
        r_y = y - B @ beta_y
        prccs[col] = np.corrcoef(r_x, r_y)[0, 1]
    return prccs

# Weight sweep with CIs
def weight_sweep(df, weights=np.linspace(0, 1, 41), n_boot=1000):
    results = []
    for w in weights:
        combs = []
        for _ in range(n_boot):
            boot_df = df.sample(frac=1, replace=True)
            combined = w * boot_df['coherence'] + (1 - w) * boot_df['fertility']
            combs.append(combined.max())
        mean_comb = np.mean(combs)
        ci_l, ci_h = np.percentile(combs, [2.5, 97.5])
        results.append({'w_coh': w, 'mean_combined': mean_comb, 'ci_low': ci_l, 'ci_high': ci_h})
    return pd.DataFrame(results)

# w_min for coh threshold with bootstrap CI (from v3)
def find_w_min_for_coh_threshold(df, threshold=0.8, n_boot=1000):
    w_mins = []
    for _ in range(n_boot):
        boot_df = df.sample(frac=1, replace=True).reset_index(drop=True)
        ws = np.linspace(0, 1, 101)
        for w in ws:
            combined = w * boot_df['coherence'] + (1 - w) * boot_df['fertility']
            idx = combined.idxmax()
            if boot_df.iloc[idx]['coherence'] >= threshold:
                w_mins.append(w)
                break
    if not w_mins:
        return None, None, None
    mean_w = np.mean(w_mins)
    ci_l, ci_h = np.percentile(w_mins, [2.5, 97.5])
    return mean_w, ci_l, ci_h

# Stat tests (enhanced with proxy-specific KS)
def stat_tests(df):
    tests = {}
    for col in ['coherence', 'fertility']:
        data = df[col].dropna()
        _, p_shap = shapiro(data[:5000])  # Subsample for Shapiro
        _, p_ks = kstest(data, 'norm', args=(data.mean(), data.std()))
        tests[col] = {'shapiro_p': p_shap, 'ks_p': p_ks}
    return tests

# Universe scores
def universe_scores(noise=0.02):
    p = Params(alpha_0, mu_0, alpha_s_0, G_0, G_F_0)
    coh = coherence_score(p, noise)
    fert = fertility_score(p, noise)
    return coh, fert

def generate_plots(df, pareto, mid_bins, p_means, p_lows, p_highs, ws_df, w_min, w_ci_l, w_ci_h, universe_coh, universe_fert):
    # Plot 1: Coherence and Fertility Histograms (unchanged)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Coh hist
    counts, bins = np.histogram(df['coherence'], bins=50)
    errs = np.sqrt(counts)  # Poisson
    centers = (bins[:-1] + bins[1:]) / 2
    axs[0].bar(centers, counts, width=bins[1]-bins[0], color='skyblue', alpha=0.75)
    axs[0].errorbar(centers, counts, yerr=errs, fmt='none', ecolor='black')
    axs[0].axvline(universe_coh, color='red', ls='--')
    axs[0].set_title('Coherence Distribution (± Poisson Error)')
    axs[0].set_xlabel('Coherence')

    # Fert hist
    counts, bins = np.histogram(df['fertility'], bins=50)
    errs = np.sqrt(counts)
    centers = (bins[:-1] + bins[1:]) / 2
    axs[1].bar(centers, counts, width=bins[1]-bins[0], color='lightgreen', alpha=0.75)
    axs[1].errorbar(centers, counts, yerr=errs, fmt='none', ecolor='black')
    axs[1].axvline(universe_fert, color='red', ls='--')
    axs[1].set_title('Fertility Distribution (± Poisson Error)')
    axs[1].set_xlabel('Fertility')

    plt.tight_layout()
    plt.savefig('sfh_plots_v6_histograms.png')

    # Plot 2: Pareto Frontier with 2D Histogram (enlarged and enhanced)
    plt.figure(figsize=(14, 12))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create the 2D histogram
    plt.hist2d(df['coherence'], df['fertility'], bins=200, cmap='YlOrRd', zorder=1)

    # Plot the Pareto Frontier line
    plt.plot(pareto['coherence'], pareto['fertility'], color='black', linewidth=3, linestyle='-', label='Pareto Frontier', zorder=2)

    # Plot Our Universe's position
    plt.scatter(universe_coh, universe_fert, marker='*', s=800, color='red', edgecolor='black', zorder=3, label='Our Universe')

    # Add a curve and label for Our Universe's position, as in the example image
    annotation_text = f'Our Universe\n(Coherence: {universe_coh:.2f}, Fertility: {universe_fert:.2f})'
    plt.annotate(
        annotation_text,
        xy=(universe_coh, universe_fert),
        xytext=(0.8, 0.85),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8, headlength=8),
        fontsize=12,
        ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1)
    )

    plt.title('Our Universe\'s Position on the Coherence-Fertility Frontier', fontsize=20)
    plt.xlabel('Coherence Score', fontsize=14)
    plt.ylabel('Fertility Score', fontsize=14)
    plt.xlim(0, 1.05)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('sfh_plots_v6_2d_hist.png')


    # Plot 3: Weight sweep (unchanged)
    plt.figure(figsize=(8, 6))
    plt.plot(ws_df['w_coh'], ws_df['mean_combined'], label='Mean Optimal Score')
    plt.fill_between(ws_df['w_coh'], ws_df['ci_low'], ws_df['ci_high'], alpha=0.2, label='95% CI')
    plt.axvline(w_min, color='green', ls='--', label=f'w_min for coh>=0.8: {w_min:.2f} [{w_ci_l:.2f}, {w_ci_h:.2f}]')
    plt.xlabel('w_coh'); plt.ylabel('Score')
    plt.title('Weight Sweep with CIs')
    plt.legend()
    plt.savefig('weight_sweep_v6.png')

def main():
    noise = 0.02
    n_boot = 1000
    N = 20000
    df = sample_mc(N=N, noise=noise)
    df.to_csv('samples_v6.csv', index=False)

    pareto = pareto_frontier(df)
    pareto.to_csv('pareto_v6.csv', index=False)

    mid_bins, p_means, p_lows, p_highs = bootstrap_pareto(df, n_boot=n_boot)

    ws_df = weight_sweep(df, n_boot=n_boot)
    ws_df.to_csv('weight_sweep_v6.csv', index=False)

    w_min, w_ci_l, w_ci_h = find_w_min_for_coh_threshold(df, n_boot=n_boot)

    params_list = ['alpha', 'mu', 'alpha_s', 'G', 'G_F']
    prcc_coh = prcc(df, 'coherence', params_list)
    prcc_fert = prcc(df, 'fertility', params_list)

    tests = stat_tests(df)

    universe_coh, universe_fert = universe_scores(noise)
    # Uniqueness confidence: z-score vs mean
    z_coh = (universe_coh - df['coherence'].mean()) / df['coherence'].std()
    conf_coh = norm.cdf(z_coh) * 100 if z_coh > 0 else (1 - norm.cdf(-z_coh)) * 100
    z_fert = (universe_fert - df['fertility'].mean()) / df['fertility'].std()
    conf_fert = norm.cdf(z_fert) * 100 if z_fert > 0 else (1 - norm.cdf(-z_fert)) * 100

    generate_plots(df, pareto, mid_bins, p_means, p_lows, p_highs, ws_df, w_min, w_ci_l, w_ci_h, universe_coh, universe_fert)

    report = {
        'prcc_coherence': prcc_coh,
        'prcc_fertility': prcc_fert,
        'stat_tests': tests,
        'universe_scores': {'coherence': universe_coh, 'fertility': universe_fert, 'coh_confidence_%': conf_coh, 'fert_confidence_%': conf_fert},
        'w_min_coh0.8': {'mean': w_min, 'ci_95': [w_ci_l, w_ci_h]}
    }
    with open('sfh_report_v6.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("Simulation complete. Outputs: CSVs, PNGs, JSON.")
    print("Universe Coherence:", universe_coh, "Confidence:", conf_coh, "%")
    print("Universe Fertility:", universe_fert, "Confidence:", conf_fert, "%")
    print("Stat Tests:", tests)
    print("PRCC Coherence:", prcc_coh)
    print("PRCC Fertility:", prcc_fert)

if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        main()
