
"""
Rebuttal Experiment: RSS-Dimension Partial Correlation

Question: Does the RSS-LEMB correlation survive after controlling for
          embedding dimensionality?

          r(RSS, LEMB | dim) = ?  (Pearson + Spearman)

FILL-IN VALUES for ac_final.md:
  [A]   Dimension-controlled partial correlation RSS(L=15)-LEMB
  [A2]  Zero-order correlation: dim-RSS(L=15)
  [B]   Multi-controlled: RSS(L=15)-LEMB | dim, params (conditional on collinearity check)
  [C]   Within-dimension spread: dim==4096 group RSS(L=15) stats

Data sources:
  - RSS/SRS/LEMB: benchmarks/ JSON files (via tbl4's gather_all_data)
  - MODEL_DIM_PARAMS: hardcoded table (dim + params_B) — fill in manually!

Outputs:
  1. dim_partial_correl_results.tsv   — all lengths L=4..16 + SRS_neg
  2. rebuttal_dim_partial_correl_summary.txt — copy-paste-ready numbers
"""

import csv
import json
import math
import os
import re
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# Reuse tbl4 infrastructure
try:
    from scripts.tbl4_master_correl import (
        gather_all_data,
        extract_lemb_task_score,
        extract_rss_l15_mean,
        extract_srs_negative_only_score,
    )
except:
    from tbl4_master_correl import (
        gather_all_data,
        extract_lemb_task_score,
        extract_rss_l15_mean,
        extract_srs_negative_only_score,
    )

# =============================================================================
# Constants
# =============================================================================

BASELINE_MODEL = "average-synth_multilingual-e5-base"
TARGET_LENGTHS = list(range(4, 17))
HIGHLIGHT_LENGTH = 15

LEMB_TASK_KEYS = [
    'LEMBNeedleRetrieval',
    'LEMBPasskeyRetrieval',
    'LEMBSummScreenFDRetrieval',
    'LEMBQMSumRetrieval',
    'LEMBWikimQARetrieval',
    'LEMBNarrativeQARetrieval',
]

# =============================================================================
# MODEL DIMENSION & PARAMETER TABLE (FILL IN MANUALLY!)
# =============================================================================
# Format: {model_name: (output_dim, params_Billions)}
# dim=0 and params=None are PLACEHOLDERS. Replace with real values.
# params unit: Billions (e.g., 0.6 for 600M, 7.0 for 7B)
# Data source: MTEB table (params) + model cards (dim)

MODEL_DIM_PARAMS: dict[str, tuple[int, float | None]] = {
    "Alibaba-NLP_gte-modernbert-base":             (768, 0.149),
    "BAAI_bge-m3":                                  (1024, 0.568),
    "BAAI_bge-m3-unsupervised":                     (1024, 0.568),
    "Haon-Chen_speed-embedding-7b-instruct":        (4096, 7.111),
    "ICT-TIME-and-Querit_BOOM_4B_v1":              (2560, 4.022),
    "Linq-AI-Research_Linq-Embed-Mistral":          (4096, 7.111),
    "Qwen_Qwen3-Embedding-0.6B":                    (1024, 0.596),
    "Qwen_Qwen3-Embedding-4B":                      (2560, 4.022),
    "Qwen_Qwen3-Embedding-8B":                      (4096, 7.567),
    "Salesforce_SFR-Embedding-2_R":                 (4096, 7.111),
    "Salesforce_SFR-Embedding-Mistral":             (4096, 7.111),
    "Snowflake_snowflake-arctic-embed-l-v2.0":      (1024, 0.568),
    "annamodels_LGAI-Embedding-Preview":            (4096, 7.11),
    "bflhc_MoD-Embedding":                          (2560, 4.022),
    "bflhc_Octen-Embedding-0.6B":                   (1024, 0.596),
    "bflhc_Octen-Embedding-4B":                     (2560, 4.022),
    "bflhc_Octen-Embedding-8B":                     (4096, 7.567),
    "codefuse-ai_F2LLM-0.6B":                       (1024, 0.596),
    "codefuse-ai_F2LLM-1.7B":                       (2048, 1.721),
    "codefuse-ai_F2LLM-4B":                         (2560, 4.022),
    "codefuse-ai_F2LLM-v2-0.6B":                    (1024, 0.596),
    "codefuse-ai_F2LLM-v2-1.7B":                    (2048, 1.721),
    "codefuse-ai_F2LLM-v2-4B":                      (2560, 4.022),
    "codefuse-ai_F2LLM-v2-8B":                      (4096, 7.568),
    "ibm-granite_granite-embedding-english-r2":      (768, 0.149),
    "ibm-granite_granite-embedding-small-english-r2": (384, 0.048),
    "jinaai_jina-embeddings-v5-text-nano":          (768, 0.212),
    "jinaai_jina-embeddings-v5-text-small":         (1024, 0.596),
    "nomic-ai_modernbert-embed-base":               (768, 0.149),
    "nomic-ai_nomic-embed-text-v1":                 (768, 0.137),
    "nomic-ai_nomic-embed-text-v1.5":               (768, 0.137),
    "nvidia_llama-embed-nemotron-8b":               (4096, 7.505),
    "sbintuitions_sarashina-embedding-v1-1b":       (1792, 1.224),
    "zeta-alpha-ai_Zeta-Alpha-E5-Mistral":          (4096, 7.111),
}

# =============================================================================
# Metric Extraction (beyond tbl4)
# =============================================================================


def extract_rss_mean(data: dict, length: int) -> float:
    """Extract mean RSS score for a specific sequence length."""
    if not data:
        return None
    length_scores = data.get('length_scores', {})
    length_key = str(length)
    if length_key not in length_scores:
        return None
    rss_scores = length_scores[length_key].get('rss_scores', [])
    if not rss_scores:
        return None
    return float(np.mean(rss_scores))


def extract_lemb_avg(data: dict) -> float:
    """Calculate average of all 6 LEMB task scores."""
    if not data:
        return None
    scores = []
    for key in LEMB_TASK_KEYS:
        score = extract_lemb_task_score(data, key)
        if score is not None:
            scores.append(float(score))
    return float(np.mean(scores)) if scores else None


# =============================================================================
# Partial Correlation (Pearson + Spearman)
# =============================================================================

def partial_correlation_pearson(x, y, z):
    """
    Compute Pearson partial correlation r(x,y.z).

    Formula:
        r(x,y.z) = (r_xy - r_xz * r_yz) / sqrt((1 - r_xz**2)(1 - r_yz**2))

    Returns:
        (r_partial, p_value, n)
        p-value computed via t-distribution with df = n - 3
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)
    n = len(x)

    if n < 4:
        return None, None, n

    r_xy, _ = scipy_stats.pearsonr(x, y)
    r_xz, _ = scipy_stats.pearsonr(x, z)
    r_yz, _ = scipy_stats.pearsonr(y, z)

    denom = math.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    if denom < 1e-10:
        return None, None, n

    r_partial = (r_xy - r_xz * r_yz) / denom

    if abs(1 - r_partial**2) < 1e-10:
        return r_partial, 0.0, n

    df = n - 3
    t_stat = r_partial * math.sqrt(df / (1 - r_partial**2))
    p_value = 2 * scipy_stats.t.sf(abs(t_stat), df)

    return r_partial, p_value, n


def partial_correlation_spearman(x, y, z):
    """
    Compute Spearman partial correlation: rank-transform x,y,z first,
    then apply Pearson partial correlation formula.

    This is Fable5's recommendation: dim is discrete (few values: 384/768/1024/4096),
    so rank-based correlation is more defensible.
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    z = np.array(z, dtype=float)

    # Rank transform
    x_rank = scipy_stats.rankdata(x)
    y_rank = scipy_stats.rankdata(y)
    z_rank = scipy_stats.rankdata(z)

    return partial_correlation_pearson(x_rank, y_rank, z_rank)


def partial_correlation_multi_control(x, y, controls: list):
    """
    Compute partial correlation r(x,y | z1, z2, ...) via residual method.

    Regress x on controls → residuals rx
    Regress y on controls → residuals ry
    Then Pearson r(rx, ry).

    Returns (r_partial, p_value, n)
    """
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    n = len(x)

    if n < 4 + len(controls):
        return None, None, n

    # Build control matrix [z1, z2, ..., 1]
    Z = np.column_stack([np.array(c, dtype=float) for c in controls])
    Z = np.column_stack([Z, np.ones(n)])  # add intercept

    # Residuals
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]

    r_partial, p_value = scipy_stats.pearsonr(rx, ry)
    return r_partial, p_value, n


# =============================================================================
# Main Analysis
# =============================================================================

def generate_tbl_dim_partial_correl(
    data_dir: str,
    output_dir: str,
    config: dict,
) -> dict:
    """
    Public API for reproduce.py integration.

    Args:
        data_dir: Directory containing benchmark JSON files
        output_dir: Directory to save output files
        config: Configuration dict with keys:
            - target_lengths: List of RSS lengths (default: range(4,17))
            - highlight_length: Length to highlight in summary (default: 15)
            - exclude_models: List of models to exclude (default: [baseline])
            - dim_params_map: Dict of model→(dim, params) (default: MODEL_DIM_PARAMS)

    Returns:
        Dictionary with paths to generated files
    """
    print("=" * 70)
    print("REBUTTAL: RSS-Dimension Partial Correlation")
    print("=" * 70)

    target_lengths = config.get('target_lengths', list(range(4, 17)))
    highlight_length = config.get('highlight_length', 15)
    exclude_models = config.get('exclude_models', [BASELINE_MODEL])
    dim_params_map = config.get('dim_params_map', MODEL_DIM_PARAMS)

    # --- 1. Gather benchmark data (LEMB + RSS + SRS) ---
    print(f"\nGathering benchmark data from '{data_dir}'...")
    all_data = gather_all_data(data_dir)
    print(f"  Loaded {len(all_data)} models with complete LEMB+RSS+SRS")

    # --- 2. Filter models ---
    model_pool = []
    for name in sorted(all_data.keys()):
        if name in exclude_models:
            continue
        if name not in dim_params_map:
            print(f"  Skipping {name} (no dim/params data)")
            continue
        model_pool.append(name)

    n_total = len(model_pool)
    print(
        f"  Final model pool: {n_total} models (baseline + models without dim excluded)")

    # --- 3. Extract metric vectors ---
    model_metrics = {}
    for name in model_pool:
        metrics = all_data[name]
        lemb_avg = extract_lemb_avg(metrics['lemb'])
        srs_neg = extract_srs_negative_only_score(metrics['srs'])
        dim_val, params_val = dim_params_map[name]

        rss_vals = {}
        for L in target_lengths:
            rss_vals[L] = extract_rss_mean(metrics['rss'], L)

        model_metrics[name] = {
            'lemb_avg': lemb_avg,
            'srs_neg': srs_neg,
            'dim': dim_val,
            'params': params_val,
            'rss': rss_vals,
        }

    # --- 4. Compute correlations for each L ---
    results = []

    for L in target_lengths:
        # Build vectors for models with complete data
        valid_names = [n for n in model_pool
                       if model_metrics[n]['rss'][L] is not None
                       and model_metrics[n]['lemb_avg'] is not None]

        x_rss = np.array([model_metrics[n]['rss'][L]
                         for n in valid_names], dtype=float)
        y_lemb = np.array([model_metrics[n]['lemb_avg']
                          for n in valid_names], dtype=float)
        z_dim = np.array([model_metrics[n]['dim']
                         for n in valid_names], dtype=float)

        n = len(x_rss)
        if n < 4:
            continue

        # --- Zero-order correlations ---
        r_rss_lemb, p_rss_lemb = scipy_stats.pearsonr(x_rss, y_lemb)
        rho_rss_lemb, p_rho_rss_lemb = scipy_stats.spearmanr(x_rss, y_lemb)

        r_rss_dim, p_rss_dim = scipy_stats.pearsonr(x_rss, z_dim)
        rho_rss_dim, p_rho_rss_dim = scipy_stats.spearmanr(x_rss, z_dim)

        r_dim_lemb, p_dim_lemb = scipy_stats.pearsonr(z_dim, y_lemb)
        rho_dim_lemb, p_rho_dim_lemb = scipy_stats.spearmanr(z_dim, y_lemb)

        # --- Partial correlation (Pearson) ---
        r_partial_pearson, p_partial_pearson, _ = partial_correlation_pearson(
            x_rss, y_lemb, z_dim)

        # --- Partial correlation (Spearman) ---
        r_partial_spearman, p_partial_spearman, _ = partial_correlation_spearman(
            x_rss, y_lemb, z_dim)

        results.append({
            'metric': f'RSS_L{L}',
            'length': L,
            'n': n,
            # Zero-order Pearson
            'r_rss_lemb': r_rss_lemb,
            'p_rss_lemb': p_rss_lemb,
            'r_rss_dim': r_rss_dim,
            'p_rss_dim': p_rss_dim,
            'r_dim_lemb': r_dim_lemb,
            'p_dim_lemb': p_dim_lemb,
            # Zero-order Spearman
            'rho_rss_lemb': rho_rss_lemb,
            'p_rho_rss_lemb': p_rho_rss_lemb,
            'rho_rss_dim': rho_rss_dim,
            'p_rho_rss_dim': p_rho_rss_dim,
            'rho_dim_lemb': rho_dim_lemb,
            'p_rho_dim_lemb': p_rho_dim_lemb,
            # Partial Pearson
            'r_partial_pearson': r_partial_pearson,
            'p_partial_pearson': p_partial_pearson,
            # Partial Spearman
            'r_partial_spearman': r_partial_spearman,
            'p_partial_spearman': p_partial_spearman,
        })

    # --- 4b. SRS_neg ---
    srs_valid = [n for n in model_pool
                 if model_metrics[n]['srs_neg'] is not None
                 and model_metrics[n]['lemb_avg'] is not None]

    if len(srs_valid) >= 4:
        x_srs = np.array([model_metrics[n]['srs_neg']
                         for n in srs_valid], dtype=float)
        y_lemb_srs = np.array([model_metrics[n]['lemb_avg']
                              for n in srs_valid], dtype=float)
        z_dim_srs = np.array([model_metrics[n]['dim']
                             for n in srs_valid], dtype=float)

        n_srs = len(x_srs)

        r_srs_lemb, p_srs_lemb = scipy_stats.pearsonr(x_srs, y_lemb_srs)
        rho_srs_lemb, p_rho_srs_lemb = scipy_stats.spearmanr(x_srs, y_lemb_srs)

        r_srs_dim, p_srs_dim = scipy_stats.pearsonr(x_srs, z_dim_srs)
        rho_srs_dim, p_rho_srs_dim = scipy_stats.spearmanr(x_srs, z_dim_srs)

        r_dim_lemb_srs, p_dim_lemb_srs = scipy_stats.pearsonr(
            z_dim_srs, y_lemb_srs)
        rho_dim_lemb_srs, p_rho_dim_lemb_srs = scipy_stats.spearmanr(
            z_dim_srs, y_lemb_srs)

        r_partial_pearson_srs, p_partial_pearson_srs, _ = partial_correlation_pearson(
            x_srs, y_lemb_srs, z_dim_srs)
        r_partial_spearman_srs, p_partial_spearman_srs, _ = partial_correlation_spearman(
            x_srs, y_lemb_srs, z_dim_srs)

        results.append({
            'metric': 'SRS_neg',
            'length': None,
            'n': n_srs,
            'r_rss_lemb': r_srs_lemb,
            'p_rss_lemb': p_srs_lemb,
            'r_rss_dim': r_srs_dim,
            'p_rss_dim': p_srs_dim,
            'r_dim_lemb': r_dim_lemb_srs,
            'p_dim_lemb': p_dim_lemb_srs,
            'rho_rss_lemb': rho_srs_lemb,
            'p_rho_rss_lemb': p_rho_srs_lemb,
            'rho_rss_dim': rho_srs_dim,
            'p_rho_rss_dim': p_rho_srs_dim,
            'rho_dim_lemb': rho_dim_lemb_srs,
            'p_rho_dim_lemb': p_rho_dim_lemb_srs,
            'r_partial_pearson': r_partial_pearson_srs,
            'p_partial_pearson': p_partial_pearson_srs,
            'r_partial_spearman': r_partial_spearman_srs,
            'p_partial_spearman': p_partial_spearman_srs,
        })

    # --- 5. Multicollinearity check for B ---
    # Collect dim and params for all models that have both
    dims_for_check = []
    params_for_check = []
    for name in model_pool:
        dim_val, params_val = dim_params_map[name]
        if params_val is not None and dim_val > 0:
            dims_for_check.append(float(dim_val))
            params_for_check.append(float(params_val))

    collinearity_info = {}
    if len(dims_for_check) >= 4:
        r_dim_params, p_dim_params = scipy_stats.pearsonr(
            dims_for_check, params_for_check)
        rho_dim_params, p_rho_dim_params = scipy_stats.spearmanr(
            dims_for_check, params_for_check)
        collinearity_info = {
            'n': len(dims_for_check),
            'r_dim_params': r_dim_params,
            'p_dim_params': p_dim_params,
            'rho_dim_params': rho_dim_params,
            'p_rho_dim_params': p_rho_dim_params,
            'high_collinearity': abs(r_dim_params) >= 0.8,
        }

        # Compute B (multi-controlled) only if collinearity is acceptable
        # and we have enough models with both dim and params
        if not collinearity_info['high_collinearity']:
            L = highlight_length
            valid_b = [n for n in model_pool
                       if model_metrics[n]['rss'][L] is not None
                       and model_metrics[n]['lemb_avg'] is not None
                       and model_metrics[n]['params'] is not None
                       and model_metrics[n]['dim'] > 0]

            if len(valid_b) >= 4:
                x_b = np.array([model_metrics[n]['rss'][L]
                               for n in valid_b], dtype=float)
                y_b = np.array([model_metrics[n]['lemb_avg']
                               for n in valid_b], dtype=float)
                z1_dim = np.array([model_metrics[n]['dim']
                                  for n in valid_b], dtype=float)
                z2_params = np.array([model_metrics[n]['params']
                                     for n in valid_b], dtype=float)

                r_b, p_b, n_b = partial_correlation_multi_control(
                    x_b, y_b, [z1_dim, z2_params])
                collinearity_info['r_partial_multi'] = r_b
                collinearity_info['p_partial_multi'] = p_b
                collinearity_info['n_multi'] = n_b

    # --- 6. Within-dimension spread (C): filter dim==4096, L=4,8,16 ---
    C_LENGTHS = [4, 8, 16]
    c_info = {}
    for cL in C_LENGTHS:
        group_4096 = [n for n in model_pool
                      if model_metrics[n]['dim'] == 4096
                      and model_metrics[n]['rss'][cL] is not None]
        if len(group_4096) >= 2:
            rss_4096 = np.array([model_metrics[n]['rss'][cL]
                                for n in group_4096], dtype=float)
            c_info[cL] = {
                'n': len(group_4096),
                'models': group_4096,
                'mean': float(np.mean(rss_4096)),
                'std': float(np.std(rss_4096, ddof=1)),
                'min': float(np.min(rss_4096)),
                'max': float(np.max(rss_4096)),
                'range': float(np.max(rss_4096) - np.min(rss_4096)),
                'iqr': float(np.percentile(rss_4096, 75) - np.percentile(rss_4096, 25)),
            }

    # --- 7. Save outputs ---
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # TSV
    tsv_path = output_path / "dim_partial_correl_results.tsv"
    headers = [
        'metric', 'length', 'n',
        'r_rss_lemb', 'p_rss_lemb',
        'rho_rss_lemb', 'p_rho_rss_lemb',
        'r_rss_dim', 'p_rss_dim',
        'rho_rss_dim', 'p_rho_rss_dim',
        'r_dim_lemb', 'p_dim_lemb',
        'rho_dim_lemb', 'p_rho_dim_lemb',
        'r_partial_pearson', 'p_partial_pearson',
        'r_partial_spearman', 'p_partial_spearman',
    ]
    with open(tsv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers, delimiter='\t')
        writer.writeheader()
        for row in results:
            clean = {}
            for k, v in row.items():
                if isinstance(v, float):
                    clean[k] = f'{v:.6f}'
                elif v is None:
                    clean[k] = 'n/a'
                else:
                    clean[k] = str(v)
            writer.writerow(clean)
    print(f"\n  Saved: {tsv_path}")

    # Summary text
    summary = _generate_summary(
        results, highlight_length, collinearity_info, c_info, n_total)
    summary_path = output_path / "rebuttal_dim_partial_correl_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    print(f"  Saved: {summary_path}")

    # LaTeX table (appendix b)
    latex_str = _build_latex_table(results, highlight_length)
    latex_path = output_path / "table.tex"
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_str)
    print(f"  Saved: {latex_path}")

    print("\n" + summary)

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")

    return {
        'results_tsv': str(tsv_path),
        'summary_txt': str(summary_path),
        'latex_table': latex_str,
        'latex_path': str(latex_path),
    }


def _format_r(v) -> str:
    """Signed 4-decimal correlation, e.g. +0.4817 / -0.1234."""
    if v is None:
        return "--"
    return f"{v:+.4f}"


def _format_p(v) -> str:
    """4-decimal p-value; <0.0001 notation for tiny p."""
    if v is None:
        return "--"
    if v < 0.0001:
        return r"$<$0.0001"
    return f"{v:.4f}"


def _format_metric_latex(metric: str) -> str:
    """RSS_L4 -> RSS $L{=}4$; SRS_neg -> SRS$_\\mathrm{neg}$."""
    if metric == "SRS_neg":
        return r"SRS$_\mathrm{neg}$"
    m = re.match(r"^RSS_L(\d+)$", metric)
    if m:
        return f"RSS $L{{=}}{m.group(1)}$"
    return metric.replace("_", r"\_")


def _build_latex_zero_order(results: list) -> str:
    """Zero-order correlations: r(X,LEMB), r(X,dim), r(dim,LEMB)."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\footnotesize")
    lines.append(r"\centering")
    lines.append(r"\caption{Zero-order correlations between "
                 r"\textsc{RSS}/\textsc{SRS}, embedding dimension, and "
                 r"\textsc{LEMB}.}")
    lines.append(r"\label{tbl:zero-order}")
    lines.append(r"\begin{tabular}{@{}l r r r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & $n$ & $r(X,\mathrm{LEMB})$ & $p$ & $r(X,\mathrm{dim})$ & $p$ & "
        r"$r(\mathrm{dim},\mathrm{LEMB})$ & $p$ \\")
    lines.append(r"\midrule")

    for r in results:
        metric_disp = _format_metric_latex(r["metric"])
        cells = [
            metric_disp,
            str(r["n"]),
            _format_r(r["r_rss_lemb"]), _format_p(r["p_rss_lemb"]),
            _format_r(r["r_rss_dim"]), _format_p(r["p_rss_dim"]),
            _format_r(r["r_dim_lemb"]), _format_p(r["p_dim_lemb"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _build_latex_partial(results: list) -> str:
    """Dimension-controlled partial correlations (Pearson + Spearman)."""
    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"\footnotesize")
    lines.append(r"\centering")
    lines.append(r"\caption{Dimension-controlled partial correlations between "
                 r"\textsc{RSS}/\textsc{SRS} and \textsc{LEMB}, controlling for "
                 r"model-wise embedding dimension.}")
    lines.append(r"\label{tbl:dim-correlate}")
    lines.append(r"\begin{tabular}{@{}l r r r r r@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Metric & $n$ & $r_{\mathrm{partial}}(P)$ & $p$ & "
        r"$r_{\mathrm{partial}}(S)$ & $p$ \\")
    lines.append(r"\midrule")

    for r in results:
        metric_disp = _format_metric_latex(r["metric"])
        cells = [
            metric_disp,
            str(r["n"]),
            _format_r(r["r_partial_pearson"]), _format_p(
                r["p_partial_pearson"]),
            _format_r(r["r_partial_spearman"]), _format_p(
                r["p_partial_spearman"]),
        ]
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _build_latex_table(results: list, highlight_length: int) -> str:
    """Two-table LaTeX: zero-order + dimension-controlled partial."""
    zero = _build_latex_zero_order(results)
    partial = _build_latex_partial(results)
    return zero + "\n\n" + partial


def _generate_summary(
    results: list,
    highlight_length: int,
    collinearity_info: dict,
    c_info: dict,
    n_total: int,
) -> str:
    """Generate copy-paste-ready summary with FILL-IN VALUES block at top."""

    # Find highlight row
    highlight_row = None
    for r in results:
        if r['metric'] == f'RSS_L{highlight_length}':
            highlight_row = r
            break

    lines = []
    lines.append("=" * 70)
    lines.append("=== FILL-IN VALUES FOR ac_final.md ===")
    lines.append("=" * 70)
    lines.append("")

    # --- [A] Dimension-controlled partial correlation ---
    if highlight_row:
        r = highlight_row
        lines.append(
            f"[A]  Dimension-controlled partial correlation: RSS(L={highlight_length})-LEMB | dim")
        lines.append(f"     n = {r['n']} models")
        lines.append("")
        if r['r_partial_pearson'] is not None:
            lines.append(
                f"     Pearson:  r_partial = {r['r_partial_pearson']:+.4f}  (p = {r['p_partial_pearson']:.4f})")
        else:
            lines.append(f"     Pearson:  (could not compute)")
        if r['r_partial_spearman'] is not None:
            lines.append(
                f"     Spearman: rho_partial = {r['r_partial_spearman']:+.4f}  (p = {r['p_partial_spearman']:.4f})")
        else:
            lines.append(f"     Spearman: (could not compute)")
        lines.append("")

    # --- [A2] Zero-order dim-RSS correlation ---
    if highlight_row:
        r = highlight_row
        lines.append(
            f"[A2] Zero-order correlation: dim - RSS(L={highlight_length})")
        lines.append(
            f"     Pearson:  r = {r['r_rss_dim']:+.4f}  (p = {r['p_rss_dim']:.4f})")
        lines.append(
            f"     Spearman: rho = {r['rho_rss_dim']:+.4f}  (p = {r['p_rho_rss_dim']:.4f})")
        lines.append("")

    # --- [B] Multi-controlled (conditional) ---
    if collinearity_info:
        ci = collinearity_info
        lines.append(
            f"[B]  Multi-controlled: RSS(L={highlight_length})-LEMB | dim, params")
        lines.append(
            f"     Multicollinearity check: r(dim, params) = {ci['r_dim_params']:+.4f} (Pearson)")
        lines.append(
            f"                              rho(dim, params) = {ci['rho_dim_params']:+.4f} (Spearman)")
        lines.append(
            f"                              n = {ci['n']} models with both dim and params")
        lines.append("")
        if ci['high_collinearity']:
            lines.append(
                f"     WARNING: |r(dim, params)| = {abs(ci['r_dim_params']):.3f} >= 0.8")
            lines.append(
                f"     dim and params are strongly collinear. Multi-controlled partial r")
            lines.append(
                f"     is unstable at n={n_total}. Recommend: report as caveat only, do NOT")
            lines.append(
                f"     feature in ac_final.md body. Use footnote: 'dim and params are")
            lines.append(
                f"     tightly coupled (r={ci['r_dim_params']:+.2f}), so individual")
            lines.append(f"     separation is unreliable at n={n_total}.'")
            lines.append("")
            if 'r_partial_multi' in ci:
                lines.append(
                    f"     (Reference only) r(RSS, LEMB | dim, params) = {ci['r_partial_multi']:+.4f} (p={ci['p_partial_multi']:.4f})")
        else:
            if 'r_partial_multi' in ci:
                lines.append(
                    f"     r(RSS, LEMB | dim, params) = {ci['r_partial_multi']:+.4f}  (p = {ci['p_partial_multi']:.4f}, n = {ci['n_multi']})")
            else:
                lines.append(
                    f"     (could not compute — insufficient models with params data)")
        lines.append("")

    # --- [C] 4096-dim group for L=4,8,16 ---
    lines.append(f"[C]  Within-dimension spread: dim == 4096")
    lines.append("")
    if c_info and len(c_info) > 0:
        first_key = list(c_info.keys())[0]
        n_models = c_info[first_key]['n']
        model_list = ', '.join(c_info[first_key]['models'])
        lines.append(f"     n = {n_models} models")
        lines.append(f"     Models: {model_list}")
        lines.append("")
        for cL in [4, 8, 16]:
            if cL in c_info:
                ci = c_info[cL]
                lines.append(f"     L={cL}:  mean={ci['mean']:.2f}, std={ci['std']:.2f}, "
                             f"range=[{ci['min']:.2f}, {ci['max']:.2f}] (span={ci['range']:.2f}), "
                             f"IQR={ci['iqr']:.2f}")
        lines.append("")
        lines.append(
            "     Interpretation: Same-dimension models show substantial RSS spread,")
        lines.append("     confirming that dim alone does not determine RSS.")
    else:
        lines.append(
            f"     (insufficient models with dim=4096, or placeholder dims not yet filled)")
    lines.append("")
    lines.append("")
    lines.append("=" * 70)
    lines.append("FULL CORRELATION TABLE: All lengths L=4..16")
    lines.append("=" * 70)
    lines.append("")
    header = (
        f"{'Metric':12s} {'n':>3s}  "
        f"{'r(X,LEMB)':>10s} {'p':>8s}  "
        f"{'r(X,dim)':>10s} {'p':>8s}  "
        f"{'r(dim,LEMB)':>12s} {'p':>8s}  "
        f"{'r_partial':>10s} {'p':>8s}  "
        f"{'rho_partial':>10s} {'p':>8s}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for r in results:
        def fmt(v, width=10):
            if v is None:
                return '      n/a'.rjust(width)
            return f'{v:{width}.4f}'

        def fmtp(v, width=8):
            if v is None:
                return '    n/a'.rjust(width)
            return f'{v:{width}.4f}'

        lines.append(
            f"{r['metric']:12s} {r['n']:3d}  "
            f"{fmt(r['r_rss_lemb'])} {fmtp(r['p_rss_lemb'])}  "
            f"{fmt(r['r_rss_dim'])} {fmtp(r['p_rss_dim'])}  "
            f"{fmt(r['r_dim_lemb'], 12)} {fmtp(r['p_dim_lemb'])}  "
            f"{fmt(r['r_partial_pearson'])} {fmtp(r['p_partial_pearson'])}  "
            f"{fmt(r['r_partial_spearman'])} {fmtp(r['p_partial_spearman'])}"
        )

    lines.append("")
    lines.append(
        "Note: r_partial = Pearson partial r(X,LEMB | dim); rho_partial = Spearman partial rho")
    lines.append(f"      n_total = {n_total} models in pool")
    lines.append("")

    # --- Markdown table (paste-ready for rebuttal) ---
    lines.append("=" * 70)
    lines.append("MARKDOWN TABLE (paste-ready for rebuttal)")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "| Metric | n | r(X,LEMB) | p | r(X,dim) | p | r(dim,LEMB) | p | r_partial(P) | p | r_partial(S) | p |")
    lines.append(
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        metric_disp = f"**{r['metric']}**" if r['metric'] == f'RSS_L{highlight_length}' else r['metric']

        def fmt(v, p=False):
            if v is None:
                return "  n/a  " if p else "   n/a   "
            return f"{v:8.4f}" if p else f"{v:+.4f}"

        def fmtp(v):
            if v is None:
                return "   n/a   "
            return f"{v:8.4f}"
        lines.append(
            f"| {metric_disp:<8s} | {r['n']:>3d} | "
            f"{fmt(r['r_rss_lemb']):>10s} | {fmtp(r['p_rss_lemb']):>6s} | "
            f"{fmt(r['r_rss_dim']):>10s} | {fmtp(r['p_rss_dim']):>6s} | "
            f"{fmt(r['r_dim_lemb']):>12s} | {fmtp(r['p_dim_lemb']):>6s} | "
            f"{fmt(r['r_partial_pearson']):>10s} | {fmtp(r['p_partial_pearson']):>6s} | "
            f"{fmt(r['r_partial_spearman']):>10s} | {fmtp(r['p_partial_spearman']):>6s} |"
        )
    lines.append("")
    lines.append(
        "Note: P = Pearson partial, S = Spearman partial. Both control for dim.")
    lines.append("")

    # --- Interpretation guide ---
    lines.append("=" * 70)
    lines.append("INTERPRETATION GUIDE")
    lines.append("=" * 70)
    lines.append("")
    lines.append(
        "If r_partial remains significant (p < 0.05) and close to r(X,LEMB):")
    lines.append("  → Dimension does NOT explain the RSS-LEMB correlation.")
    lines.append(
        "  → RSS captures structural sensitivity beyond dimensionality effects.")
    lines.append("")
    lines.append("If r_partial drops substantially (|Delta| > 0.1):")
    lines.append(
        "  → Some of the RSS-LEMB correlation is mediated by dimension.")
    lines.append("  → Report honestly as a limitation.")
    lines.append("")
    lines.append(
        "Spearman partial is more robust because dim is near-discrete (few values).")
    lines.append(
        "If Pearson and Spearman point in the same direction, the conclusion is solid.")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    config = {
        'target_lengths': list(range(4, 17)),
        'highlight_length': 15,
        'exclude_models': [BASELINE_MODEL],
        'dim_params_map': MODEL_DIM_PARAMS,
    }
    generate_tbl_dim_partial_correl(
        "benchmarks",
        "source/compiled/rebuttal_dim_partial_correl",
        config,
    )
