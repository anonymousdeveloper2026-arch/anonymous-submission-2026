import os
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# =============================================================================
# Helper Functions
# =============================================================================

def _to_serializable(obj):
    """
    Recursively convert NumPy types and pandas types to JSON-serializable Python types.
    Handles np.int*, np.float*, np.nan, and DataFrames.
    """
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# =============================================================================
# Configuration
# =============================================================================


# Target models for Table 3 (per-model comparison)


# =============================================================================
# Core Calculation
# =============================================================================

def _calculate_metrics(
    input_parquet: str,
    config: dict
) -> Dict[str, Any]:
    """
    Calculate all metrics from the input parquet file.
    
    Args:
        input_parquet: Path to the input parquet file.
        config: Configuration dictionary with optional keys:
            - include_models: Comma-separated string of model names to include
            - target_models: List of model names for per-model comparison table
    
    Returns a dictionary containing:
    - quadrant_counts: dict with counts for each quadrant
    - wilcoxon_results: dict with W-statistic, p-value, etc.
    - per_model_data: DataFrame with pos1_rate, neg1_rate, difference
    - metadata: n_pos, n_neg, total, midpoint, etc.
    """
    include_models = config.get("include_models")
    target_models = config.get("target_models", [])
    
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Input not found: {input_parquet}")
    
    df = pd.read_parquet(input_parquet)
    
    # Filter by included models if specified
    if include_models:
        model_set = set(m.strip() for m in include_models.split(","))
        df = df[df["model"].isin(model_set)]
        if df.empty:
            raise ValueError("No data remaining after model filter.")
    
    total = len(df)
    
    # Assert fixed cluster sizes
    uniq_max_pos = df["max_rank_pos"].unique()
    uniq_max_neg = df["max_rank_neg"].unique()
    assert len(uniq_max_pos) == 1, f"max_rank_pos not fixed: {sorted(uniq_max_pos.tolist())}"
    assert len(uniq_max_neg) == 1, f"max_rank_neg not fixed: {sorted(uniq_max_neg.tolist())}"
    
    max_rank_pos = int(uniq_max_pos[0])
    max_rank_neg = int(uniq_max_neg[0])
    n_pos = max_rank_pos - 1
    n_neg = max_rank_neg - 1
    midpoint = max_rank_pos // 2
    
    # Quadrant Analysis
    pos_far = df["rank_pos"] <= midpoint
    pos_close = df["rank_pos"] > midpoint
    neg_far = df["rank_neg"] <= midpoint
    neg_close = df["rank_neg"] > midpoint
    
    q_lower_left = df[pos_far & neg_far]      # Both far
    q_upper_right = df[pos_close & neg_close] # Both close
    q_lower_right = df[pos_close & neg_far]   # Pos close, Neg far
    q_upper_left = df[pos_far & neg_close]    # Pos far, Neg close
    
    quadrant_counts = {
        "lower_left": len(q_lower_left),
        "upper_right": len(q_upper_right),
        "lower_right": len(q_lower_right),
        "upper_left": len(q_upper_left),
        "total": total,
    }
    
    # Per-model rates for Wilcoxon test
    per_model = (
        df.groupby("model", observed=True)
        .apply(lambda g: pd.Series({
            "pos1_rate": ((g["rank_pos"] > midpoint) & (g["rank_neg"] <= midpoint)).mean(),
            "neg1_rate": ((g["rank_pos"] <= midpoint) & (g["rank_neg"] > midpoint)).mean(),
            "cluster_separation": (1 - g["sim_centroids"]).mean(),
        }))
    )
    
    # Filter out models where pos1_rate == neg1_rate
    valid = per_model[per_model["pos1_rate"] != per_model["neg1_rate"]]
    n_models_valid = len(valid)
    
    if n_models_valid >= 2:
        w_stat, p_val = wilcoxon(
            valid["pos1_rate"],
            valid["neg1_rate"],
            alternative="greater"
        )
    else:
        w_stat, p_val = np.nan, np.nan
    
    consistent = valid[valid["pos1_rate"] > valid["neg1_rate"]]
    n_consistent = len(consistent)
    inconsistent = valid[valid["pos1_rate"] <= valid["neg1_rate"]]
    inconsistent_models = inconsistent.index.tolist()
    
    wilcoxon_results = {
        "w_statistic": w_stat,
        "p_value": p_val,
        "n_models_valid": n_models_valid,
        "n_consistent": n_consistent,
        "inconsistent_models": inconsistent_models,
        "median_pos1_rate": valid["pos1_rate"].median() if n_models_valid > 0 else np.nan,
        "median_neg1_rate": valid["neg1_rate"].median() if n_models_valid > 0 else np.nan,
    }
    
    # Per-model comparison data (sorted by difference)
    per_model_sorted = per_model.copy()
    per_model_sorted['difference'] = per_model_sorted['pos1_rate'] - per_model_sorted['neg1_rate']
    per_model_sorted = per_model_sorted.sort_values(by='difference', ascending=True)
    
    # Extract target models if they exist
    target_model_data = []
    for model in target_models:
        if model in per_model_sorted.index:
            row = per_model_sorted.loc[model]
            target_model_data.append({
                "model": model,
                "pos1_rate": row["pos1_rate"],
                "neg1_rate": row["neg1_rate"],
                "difference": row["difference"],
            })
    
    metadata = {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "midpoint": midpoint,
        "total": total,
        "n_models": df["model"].nunique(),
    }
    
    # Convert DataFrame to list of dicts for JSON serialization
    per_model_data_list = per_model_sorted.reset_index().to_dict('records')
    
    result = {
        "quadrant_counts": quadrant_counts,
        "wilcoxon_results": wilcoxon_results,
        "per_model_data": per_model_data_list,
        "target_model_data": target_model_data,
        "metadata": metadata,
    }
    
    # Ensure all values are JSON-serializable
    return _to_serializable(result)


# =============================================================================
# LaTeX Rendering
# =============================================================================

def _render_latex(raw_data: Dict[str, Any]) -> str:
    """Render all three tables as LaTeX code."""
    
    # Table 1: Quadrant Analysis
    qc = raw_data["quadrant_counts"]
    total = qc["total"]
    
    def _pct(count):
        return 100.0 * count / total if total > 0 else 0
    
    # Find max proportion for bold formatting
    proportions = {
        "lower_left": _pct(qc["lower_left"]),
        "upper_right": _pct(qc["upper_right"]),
        "lower_right": _pct(qc["lower_right"]),
        "upper_left": _pct(qc["upper_left"]),
    }
    max_prop = max(proportions.values())
    
    def _fmt_prop(key, count):
        prop = proportions[key]
        fmt = f"{prop:.2f}\\%"
        if prop == max_prop:
            return f"$\\mathbf{{{prop:.2f}\\%}}$"
        return f"${fmt}$"
    
    latex_table1 = f"""\\begin{{table}}[ht!]
\\centering
\\caption{{Quadrant analysis results}}
\\label{{tbl:d-r-centroid-all-heatmap-quadrant}}
\\begin{{tabular}}{{lcc}}
\\toprule
Quadrant & Vector Count & Proportion \\\\
\\midrule
lower-left & {qc["lower_left"]:,} & {_fmt_prop("lower_left", qc["lower_left"])} \\\\
upper-right & {qc["upper_right"]:,} & {_fmt_prop("upper_right", qc["upper_right"])} \\\\
lower-right & {qc["lower_right"]:,} & {_fmt_prop("lower_right", qc["lower_right"])} \\\\
upper-left & {qc["upper_left"]:,} & {_fmt_prop("upper_left", qc["upper_left"])} \\\\
total & {qc["total"]:,} & $100\\%$ \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    # Table 2: Wilcoxon Results
    wr = raw_data["wilcoxon_results"]
    
    if np.isnan(wr["w_statistic"]):
        w_stat_str = "N/A"
        p_val_str = "N/A"
    else:
        w_stat_str = f"{wr['w_statistic']:.3f}"
        p_val_str = "< 0.0001" if wr["p_value"] < 0.0001 else f"{wr['p_value']:.4f}"
    
    consistent_pct = 100.0 * wr["n_consistent"] / wr["n_models_valid"] if wr["n_models_valid"] > 0 else 0
    inconsistent_pct = 100.0 * (wr["n_models_valid"] - wr["n_consistent"]) / wr["n_models_valid"] if wr["n_models_valid"] > 0 else 0
    
    latex_table2 = f"""\\begin{{table}}[ht!]
\\centering
\\caption{{Wilcoxon signed-rank test results}}
\\label{{tbl:d-r-centroid-wilcoxon-results}}
\\begin{{tabular}}{{lc}}
\\toprule
Metric & Value \\\\
\\midrule
W-statistic & {w_stat_str} \\\\
p-value & {p_val_str} \\\\
Number of valid models ($N_{{valid}}$) & {wr["n_models_valid"]} \\\\
Number of consistent models ($r_{{pos}} > r_{{neg}}$) & {wr["n_consistent"]} ({consistent_pct:.1f}\\%) \\\\
Number of inconsistent models ($r_{{pos}} \\leq r_{{neg}}$) & {wr["n_models_valid"] - wr["n_consistent"]} ({inconsistent_pct:.1f}\\%) \\\\
Median $r_{{pos}}$ (overall) & {wr["median_pos1_rate"]:.4f} \\\\
Median $r_{{neg}}$ (overall) & {wr["median_neg1_rate"]:.4f} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    # Table 3: Per-Model Comparison
    target_data = raw_data["target_model_data"]
    
    rows = []
    for item in target_data:
        model_escaped = item["model"].replace("_", "\\_")
        rows.append(
            f"\\texttt{{{model_escaped}}} & {item['pos1_rate']:.6f} & "
            f"{item['neg1_rate']:.6f} & {item['difference']:.6f} \\\\"
        )
    
    rows_str = "\n".join(rows)
    
    latex_table3 = f"""\\begin{{table}}[ht!]
\\centering
\\caption{{Per-model comparison of $r_{{pos}}$ and $r_{{neg}}$}}
\\label{{tbl:d-r-centroid-nemotron-comparison}}
\\begin{{tabular}}{{lccc}}
\\toprule
Model & $r_{{pos}}$ & $r_{{neg}}$ & $r_{{pos}} - r_{{neg}}$ \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    
    # Combine all tables
    full_latex = f"{latex_table1}\n\n{latex_table2}\n\n{latex_table3}"
    return full_latex


# =============================================================================
# Markdown Rendering
# =============================================================================

def _render_markdown(raw_data: Dict[str, Any]) -> str:
    """Render all three tables as Markdown code."""
    
    # Table 1: Quadrant Analysis
    qc = raw_data["quadrant_counts"]
    total = qc["total"]
    
    def _pct(count):
        return 100.0 * count / total if total > 0 else 0
    
    md_table1 = f"""## Table 1: Quadrant Analysis Results

| Quadrant | Vector Count | Proportion |
|----------|-------------|------------|
| lower-left | {qc["lower_left"]:,} | {_pct(qc["lower_left"]):.2f}% |
| upper-right | {qc["upper_right"]:,} | {_pct(qc["upper_right"]):.2f}% |
| lower-right | {qc["lower_right"]:,} | {_pct(qc["lower_right"]):.2f}% |
| upper-left | {qc["upper_left"]:,} | {_pct(qc["upper_left"]):.2f}% |
| **total** | **{qc["total"]:,}** | **100%** |
"""
    
    # Table 2: Wilcoxon Results
    wr = raw_data["wilcoxon_results"]
    
    if np.isnan(wr["w_statistic"]):
        w_stat_str = "N/A"
        p_val_str = "N/A"
    else:
        w_stat_str = f"{wr['w_statistic']:.3f}"
        p_val_str = "< 0.0001" if wr["p_value"] < 0.0001 else f"{wr['p_value']:.4f}"
    
    consistent_pct = 100.0 * wr["n_consistent"] / wr["n_models_valid"] if wr["n_models_valid"] > 0 else 0
    inconsistent_pct = 100.0 * (wr["n_models_valid"] - wr["n_consistent"]) / wr["n_models_valid"] if wr["n_models_valid"] > 0 else 0
    
    md_table2 = f"""## Table 2: Wilcoxon Signed-Rank Test Results

| Metric | Value |
|--------|-------|
| W-statistic | {w_stat_str} |
| p-value | {p_val_str} |
| Number of valid models (N_valid) | {wr["n_models_valid"]} |
| Consistent models (r_pos > r_neg) | {wr["n_consistent"]} ({consistent_pct:.1f}%) |
| Inconsistent models (r_pos ≤ r_neg) | {wr["n_models_valid"] - wr["n_consistent"]} ({inconsistent_pct:.1f}%) |
| Median r_pos (overall) | {wr["median_pos1_rate"]:.4f} |
| Median r_neg (overall) | {wr["median_neg1_rate"]:.4f} |
"""
    
    # Table 3: Per-Model Comparison
    target_data = raw_data["target_model_data"]
    
    rows = []
    for item in target_data:
        rows.append(
            f"| `{item['model']}` | {item['pos1_rate']:.6f} | "
            f"{item['neg1_rate']:.6f} | {item['difference']:.6f} |"
        )
    
    rows_str = "\n".join(rows)
    
    md_table3 = f"""## Table 3: Per-Model Comparison of r_pos and r_neg

| Model | r_pos | r_neg | r_pos - r_neg |
|-------|-------|-------|---------------|
{rows_str}
"""
    
    # Combine all tables
    full_md = f"{md_table1}\n\n{md_table2}\n\n{md_table3}"
    return full_md


# =============================================================================
# Master Orchestrator
# =============================================================================

def generate_table_7_analysis(
    input_parquet: str,
    config: dict = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Generate Table 7 analysis: quadrant, Wilcoxon, and per-model comparison.
    
    Args:
        input_parquet: Path to the input parquet file.
        config: Configuration dictionary (optional). Keys:
            - include_models: Comma-separated string of model names to include
            - target_models: List of model names for per-model comparison table
    
    Returns:
        (raw_data, latex_string, markdown_string) tuple
    """
    if config is None:
        config = {}
    
    raw_data = _calculate_metrics(input_parquet, config)
    latex_string = _render_latex(raw_data)
    markdown_string = _render_markdown(raw_data)
    return raw_data, latex_string, markdown_string
