import json
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_TARGET_TRANSITIONS = [
    (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16)
]
SUMMARY_TRANSITIONS = [(4, 8), (8, 16)]


# =============================================================================
# Data Processing
# =============================================================================

def _extract_rss_and_latency(data: dict) -> Dict[int, Tuple[List[float], List[float]]]:
    """Extract raw RSS scores and latency for all sequence lengths."""
    length_scores = data.get('length_scores', {})
    results = {}
    
    for length_str, metrics in length_scores.items():
        try:
            length = int(length_str)
            rss_scores = metrics.get('rss_scores', [])
            latency_scores = metrics.get('total_latency_scores', [])
            
            if rss_scores and latency_scores:
                results[length] = (rss_scores, latency_scores)
        except (ValueError, TypeError):
            continue
    
    return results


def _calculate_metrics(directory: str, config: dict) -> Dict[str, Any]:
    """
    Process RSS directory and perform Welch's t-tests for length transitions.
    
    Returns dictionary with:
    - granular_results: List of transition analyses (unit-spaced)
    - summary_results: List of transition analyses (doubling)
    - metadata: Processing information
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    # Get exclude models from config
    exclude_models = set(config.get('exclude_models', []))
    
    json_files = list(dir_path.rglob('benchmark_results.json'))
    if not json_files:
        raise ValueError(f"No benchmark_results.json files found in '{directory}'.")
    
    # Collect all model data
    all_models_data = {}
    
    for json_file in json_files:
        model_name = json_file.parent.parent.name
        
        # Skip baseline and excluded models
        if  model_name in exclude_models:
            print(f"  ⊘ Excluded model: {model_name}")
            continue
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading '{json_file}': {e}")
            continue
        
        # Verify this is RSS data
        if 'average_rss' not in data:
            print(f"  ⊘ Skipped '{json_file.name}' (not RSS data)")
            continue
        
        length_data = _extract_rss_and_latency(data)
        if length_data:
            all_models_data[model_name] = length_data
            print(f"  ✓ {model_name:<30} | Lengths: {sorted(length_data.keys())}")
        else:
            print(f"  ✗ {model_name:<30} | No valid data")
    
    if not all_models_data:
        raise ValueError("No valid RSS data found after filtering.")
    
    total_models = len(all_models_data)
    ALPHA = config.get('alpha', 0.05)
    
    # Perform granular analysis (all adjacent transitions)
    granular_results = _analyze_transitions(
        all_models_data, DEFAULT_TARGET_TRANSITIONS, ALPHA
    )
    
    # Perform summary analysis (doubling transitions)
    summary_results = _analyze_transitions(
        all_models_data, SUMMARY_TRANSITIONS, ALPHA
    )
    
    return {
        'granular_results': granular_results,
        'summary_results': summary_results,
        'metadata': {
            'n_models': total_models,
            'alpha': ALPHA,
            'excluded_models': list(exclude_models),
            'directory': directory,
        }
    }


def _analyze_transitions(
    models_data: Dict[str, Dict[int, Tuple[List[float], List[float]]]],
    transitions: List[Tuple[int, int]],
    alpha: float
) -> List[Dict]:
    """Perform Welch's t-tests for specified transitions across all models."""
    
    results = []
    
    for l_current, l_next in transitions:
        transition_key = (f'L{l_current}', f'L{l_next}')
        
        # Collect p-values and track worst case
        all_p_values = []
        sig_count = 0
        worst_case = {'max_p': 0.0, 'model': None}
        
        for model_name, length_data in models_data.items():
            # Check if both lengths exist for this model
            if l_current not in length_data or l_next not in length_data:
                continue
            
            rss_current, _ = length_data[l_current]
            rss_next, _ = length_data[l_next]
            
            # Welch's t-test: H0: RSS_current <= RSS_next, H1: RSS_current > RSS_next
            # We expect RSS to decrease as length increases
            rss_tstat, rss_pval = stats.ttest_ind(
                rss_current, rss_next, 
                alternative='greater', 
                equal_var=False
            )
            
            all_p_values.append(rss_pval)
            
            if rss_pval < alpha:
                sig_count += 1
            
            # Track worst case (largest p-value)
            if rss_pval > worst_case['max_p']:
                worst_case['max_p'] = rss_pval
                worst_case['model'] = model_name.replace("_", r"\_")
        
        if all_p_values:
            results.append({
                'transition': transition_key,
                'l_current': l_current,
                'l_next': l_next,
                'sig_count': sig_count,
                'total_models': len(all_p_values),
                'median_p': np.median(all_p_values),
                'max_p': worst_case['max_p'],
                'worst_case_model': worst_case['model'],
                'all_p_values': all_p_values,
            })
    
    return results


# =============================================================================
# LaTeX Rendering
# =============================================================================

def _format_scientific(val: float) -> str:
    """Format a float as LaTeX scientific notation like $1.95 \\times 10^{-5}$."""
    if val <= 0:
        return r"$< 10^{-15}$"
    
    # Convert to scientific notation string
    sci_str = f"{val:.2e}"  # e.g., "1.95e-05"
    
    # Split mantissa and exponent
    if 'e' in sci_str:
        mantissa, exponent = sci_str.split('e')
        exponent = int(exponent)
        return f"${mantissa} \\times 10^{{{exponent}}}$"
    else:
        # No exponent needed
        return f"${val:.2f}$"


def _render_latex(raw_data: Dict[str, Any]) -> str:
    """Render comprehensive Welch's t-test table as LaTeX."""
    
    granular = raw_data['granular_results']
    summary = raw_data['summary_results']
    n_models = raw_data['metadata']['n_models']
    
    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        f"\\caption{{Welch's one-tailed t-tests for $\\textsc{{RSS}}$ degradation at all adjacent length transitions, across {n_models} models. ``\\# Sig.'' is the number of models for which the transition reached significance at $\\alpha = 0.05$. ``Median $p$'' and ``Max $p$'' are computed across the {n_models} per-model p-values. The top block reports all 12 unit-spaced transitions; the bottom block reports the two doubling transitions used as the primary analysis scale in the main text. Note the approximately eight orders of magnitude gap between the median p-values of the two blocks.}}",
        r"\begin{tabular}{lcccl}",
        r"\toprule",
        r"Transition & \# Sig. / Total & Median $p$ & Max $p$ & Worst-case model \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Unit-spaced transitions}} \\",
        r"\midrule"
    ]
    
    # Granular (unit-spaced) transitions
    for result in granular:
        l_curr, l_next = result['transition']
        trans_label = f"${l_curr} \\to {l_next}$"
        sig_str = f"{result['sig_count']} / {result['total_models']}"
        
        # Format p-values in scientific notation
        median_p = result['median_p']
        max_p = result['max_p']
        
        if median_p > 0:
            median_str = _format_scientific(median_p)
        else:
            median_str = r"$< 10^{-15}$"
        
        if max_p > 0:
            max_str = _format_scientific(max_p)
        else:
            max_str = r"$< 10^{-15}$"
        
        model_str = f"\\texttt{{{result['worst_case_model']}}}"
        
        lines.append(
            f"{trans_label:<14} & {sig_str:<12} & {median_str:<20} & {max_str:<20} & {model_str} \\\\"
        )
    
    # Separator and summary (doubling) transitions
    lines.extend([
        r"\midrule",
        r"\multicolumn{5}{l}{\textit{Doubling transitions (primary scale)}} \\",
        r"\midrule"
    ])
    
    for result in summary:
        l_curr, l_next = result['transition']
        trans_label = f"${l_curr} \\to {l_next}$"
        sig_str = f"{result['sig_count']} / {result['total_models']}"
        
        median_p = result['median_p']
        max_p = result['max_p']
        
        if median_p > 0:
            median_str = _format_scientific(median_p)
        else:
            median_str = r"$< 10^{-15}$"
        
        if max_p > 0:
            max_str = _format_scientific(max_p)
        else:
            max_str = r"$< 10^{-15}$"
        
        model_str = f"\\texttt{{{result['worst_case_model']}}}"
        
        lines.append(
            f"{trans_label:<14} & {sig_str:<12} & {median_str:<20} & {max_str:<20} & {model_str} \\\\"
        )
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])
    
    return "\n".join(lines)


# =============================================================================
# Markdown Rendering
# =============================================================================

def _render_markdown(raw_data: Dict[str, Any]) -> str:
    """Render Welch's t-test results as Markdown table."""
    
    granular = raw_data['granular_results']
    summary = raw_data['summary_results']
    n_models = raw_data['metadata']['n_models']
    
    lines = [
        f"## Table: Welch's t-tests for RSS degradation ({n_models} models)",
        "",
        "### Unit-spaced transitions",
        "",
        "| Transition | # Sig. / Total | Median p | Max p | Worst-case model |",
        "|------------|----------------|----------|-------|------------------|"
    ]
    
    for result in granular:
        l_curr, l_next = result['transition']
        trans_label = f"{l_curr} → {l_next}"
        sig_str = f"{result['sig_count']} / {result['total_models']}"
        median_str = f"{result['median_p']:.2e}" if result['median_p'] > 0 else "< 1e-15"
        max_str = f"{result['max_p']:.2e}" if result['max_p'] > 0 else "< 1e-15"
        
        lines.append(
            f"| {trans_label} | {sig_str} | {median_str} | {max_str} | `{result['worst_case_model']}` |"
        )
    
    lines.extend([
        "",
        "### Doubling transitions (primary scale)",
        "",
        "| Transition | # Sig. / Total | Median p | Max p | Worst-case model |",
        "|------------|----------------|----------|-------|------------------|"
    ])
    
    for result in summary:
        l_curr, l_next = result['transition']
        trans_label = f"{l_curr} → {l_next}"
        sig_str = f"{result['sig_count']} / {result['total_models']}"
        median_str = f"{result['median_p']:.2e}" if result['median_p'] > 0 else "< 1e-15"
        max_str = f"{result['max_p']:.2e}" if result['max_p'] > 0 else "< 1e-15"
        
        lines.append(
            f"| {trans_label} | {sig_str} | {median_str} | {max_str} | `{result['worst_case_model']}` |"
        )
    
    return "\n".join(lines)


# =============================================================================
# Master Orchestrator
# =============================================================================

def generate_table_16_welch_ttest(
    directory: str,
    config: dict = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Generate Table 16 analysis: Welch's t-tests for RSS degradation.
    
    Args:
        directory: Directory containing RSS benchmark results
        config: Configuration dictionary (optional). Keys:
            - exclude_models: List of models to exclude
            - alpha: Significance threshold (default: 0.05)
    
    Returns:
        (raw_data, latex_string, markdown_string) tuple
    """
    if config is None:
        config = {}
    
    print(f"=" * 70)
    print("TABLE 16: WELCH'S T-TEST ANALYSIS FOR RSS DEGRADATION")
    print(f"=" * 70)
    print(f"Scanning directory: {directory}")
    print(f"Alpha threshold: {config.get('alpha', 0.05)}")
    print()
    
    raw_data = _calculate_metrics(directory, config)
    
    print(f"\nProcessed {raw_data['metadata']['n_models']} models.")
    print(f"Granular transitions: {len(raw_data['granular_results'])}")
    print(f"Summary transitions: {len(raw_data['summary_results'])}")
    
    print(f"\nRendering LaTeX and Markdown...")
    latex_string = _render_latex(raw_data)
    markdown_string = _render_markdown(raw_data)
    
    return raw_data, latex_string, markdown_string
