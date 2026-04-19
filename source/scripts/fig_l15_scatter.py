import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple


LEMB_TASK_MAP = {
    'needle': 'LEMBNeedleRetrieval',
    'passkey': 'LEMBPasskeyRetrieval',
    'summscreen': 'LEMBSummScreenFDRetrieval',
    'qmsum': 'LEMBQMSumRetrieval',
    'wikimqa': 'LEMBWikimQARetrieval',
    'narrativeqa': 'LEMBNarrativeQARetrieval',
}


def gather_benchmark_data(directory: str) -> Dict[str, Dict]:
    """Gather LEMB and RSS benchmark data from directory."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    json_files = list(dir_path.rglob("overall_results.json")) + \
        list(dir_path.rglob("benchmark_results.json"))

    all_data = {}

    for json_file in json_files:
        model_name = json_file.parent.name if 'lemb' in str(
            json_file).lower() else json_file.parent.parent.name

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            continue

        if model_name not in all_data:
            all_data[model_name] = {'lemb': None, 'rss': None}

        # Detect LEMB data
        if any(k.startswith('LEMB') for k in data.keys()):
            all_data[model_name]['lemb'] = data
        # Detect RSS data
        elif 'average_rss' in data:
            all_data[model_name]['rss'] = data

    # Filter complete data only
    return {k: v for k, v in all_data.items()
            if v['lemb'] is not None and v['rss'] is not None}


def extract_rss_l15_mean(rss_data: dict) -> float:
    """Extract mean RSS score at L=15."""
    if not rss_data:
        return None

    length_scores = rss_data.get('length_scores', {})
    l15_data = length_scores.get('15')

    if not l15_data:
        return None

    rss_scores = l15_data.get('rss_scores', [])
    if not rss_scores:
        return None

    return np.mean(rss_scores)


def extract_lemb_task_score(lemb_data: dict, task_key: str) -> float:
    """Extract score for a specific LEMB task."""
    if not lemb_data or task_key not in lemb_data:
        return None

    task_data = lemb_data[task_key]
    if isinstance(task_data, dict):
        if 'needle' in task_key.lower() or 'passkey' in task_key.lower():
            return task_data.get('avg')
        else:
            return task_data.get('ndcg@10')
    return float(task_data) if task_data is not None else None


def _format_pvalue(p: float) -> str:
    """Format p-value in LaTeX scientific notation."""
    if p == 0:
        return r"$< 10^{-10}$"
    s = f"{p:.1e}"
    mantissa, exponent = s.split('e')
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def generate_l15_scatter_plot(model_data: Dict[str, Dict], task_short: str,
                              task_full: str, output_dir: str, config: dict = None,
                              baseline_model: str = None) -> str:
    """
    Generate scatter plot for RSS L=15 vs specific LEMB task.

    Args:
        model_data: Dictionary of model benchmark data
        task_short: Short task name (e.g., 'needle')
        task_full: Full LEMB task key (e.g., 'LEMBNeedleRetrieval')
        output_dir: Directory to save output

    Returns:
        Path to generated SVG file
    """
    # Collect paired data
    rss_values = []
    lemb_values = []
    model_names = []

    for model_name, metrics in model_data.items():
        rss_score = extract_rss_l15_mean(metrics['rss'])
        lemb_score = extract_lemb_task_score(metrics['lemb'], task_full)

        if rss_score is not None and lemb_score is not None:
            rss_values.append(rss_score)
            lemb_values.append(lemb_score)
            model_names.append(model_name)

    if len(rss_values) < 2:
        raise ValueError(f"Insufficient data for {task_short}")

    # Create filtered lists for calculation (excluding baseline model)
    calc_rss_values = rss_values
    calc_lemb_values = lemb_values

    if baseline_model:
        calc_rss_values = []
        calc_lemb_values = []
        for i, m in enumerate(model_names):
            if m != baseline_model:
                calc_rss_values.append(rss_values[i])
                calc_lemb_values.append(lemb_values[i])
        print(
            f"  Excluded baseline from calculation: {baseline_model} (n={len(calc_rss_values)})")

    # Calculate correlations using filtered data
    pearson_r, pearson_p = scipy_stats.pearsonr(
        calc_rss_values, calc_lemb_values)
    spearman_r, spearman_p = scipy_stats.spearmanr(
        calc_rss_values, calc_lemb_values)

    # Create plot
    plt.figure(figsize=(12, 9))

    # Plot all points first with default markers
    plt.scatter(x=rss_values, y=lemb_values, s=80, c='dimgrey', marker='o',
                alpha=0.5, edgecolor='black', linewidth=0.3, label='Other models')

    # Define color palette for aliased models
    alias_colors = ['#1f77b4', '#ff7f0e',
                    '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Re-plot with differentiated markers by model type
    aliases = config.get('aliases', {}) if config else {}
    alias_list = list(aliases.keys())  # Ordered list of aliased model names

    for i, model in enumerate(model_names):
        if model == baseline_model:
            # Baseline: black square
            plt.scatter(rss_values[i], lemb_values[i], s=100, c='black', marker='s',
                        alpha=0.9, edgecolor='black', linewidth=1.5,
                        label='Baseline' if i == 0 else "", zorder=5)
        elif model in aliases:
            # Aliased models: colored star using palette
            color_idx = alias_list.index(model) % len(alias_colors)
            model_color = alias_colors[color_idx]
            plt.scatter(rss_values[i], lemb_values[i], s=200, c=model_color, marker='*',
                        alpha=0.9, edgecolor='black', linewidth=1.5,
                        label='Highlighted' if i == 0 else "", zorder=5)
        # Default markers already plotted above

    # Add annotations with matching colors (intelligent placement)
    # Calculate median for smart offset direction
    median_x = np.median(rss_values)
    median_y = np.median(lemb_values)
    
    for i, model in enumerate(model_names):
        if model == baseline_model:
            # Annotation for baseline
            plt.text(rss_values[i], lemb_values[i] - 0.02, "Baseline",
                     fontsize=10, ha='center', fontweight='bold', color='black')
        elif model in aliases:
            # Aliased model annotation with matching color
            display_name = aliases[model]
            color_idx = alias_list.index(model) % len(alias_colors)
            model_color = alias_colors[color_idx]
            
            # Determine intelligent offset based on point position relative to median
            x_val = rss_values[i]
            y_val = lemb_values[i]
            
            # If point is on right side (> median), place label to the left
            if x_val > median_x:
                x_offset = -150
            else:
                x_offset = 10
            
            # If point is on top side (> median), place label below
            if y_val > median_y:
                y_offset = -20
            else:
                y_offset = 10
            
            plt.annotate(display_name,
                         (x_val, y_val),
                         fontsize=12, alpha=0.9, fontweight='bold',
                         xytext=(x_offset, y_offset), textcoords='offset points',
                         color=model_color,
                         arrowprops=dict(arrowstyle='-', color='gray', alpha=0.3))

    # Add trend line (using filtered data)
    z = np.polyfit(calc_rss_values, calc_lemb_values, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(rss_values), max(rss_values), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend line')

    # Labels and title
    plt.xlabel('RSS(L=15)', fontsize=12, fontweight='bold')
    plt.ylabel(f'{task_short.lower()} Score', fontsize=12, fontweight='bold')

    # Annotate with correlation coefficients (p-values in scientific notation)
    textstr = f'Pearson r = {pearson_r:.3f} (p = {_format_pvalue(pearson_p)})\n' \
        f'Spearman ρ = {spearman_r:.3f} (p = {_format_pvalue(spearman_p)})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()

    # Save as SVG
    output_path = Path(output_dir) / f'l15_rss_vs_{task_short}_scatter.svg'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_path}")
    print(f"    Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}")

    return str(output_path)


def generate_l15_lemb_scatters(data_dir: str, output_dir: str, config: dict) -> Dict[str, str]:
    """
    Generate L=15 RSS vs LEMB task scatter plots for specified tasks.

    Args:
        data_dir: Directory containing benchmark data
        output_dir: Directory to save output SVG files
        config: Configuration dictionary with keys:
            - lemb_tasks: List of task short names (e.g., ['needle', 'qmsum'])
            - exclude_models: Optional list of models to exclude

    Returns:
        Dictionary mapping task names to generated file paths
    """
    print(f"=" * 60)
    print("L=15 LEMB SCATTER PLOT GENERATION")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Gather data
    all_data = gather_benchmark_data(data_dir)

    # Get baseline model (optional - used to skip annotation but still plot)
    baseline_model = config.get('baseline_model', None)
    if baseline_model:
        print(f"Baseline model (annotation skipped): {baseline_model}")

    if not all_data:
        raise ValueError("No valid data after filtering.")

    print(f"Processing {len(all_data)} models")

    # Get task list from config
    task_list = config.get('lemb_tasks', list(LEMB_TASK_MAP.keys()))

    # Generate plots for each task
    results = {}

    for task_short in task_list:
        if task_short not in LEMB_TASK_MAP:
            print(f"  ⚠ Unknown task: {task_short}, skipping")
            continue

        task_full = LEMB_TASK_MAP[task_short]
        print(f"\nGenerating plot for {task_short}...")

        try:
            output_path = generate_l15_scatter_plot(
                all_data, task_short, task_full, output_dir, config, baseline_model)
            results[task_short] = output_path
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Generated {len(results)} plots")

    return results
