import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict


LEMB_TASK_MAP = {
    'needle': 'LEMBNeedleRetrieval',
    'passkey': 'LEMBPasskeyRetrieval',
    'summscreen': 'LEMBSummScreenFDRetrieval',
    'qmsum': 'LEMBQMSumRetrieval',
    'wikimqa': 'LEMBWikimQARetrieval',
    'narrativeqa': 'LEMBNarrativeQARetrieval',
}


def _gather_benchmark_data(directory: str) -> Dict[str, Dict]:
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


def _extract_rss_at_l(rss_data: dict, length: int) -> float:
    """Extract mean RSS score at specific length L."""
    if not rss_data:
        return None

    length_scores = rss_data.get('length_scores', {})
    l_data = length_scores.get(str(length))

    if not l_data:
        return None

    rss_scores = l_data.get('rss_scores', [])
    if not rss_scores:
        return None

    return np.mean(rss_scores)


def _extract_lemb_task_score(lemb_data: dict, task_key: str) -> float:
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


def _generate_l_curve_data(data_dir: str, config: dict = None) -> Dict[str, pd.DataFrame]:
    """
    Calculate correlation data between RSS at each L and LEMB task scores.

    Args:
        data_dir: Directory containing benchmark data
        config: Optional configuration with 'exclude_models' list

    Returns:
        Dictionary mapping task names to DataFrames with columns:
        [L, pearson_r, pearson_p, spearman_r, spearman_p, n_models]
    """
    # Gather data
    all_data = _gather_benchmark_data(data_dir)

    # Apply exclusions
    exclude_models = config.get('exclude_models', []) if config else []
    filtered_data = {k: v for k,
                     v in all_data.items() if k not in exclude_models}

    if not filtered_data:
        raise ValueError("No valid data after filtering.")

    print(
        f"Processing {len(filtered_data)} models (excluded: {len(exclude_models)})")

    # Target lengths
    lengths = list(range(4, 17))

    # Calculate correlations for each task and each L
    results = {}

    for task_short, task_full in LEMB_TASK_MAP.items():
        task_results = []

        for length in lengths:
            # Collect paired data
            rss_values = []
            lemb_values = []

            for model_name, metrics in filtered_data.items():
                rss_score = _extract_rss_at_l(metrics['rss'], length)
                lemb_score = _extract_lemb_task_score(
                    metrics['lemb'], task_full)

                if rss_score is not None and lemb_score is not None:
                    rss_values.append(rss_score)
                    lemb_values.append(lemb_score)

            # Calculate correlations if enough data
            if len(rss_values) >= 2:
                pearson_r, pearson_p = scipy_stats.pearsonr(
                    rss_values, lemb_values)
                spearman_r, spearman_p = scipy_stats.spearmanr(
                    rss_values, lemb_values)
            else:
                pearson_r = pearson_p = spearman_r = spearman_p = np.nan

            task_results.append({
                'L': length,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'n_models': len(rss_values),
            })

        results[task_short] = pd.DataFrame(task_results)
        print(f"  {task_short:12s}: {len(task_results)} L-values calculated")

    return results

def _render_single_plot(ax, data_dict: Dict[str, pd.DataFrame], metric_type: str):
    """
    Render a single L-curve plot (Pearson or Spearman).

    Uses hybrid visualization:
    - Line segments: solid for significant (p<0.05), dotted for non-significant
    - Scatter markers: size based on p-value tier, hollow for non-significant

    Args:
        ax: Matplotlib axis to render on
        data_dict: Dictionary mapping task names to DataFrames
        metric_type: 'pearson' or 'spearman'
    """
    color_map = plt.cm.tab10
    all_r_values = []

    for idx, (task_name, df) in enumerate(data_dict.items()):
        base_color_rgb = color_map(idx % 10)[:3]

        r_col = f"{metric_type}_r"
        p_col = f"{metric_type}_p"

        all_r_values.extend(df[r_col].tolist())

        # Plot line segment by segment based on significance
        for i in range(len(df) - 1):
            row1 = df.iloc[i]
            row2 = df.iloc[i + 1]
            
            # Determine line style based on both endpoints
            p1 = row1[p_col]
            p2 = row2[p_col]
            
            # Segment is significant if BOTH endpoints are significant (p < 0.05)
            if p1 < 0.05 and p2 < 0.05:
                linestyle = '-'
                linewidth = 2.5
            else:
                linestyle = ':'
                linewidth = 1.5
            
            ax.plot([row1['L'], row2['L']], [row1[r_col], row2[r_col]],
                    color=base_color_rgb, linestyle=linestyle, linewidth=linewidth,
                    alpha=0.85, zorder=1)

        # Plot markers with enhanced visualization based on p-value tier
        for _, row in df.iterrows():
            p_val = row[p_col]
            l_val = row['L']
            r_val = row[r_col]
            
            if p_val >= 0.05:
                # Non-significant: small white-filled marker
                ax.scatter(l_val, r_val, s=40, c='white',
                          edgecolors=base_color_rgb, linewidth=1.5, alpha=0.85,
                          zorder=3)
            elif p_val < 0.001:
                # Highly significant (***): double circle effect
                # Outer circle (thick colored edge, white fill)
                ax.scatter(l_val, r_val, s=230, c='white',
                          edgecolors=base_color_rgb, linewidth=1.5, alpha=0.95,
                          zorder=4)
                # Inner circle (smaller, filled with color)
                ax.scatter(l_val, r_val, s=80, c=base_color_rgb,
                          edgecolors=base_color_rgb, linewidth=1, alpha=0.9,
                          zorder=5)
            elif p_val < 0.01:
                # Very significant (**): medium filled marker
                ax.scatter(l_val, r_val, s=100, c=base_color_rgb,
                          edgecolors=base_color_rgb, linewidth=2, alpha=0.85,
                          zorder=3)
            else:
                # Significant (*): small filled marker
                ax.scatter(l_val, r_val, s=40, c=base_color_rgb,
                          edgecolors=base_color_rgb, linewidth=1.5, alpha=0.85,
                          zorder=3)
        
        # Add legend entry for this task (use a representative marker)
        ax.scatter([], [], s=100, c=base_color_rgb, edgecolors=base_color_rgb,
                  linewidth=2, alpha=0.85, label=task_name)

    # Reference line
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.6, label='Threshold (r=0.5)', zorder=0)

    # Labels
    ylabel = 'Pearson r' if metric_type == 'pearson' else 'Spearman ρ'
    ax.set_xlabel('Sequence Length (L)', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10, ncol=1, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)

    # Set limits
    if all_r_values:
        ax.set_ylim(-0.1, 1.0)

    all_l = [df['L'].tolist() for df in data_dict.values()]
    all_l_flat = [l for sublist in all_l for l in sublist]
    if all_l_flat:
        x_min, x_max = min(all_l_flat) - 0.5, max(all_l_flat) + 0.5
        ax.set_xlim(x_min, x_max)


def generate_lx_master_figure(data_dir: str, output_dir: str, config: dict = None) -> Dict[str, str]:
    """
    Master orchestrator for L-curve figure generation.

    Generates separate Pearson and Spearman plots.

    Args:
        data_dir: Directory containing benchmark data
        output_dir: Directory to save output SVG files
        config: Optional configuration dict with 'exclude_models'

    Returns:
        Dictionary with paths to generated files
    """
    print(f"=" * 60)
    print("L-CURVE MASTER FIGURE GENERATION")
    print(f"=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Generate correlation data
    data_dict = _generate_l_curve_data(data_dir, config)

    results = {}

    # Generate Pearson plot
    print(f"\nGenerating Pearson plot...")
    fig, ax = plt.subplots(figsize=(14, 9))
    _render_single_plot(ax, data_dict, 'pearson')
    plt.tight_layout()

    pearson_path = Path(output_dir) / 'lx_correlation_pearson.svg'
    pearson_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(pearson_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved: {pearson_path}")
    plt.close()
    results['pearson'] = str(pearson_path)

    # Generate Spearman plot
    print(f"\nGenerating Spearman plot...")
    fig, ax = plt.subplots(figsize=(14, 9))
    _render_single_plot(ax, data_dict, 'spearman')
    plt.tight_layout()

    spearman_path = Path(output_dir) / 'lx_correlation_spearman.svg'
    plt.savefig(spearman_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Saved: {spearman_path}")
    plt.close()
    results['spearman'] = str(spearman_path)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    for key, path in results.items():
        print(f"  {key}: {path}")

    return results

