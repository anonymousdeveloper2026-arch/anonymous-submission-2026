import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List


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


def extract_rss_at_l(rss_data: dict, length: int) -> float:
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


def extract_lemb_avg(lemb_data: dict) -> float:
    """Calculate average of all LEMB task scores."""
    if not lemb_data:
        return None

    scores = []
    for short_key, full_key in LEMB_TASK_MAP.items():
        if full_key in lemb_data:
            task_data = lemb_data[full_key]
            if isinstance(task_data, dict):
                score = task_data.get('avg') if short_key in [
                    'needle', 'passkey'] else task_data.get('ndcg@10')
            else:
                score = task_data
            if score is not None:
                scores.append(float(score))

    return np.mean(scores) if scores else None


def perform_jackknife_analysis(model_data: Dict[str, Dict], length: int,
                               exclude_models: List[str] = None) -> pd.DataFrame:
    """
    Perform jackknife (leave-one-out) correlation analysis for a specific length.

    Args:
        model_data: Dictionary of model benchmark data
        length: Sequence length L to analyze
        exclude_models: List of models to permanently exclude from analysis

    Returns:
        DataFrame with jackknife results
    """
    # Filter excluded models
    if exclude_models:
        model_data = {k: v for k, v in model_data.items()
                      if k not in exclude_models}

    # Collect baseline data (all models)
    model_names = []
    rss_values = []
    lemb_values = []

    for model_name, metrics in model_data.items():
        rss_score = extract_rss_at_l(metrics['rss'], length)
        lemb_score = extract_lemb_avg(metrics['lemb'])

        if rss_score is not None and lemb_score is not None:
            model_names.append(model_name)
            rss_values.append(rss_score)
            lemb_values.append(lemb_score)

    if len(model_names) < 3:
        raise ValueError(
            f"Insufficient models for jackknife analysis at L={length}")
    # Calculate baseline (all models)
    baseline_pearson_r, baseline_pearson_p = scipy_stats.pearsonr(
        rss_values, lemb_values)
    baseline_spearman_r, baseline_spearman_p = scipy_stats.spearmanr(
        rss_values, lemb_values)

    results = []

    # Add FULL_SET row: represents analysis using ALL models (no exclusion)
    results.append({
        'length': length,
        'excluded_model': 'FULL_SET',
        'pearson_r': baseline_pearson_r,
        'pearson_p': baseline_pearson_p,
        'spearman_r': baseline_spearman_r,
        'spearman_p': baseline_spearman_p,
        'n_models': len(model_names),
    })
    # Jackknife iterations (leave-one-out)
    for i, excluded_model in enumerate(model_names):
        # Create dataset without this model
        jack_rss = [rss_values[j] for j in range(len(model_names)) if j != i]
        jack_lemb = [lemb_values[j] for j in range(len(model_names)) if j != i]

        # Calculate correlations
        pearson_r, pearson_p = scipy_stats.pearsonr(jack_rss, jack_lemb)
        spearman_r, spearman_p = scipy_stats.spearmanr(jack_rss, jack_lemb)

        results.append({
            'length': length,
            'excluded_model': excluded_model,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_models': len(jack_rss),
        })

    return pd.DataFrame(results)


def plot_correlation_stability_grid(results_dict: Dict[int, pd.DataFrame], output_path: str):
    """Create grid of boxplots showing correlation stability across lengths."""
    lengths = sorted(results_dict.keys())
    n_lengths = len(lengths)

    # Calculate grid dimensions
    n_cols = min(4, n_lengths)
    n_rows = (n_lengths + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_lengths == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes.flatten()

    for idx, length in enumerate(lengths):
        ax = axes[idx] if n_lengths > 1 else axes[0]
        df = results_dict[length]

        # Filter out FULL_SET for boxplot (show only jackknife leave-one-out results)
        jack_df = df[df['excluded_model'] != 'FULL_SET']

        # Prepare data for boxplot
        data_to_plot = [jack_df['pearson_r'].values,
                        jack_df['spearman_r'].values]

        # Create boxplot
        bp = ax.boxplot(data_to_plot, labels=[
                        'Pearson r', 'Spearman ρ'], patch_artist=True)

        # Color boxes
        colors = ['skyblue', 'lightgreen']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add FULL_SET reference lines using segmented ax.plot
        baseline = df[df['excluded_model'] == 'FULL_SET'].iloc[0]
        # Pearson: from x=0.5 to x=1.5
        ax.plot([0.5, 1.5], [baseline['pearson_r'], baseline['pearson_r']],
                color='red', linestyle='--', linewidth=1.5,
                label=f'not-ablated r={baseline["pearson_r"]:.3f}')
        # Spearman: from x=1.5 to x=2.5
        ax.plot([1.5, 2.5], [baseline['spearman_r'], baseline['spearman_r']],
                color='darkred', linestyle=':', linewidth=1.5,
                label=f'not-ablated ρ={baseline["spearman_r"]:.3f}')

        ax.set_title(f'L={length}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Correlation', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0.3, 0.8])
        ax.legend(fontsize=8, loc='lower right')

    # Hide unused subplots
    for idx in range(n_lengths, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].set_visible(False)

    plt.tight_layout()

    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved correlation stability plot: {output_path}")


def plot_pvalue_significance_grid(results_dict: Dict[int, pd.DataFrame], output_path: str):
    """Create grid of strip plots showing p-value significance across lengths."""
    lengths = sorted(results_dict.keys())
    n_lengths = len(lengths)

    # Calculate grid dimensions
    n_cols = min(4, n_lengths)
    n_rows = (n_lengths + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_lengths == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes.flatten()

    for idx, length in enumerate(lengths):
        ax = axes[idx] if n_lengths > 1 else axes[0]
        df = results_dict[length]

        # Separate jackknife data and FULL_SET reference
        jack_df = df[df['excluded_model'] != 'FULL_SET']
        full_set_row = df[df['excluded_model'] == 'FULL_SET'].iloc[0]

        # Prepare data
        pearson_data = jack_df['pearson_p'].values
        spearman_data = jack_df['spearman_p'].values

        positions = [1, 2]
        data_list = [pearson_data, spearman_data]
        labels = ['Pearson p', 'Spearman p']

        # Create strip plots with jitter
        for pos, data, label in zip(positions, data_list, labels):
            jittered_x = np.random.normal(pos, 0.04, size=len(data))
            ax.scatter(jittered_x, data, alpha=0.6, s=30, label=label)

        # Plot FULL_SET as a prominent marker on top
        ax.scatter(1, full_set_row['pearson_p'], s=60, c='red', edgecolor='black',
                   linewidth=2, zorder=10, label='Full Set (baseline)')
        ax.scatter(2, full_set_row['spearman_p'], s=60, c='red', edgecolor='black',
                   linewidth=2, zorder=10)

        # Add significance threshold
        ax.axhline(0.05, color='red', linestyle='--',
                   linewidth=2, label='p=0.05', alpha=0.8)

        # Fill significant region
        ax.fill_between([0.5, 2.5], 0, 0.05, alpha=0.1, color='green')

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(f'L={length}', fontsize=10, fontweight='bold')
        ax.set_ylabel('p-value', fontsize=9)
        ax.set_ylim([0, 0.075])
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_lengths, len(axes) if isinstance(axes, np.ndarray) else 1):
        if isinstance(axes, np.ndarray):
            axes[idx].set_visible(False)

    plt.tight_layout()

    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved p-value significance plot: {output_path}")


def generate_rss_jackknife_figures(data_dir: str, output_dir: str, config: dict) -> Dict[str, str]:
    """
    Master orchestrator for jackknife analysis figure generation.

    Args:
        data_dir: Directory containing benchmark data
        output_dir: Directory to save output files
        config: Configuration dictionary with keys:
            - target_lengths: List of sequence lengths to analyze (e.g., [4, 5, ..., 16])
            - exclude_models: List of models to permanently exclude from analysis

    Returns:
        Dictionary with paths to generated files
    """
    print(f"=" * 70)
    print("JACKKNIFE CORRELATION STABILITY ANALYSIS")
    print(f"=" * 70)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Fix random seed
    np.random.seed(42)

    # Extract config
    target_lengths = config.get('target_lengths', list(range(4, 17)))
    exclude_models = config.get('exclude_models', [])

    print(f"Target lengths: {target_lengths}")
    if exclude_models:
        print(f"Excluded models: {exclude_models}")

    # Gather data
    model_data = gather_benchmark_data(data_dir)
    print(f"\nLoaded data for {len(model_data)} models")

    # Perform jackknife analysis for each length
    results_dict = {}
    all_results = []

    for length in target_lengths:
        print(f"\nAnalyzing L={length}...")
        try:
            df = perform_jackknife_analysis(model_data, length, exclude_models)
            results_dict[length] = df
            all_results.append(df)
            print(f"  Completed: {len(df)-1} jackknife iterations")
        except Exception as e:
            print(f"  Error: {e}")

    if not results_dict:
        raise ValueError("No successful jackknife analyses completed")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save combined results to TSV
    combined_df = pd.concat(all_results, ignore_index=True)
    tsv_path = output_path / 'rss_lemb_jackknife_results.tsv'
    combined_df.to_csv(tsv_path, sep='\t', index=False, float_format='%.6f')
    print(f"\nSaved combined results: {tsv_path}")

    # Generate correlation stability figure
    print(f"\nGenerating correlation stability figure...")
    corr_svg_path = output_path / 'rss_lemb_jackknife_correlation.svg'
    plot_correlation_stability_grid(results_dict, str(corr_svg_path))

    # Generate p-value significance figure
    print(f"\nGenerating p-value significance figure...")
    pval_svg_path = output_path / 'rss_lemb_jackknife_pvalue.svg'
    plot_pvalue_significance_grid(results_dict, str(pval_svg_path))

    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")

    return {
        'results_tsv': str(tsv_path),
        'correlation_svg': str(corr_svg_path),
        'pvalue_svg': str(pval_svg_path),
    }

