import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from typing import Dict, List


def plot_rss_metric(data: dict, metric_name: str, output_path: str, 
                     model_name: str = None, config: dict = None):
    """
    Generate boxplot for RSS metric data across sequence lengths.
    
    Args:
        data: JSON data dictionary with length_scores
        metric_name: Metric key to extract (e.g., 'rss_scores')
        output_path: Path to save the plot
        model_name: Optional model name for title and watermark
    """
    length_scores = data.get('length_scores', {})
    plot_data = []
    labels = []
    
    # Sort lengths numerically
    sorted_lengths = sorted([int(k) for k in length_scores.keys()])
    
    for length in sorted_lengths:
        scores = length_scores.get(str(length), {}).get(metric_name, [])
        if scores:
            plot_data.append(scores)
            labels.append(str(length))
    
    if not plot_data:
        print(f"No valid data for metric '{metric_name}' to plot.")
        return
    
    plt.figure(figsize=(12, 8))
    plt.boxplot(plot_data, labels=labels)
    
    # Build title
    title = f'RSS Metric - {metric_name} per Sequence Length'
    if model_name:
        title = f'{model_name} - {title}'
    # plt.title(title, fontsize=14, fontweight='bold')
    
    # Configure axes
    if 'latency' in metric_name.lower():
        plt.ylabel(f'{metric_name} (ms, Lower is better)', fontsize=12)
        plt.yscale('log')
        plt.ylim((0.1, 5000))
        plt.yticks([1, 10, 100, 1000])
    else:
        plt.ylabel(f'{metric_name} (Higher is better)', fontsize=12)
        plt.ylim((-50, 550))
    
    plt.xlabel('Sequence Length (tokens)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add watermark (use alias if available)
    if model_name:
        aliases = config.get('aliases', {}) if config else {}
        watermark_text = aliases.get(model_name, model_name)
        plt.text(0.98, 0.88, watermark_text, transform=plt.gca().transAxes,
                 fontsize=24, fontweight='bold', ha='right', va='bottom',
                 alpha=0.5, bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"RSS plot saved: {output_path}")


def plot_srs_heatmap(data: dict, output_path: str, model_name: str = None, 
                     p_thresh: float = None, config: dict = None):
    """
    Generate heatmap for SRS mean scores with statistical significance overlay.
    
    Args:
        data: JSON data dictionary with length_scores
        output_path: Path to save the heatmap
        model_name: Optional model name for title and watermark
        p_thresh: P-value threshold for significance (cells with p >= p_thresh get hatching)
    """
    length_scores = data.get('length_scores', {})
    
    # Use first available length
    sorted_keys = sorted([int(k) for k in length_scores.keys()])
    if not sorted_keys:
        print("No SRS length data found.")
        return
    
    target_length = sorted_keys[0]
    sample_results = length_scores.get(str(target_length), {}).get('sample_results', [])
    
    if not sample_results:
        print(f"No sample results for length {target_length}.")
        return
    
    # Collect all probe lengths and max positions
    all_probe_lens = set()
    max_pos_map = {}
    
    for sample in sample_results:
        for probe_len_str in sample.keys():
            try:
                probe_len = int(probe_len_str)
                all_probe_lens.add(probe_len)
                max_pos_map.setdefault(probe_len, 0)
                max_pos_map[probe_len] = max(max_pos_map[probe_len], 
                                              len(sample[probe_len_str]))
            except (ValueError, TypeError):
                continue
    
    sorted_probe_lens = sorted(list(all_probe_lens))
    if not sorted_probe_lens:
        print("No valid probe data found.")
        return
    
    max_pos = max(max_pos_map.values()) if max_pos_map else 0
    if max_pos == 0:
        print("No valid positions found.")
        return
    
    # Collect scores into cell lists
    score_lists = [[[] for _ in range(max_pos)] for _ in range(len(sorted_probe_lens))]
    
    for sample in sample_results:
        for r_idx, probe_len in enumerate(sorted_probe_lens):
            scores = sample.get(str(probe_len), [])
            for c_idx, score in enumerate(scores):
                if c_idx < max_pos:
                    score_lists[r_idx][c_idx].append(score)
    
    # Calculate mean for each cell
    result_matrix = np.zeros((len(sorted_probe_lens), max_pos))
    p_value_matrix = np.zeros((len(sorted_probe_lens), max_pos))
    
    for r_idx in range(len(sorted_probe_lens)):
        for c_idx in range(max_pos):
            cell_scores = score_lists[r_idx][c_idx]
            if cell_scores:
                result_matrix[r_idx, c_idx] = np.mean(cell_scores)
                # One-sample t-test against 0
                if len(cell_scores) > 1:
                    _, p_val = stats.ttest_1samp(cell_scores, 0)
                    p_value_matrix[r_idx, c_idx] = p_val
                else:
                    p_value_matrix[r_idx, c_idx] = 1.0
            else:
                result_matrix[r_idx, c_idx] = np.nan
                p_value_matrix[r_idx, c_idx] = 1.0
    
    # Get p_thresh from config if not provided as argument
    if p_thresh is None and config:
        p_thresh = config.get('p_thresh', 0.05)
    elif p_thresh is None:
        p_thresh = 0.05
    
    # Create heatmap
    plt.figure(figsize=(14, 10))
    
    title = f'SRS Mean Heatmap (Target Length: {target_length})'
    if model_name:
        title = f'{model_name} - {title}'
    
    limit = 100
    ax = sns.heatmap(result_matrix, annot=True, fmt=".2f", cmap='RdBu',
                     vmin=-limit, vmax=limit, 
                     xticklabels=range(max_pos),
                     yticklabels=sorted_probe_lens,
                     annot_kws={'size': 12},
                     cbar_kws={'label': 'Mean Score'})
    
    # plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Probe Position Index', fontsize=12)
    plt.ylabel('Probe Length', fontsize=12)
    
    # Add hatching and text overlay for non-significant cells (only if p_thresh is set)
    if p_thresh is not None:
        for r_idx in range(len(sorted_probe_lens)):
            for c_idx in range(max_pos):
                # Only hatch if statistically non-significant AND within valid triangle region
                if (p_value_matrix[r_idx, c_idx] >= p_thresh and 
                    target_length - sorted_probe_lens[r_idx] + 1 > c_idx):
                    ax.add_patch(plt.Rectangle((c_idx, r_idx), 1, 1, 
                                               fill=False, hatch='//', 
                                               edgecolor='black', linewidth=0))
                    
                    # Add text on top of hatching with white background for contrast
                    cell_val = result_matrix[r_idx, c_idx]
                    if not np.isnan(cell_val):
                        ax.text(c_idx + 0.5, r_idx + 0.5, f"{cell_val:.2f}",
                                ha='center', va='center', fontsize=12,
                                color='black',
                                bbox=dict(boxstyle='round,pad=0.15', 
                                          facecolor='white', edgecolor='none', alpha=0.9),
                                zorder=10)
    
    # Add watermark (use alias if available)
    if model_name:
        aliases = config.get('aliases', {}) if config else {}
        watermark_text = aliases.get(model_name, model_name)
        plt.text(0.98, 0.02, watermark_text, transform=plt.gca().transAxes,
                 fontsize=24, fontweight='bold', ha='right', va='bottom',
                 alpha=0.5, bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor='white', alpha=0.8, edgecolor='none'))
    
    plt.tight_layout()
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"SRS heatmap saved: {output_path}")


def generate_srs_rss_figures(data_dir: str, output_dir: str, config: dict = None) -> Dict[str, List[str]]:
    """
    Generate all SRS heatmaps and RSS boxplots from benchmark data.
    
    Args:
        data_dir: Directory to recursively search for benchmark_results.json files
        output_dir: Directory to save generated figures
    
    Returns:
        Dictionary with 'rss' and 'srs' keys containing lists of generated file paths
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")
    
    # Create organized output directories
    rss_dir = output_path / 'rss'
    srs_dir = output_path / 'srs'
    rss_dir.mkdir(parents=True, exist_ok=True)
    srs_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all benchmark results
    json_files = list(data_path.rglob('benchmark_results.json'))
    
    if not json_files:
        print(f"No benchmark_results.json files found in '{data_dir}'.")
        return {'rss': [], 'srs': []}
    
    print(f"\n{'='*60}")
    print(f"FIGURE GENERATION: SRS Heatmaps & RSS Boxplots")
    print(f"{'='*60}")
    print(f"Found {len(json_files)} benchmark file(s)")
    print(f"Output directory: {output_dir}")
    print()
    
    generated_files = {'rss': [], 'srs': []}
    
    for json_file in json_files:
        model_name = json_file.parent.parent.name
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading '{json_file}': {e}")
            continue
        
        # Determine benchmark type and generate appropriate figure
        if 'average_rss' in data:
            # RSS data - generate boxplot
            print(f"  → RSS: {model_name}")
            
            # Check available metrics
            first_length = next(iter(data.get('length_scores', {}).keys()), None)
            if first_length:
                length_data = data['length_scores'][first_length]
                
                if 'rss_scores' in length_data:
                    output_file = rss_dir / f"{model_name}_rss_boxplot.svg"
                    plot_rss_metric(data, 'rss_scores', str(output_file), model_name, config)
                    generated_files['rss'].append(str(output_file))
                
        elif 'average_srs' in data:
            # SRS data - generate heatmap
            print(f"  → SRS: {model_name}")
            output_file = srs_dir / f"{model_name}_srs_heatmap.svg"
            plot_srs_heatmap(data, str(output_file), model_name, config=config)
            generated_files['srs'].append(str(output_file))
        
        else:
            print(f"  ⊘ Unknown type: {model_name}")
    
    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"RSS figures: {len(generated_files['rss'])}")
    print(f"SRS figures: {len(generated_files['srs'])}")
    print(f"Total: {len(generated_files['rss']) + len(generated_files['srs'])}")
    
    return generated_files