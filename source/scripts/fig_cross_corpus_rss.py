import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, List, Tuple


def gather_rss_data(directory: str) -> Dict[str, dict]:
    """Parse a results directory and return a dict of {model_name: rss_json_data}.
    
    Traverses the directory to find benchmark_results.json files for RSS benchmarks.
    
    Args:
        directory: Root directory to search for RSS results
        
    Returns:
        Dictionary mapping model names to their RSS data
    """
    print(f"Gathering RSS data from '{directory}'...")
    
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")
    
    # Find all RSS result files
    rss_files = list(dir_path.rglob("benchmark_results.json"))
    
    rss_data = {}
    
    for json_file in rss_files:
        # Extract model name from parent directory structure
        model_name = json_file.parent.parent.name
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this is RSS data
            if 'average_rss' in data:
                rss_data[model_name] = data
                print(f"  ✓ RSS: {model_name}")
                
        except Exception as e:
            print(f"  ✗ Error loading '{json_file.name}': {e}")
    
    print(f"  Found RSS data for {len(rss_data)} models")
    return rss_data


def extract_rss_l15_mean(rss_data: dict) -> float:
    """Extract the L=15 mean RSS score from RSS data.
    
    Args:
        rss_data: RSS JSON data dictionary
        
    Returns:
        Mean RSS score at L=15, or None if not found
    """
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


def _format_pvalue(p: float) -> str:
    """Format p-value in LaTeX scientific notation."""
    if p == 0:
        return r"$< 10^{-10}$"
    s = f"{p:.1e}"
    mantissa, exponent = s.split('e')
    return f"${mantissa} \\times 10^{{{int(exponent)}}}$"


def generate_cross_corpus_figure(config: dict, output_dir: str) -> Tuple[str, str]:
    """
    Analyze cross-corpus RSS correlation at L=15 and generate scatter plot.
    
    Args:
        output_dir: Directory to save output files
        config: Configuration dictionary with keys:
            - corpus_a_dir: Directory containing RSS results for Corpus A
            - corpus_b_dir: Directory containing RSS results for Corpus B
            - corpus_a_name: Display name for Corpus A
            - corpus_b_name: Display name for Corpus B
    
    Returns:
        Tuple of (path_to_svg, path_to_tsv)
    """
    # Extract config parameters
    corpus_a_dir = config.get("corpus_a_dir")
    corpus_b_dir = config.get("corpus_b_dir")
    corpus_a_name = config.get("corpus_a_name", "Corpus A")
    corpus_b_name = config.get("corpus_b_name", "Corpus B")
    
    # Validate required parameters
    if not all([corpus_a_dir, corpus_b_dir]):
        raise ValueError("Missing required config: corpus_a_dir, corpus_b_dir")
    
    print(f"\n{'='*60}")
    print("Cross-Corpus RSS Correlation Analysis (L=15)")
    print(f"{'='*60}")
    print(f"Corpus A: {corpus_a_name} ({corpus_a_dir})")
    print(f"Corpus B: {corpus_b_name} ({corpus_b_dir})")
    
    # Gather data from both corpora
    corpus_a_data = gather_rss_data(corpus_a_dir)
    corpus_b_data = gather_rss_data(corpus_b_dir)
    
    if not corpus_a_data or not corpus_b_data:
        raise ValueError("Could not load RSS data from one or both corpora.")
    
    # Find common models
    common_models = set(corpus_a_data.keys()) & set(corpus_b_data.keys())
    
    # Exclude models specified in config
    exclude_models = config.get('exclude_models', [])
    if exclude_models:
        print(f"  Excluding {len(exclude_models)} models: {exclude_models}")
        common_models = common_models - set(exclude_models)
    
    if not common_models:
        raise ValueError(f"No common models found between the two corpora.")
    
    print(f"\nFound {len(common_models)} common models")
    
    # Extract L=15 mean RSS scores
    scores_a = []
    scores_b = []
    model_names = []
    
    for model in sorted(common_models):
        score_a = extract_rss_l15_mean(corpus_a_data[model])
        score_b = extract_rss_l15_mean(corpus_b_data[model])
        
        if score_a is not None and score_b is not None:
            scores_a.append(score_a)
            scores_b.append(score_b)
            model_names.append(model)
            print(f"  ✓ {model}: {corpus_a_name}={score_a:.2f}, {corpus_b_name}={score_b:.2f}")
        else:
            print(f"  ✗ {model}: Missing L=15 data")
    
    if len(scores_a) < 2:
        raise ValueError("Insufficient data points for correlation analysis.")
    
    # Calculate correlations
    pearson_r, pearson_p = scipy_stats.pearsonr(scores_a, scores_b)
    spearman_r, spearman_p = scipy_stats.spearmanr(scores_a, scores_b)
    
    print(f"\n{'='*60}")
    print("Correlation Results:")
    print(f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.6f})")
    print(f"  Spearman ρ = {spearman_r:.4f} (p = {spearman_p:.6f})")
    print(f"{'='*60}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filenames
    # Clean corpus names for filename (replace spaces/special chars)
    safe_a = "".join(c for c in corpus_a_name if c.isalnum()).lower()
    safe_b = "".join(c for c in corpus_b_name if c.isalnum()).lower()
    
    svg_filename = f"cross_corpus_{safe_a}_vs_{safe_b}_scatter.svg"
    tsv_filename = f"cross_corpus_{safe_a}_vs_{safe_b}_data.tsv"
    
    svg_path = os.path.join(output_dir, svg_filename)
    tsv_path = os.path.join(output_dir, tsv_filename)
    
    # Create scatter plot
    plt.figure(figsize=(12, 9))
    
    # Plot scatter points
    sns.scatterplot(x=scores_a, y=scores_b, s=80, alpha=0.7, c='dimgrey',
                    edgecolor='black', linewidth=0.5)

    # Add trend line
    z = np.polyfit(scores_a, scores_b, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(scores_a), max(scores_a), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.5, label='Trend line')
    
    # Add diagonal reference line
    min_val = min(min(scores_a), min(scores_b))
    max_val = max(max(scores_a), max(scores_b))
    plt.plot([min_val, max_val], [min_val, max_val], 'k:', alpha=0.3, 
             label='Perfect correlation (y=x)')
    
    # Labels and title
    plt.xlabel(f'{corpus_a_name} RSS (L=15)', fontsize=12, fontweight='bold')
    plt.ylabel(f'{corpus_b_name} RSS (L=15)', fontsize=12, fontweight='bold')
    
    # Annotate with correlation coefficients (p-values in scientific notation)
    textstr = f'Pearson r = {pearson_r:.3f} (p = {_format_pvalue(pearson_p)})\nSpearman ρ = {spearman_r:.3f} (p = {_format_pvalue(spearman_p)})'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right')
    plt.tight_layout()
    
    # Save plot as SVG
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nScatter plot saved to: {svg_path}")
    
    # Save data as TSV
    with open(tsv_path, 'w') as f:
        f.write(f"model	{corpus_a_name}_rss_l15	{corpus_b_name}_rss_l15\n")
        for model, score_a, score_b in zip(model_names, scores_a, scores_b):
            f.write(f"{model}\t{score_a:.6f}\t{score_b:.6f}\n")
    
    print(f"Correlation data saved to: {tsv_path}")
    
    return svg_path, tsv_path
