import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats as scipy_stats
import numpy as np


def extract_srs_negative_only_score(data: dict) -> Optional[float]:
    """Calculate SRS negative-only score (mean of negative sample similarities)."""
    if not data:
        return None

    length_scores = data.get('length_scores', {})
    negative_only_scores = []
    sample_count = 0

    for length_data in length_scores.values():
        sample_results = length_data.get('sample_results', [])
        for sample in sample_results:
            for probe_len, scores in sample.items():
                if scores:
                    neg_only = [s for s in scores if s < 0]
                    if neg_only:
                        negative_only_scores.append(np.mean(neg_only))
                    sample_count += 1

    if not negative_only_scores:
        return None

    return abs(np.sum(negative_only_scores) / sample_count)


def gather_rss_lemb_srs_data(directory: str) -> Dict[str, Dict]:
    """
    Layer 1: Data Loading.
    Traverse directory and extract RSS, LEMB, and SRS metrics for each model.
    """
    print(f"\n{'='*60}")
    print(f"Gathering RSS, LEMB, and SRS data from '{directory}'...")
    print(f"{'='*60}")

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return {}

    # Find all JSON files recursively
    json_files = list(dir_path.rglob("overall_results.json")) + \
        list(dir_path.rglob("benchmark_results.json"))

    if not json_files:
        print(f"No JSON files found.")
        return {}

    print(f"Found {len(json_files)} JSON file(s).")

    all_data = {}

    for json_file in json_files:

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load '{json_file.name}': {e}")
            continue

        # Detect and extract RSS data
        if 'average_rss' in data:
            model_name = json_file.parent.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'rss': None, 'lemb': None, 'srs': None}
            all_data[model_name]['rss'] = data
            print(f"  ✓ RSS: {json_file.name} -> {model_name}")

        # Detect and extract SRS data
        elif 'average_srs' in data:
            model_name = json_file.parent.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'rss': None, 'lemb': None, 'srs': None}
            all_data[model_name]['srs'] = data
            print(f"  ✓ SRS: {json_file.name} -> {model_name}")

        # Detect and extract LEMB data
        elif any(k.startswith('LEMB') for k in data.keys()):
            model_name = json_file.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'rss': None, 'lemb': None, 'srs': None}
            all_data[model_name]['lemb'] = data
            print(f"  ✓ LEMB: {json_file.name} -> {model_name}")

    # Filter to only models with BOTH rss and lemb data
    # SRS is optional but stored if available
    complete_data = {}
    for model_name, metrics in all_data.items():
        if metrics['rss'] is not None and metrics['lemb'] is not None:
            complete_data[model_name] = metrics
            has_srs = " + SRS" if metrics['srs'] is not None else ""
            print(f"  ✓ Complete data for: {model_name}{has_srs}")
        else:
            missing = []
            if metrics['rss'] is None:
                missing.append('RSS')
            if metrics['lemb'] is None:
                missing.append('LEMB')
            print(f"  ✗ Skipping {model_name} (missing: {', '.join(missing)})")

    print(
        f"\nSuccessfully loaded data for {len(complete_data)} complete model(s).")
    srs_count = sum(1 for m in complete_data.values() if m['srs'] is not None)
    print(f"  ({srs_count} models also have SRS data)")
    return complete_data


def extract_rss_mean(data: Dict, length: int) -> Optional[float]:
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

    return sum(rss_scores) / len(rss_scores)


def extract_lemb_avg(data: Dict) -> Optional[float]:
    """Calculate average of all LEMB task scores."""
    if not data:
        return None

    lemb_keys = [
        'LEMBNeedleRetrieval',
        'LEMBPasskeyRetrieval',
        'LEMBSummScreenFDRetrieval',
        'LEMBQMSumRetrieval',
        'LEMBWikimQARetrieval',
        'LEMBNarrativeQARetrieval'
    ]

    scores = []
    for key in lemb_keys:
        if key in data:
            val = data[key]
            if isinstance(val, dict):
                score = val.get('avg') or val.get('ndcg@10')
            else:
                score = val
            if score is not None:
                scores.append(float(score))

    return sum(scores) / len(scores) if scores else None


def _process_data_into_ir(all_data: Dict[str, Dict], config: Dict) -> List[Dict]:
    """
    Layer 2: Processing & Intermediate Representation.
    Compute correlations between RSS(L=x) and LEMB avg for x in config['l_range'].
    Optionally compute SRS(neg) vs LEMB correlation if 'include_srs_neg_correlation' is True.
    """
    ir_rows = []
    
    # Filter out excluded models
    exclude_models = config.get('exclude_models', [])
    filtered_data = {k: v for k, v in all_data.items() if k not in exclude_models}
    
    if not filtered_data:
        print("Warning: No models remaining after exclusion filter.")
        return []
    
    print(f"\nProcessing {len(filtered_data)} models (excluded: {len(exclude_models)})")
    
    # Get L-range from config
    l_range = config.get('l_range', range(4, 17))
    
    # Check if SRS negative correlation is requested
    include_srs_neg = config.get('include_srs_neg_correlation', False)
    
    # Loop through specified lengths
    for length in l_range:
        # Collect paired data points
        rss_values = []
        lemb_values = []

        for model_name, metrics in filtered_data.items():
            rss_val = extract_rss_mean(metrics['rss'], length)
            lemb_val = extract_lemb_avg(metrics['lemb'])

            if rss_val is not None and lemb_val is not None:
                rss_values.append(rss_val)
                lemb_values.append(lemb_val)

        if len(rss_values) < 2:
            print(
                f"Warning: Insufficient data for L={length} (n={len(rss_values)})")
            continue

        # Compute correlations
        pearson_r, pearson_p = scipy_stats.pearsonr(rss_values, lemb_values)
        spearman_r, spearman_p = scipy_stats.spearmanr(rss_values, lemb_values)

        ir_row = {
            'length': length,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'n_samples': len(rss_values)
        }
        ir_rows.append(ir_row)

        print(
            f"  L={length:2d}: n={len(rss_values):2d}, r={pearson_r:.4f}, ρ={spearman_r:.4f}")

    # Compute SRS negative-only correlation with LEMB if requested
    if include_srs_neg:
        srs_neg_values = []
        lemb_values_srs = []
        
        for model_name, metrics in filtered_data.items():
            srs_neg_val = extract_srs_negative_only_score(metrics['srs'])
            lemb_val = extract_lemb_avg(metrics['lemb'])
            
            if srs_neg_val is not None and lemb_val is not None:
                srs_neg_values.append(srs_neg_val)
                lemb_values_srs.append(lemb_val)
        
        if len(srs_neg_values) >= 2:
            pearson_r_srs, pearson_p_srs = scipy_stats.pearsonr(srs_neg_values, lemb_values_srs)
            spearman_r_srs, spearman_p_srs = scipy_stats.spearmanr(srs_neg_values, lemb_values_srs)
            
            srs_row = {
                'length': 'SRS(neg)',
                'pearson_r': pearson_r_srs,
                'pearson_p': pearson_p_srs,
                'spearman_r': spearman_r_srs,
                'spearman_p': spearman_p_srs,
                'n_samples': len(srs_neg_values),
                'is_srs_neg': True
            }
            ir_rows.append(srs_row)
            print(f"  SRS(neg): n={len(srs_neg_values):2d}, r={pearson_r_srs:.4f}, ρ={spearman_r_srs:.4f}")
        else:
            print(f"  SRS(neg): Insufficient data (n={len(srs_neg_values)})")

    # Identify best Pearson correlation (highest absolute r) among RSS rows
    rss_rows = [r for r in ir_rows if not r.get('is_srs_neg', False)]
    if rss_rows:
        best_idx = max(range(len(rss_rows)),
                       key=lambda i: abs(rss_rows[i]['pearson_r']))
        rss_rows[best_idx]['is_best'] = True
        print(
            f"\n  ★ Best RSS correlation at L={rss_rows[best_idx]['length']} (r={rss_rows[best_idx]['pearson_r']:.4f})")

    return ir_rows


def _render_markdown(ir_rows: List[Dict]) -> str:
    """
    Layer 3a: Markdown Renderer.
    """
    lines = [
        "| Metric | r_pearson | p_pearson | ρ_spearman | p_spearman |",
        "|--------|-----------|-----------|------------|------------|"
    ]

    for row in ir_rows:
        length = row['length']
        pearson_r = row['pearson_r']
        pearson_p = row['pearson_p']
        spearman_r = row['spearman_r']
        spearman_p = row['spearman_p']
        is_best = row.get('is_best', False)
        is_srs_neg = row.get('is_srs_neg', False)

        # Format p-values
        pearson_p_str = f"{pearson_p:.4f}" if pearson_p >= 0.0001 else "< 0.0001"
        spearman_p_str = f"{spearman_p:.4f}" if spearman_p >= 0.0001 else "< 0.0001"

        # Bold best values
        r_str = f"**{pearson_r:.4f}**" if is_best else f"{pearson_r:.4f}"
        rho_str = f"**{spearman_r:.4f}**" if is_best else f"{spearman_r:.4f}"

        # Format row label
        if is_srs_neg:
            label = "SRS(neg)"
        else:
            label = f"L={length}"

        lines.append(
            f"| {label} | {r_str} | {pearson_p_str} | {rho_str} | {spearman_p_str} |")

    return "\n".join(lines)


def _render_latex(ir_rows: List[Dict]) -> str:
    """
    Layer 3b: LaTeX Renderer.
    Produces the correlation table with proper formatting.
    """
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{Correlation analysis: $\text{RSS}(L=x)$ vs \textsc{LEMB} Avg.}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"$\text{RSS}(L=x)$ & $r_{pearson}$ & $p_{pearson}$ & $\rho_{spearman}$ & $p_{spearman}$ \\",
        r"\midrule"
    ]

    for row in ir_rows:
        length = row['length']
        pearson_r = row['pearson_r']
        pearson_p = row['pearson_p']
        spearman_r = row['spearman_r']
        spearman_p = row['spearman_p']
        is_best = row.get('is_best', False)
        is_srs_neg = row.get('is_srs_neg', False)

        # Format p-values
        pearson_p_str = f"{pearson_p:.4f}" if pearson_p >= 0.0001 else r"< 0.0001"
        spearman_p_str = f"{spearman_p:.4f}" if spearman_p >= 0.0001 else r"< 0.0001"

        # Bold best values
        r_str = f"\\textbf{{{pearson_r:.4f}}}" if is_best else f"{pearson_r:.4f}"
        rho_str = f"\\textbf{{{spearman_r:.4f}}}" if is_best else f"{spearman_r:.4f}"

        # Format row label
        if is_srs_neg:
            lines.append(r"\midrule")
            label = r"$\mathbf{srs}_{\text{neg}}$"
        else:
            label = f"$L={length}$"

        lines.append(
            f"{label} & {r_str} & {pearson_p_str} & {rho_str} & {spearman_p_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_table_3_rss_lemb_correl(directory: str, config: Optional[Dict] = None) -> Tuple[List[Dict], str, str]:
    """
    Public API: Orchestrates the 3-layer pipeline.
    
    Args:
        directory: Root directory containing benchmark JSON files
        config: Optional configuration dict with:
            - 'exclude_models': List of model names to exclude from analysis
            - 'l_range': Iterable of sequence lengths to analyze (default: range(4, 17))
    
    Returns:
        (raw_data, latex_table, markdown_table) tuple
    """
    # Set default config if not provided
    if config is None:
        config = {
            'exclude_models': [],
            'l_range': range(4, 17)
        }
    
    # Layer 1: Load data
    all_data = gather_rss_lemb_srs_data(directory)

    if not all_data:
        print("No data found.")
        return [], "", ""

    # Layer 2: Process into IR (with config)
    ir_rows = _process_data_into_ir(all_data, config)

    # Layer 3: Render
    md_table = _render_markdown(ir_rows)
    latex_table = _render_latex(ir_rows)

    return ir_rows, latex_table, md_table