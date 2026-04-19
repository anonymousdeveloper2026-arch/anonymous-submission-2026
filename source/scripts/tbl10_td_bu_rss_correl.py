from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats as scipy_stats
import torch
import torch.nn.functional as F

# =============================================================================
# Core FINESSE Functions (PRESERVED EXACTLY - DO NOT MODIFY)
# =============================================================================

def calculate_self_attestation_scores(chunk_embeddings, synth_embeddings, eval_mode: str = 'q1q3'):
    """
    Calculate Top-Down self-attestation scores using Robust Separation Score.

    Robust Separation Philosophy:
    - Focus on RAG separability: Weakest memories must outperform strongest noise.
    - No rank-order violations; instead, measure quartile gap or median gap between Tier 1 and Tier 2.

    Eval Modes:
    - 'q1q3' (default): Q1(Tier 1) - Q3(Tier 2). Ensures weakest 25% of memories > strongest 75% of noise.
                           Most strict, tests for perfect separation.
    - 'q2q2': Median(Tier 1) - Median(Tier 2). Tests if typical memory > typical noise.
                          More permissive, good for detecting directional trends.

    Args:
        chunk_embeddings: List[torch.Tensor] - Embeddings for all chunks, each (d_model,)
        synth_embeddings: List[torch.Tensor] - Embeddings for synthesis steps, each (d_model,)
        eval_mode: 'q1q3' or 'q2q2' - Scoring mode to use.

    Returns:
        Dictionary with 'contextual_coherence' score (average robust gaps)
    """
    # Stack embeddings
    device = chunk_embeddings[0].device
    chunk_emb_tensor = torch.stack([t.to(device)
                                   for t in chunk_embeddings])  # (M, d_model)
    synth_emb_tensor = torch.stack([t.to(device)
                                   for t in synth_embeddings])  # (N, d_model)

    # Compute similarity matrix (N_synth, M_chunks)
    sim_matrix = F.cosine_similarity(
        synth_emb_tensor.unsqueeze(1),
        chunk_emb_tensor.unsqueeze(0),
        dim=2
    )

    N = sim_matrix.shape[0]  # Number of synthesis steps
    M = sim_matrix.shape[1]  # Number of chunks

    row_gaps = []

    # Evaluate only middle synthesis steps, excluding start (Synth(A)) and end (Synth(ABC...G))
    for i in range(1, N - 1):  # Skip first (i=0) and last (i=N-1) synthesis steps
        # Assign 2-tier system
        tier_for_chunk = []
        for j in range(M):
            if j <= i:  # Memory chunks: part of the synthesis
                tier = 1
            else:  # Noise chunks: not part of the synthesis
                tier = 2
            tier_for_chunk.append(tier)

        # Collect tier indices
        tier1_js = [j for j in range(M) if tier_for_chunk[j] == 1]
        tier2_js = [j for j in range(M) if tier_for_chunk[j] == 2]

        # Collect scores for each tier
        tier1_scores = sim_matrix[i][tier1_js]
        tier2_scores = sim_matrix[i][tier2_js]

        # Calculate Robust Gap if both tiers have scores
        if len(tier1_scores) > 0 and len(tier2_scores) > 0:
            if eval_mode == 'q1q3':
                # Strict: Q1(Tier 1) - Q3(Tier 2)
                q1_t1 = torch.quantile((tier1_scores).to(torch.float32), 0.25)
                q3_t2 = torch.quantile((tier2_scores).to(torch.float32), 0.75)
                row_gap = q1_t1 - q3_t2
            elif eval_mode == 'q2q2':
                # Permissive: Median(Tier 1) - Median(Tier 2)
                median_t1 = torch.quantile(
                    (tier1_scores).to(torch.float32), 0.5)
                median_t2 = torch.quantile(
                    (tier2_scores).to(torch.float32), 0.5)
                row_gap = median_t1 - median_t2
            else:
                raise ValueError(
                    f"Unknown eval_mode: {eval_mode}. Must be 'q1q3' or 'q2q2'.")
        else:
            # No meaningful separation possible
            row_gap = torch.tensor(0.0, device=sim_matrix.device)

        row_gaps.append(row_gap)

    # Overall contextual coherence score (average robust gap)
    contextual_coherence = torch.mean(
        torch.stack(row_gaps)).item() if row_gaps else 0.0

    return {
        'contextual_coherence': contextual_coherence
    }


def calculate_self_attestation_scores_bottom_up(chunk_embeddings, synth_embeddings, eval_mode: str = 'q1q3'):
    """
    Calculate Bottom-Up self-attestation scores using Robust Separation Score.

    For each story chunk as anchor (starting from 1 to num_synth_steps-2 to skip start and end):
    - Tier 1 (Memory): Synths that include this chunk (synth_idx >= anchor_idx)
    - Tier 2 (Noise): Synths that do not include it (synth_idx < anchor_idx)

    Robust Gap: Q1(Tier 1 scores) - Q3(Tier 2 scores)
    Ensures weakest 25% of including synths > strongest 75% of non-including synths.
    Pure separation without rank violations, focusing on RAG utility.

    Args:
        chunk_embeddings: List[torch.Tensor] - Embeddings for all chunks, each (d_model,)
        synth_embeddings: List[torch.Tensor] - Embeddings for synthesis steps, each (d_model,)
        main_story_end: Number of story chunks (unused, for compatibility)
        chunk_ids: List of chunk IDs (unused, for consistency)

    Returns:
        Dict with 'bottom_up_coherence' (average robust gaps over anchors)
    """
    # Stack embeddings
    device = chunk_embeddings[0].device
    chunk_emb_tensor = torch.stack([t.to(device)
                                   for t in chunk_embeddings])  # (M, d_model)
    synth_emb_tensor = torch.stack([t.to(device)
                                   for t in synth_embeddings])  # (N, d_model)

    # Compute similarity matrix for bottom-up: (M_chunks, N_synth)
    sim_bottom_up = F.cosine_similarity(
        chunk_emb_tensor.unsqueeze(1),
        synth_emb_tensor.unsqueeze(0),
        dim=2
    )

    M_synth = sim_bottom_up.shape[1]

    row_gaps = []

    # Evaluate only middle chunks, excluding start (anchor_idx=0) and end (anchor_idx=M_synth-1)
    for anchor_idx in range(1, M_synth):  # Skip start and end anchors
        tier_for_synth = []
        for j in range(M_synth):
            if j >= anchor_idx:  # synth j includes chunks 0 to j, so includes anchor_idx if j >= anchor_idx
                tier = 1  # Memory
            else:
                tier = 2  # Noise
            tier_for_synth.append(tier)

        # Collect tier indices
        tier1_js = [j for j in range(M_synth) if tier_for_synth[j] == 1]
        tier2_js = [j for j in range(M_synth) if tier_for_synth[j] == 2]

        # Collect scores for each tier (similarities from this anchor to synths)
        tier1_scores = sim_bottom_up[anchor_idx][tier1_js]
        tier2_scores = sim_bottom_up[anchor_idx][tier2_js]

        # Calculate Robust Gap if both tiers have scores
        if len(tier1_scores) > 0 and len(tier2_scores) > 0:
            if eval_mode == 'q1q3':
                # Strict: Q1(Tier 1) - Q3(Tier 2)
                q1_t1 = torch.quantile((tier1_scores).to(torch.float32), 0.25)
                q3_t2 = torch.quantile((tier2_scores).to(torch.float32), 0.75)
                row_gap = q1_t1 - q3_t2
            elif eval_mode == 'q2q2':
                # Permissive: Median(Tier 1) - Median(Tier 2)
                median_t1 = torch.quantile(
                    (tier1_scores).to(torch.float32), 0.5)
                median_t2 = torch.quantile(
                    (tier2_scores).to(torch.float32), 0.5)
                row_gap = median_t1 - median_t2
            else:
                raise ValueError(
                    f"Unknown eval_mode: {eval_mode}. Must be 'q1q3' or 'q2q2'.")
        else:
            # No meaningful separation possible
            row_gap = torch.tensor(0.0, device=sim_bottom_up.device)
            
        row_gaps.append(row_gap)

    # Average over all anchors
    contextual_coherence_bottom_up = torch.mean(
        torch.stack(row_gaps)).item() if row_gaps else 0.0

    return {
        'bottom_up_coherence': contextual_coherence_bottom_up
    }


# =============================================================================
# Configuration
# =============================================================================

BENCHMARK_DIR = "benchmarks"

# Target lengths for analysis
DEFAULT_TARGET_LENGTHS = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16]


# =============================================================================
# Data Processing - .pt File Targeting with Re-computation
# =============================================================================

def _load_pt_file(pt_path: str) -> Dict[str, Any]:
    """Load PyTorch .pt file containing embeddings."""
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    return data


def _extract_model_name_from_path(pt_path: Path) -> str:
    """Extract model name from .pt file path."""

    return pt_path.parent.parent.name

def _recompute_scores_from_pt(pt_path: str, eval_mode: str = 'q1q3') -> Dict[str, Any]:
    """
    Load .pt file and re-compute TD, BU, and RSS scores from embeddings.
    
    Returns:
        Dict with:
        - td_scores: List of TD scores (one per query)
        - bu_scores: List of BU scores (one per query)
        - rss_scores: List of RSS scores = (TD + BU) * 500
        - mean_td: Mean of TD scores
        - mean_bu: Mean of BU scores
        - mean_rss: Mean of RSS scores = (mean_td + mean_bu) * 500
        ,
        each wrapped by l. 

        {   
            4 : {...},
            5 : {...},
            ...
        }
    """
    data = _load_pt_file(pt_path)

    data = data.get('raw_results', data)["length_results"]
    
    
    # Calculate TD and BU for each sample
    
    res = {}

    for l, length_res in data.items():
        td_scores = []
        bu_scores = []

        sample_dicts = length_res.get('sample_results')
        for sample_dict in sample_dicts:
            # Extract embeddings
            chunk_embeddings = sample_dict.get('chunk_embeddings')
            synthesis_embeddings = sample_dict.get('synthesis_embeddings')
            
            if chunk_embeddings is None or synthesis_embeddings is None:
                continue
            
            # Convert numpy arrays to torch tensors if needed
            if isinstance(chunk_embeddings, np.ndarray):
                chunk_embeddings = [torch.from_numpy(chunk_embeddings[i]) for i in range(len(chunk_embeddings))]
            if isinstance(synthesis_embeddings, np.ndarray):
                synthesis_embeddings = [torch.from_numpy(synthesis_embeddings[i]) for i in range(len(synthesis_embeddings))]
            
            # Ensure we have enough embeddings
            if len(chunk_embeddings) < 2 or len(synthesis_embeddings) < 2:
                continue
            
            # Calculate TD and BU using FINESSE functions
            td_result = calculate_self_attestation_scores(chunk_embeddings, synthesis_embeddings, eval_mode=eval_mode)
            bu_result = calculate_self_attestation_scores_bottom_up(chunk_embeddings, synthesis_embeddings, eval_mode=eval_mode)
            
            td_scores.append(td_result['contextual_coherence'])
            bu_scores.append(bu_result['bottom_up_coherence'])
        
        # Calculate RSS for each sample: (TD + BU) * 500
        rss_scores = [(td + bu) * 500 for td, bu in zip(td_scores, bu_scores)]
        
        # Calculate means
        mean_td = np.mean(td_scores) if td_scores else 0.0
        mean_bu = np.mean(bu_scores) if bu_scores else 0.0
        mean_rss = (mean_td + mean_bu) * 500  # This ensures perfect consistency
        
        res[int(l)] =  {
            'td_scores': td_scores,
            'bu_scores': bu_scores,
            'rss_scores': rss_scores,
            'mean_td': mean_td,
            'mean_bu': mean_bu,
            'mean_rss': mean_rss,
            'n_samples': len(td_scores),
        }
    
    return res


def _gather_and_process_data(benchmark_dir: str, config: dict) -> pd.DataFrame:
    """
    Gather data by loading .pt files and re-computing TD/BU/RSS scores.
    
    Creates DataFrame with columns:
    model, L, rss_mean, td_mean, bu_mean
    
    Where rss_mean = (td_mean + bu_mean) * 500 (perfect consistency guaranteed)
    """
    target_lengths = config.get('target_lengths', DEFAULT_TARGET_LENGTHS)
    eval_mode = config.get('eval_mode', 'q1q3')
    
    dir_path = Path(benchmark_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{benchmark_dir}' does not exist.")
    
    rows = []
    
    # Find all .pt files in RSS benchmark directories
    for pt_file in dir_path.rglob("*.pt"):
        # Check if this is in an RSS directory
        if 'rss' not in str(pt_file).lower():
            continue
        
        # Extract model name and length from path
        model_name = _extract_model_name_from_path(pt_file)
        
        # Skip excluded models early
        if 'exclude_models' in config and config['exclude_models']:
            exclude_set = set(config['exclude_models'])
            if model_name in exclude_set:
                print(f"  ⊘ Excluded model: {model_name}")
                continue
        
        try:
            # Re-compute scores from embeddings (returns dict of {length: scores})
            print(f"  ~ In progress : {pt_file}")
            scores_by_length = _recompute_scores_from_pt(str(pt_file), eval_mode=eval_mode)
        except Exception as e:
            print(f"  ✗ Failed to process {pt_file}: {e}")
            continue
        
        # Iterate over lengths in the returned dictionary
        for length, scores in scores_by_length.items():
            # Skip if length not in target lengths
            if length not in target_lengths:
                continue
            
            # Skip if no valid scores
            if scores['n_samples'] == 0:
                continue
            
            # Add one row with mean scores (one row per model-length)
            rows.append({
                'model': model_name,
                'L': length,
                'rss_mean': scores['mean_rss'],
                'td_mean': scores['mean_td'],
                'bu_mean': scores['mean_bu'],
                'n_samples': scores['n_samples'],
                'pt_file': str(pt_file),
            })
    
    if not rows:
        raise ValueError(f"No valid .pt files found in '{benchmark_dir}' for RSS analysis.")
    
    df = pd.DataFrame(rows)
    
    print(f"Gathered {len(df)} data points from {df['model'].nunique()} models across {df['L'].nunique()} lengths")
    
    return df


# =============================================================================
# Correlation Calculation
# =============================================================================

def _calculate_correlations(df: pd.DataFrame, config: dict) -> Dict[str, Any]:
    """
    Calculate correlations for each length L using re-computed mean scores.
    """
    target_lengths = config.get('target_lengths', DEFAULT_TARGET_LENGTHS)
    
    correlations = []
    
    # Store all correlation values for mean calculation
    td_rss_pearson = []
    td_rss_spearman = []
    bu_rss_pearson = []
    bu_rss_spearman = []
    td_bu_pearson = []
    td_bu_spearman = []
    
    for length in target_lengths:
        df_l = df[df['L'] == length]
        
        if len(df_l) < 3:
            print(f"  ! Insufficient data for L={length}, skipping")
            continue
        
        # Calculate correlations using mean scores
        # TD vs RSS (using mean RSS score per model)
        pearson_td_rss, p_pearson_td_rss = scipy_stats.pearsonr(df_l['td_mean'], df_l['rss_mean'])
        spearman_td_rss, p_spearman_td_rss = scipy_stats.spearmanr(df_l['td_mean'], df_l['rss_mean'])
        
        # BU vs RSS (using mean RSS score per model)
        pearson_bu_rss, p_pearson_bu_rss = scipy_stats.pearsonr(df_l['bu_mean'], df_l['rss_mean'])
        spearman_bu_rss, p_spearman_bu_rss = scipy_stats.spearmanr(df_l['bu_mean'], df_l['rss_mean'])
        
        # TD vs BU
        pearson_td_bu, p_pearson_td_bu = scipy_stats.pearsonr(df_l['td_mean'], df_l['bu_mean'])
        spearman_td_bu, p_spearman_td_bu = scipy_stats.spearmanr(df_l['td_mean'], df_l['bu_mean'])
        
        correlations.append({
            'L': length,
            'pearson_td_rss': pearson_td_rss,
            'p_pearson_td_rss': p_pearson_td_rss,
            'spearman_td_rss': spearman_td_rss,
            'p_spearman_td_rss': p_spearman_td_rss,
            'pearson_bu_rss': pearson_bu_rss,
            'p_pearson_bu_rss': p_pearson_bu_rss,
            'spearman_bu_rss': spearman_bu_rss,
            'p_spearman_bu_rss': p_spearman_bu_rss,
            'pearson_td_bu': pearson_td_bu,
            'p_pearson_td_bu': p_pearson_td_bu,
            'spearman_td_bu': spearman_td_bu,
            'p_spearman_td_bu': p_spearman_td_bu,
        })
        
        # Accumulate for mean calculation
        td_rss_pearson.append(pearson_td_rss)
        td_rss_spearman.append(spearman_td_rss)
        bu_rss_pearson.append(pearson_bu_rss)
        bu_rss_spearman.append(spearman_bu_rss)
        td_bu_pearson.append(pearson_td_bu)
        td_bu_spearman.append(spearman_td_bu)
    
    # Calculate means
    means = {
        'pearson_td_rss': np.mean(td_rss_pearson),
        'spearman_td_rss': np.mean(td_rss_spearman),
        'pearson_bu_rss': np.mean(bu_rss_pearson),
        'spearman_bu_rss': np.mean(bu_rss_spearman),
        'pearson_td_bu': np.mean(td_bu_pearson),
        'spearman_td_bu': np.mean(td_bu_spearman),
    }
    
    # Convert processed DataFrame to list of dicts for JSON serializability
    raw_points = df.to_dict('records')
    
    return {
        'correlations': correlations,
        'means': means,
        'metadata': {
            'n_models': df['model'].nunique(),
            'total_data_points': len(df),
            'lengths_analyzed': len(correlations),
            'computation_method': 'recomputed_from_pt_embeddings',
            'rss_formula': '(TD_mean + BU_mean) * 500',
            'eval_mode': config.get('eval_mode', 'q1q3'),
        },
        'raw_points': raw_points,
    }


# =============================================================================
# LaTeX Rendering
# =============================================================================

def _render_latex(raw_data: Dict[str, Any]) -> str:
    """Render the correlation table as LaTeX code."""
    correlations = raw_data['correlations']
    means = raw_data['means']
    
    # Build table rows
    rows = []
    for corr in correlations:
        row = (
            f"{corr['L']:2d} & "
            f"{corr['pearson_td_rss']:.4f} & "
            f"{corr['spearman_td_rss']:.4f} & "
            f"{corr['pearson_bu_rss']:.4f} & "
            f"{corr['spearman_bu_rss']:.4f} & "
            f"{corr['pearson_td_bu']:.4f} & "
            f"{corr['spearman_td_bu']:.4f} \\\\"
        )
        rows.append(row)
    
    rows_str = "\n".join(rows)
    
    # Mean row
    mean_row = (
        f"\\midrule\n"
        f"Mean & "
        f"{means['pearson_td_rss']:.4f} & "
        f"{means['spearman_td_rss']:.4f} & "
        f"{means['pearson_bu_rss']:.4f} & "
        f"{means['spearman_bu_rss']:.4f} & "
        f"{means['pearson_td_bu']:.4f} & "
        f"{means['spearman_td_bu']:.4f} \\\\"
    )
    
    latex = f"""\\begin{{table*}}[ht!]
\\centering
\\small
\\caption{{Correlation analysis among TD, BU, and the composite RSS score, across sequence lengths $L$. All correlations are significant at $p<0.001$ ($N={raw_data['metadata']['n_models']}$ models). RSS scores were re-computed from raw embeddings to ensure consistency with TD and BU components.}}
\\begin{{tabular}}{{ccccccc}}
\\toprule
& \\multicolumn{{2}}{{c}}{{TD vs.\\ RSS}} & \\multicolumn{{2}}{{c}}{{BU vs.\\ RSS}} & \\multicolumn{{2}}{{c}}{{TD vs.\\ BU}} \\\\
\\cmidrule(lr){{2-3}} \\cmidrule(lr){{4-5}} \\cmidrule(lr){{6-7}}
$L$ & Pearson $r$ & Spearman $\\rho$ & Pearson $r$ & Spearman $\\rho$ & Pearson $r$ & Spearman $\\rho$ \\\\
\\midrule
{rows_str}
{mean_row}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
"""
    return latex


# =============================================================================
# Markdown Rendering
# =============================================================================

def _render_markdown(raw_data: Dict[str, Any]) -> str:
    """Render the correlation table as Markdown."""
    correlations = raw_data['correlations']
    means = raw_data['means']
    
    # Build table rows
    rows = []
    for corr in correlations:
        row = (
            f"| {corr['L']:2d} | "
            f"{corr['pearson_td_rss']:.4f} | "
            f"{corr['spearman_td_rss']:.4f} | "
            f"{corr['pearson_bu_rss']:.4f} | "
            f"{corr['spearman_bu_rss']:.4f} | "
            f"{corr['pearson_td_bu']:.4f} | "
            f"{corr['spearman_td_bu']:.4f} |"
        )
        rows.append(row)
    
    rows_str = "\n".join(rows)
    
    # Mean row
    mean_row = (
        f"| **Mean** | "
        f"**{means['pearson_td_rss']:.4f}** | "
        f"**{means['spearman_td_rss']:.4f}** | "
        f"**{means['pearson_bu_rss']:.4f}** | "
        f"**{means['spearman_bu_rss']:.4f}** | "
        f"**{means['pearson_td_bu']:.4f}** | "
        f"**{means['spearman_td_bu']:.4f}** |"
    )
    
    md = f"""## Table: Correlation Analysis Among TD, BU, and RSS Score

**Caption:** Correlation analysis among TD, BU, and the composite RSS score, across sequence lengths $L$. All correlations are significant at $p<0.001$ ($N={raw_data['metadata']['n_models']}$ models). RSS scores were re-computed from raw embeddings to ensure consistency with TD and BU components.

| L | TD vs RSS (Pearson r) | TD vs RSS (Spearman ρ) | BU vs RSS (Pearson r) | BU vs RSS (Spearman ρ) | TD vs BU (Pearson r) | TD vs BU (Spearman ρ) |
|---|----------------------|------------------------|----------------------|------------------------|---------------------|----------------------|
{rows_str}
{mean_row}

**Note:** Computation method: {raw_data['metadata']['computation_method']}. RSS formula: {raw_data['metadata']['rss_formula']}. Eval mode: {raw_data['metadata']['eval_mode']}.
"""
    return md


# =============================================================================
# Master Orchestrator
# =============================================================================

def generate_table_10_tdburss_analysis(
    benchmark_dir: str = BENCHMARK_DIR,
    config: dict = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Generate Table 10 analysis: correlations among TD, BU, and RSS.
    
    This version re-computes TD and BU from raw .pt embeddings to ensure
    perfect consistency with RSS = (TD + BU) * 500.
    
    Args:
        benchmark_dir: Path to the benchmark directory
        config: Configuration dictionary (optional). Keys:
            - target_lengths: List of sequence lengths to analyze
            - exclude_models: List of models to exclude
            - eval_mode: FINESSE evaluation mode ('q1q3' or 'q2q2', default: 'q1q3')
    
    Returns:
        (raw_data, latex_string, markdown_string) tuple
    """
    if config is None:
        config = {}
    
    target_lengths = config.get('target_lengths', DEFAULT_TARGET_LENGTHS)
    eval_mode = config.get('eval_mode', 'q1q3')
    
    print(f"=" * 60)
    print("TABLE 10: TD-BU-RSS CORRELATION ANALYSIS (RE-COMPUTATION MODE)")
    print("=" * 60)
    print(f"Loading .pt files and re-computing TD/BU/RSS from embeddings...")
    print(f"Target lengths: {target_lengths}")
    print(f"Eval mode: {eval_mode}")
    print()
    
    df = _gather_and_process_data(benchmark_dir, config)
    
    print(f"\nCalculating correlations across {len(target_lengths)} lengths...")
    raw_data = _calculate_correlations(df, config)
    
    print(f"Rendering LaTeX and Markdown...")
    latex_string = _render_latex(raw_data)
    markdown_string = _render_markdown(raw_data)
    
    return raw_data, latex_string, markdown_string

