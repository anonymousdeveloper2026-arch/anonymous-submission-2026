import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any


# =============================================================================
# LEMB Task Configuration
# =============================================================================
LEMB_TASK_MAP = {
    'needle': 'LEMBNeedleRetrieval',
    'passkey': 'LEMBPasskeyRetrieval',
    'summscreen': 'LEMBSummScreenFDRetrieval',
    'qmsum': 'LEMBQMSumRetrieval',
    'wikimqa': 'LEMBWikimQARetrieval',
    'narrativeqa': 'LEMBNarrativeQARetrieval',
}

# Task display order and categories
SYNTHETIC_TASKS = ['needle', 'passkey']
REAL_TASKS = ['summscreen', 'qmsum', 'wikimqa', 'narrativeqa']
TASK_ORDER = SYNTHETIC_TASKS + REAL_TASKS


# =============================================================================
# Data Processing
# =============================================================================

def _calculate_metrics(directory: str, config: dict = None) -> Dict[str, Any]:
    """
    Calculate n, mean, and IQR for each LEMB task.

    Returns:
        Dictionary with task metrics and metadata
    """
    # Prepare exclude set
    exclude_set = set(config.get('exclude_models', []))

    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' does not exist.")

    # Initialize score collection
    task_scores: Dict[str, List[float]] = {
        task_key: [] for task_key in LEMB_TASK_MAP.values()
    }

    # Find all overall_results.json files
    json_files = list(dir_path.rglob("overall_results.json"))

    if not json_files:
        raise ValueError(
            f"No overall_results.json files found in '{directory}'.")

    lemb_files_found = 0

    for json_file in json_files:
        # Extract model name from path
        model_name = json_file.parent.name

        # Skip excluded models
        if model_name in exclude_set:
            print(f"  ⊘ Excluded model: {model_name}")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if this is a LEMB result file
            is_lemb = any(
                task_key in data for task_key in LEMB_TASK_MAP.values())
            if not is_lemb:
                continue

            lemb_files_found += 1

            # Extract scores for each task
            for short_key, full_key in LEMB_TASK_MAP.items():
                if full_key in data:
                    task_data = data[full_key]

                    # Extract score based on task type
                    if short_key in ['needle', 'passkey']:
                        score = task_data.get('avg') if isinstance(
                            task_data, dict) else task_data
                    else:
                        score = task_data.get(
                            'ndcg@10') if isinstance(task_data, dict) else task_data

                    if score is not None:
                        task_scores[full_key].append(float(score))

        except Exception as e:
            print(f"  ✗ Error loading '{json_file.name}': {e}")

    # Calculate metrics for each task
    metrics = []

    for short_key in TASK_ORDER:
        full_key = LEMB_TASK_MAP[short_key]
        scores = task_scores[full_key]

        if scores:
            n = len(scores)
            mean_score = np.mean(scores)
            q3, q1 = np.percentile(scores, [75, 25])
            iqr = q3 - q1

            metrics.append({
                'task': short_key,
                'full_name': full_key,
                'n': n,
                'mean': mean_score,
                'iqr': iqr,
                'is_synthetic': short_key in SYNTHETIC_TASKS,
            })

    return {
        'metrics': metrics,
        'metadata': {
            'n_models': lemb_files_found,
            'excluded_models': list(exclude_set),
            'directory': directory, }
    }


# =============================================================================
# LaTeX Rendering
# =============================================================================

def _render_latex(raw_data: Dict[str, Any]) -> str:
    """Render the LEMB metrics table as LaTeX code."""
    metrics = raw_data['metrics']

    # Build table rows
    rows = []
    for i, m in enumerate(metrics):
        # Check if we need to insert midrule after passkey (last synthetic task)
        midrule = ""
        if i > 0 and not m['is_synthetic'] and metrics[i-1]['is_synthetic']:
            midrule = "\\midrule\n"

        row = f"{midrule}{m['task']:15s} & {m['n']:3d} & {m['mean']:.4f} & {m['iqr']:.4f} \\\\"
        rows.append(row)

    rows_str = "\n".join(rows)

    latex = f"""\\begin{{table}}[ht!]
\\centering
\\caption{{LEMB Task Score Mean and IQR}}
\\begin{{tabular}}{{lccc}}
\\toprule
Task & $n$ & Mean & IQR \\\\
\\midrule
{rows_str}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return latex


# =============================================================================
# Markdown Rendering
# =============================================================================

def _render_markdown(raw_data: Dict[str, Any]) -> str:
    """Render the LEMB metrics table as Markdown."""
    metrics = raw_data['metrics']

    # Build table rows
    rows = []
    for i, m in enumerate(metrics):
        # Add separator line after synthetic tasks
        separator = ""
        if i > 0 and not m['is_synthetic'] and metrics[i-1]['is_synthetic']:
            separator = "| **---** | **---** | **---** | **---** |\n"

        row = f"| {separator}| {m['task']:15s} | {m['n']:3d} | {m['mean']:.4f} | {m['iqr']:.4f} |"
        rows.append(row)

    rows_str = "\n".join(rows)

    md = f"""## Table: LEMB Task Score Mean and IQR

| Task            |   n |   Mean |   IQR |
|:----------------|----:|-------:|------:|
{rows_str}

**Note:** Based on {raw_data['metadata']['n_models']} models. Synthetic tasks (needle, passkey) are visually separated from real-world tasks.
"""
    return md


# =============================================================================
# Master Orchestrator
# =============================================================================

def generate_table_11_lemb_iqr(
    directory: str,
    config: dict = None
) -> Tuple[Dict[str, Any], str, str]:
    """
    Generate Table 11 analysis: LEMB task mean and IQR.

    Args:
        directory: Directory containing overall_results.json files
        config: Configuration dictionary (optional)

    Returns:
        (raw_data, latex_string, markdown_string) tuple
    """
    if config is None:
        config = {}

    print(f"=" * 60)
    print("TABLE 11: LEMB TASK MEAN AND IQR ANALYSIS")
    print(f"=" * 60)
    print(f"Scanning directory: {directory}")
    print()

    raw_data = _calculate_metrics(directory, config)

    print(f"\nProcessed {raw_data['metadata']['n_models']} LEMB result files.")
    print(f"\nMetrics calculated:")
    for m in raw_data['metrics']:
        category = "[Synthetic]" if m['is_synthetic'] else "[Real]"
        print(
            f"  {m['task']:15s} {category}: n={m['n']:3d}, mean={m['mean']:.4f}, IQR={m['iqr']:.4f}")

    print(f"\nRendering LaTeX and Markdown...")
    latex_string = _render_latex(raw_data)
    markdown_string = _render_markdown(raw_data)

    return raw_data, latex_string, markdown_string
