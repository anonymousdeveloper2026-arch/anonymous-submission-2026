import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def extract_rss_for_tradeoff(data, model_name):
    """Extract RSS scores and latency for tradeoff analysis.

    Args:
        data: The JSON data dictionary
        model_name: Name of the model

    Returns:
        dict with model_name and list of (length, rss_score, latency) tuples, or None if no RSS data
    """
    length_scores = data.get('length_scores', {})
    if not length_scores:
        return None

    model_raw_data = []
    for length_str, metrics in length_scores.items():
        try:
            length = int(length_str)
            rss_scores = metrics.get('rss_scores', [])
            latency_scores = metrics.get('total_latency_scores', [])

            if rss_scores and latency_scores:
                mean_rss = np.mean(rss_scores)
                mean_latency = np.mean(latency_scores)
                model_raw_data.append((length, mean_rss, mean_latency))
        except (ValueError, TypeError):
            continue

    if not model_raw_data:
        return None

    return {'model_name': model_name, 'model_raw_data': model_raw_data}


def process_rss_directory(directory, exclude_models=None):
    """Process all RSS benchmark files for tradeoff analysis.

    Args:
        directory: Root directory to search for benchmark_results.json files
        exclude_models: Set of model names to exclude from analysis

    Returns:
        list of dicts with model tradeoff data
    """
    if exclude_models is None:
        exclude_models = set()
    print(f"\n{'='*60}")
    print(
        f"RSS Tradeoff Mode: Searching for benchmark_results.json in '{directory}'...")
    print(f"{'='*60}")

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return []

    json_files = list(dir_path.rglob('benchmark_results.json'))
    if not json_files:
        print(f"No benchmark_results.json files found in '{directory}'.")
        return []

    print(f"Found {len(json_files)} file(s).")

    all_models_data = []
    for json_file in json_files:
        model_name = json_file.parent.parent.name

        # Skip if model is in exclusion set
        if model_name in exclude_models:
            print(f"  ⊘ Excluding model '{model_name}' (in exclusion list)")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  ✗ Error loading '{json_file}': {e}")
            continue

        # Check if RSS data exists
        if 'average_rss' not in data:
            print(f"  ✗ Skipped '{json_file.name}' (no RSS data)")
            continue

        metrics = extract_rss_for_tradeoff(data, model_name)
        if metrics:
            all_models_data.append(metrics)
            print(f"  ✓ Processed '{json_file.name}' -> Model: '{model_name}'")
        else:
            print(f"  ✗ Skipped '{json_file.name}' (no valid tradeoff data)")

    print(
        f"\nSuccessfully extracted data from {len(all_models_data)} model(s).")
    return all_models_data


def _process_data_into_ir(all_models_data: List[Dict], table_config: Dict) -> List[Dict]:
    """
    Layer 2: Processing & Intermediate Representation.

    Transforms raw model data into a format-agnostic IR with computed values,
    best-score identification per column, and optional heatmap intensity data.
    """
    target_lengths = table_config["columns"]["target_lengths"]
    avg_range = table_config["columns"]["avg_range"]
    groups = table_config["groups"]
    heatmap_config = table_config.get("heatmap", {})
    heatmap_enabled = heatmap_config.get("enabled", False)

    # Build lookup: model_name -> {length: rss_score}
    model_lookup = {}
    for model_data in all_models_data:
        model_name = model_data["model_name"]
        tradeoff_dict = {length: rss for length,
                         rss, _ in model_data["model_raw_data"]}
        model_lookup[model_name] = tradeoff_dict

    # Collect all model rows
    model_rows = []
    
    if groups is None:
        # Fallback: process all models as a single group with sorting
        for model_name in model_lookup.keys():
            tradeoff_dict = model_lookup[model_name]
            row_data = {"type": "model_row",
                        "model_name": model_name, "values": {},
                        "group_name": "All Models",
                        "is_baseline": False}

            # Extract target length values
            for length in target_lengths:
                if length in tradeoff_dict:
                    row_data["values"][length] = tradeoff_dict[length]
                else:
                    row_data["values"][length] = None

            # Calculate average over avg_range
            avg_values = [tradeoff_dict[l]
                          for l in avg_range if l in tradeoff_dict]
            if avg_values:
                row_data["values"]["avg"] = np.mean(avg_values)
            else:
                row_data["values"]["avg"] = None

            model_rows.append(row_data)
        
        # Sort model rows by average score (descending)
        model_rows.sort(key=lambda r: r["values"].get("avg", float('-inf')), reverse=True)
    else:
        # Group-based iteration
        for group in groups:
            group_name = group["group_name"]
            models = group["models"]
            is_baseline = group.get("is_baseline", False)

            for model_name in models:
                if model_name not in model_lookup:
                    continue  # Skip if model not found in data

                tradeoff_dict = model_lookup[model_name]
                row_data = {"type": "model_row",
                            "model_name": model_name, "values": {},
                            "group_name": group_name,
                            "is_baseline": is_baseline}

                # Extract target length values
                for length in target_lengths:
                    if length in tradeoff_dict:
                        row_data["values"][length] = tradeoff_dict[length]
                    else:
                        row_data["values"][length] = None

                # Calculate average over avg_range
                avg_values = [tradeoff_dict[l]
                              for l in avg_range if l in tradeoff_dict]
                if avg_values:
                    row_data["values"]["avg"] = np.mean(avg_values)
                else:
                    row_data["values"]["avg"] = None

                model_rows.append(row_data)

        # Sort model rows by average score (descending) - only for non-baseline groups
        # Baseline entries stay at the bottom
        baseline_rows = [r for r in model_rows if r.get("is_baseline", False)]
        non_baseline_rows = [r for r in model_rows if not r.get("is_baseline", False)]
        non_baseline_rows.sort(key=lambda r: r["values"].get("avg", float('-inf')), reverse=True)
        model_rows = non_baseline_rows + baseline_rows

    # Build IR rows
    if groups is None:
        # No groups: directly use model_rows without group headers
        ir_rows = model_rows
    else:
        # Groups defined: build with group headers
        ir_rows = []
        current_group = None
        for row in model_rows:
            # Add group header when group changes
            if row["group_name"] != current_group:
                current_group = row["group_name"]
                ir_rows.append({
                    "type": "group_header",
                    "group_name": current_group,
                    "is_baseline": row["is_baseline"]
                })
            ir_rows.append(row)

    # Identify best scores per column (for bold formatting)
    columns_to_check = target_lengths + ["avg"]
    for col in columns_to_check:
        col_values = [row["values"].get(col) for row in ir_rows
                      if row["type"] == "model_row" and row["values"].get(col) is not None]
        if col_values:
            best_value = max(col_values)  # Higher RSS is better
            for row in ir_rows:
                if row["type"] == "model_row" and row["values"].get(col) == best_value:
                    if "is_best" not in row:
                        row["is_best"] = {}
                    row["is_best"][col] = True

    # Calculate heatmap intensities if enabled
    if heatmap_enabled:
        for col in columns_to_check:
            # Get all values for this column (excluding None)
            col_entries = [(row, row["values"].get(col)) for row in ir_rows
                          if row["type"] == "model_row" and row["values"].get(col) is not None]
            if not col_entries:
                continue

            values = [v for _, v in col_entries]
            col_min = min(values)
            col_max = max(values)
            col_range = col_max - col_min if col_max != col_min else 1.0

            # Calculate intensity for each cell (0.0 to 1.0)
            for row, val in col_entries:
                if "heatmap_intensity" not in row:
                    row["heatmap_intensity"] = {}
                # Normalize to 0-1 range
                intensity = (val - col_min) / col_range
                row["heatmap_intensity"][col] = intensity

    return ir_rows


def _render_markdown(ir_rows: List[Dict], target_lengths: List[int]) -> str:
    """
    Layer 3a: Markdown Renderer.
    Produces a readable table for terminal output.
    """
    # Build header
    header_cols = ["Model"] + \
        [f"L={l}" for l in target_lengths] + ["Avg(L4..L16)"]
    lines = ["| " + " | ".join(header_cols) + " |"]
    lines.append("|" + "|".join(["-" * (len(c) + 2)
                 for c in header_cols]) + "|")

    for row in ir_rows:
        if row["type"] == "group_header":
            group_line = f"| *{row['group_name']}* |" + \
                " |" * len(target_lengths) + " |"
            lines.append(group_line)
        else:  # model_row
            model_name = row["model_name"]
            cells = [model_name]

            for length in target_lengths:
                val = row["values"].get(length)
                is_best = row.get("is_best", {}).get(length, False)
                if val is None:
                    cell = "N/A"
                else:
                    cell = f"{val:.2f}"
                    if is_best:
                        cell = f"**{cell}**"
                cells.append(cell)

            # Avg column
            avg_val = row["values"].get("avg")
            avg_is_best = row.get("is_best", {}).get("avg", False)
            if avg_val is None:
                avg_cell = "N/A"
            else:
                avg_cell = f"{avg_val:.2f}"
                if avg_is_best:
                    avg_cell = f"**{avg_cell}**"
            cells.append(avg_cell)

            lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _render_latex(ir_rows: List[Dict], target_lengths: List[int], config: Dict = None) -> str:
    """
    Layer 3b: LaTeX Renderer.
    Produces a Booktabs-formatted LaTeX table for paper insertion.
    Supports heatmap mode with \cellcolor for intensity visualization.
    """
    # Check if heatmap mode is enabled
    heatmap_enabled = config.get('heatmap', {}).get('enabled', False) if config else False
    # Build column spec: l for model, c for each numeric column
    col_spec = "l" + "c" * (len(target_lengths) + 1)
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{RSS scores for selected model groups.}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        r"\toprule"
    ]

    # Header row
    header_parts = [
        "Model"] + [f"$L={l}$" for l in target_lengths] + [r"$\text{Avg}(L4 \dots L16)$"]
    lines.append(" & ".join(header_parts) + r" \\")
    lines.append(r"\midrule")

    first_group = True
    for row in ir_rows:
        if row["type"] == "group_header":
            # Add midrule before group header (except for first group)
            if not first_group:
                lines.append(r"\midrule")
            first_group = False

            n_cols = len(target_lengths) + 2  # Model + lengths + avg
            group_name = row["group_name"]
            lines.append(
                f"\\multicolumn{{{n_cols}}}{{l}}{{\\textit{{{group_name}}}}} \\\\")
            lines.append(r"\midrule")
        else:  # model_row
            # Escape model name for LaTeX
            model_name = row["model_name"].replace("_", r"\_")
            model_name = f"\\texttt{{{model_name}}}"

            cells = [model_name]

            for length in target_lengths:
                val = row["values"].get(length)
                is_best = row.get("is_best", {}).get(length, False)
                intensity = row.get("heatmap_intensity", {}).get(length) if heatmap_enabled else None
                
                if val is None:
                    cell = "N/A"
                else:
                    cell_content = f"{val:.2f}"
                    
                    # Apply heatmap coloring if enabled
                    if heatmap_enabled and intensity is not None:
                        # Determine color based on value sign and intensity
                        if val < 0:
                            # Negative values: fixed red intensity
                            cell = f"\\cellcolor{{red!25}}{cell_content}"
                        else:
                            # Positive values: scaled blue (0-60 range for subtlety)
                            blue_intensity = int(intensity * 45)
                            cell = f"\\cellcolor{{blue!{blue_intensity}}}{cell_content}"
                    else:
                        cell = cell_content
                    
                    # Apply bold formatting only when heatmap is disabled
                    if is_best and not heatmap_enabled:
                        cell = f"\\textbf{{{cell}}}"
                
                cells.append(cell)

            # Avg column
            avg_val = row["values"].get("avg")
            avg_is_best = row.get("is_best", {}).get("avg", False)
            intensity_avg = row.get("heatmap_intensity", {}).get("avg") if heatmap_enabled else None
            
            if avg_val is None:
                avg_cell = "N/A"
            else:
                cell_content = f"{avg_val:.2f}"
                
                # Apply heatmap coloring if enabled
                if heatmap_enabled and intensity_avg is not None:
                    if avg_val < 0:
                        # Negative values: fixed red intensity
                        avg_cell = f"\\cellcolor{{red!25}}{cell_content}"
                    else:
                        # Positive values: scaled blue (0-60 range for subtlety)
                        blue_intensity = int(intensity_avg * 60)
                        avg_cell = f"\\cellcolor{{blue!{blue_intensity}}}{cell_content}"
                else:
                    avg_cell = cell_content
                
                # Apply bold formatting only when heatmap is disabled
                if avg_is_best and not heatmap_enabled:
                    avg_cell = f"\\textbf{{{avg_cell}}}"
            
            cells.append(avg_cell)

            lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_table_1_rss(directory: str, table_config: Dict) -> str:
    """
    Public API: Orchestrates the three-layer pipeline.

    Args:
        directory: Root directory containing benchmark_results.json files
        table_config: Configuration dict defining columns and groups
        output_format: 'markdown' or 'tex'

    Returns:
        raw, formatted_tex, formatted_markdown
    """
    # Layer 1: Data Loading
    all_models_data = process_rss_directory(directory)

    if not all_models_data:
        return "No data found."

    # Layer 2: Processing & IR
    ir_rows = _process_data_into_ir(all_models_data, table_config)

    # Layer 3: Rendering
    target_lengths = table_config["columns"]["target_lengths"]

    return all_models_data, _render_latex(ir_rows, target_lengths, table_config), _render_markdown(ir_rows, target_lengths)
