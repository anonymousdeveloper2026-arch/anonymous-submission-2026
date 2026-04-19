import json
from pathlib import Path
from typing import Dict, List, Optional, Set


def extract_lemb_metrics(data: Dict, model_name: str) -> Optional[Dict]:
    """Extract LEMB metrics from JSON data.

    Args:
        data: The JSON data dictionary
        model_name: Name of the model extracted from filename

    Returns:
        dict with model_name and extracted metrics, or None if LEMB data not found
    """
    # LEMB metrics to extract
    metric_paths = {
        'LEMBNeedleRetrieval': ['LEMBNeedleRetrieval', 'avg'],
        'LEMBPasskeyRetrieval': ['LEMBPasskeyRetrieval', 'avg'],
        'LEMBSummScreenFDRetrieval': ['LEMBSummScreenFDRetrieval', 'ndcg@10'],
        'LEMBQMSumRetrieval': ['LEMBQMSumRetrieval', 'ndcg@10'],
        'LEMBWikimQARetrieval': ['LEMBWikimQARetrieval', 'ndcg@10'],
        'LEMBNarrativeQARetrieval': ['LEMBNarrativeQARetrieval', 'ndcg@10']
    }

    result = {'model_name': model_name}
    metrics_found = False

    for task_name, path in metric_paths.items():
        try:
            # Navigate through nested structure
            value = data
            for key in path:
                if isinstance(value, dict):
                    value = value.get(key)
                else:
                    value = None
                    break

            if value is not None:
                result[task_name] = value
                metrics_found = True
            else:
                result[task_name] = None
        except (KeyError, TypeError):
            result[task_name] = None

    return result if metrics_found else None


def process_lemb_directory(directory: str, exclude_set: Optional[Set[str]] = None) -> List[Dict]:
    """Process all JSON files in directory for LEMB metrics.

    Args:
        directory: Root directory to search for JSON files
        exclude_set: Optional set of model names to exclude from processing

    Returns:
        list of dicts with model metrics, or empty list if no valid data
    """
    if exclude_set is None:
        exclude_set = set()

    print(f"\n{'='*60}")
    print(f"LEMB Mode: Searching for JSON files in '{directory}'...")
    print(f"{'='*60}")

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return []

    # Find all JSON files recursively
    json_files = list(dir_path.rglob("overall_results.json"))

    if not json_files:
        print(f"No JSON files found in '{directory}'.")
        return []

    print(f"Found {len(json_files)} JSON file(s).")

    all_models_data = []

    for json_file in json_files:
        # Extract model name from parent directory name
        model_name = json_file.parent.name

        # Skip if model is in exclusion list
        if model_name in exclude_set:
            print(f"  ⊘ Excluding model '{model_name}' (in exclusion list)")
            continue

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"  Warning: Could not load '{json_file}': {e}")
            continue

        # Extract LEMB metrics
        metrics = extract_lemb_metrics(data, model_name)

        if metrics:
            all_models_data.append(metrics)
            print(f"  ✓ Processed '{json_file.name}' -> Model: '{model_name}'")
        else:
            print(f"  ✗ Skipped '{json_file.name}' (no LEMB metrics found)")

    print(
        f"\nSuccessfully extracted metrics from {len(all_models_data)} model(s).")
    return all_models_data


def _process_data_into_ir(all_models_data: List[Dict], table_config: Dict) -> List[Dict]:
    """
    Layer 2: Processing & Intermediate Representation.

    Transforms raw model data into a format-agnostic IR with computed averages.
    Supports heatmap mode and 'all' models configuration.
    """
    # Extract all 6 LEMB metrics
    metric_keys = [
        'LEMBNeedleRetrieval',
        'LEMBPasskeyRetrieval',
        'LEMBSummScreenFDRetrieval',
        'LEMBQMSumRetrieval',
        'LEMBWikimQARetrieval',
        'LEMBNarrativeQARetrieval'
    ]

    target_models_config = table_config.get("models", [])
    
    # Build lookup: model_name -> metrics dict
    model_lookup = {}
    for model_data in all_models_data:
        model_name = model_data["model_name"]
        model_lookup[model_name] = model_data
    
    # Determine target models
    if target_models_config == 'all':
        target_models = list(model_lookup.keys())
    else:
        target_models = target_models_config

    # Build IR rows
    ir_rows = []

    for model_name in target_models:
        if model_name not in model_lookup:
            continue  # Skip if model not found in data

        metrics = model_lookup[model_name]
        row_data = {
            "type": "model_row",
            "model_name": model_name,
            "values": {}
        }

        all_values = []
        for key in metric_keys:
            val = metrics.get(key)
            if val is not None:
                row_data["values"][key] = val
                all_values.append(val)
            else:
                row_data["values"][key] = None

        # Calculate average across all 6 metrics
        if all_values:
            row_data["values"]["avg"] = sum(all_values) / len(all_values)
        else:
            row_data["values"]["avg"] = None

        ir_rows.append(row_data)

    # Sort by average score (descending)
    ir_rows.sort(key=lambda r: r["values"].get("avg", float('-inf')), reverse=True)

    # Identify best scores per column
    metric_keys_with_avg = metric_keys + ["avg"]
    for col in metric_keys_with_avg:
        col_values = [row["values"].get(col) for row in ir_rows
                      if row["values"].get(col) is not None]
        if col_values:
            best_value = max(col_values)  # Higher is better
            for row in ir_rows:
                if row["values"].get(col) == best_value:
                    if "is_best" not in row:
                        row["is_best"] = {}
                    row["is_best"][col] = True

    # Calculate heatmap intensities if enabled (using global min/max for consistent scale)
    heatmap_config = table_config.get("heatmap", {})
    heatmap_enabled = heatmap_config.get("enabled", False)
    
    if heatmap_enabled:
        # First compute global min and max from all values across all columns
        all_values = []
        for row in ir_rows:
            for col in metric_keys_with_avg:
                val = row["values"].get(col)
                if val is not None:
                    all_values.append(val)
        
        if all_values:
            global_min = min(all_values)
            global_max = max(all_values)
            global_range = global_max - global_min if global_max != global_min else 1.0
        else:
            global_min = global_max = global_range = 0
        
        # Now calculate intensity using global range
        for col in metric_keys_with_avg:
            for row in ir_rows:
                val = row["values"].get(col)
                if val is not None:
                    if "heatmap_intensity" not in row:
                        row["heatmap_intensity"] = {}
                    # Normalize to 0-1 range using global min/max
                    intensity = (val - global_min) / global_range
                    row["heatmap_intensity"][col] = intensity

    return ir_rows


def _render_markdown(ir_rows: List[Dict]) -> str:
    """
    Layer 3a: Markdown Renderer.
    Produces a readable table for terminal output.
    """
    # Header structure matching the LaTeX table
    header_cols = ["Model", "Needle", "Passkey", "SummScreen",
                   "QMSum", "WikimQA", "NarrativeQA", "Avg."]

    # Build separator line
    col_widths = [max(len(h), 8) for h in header_cols]

    lines = []
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["-" * (w + 2) for w in col_widths]) + "|")

    for row in ir_rows:
        model_name = row["model_name"]
        cells = [model_name]

        metric_keys = [
            'LEMBNeedleRetrieval',
            'LEMBPasskeyRetrieval',
            'LEMBSummScreenFDRetrieval',
            'LEMBQMSumRetrieval',
            'LEMBWikimQARetrieval',
            'LEMBNarrativeQARetrieval'
        ]

        for key in metric_keys:
            val = row["values"].get(key)
            is_best = row.get("is_best", {}).get(key, False)
            if val is None:
                cell = "N/A"
            else:
                cell = f"{val:.3f}"
                if is_best:
                    cell = f"**{cell}**"
            cells.append(cell)

        # Avg column
        avg_val = row["values"].get("avg")
        avg_is_best = row.get("is_best", {}).get("avg", False)
        if avg_val is None:
            avg_cell = "N/A"
        else:
            avg_cell = f"{avg_val:.3f}"
            if avg_is_best:
                avg_cell = f"**{avg_cell}**"
        cells.append(avg_cell)

        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines)


def _render_latex(ir_rows: List[Dict], table_config: Dict = None) -> str:
    """
    Layer 3b: LaTeX Renderer.
    Produces a Booktabs-formatted LaTeX table with multirow/multicolumn headers.
    Supports heatmap mode with cellcolor rendering.
    """
    heatmap_enabled = table_config.get("heatmap", {}).get("enabled", False) if table_config else False
    
    lines = [
        r"\begin{table}[ht!]",
        r"\centering",
        r"\caption{LEMB results for the nomic-embed-text series.}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l cc cccc c}",
        r"\toprule"
    ]

    # Two-row header with multirow and multicolumn
    lines.append(
        r"\multirow{2}{*}{Model} & "
        r"\multicolumn{2}{c}{Synthetic (Acc@1)} & "
        r"\multicolumn{4}{c}{Real (nDCG@10)} & "
        r"\multirow{2}{*}{Avg.} \\"
    )

    # cmidrule lines for column groupings
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-7}")

    # Second row: individual column names
    lines.append(
        r" & Needle & Passkey & SummScreen & QMSum & WikimQA & NarrativeQA & \\"
    )

    lines.append(r"\midrule")

    for row in ir_rows:
        # Escape model name for LaTeX
        model_name = row["model_name"].replace("_", r"\_")
        model_name = f"\\texttt{{{model_name}}}"

        cells = [model_name]

        metric_keys = [
            'LEMBNeedleRetrieval',
            'LEMBPasskeyRetrieval',
            'LEMBSummScreenFDRetrieval',
            'LEMBQMSumRetrieval',
            'LEMBWikimQARetrieval',
            'LEMBNarrativeQARetrieval'
        ]

        for key in metric_keys:
            val = row["values"].get(key)
            is_best = row.get("is_best", {}).get(key, False)
            intensity = row.get("heatmap_intensity", {}).get(key) if heatmap_enabled else None
            
            if val is None:
                cell = "N/A"
            else:
                cell_content = f"{val:.3f}"
                
                # Apply heatmap coloring if enabled
                if heatmap_enabled and intensity is not None:
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
            cell_content = f"{avg_val:.3f}"
            
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


def generate_table_2_lemb(directory: str, table_config: Dict) -> str:
    """
    Public API: Orchestrates the three-layer pipeline.

    Args:
        directory: Root directory containing JSON files
        table_config: Configuration dict defining target models
        output_format: 'markdown' or 'tex'

    Returns:
        raw, formatted_tex, formatted_markdown
    """
    # Layer 1: Data Loading
    all_models_data = process_lemb_directory(directory)

    if not all_models_data:
        return "No data found."

    # Layer 2: Processing & IR
    ir_rows = _process_data_into_ir(all_models_data, table_config)

    # Layer 3: Rendering
    return all_models_data, _render_latex(ir_rows, table_config), _render_markdown(ir_rows)