import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats


LEMB_TASK_MAP = {
    'needle': 'LEMBNeedleRetrieval',
    'passkey': 'LEMBPasskeyRetrieval',
    'summscreen': 'LEMBSummScreenFDRetrieval',
    'qmsum': 'LEMBQMSumRetrieval',
    'wikimqa': 'LEMBWikimQARetrieval',
    'narrativeqa': 'LEMBNarrativeQARetrieval',
}


def gather_all_data(directory: str) -> dict:
    """
    Layer 1: Data Loading.
    Traverse directory and extract only LEMB and SRS metrics.
    """
    print(f"\n{'='*60}")
    print(f"Gathering LEMB/SRS data from '{directory}'...")
    print(f"{'='*60}")

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return {}

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

        # Detect and extract LEMB data
        if any(k.startswith('LEMB') for k in data.keys()):
            model_name = json_file.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'lemb': None, 'srs': None}
            all_data[model_name]['lemb'] = data
            print(f"  ✓ LEMB: {json_file.name} -> {model_name}")

        # Detect and extract RSS data
        elif 'average_rss' in data:
            model_name = json_file.parent.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'lemb': None, 'rss': None, 'srs': None}
            all_data[model_name]['rss'] = data
            print(f"  ✓ RSS: {json_file.name} -> {model_name}")

        # Detect and extract SRS data
        elif 'average_srs' in data:
            model_name = json_file.parent.parent.name
            if model_name not in all_data:
                all_data[model_name] = {'lemb': None, 'srs': None}
            all_data[model_name]['srs'] = data
            print(f"  ✓ SRS: {json_file.name} -> {model_name}")

    # Filter to only models with both data types
    complete_data = {}
    for model_name, metrics in all_data.items():
        if all(v is not None for v in metrics.values()):
            complete_data[model_name] = metrics
            print(f"  ✓ Complete data for: {model_name}")
        else:
            missing = [k for k, v in metrics.items() if v is None]
            print(f"  ✗ Skipping {model_name} (missing: {', '.join(missing)})")

    print(
        f"\nSuccessfully loaded data for {len(complete_data)} complete model(s).")
    return complete_data


def extract_lemb_task_score(data: dict, task_key: str) -> float:
    """Extract score for a specific LEMB task."""
    if not data or task_key not in data:
        return None

    task_data = data[task_key]
    if isinstance(task_data, dict):
        # Needle and Passkey use 'avg', others use 'ndcg@10'
        if 'needle' in task_key.lower() or 'passkey' in task_key.lower():
            return task_data.get('avg')
        else:
            return task_data.get('ndcg@10')
    return float(task_data) if task_data is not None else None


def extract_srs_negative_only_score(data: dict) -> float:
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


def extract_srs_positive_only_score(data: dict) -> float:
    """Calculate SRS positive-only score (mean of positive sample similarities)."""
    if not data:
        return None

    length_scores = data.get('length_scores', {})
    positive_only_scores = []
    sample_count = 0

    for length_data in length_scores.values():
        sample_results = length_data.get('sample_results', [])
        for sample in sample_results:
            for probe_len, scores in sample.items():
                if scores:
                    pos_only = [s for s in scores if s > 0]
                    if pos_only:
                        positive_only_scores.append(np.mean(pos_only))
                    sample_count += 1

    if not positive_only_scores:
        return None

    return np.sum(positive_only_scores) / sample_count


def extract_srs_mean_score(data: dict) -> float:
    """Calculate SRS overall mean score (mean of all sample similarities)."""
    if not data:
        return None

    length_scores = data.get('length_scores', {})
    all_means = []
    sample_count = 0

    for length_data in length_scores.values():
        sample_results = length_data.get('sample_results', [])
        for sample in sample_results:
            for probe_len, scores in sample.items():
                if scores:
                    all_means.append(np.mean(scores))
                    sample_count += 1

    if not all_means:
        return None

    return np.sum(all_means) / sample_count


def extract_rss_l15_mean(data: dict) -> float:
    """Extract mean RSS score for L=15."""
    if not data:
        return None

    length_scores = data.get('length_scores', {})
    if '15' not in length_scores:
        return None

    rss_scores = length_scores['15'].get('rss_scores', [])
    if not rss_scores:
        return None

    return np.mean(rss_scores)


def _compute_correlations(x_values: list, y_values: list) -> dict:
    """Compute Pearson and Spearman correlations between x and y."""
    if len(x_values) < 2 or len(y_values) < 2:
        return {
            'pearson_r': None, 'pearson_p': None,
            'spearman_r': None, 'spearman_p': None
        }

    pearson_r, pearson_p = scipy_stats.pearsonr(x_values, y_values)
    spearman_r, spearman_p = scipy_stats.spearmanr(x_values, y_values)

    return {
        'pearson_r': pearson_r, 'pearson_p': pearson_p,
        'spearman_r': spearman_r, 'spearman_p': spearman_p
    }


def _process_data_into_ir(all_data: dict, config: dict) -> list:
    """
    Layer 2: Processing & Intermediate Representation.

    Structure:
    - 6 LEMB tasks as main rows, each with 4 sub-rows (srs_neg, srs_pos, srs_mean, rss_l15)
    - 1 Average row at end, with same 4 sub-rows
    """
    # Filter excluded models
    exclude_models = config.get('exclude_models', [])
    filtered_data = {k: v for k,
                     v in all_data.items() if k not in exclude_models}

    if not filtered_data:
        print("Warning: No models remaining after exclusion filter.")
        return []

    print(
        f"\nProcessing {len(filtered_data)} models (excluded: {len(exclude_models)})")

    # Define the 6 LEMB tasks in order
    lemb_tasks = [
        ('needle', 'Needle'),
        ('passkey', 'Passkey'),
        ('summscreen', 'SummScreen'),
        ('qmsum', 'QMSum'),
        ('wikimqa', 'WikimQA'),
        ('narrativeqa', 'NarrativeQA'),
    ]

    ir_rows = []

    # Process each LEMB task
    for task_key, task_display in lemb_tasks:
        # Collect paired data for all four metrics (3 SRS + 1 RSS)
        srs_neg_x, srs_pos_x, srs_mean_x, rss_l15_x = [], [], [], []
        srs_y_values = []

        full_task_key = LEMB_TASK_MAP[task_key]

        for model_name, metrics in filtered_data.items():
            lemb_score = extract_lemb_task_score(
                metrics['lemb'], full_task_key)

            # Extract all three SRS metrics and RSS(L=15)
            srs_neg_score = extract_srs_negative_only_score(metrics['srs'])
            srs_pos_score = extract_srs_positive_only_score(metrics['srs'])
            srs_mean_score = extract_srs_mean_score(metrics['srs'])
            rss_l15_score = extract_rss_l15_mean(metrics['rss'])

            if lemb_score is not None:
                srs_y_values.append(lemb_score)
                if srs_neg_score is not None:
                    srs_neg_x.append(srs_neg_score)
                if srs_pos_score is not None:
                    srs_pos_x.append(srs_pos_score)
                if srs_mean_score is not None:
                    srs_mean_x.append(srs_mean_score)
                if rss_l15_score is not None:
                    rss_l15_x.append(rss_l15_score)

        # Compute correlations for each metric
        srs_neg_corr = _compute_correlations(
            srs_neg_x, srs_y_values[:len(srs_neg_x)])
        srs_pos_corr = _compute_correlations(
            srs_pos_x, srs_y_values[:len(srs_pos_x)])
        srs_mean_corr = _compute_correlations(
            srs_mean_x, srs_y_values[:len(srs_mean_x)])
        rss_l15_corr = _compute_correlations(
            rss_l15_x, srs_y_values[:len(rss_l15_x)])

        task_type = 'Synthetic' if task_key in [
            'needle', 'passkey'] else 'Real'

        ir_row = {
            'row_type': 'task_header',
            'task': task_display,
            'task_type': task_type,
            'n_samples': len(srs_neg_x),
            'sub_rows': [
                {
                    'metric': r'$\mathbf{srs}_{\text{neg}}$',
                    'metric_key': 'srs_neg',
                    **srs_neg_corr
                },
                {
                    'metric': r'$\mathbf{srs}_{\text{pos}}$',
                    'metric_key': 'srs_pos',
                    **srs_pos_corr
                },
                {
                    'metric': r'$\mathbf{srs}_{\text{mean}}$',
                    'metric_key': 'srs_mean',
                    **srs_mean_corr
                },
                {
                    'metric': r'$\text{RSS}_{L=15}$',
                    'metric_key': 'rss_l15',
                    **rss_l15_corr
                },
            ]
        }
        ir_rows.append(ir_row)

        print(f"  {task_display:12s} ({task_type:8s}): n={len(srs_neg_x):2d}")

    # Compute average LEMB score across all tasks for each model
    print("\n  Computing Average LEMB scores...")

    avg_srs_neg_x, avg_srs_pos_x, avg_srs_mean_x, avg_rss_l15_x = [], [], [], []
    avg_y_values = []

    for model_name, metrics in filtered_data.items():
        # Collect all 6 LEMB task scores for this model
        all_lemb_scores = []
        for task_key, _ in lemb_tasks:
            full_task_key = LEMB_TASK_MAP[task_key]
            score = extract_lemb_task_score(metrics['lemb'], full_task_key)
            if score is not None:
                all_lemb_scores.append(score)

        if len(all_lemb_scores) > 0:
            avg_lemb_score = np.mean(all_lemb_scores)
            avg_y_values.append(avg_lemb_score)

            # Get all four metrics
            srs_neg_score = extract_srs_negative_only_score(metrics['srs'])
            srs_pos_score = extract_srs_positive_only_score(metrics['srs'])
            srs_mean_score = extract_srs_mean_score(metrics['srs'])
            rss_l15_score = extract_rss_l15_mean(metrics['rss'])

            if srs_neg_score is not None:
                avg_srs_neg_x.append(srs_neg_score)
            if srs_pos_score is not None:
                avg_srs_pos_x.append(srs_pos_score)
            if srs_mean_score is not None:
                avg_srs_mean_x.append(srs_mean_score)
            if rss_l15_score is not None:
                avg_rss_l15_x.append(rss_l15_score)

    # Compute correlations for average
    avg_srs_neg_corr = _compute_correlations(
        avg_srs_neg_x, avg_y_values[:len(avg_srs_neg_x)])
    avg_srs_pos_corr = _compute_correlations(
        avg_srs_pos_x, avg_y_values[:len(avg_srs_pos_x)])
    avg_srs_mean_corr = _compute_correlations(
        avg_srs_mean_x, avg_y_values[:len(avg_srs_mean_x)])
    avg_rss_l15_corr = _compute_correlations(
        avg_rss_l15_x, avg_y_values[:len(avg_rss_l15_x)])

    # Add Average row
    avg_row = {
        'row_type': 'average_header',
        'task': 'Average',
        'task_type': 'Summary',
        'n_samples': len(avg_srs_neg_x),
        'sub_rows': [
            {
                'metric': r'$\mathbf{srs}_{\text{neg}}$',
                'metric_key': 'srs_neg',
                **avg_srs_neg_corr
            },
            {
                'metric': r'$\mathbf{srs}_{\text{pos}}$',
                'metric_key': 'srs_pos',
                **avg_srs_pos_corr
            },
            {
                'metric': r'$\mathbf{srs}_{\text{mean}}$',
                'metric_key': 'srs_mean',
                **avg_srs_mean_corr
            },
            {
                'metric': r'$\text{RSS}_{L=15}$',
                'metric_key': 'rss_l15',
                **avg_rss_l15_corr
            },
        ]
    }
    ir_rows.append(avg_row)

    print(f"  Average: n={len(avg_srs_neg_x):2d}")

    return ir_rows


def _render_markdown(ir_rows: list) -> str:
    """
    Layer 3a: Markdown Renderer.
    Renders hierarchical table with task headers and 3 sub-rows each.
    """
    lines = [
        "| LEMB Task | SRS Metric | Pearson r | Pearson p | Spearman ρ | Spearman p |",
        "|-----------|------------|-----------|-----------|------------|------------|"
    ]

    for row in ir_rows:
        task = row['task']
        task_type = row['task_type']

        # Add task header row
        lines.append(f"| **{task}** ({task_type}) | | | | | |")

        # Add 3 sub-rows
        for sub in row['sub_rows']:
            metric = sub['metric']
            pr = f"{sub['pearson_r']:.4f}" if sub['pearson_r'] is not None else "N/A"
            pp = f"{sub['pearson_p']:.4f}" if sub['pearson_p'] is not None else "N/A"
            sr = f"{sub['spearman_r']:.4f}" if sub['spearman_r'] is not None else "N/A"
            sp = f"{sub['spearman_p']:.4f}" if sub['spearman_p'] is not None else "N/A"

            lines.append(f"| | {metric} | {pr} | {pp} | {sr} | {sp} |")

        # Add separator after each task group
        lines.append("| | | | | | |")

    return "\n".join(lines)


def _render_latex(ir_rows: list) -> str:
    """
    Layer 3b: LaTeX Renderer.
    Produces a compact 9-column table with significance stars.
    Structure: Task | Pearson (4 metrics) | Spearman (4 metrics)
    Metrics: srs_neg, srs_pos, srs_mean, rss_l15
    Significance: * p<0.05, ** p<0.01, *** p<0.001
    Groups: Synthetic Tasks, Real Tasks, Summary (Average)
    """
    def fmt_with_stars(val, p_val):
        """Format correlation value with significance stars. Gray out non-significant values."""
        if val is None or p_val is None:
            return "--"

        # Non-significant values (p >= 0.05) are shown in gray
        if p_val >= 0.05:
            return f"\\textcolor{{gray}}{{\\textit{{{val:.4f}}}}}"

        stars = ""
        if p_val < 0.001:
            stars = "$^{***}$"
        elif p_val < 0.01:
            stars = "$^{**}$"
        elif p_val < 0.05:
            stars = "$^{*}$"

        return f"{val:.4f}{stars}"

    def fmt_bold_stars(val, p_val, is_max=False):
        """Format with bold and stars for maximum values. Gray out non-significant values."""
        if val is None or p_val is None:
            return "--"

        # Non-significant values (p >= 0.05) are shown in gray (no bold even if max)
        if p_val >= 0.05:
            return f"\\textcolor{{gray}}{{\\textit{{{val:.4f}}}}}"

        stars = ""
        if p_val < 0.001:
            stars = "$^{***}$"
        elif p_val < 0.01:
            stars = "$^{**}$"
        elif p_val < 0.05:
            stars = "$^{*}$"

        formatted = f"{val:.4f}{stars}"
        if is_max:
            return f"\\textbf{{{formatted}}}"
        return formatted
    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Correlation between LEMB task scores and sensitivity metrics ($\mathbf{srs}_{\text{neg}}$, $\mathbf{srs}_{\text{pos}}$, $\mathbf{srs}_{\text{mean}}$, $\text{RSS}_{L=15}$). Significance: $^{*}$$p$$<$0.05, $^{**}$$p$$<$0.01, $^{***}$$p$$<$0.001. Non-significant values ($p$$\geq$0.05) are shown in gray italics.}",
        r"\label{tbl:srs-rss-correl-lemb-full}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Pearson $r$} & \multicolumn{4}{c}{Spearman $\rho$} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        r"Task & $\mathbf{srs}_{\text{neg}}$ & $\mathbf{srs}_{\text{pos}}$ & $\mathbf{srs}_{\text{mean}}$ & $\text{RSS}_{L=15}$ & $\mathbf{srs}_{\text{neg}}$ & $\mathbf{srs}_{\text{pos}}$ & $\mathbf{srs}_{\text{mean}}$ & $\text{RSS}_{L=15}$ \\",
        r"\midrule"
    ]

    # Group rows by type
    synthetic_rows = [r for r in ir_rows if r['task_type'] == 'Synthetic']
    real_rows = [r for r in ir_rows if r['task_type'] == 'Real']
    summary_rows = [r for r in ir_rows if r['row_type'] == 'average_header']

    # Render Synthetic Tasks section
    if synthetic_rows:
        lines.append(r"\multicolumn{9}{l}{\textit{Synthetic Tasks}} \\")
        lines.append(r"\midrule")
        for row in synthetic_rows:
            task = row['task']
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            line = (f"{task} & "
                    f"{fmt_with_stars(srs_neg.get('pearson_r'), srs_neg.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_pos.get('pearson_r'), srs_pos.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_mean.get('pearson_r'), srs_mean.get('pearson_p'))} & "
                    f"{fmt_with_stars(rss_l15.get('pearson_r'), rss_l15.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_neg.get('spearman_r'), srs_neg.get('spearman_p'))} & "
                    f"{fmt_with_stars(srs_pos.get('spearman_r'), srs_pos.get('spearman_p'))} & "
                    f"{fmt_with_stars(srs_mean.get('spearman_r'), srs_mean.get('spearman_p'))} & "
                    f"{fmt_with_stars(rss_l15.get('spearman_r'), rss_l15.get('spearman_p'))} \\\\")
            lines.append(line)
        lines.append(r"\midrule")

    # Render Real Tasks section
    if real_rows:
        lines.append(r"\multicolumn{9}{l}{\textit{Real Tasks}} \\")
        lines.append(r"\midrule")
        for row in real_rows:
            task = row['task']
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            line = (f"{task} & "
                    f"{fmt_with_stars(srs_neg.get('pearson_r'), srs_neg.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_pos.get('pearson_r'), srs_pos.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_mean.get('pearson_r'), srs_mean.get('pearson_p'))} & "
                    f"{fmt_with_stars(rss_l15.get('pearson_r'), rss_l15.get('pearson_p'))} & "
                    f"{fmt_with_stars(srs_neg.get('spearman_r'), srs_neg.get('spearman_p'))} & "
                    f"{fmt_with_stars(srs_pos.get('spearman_r'), srs_pos.get('spearman_p'))} & "
                    f"{fmt_with_stars(srs_mean.get('spearman_r'), srs_mean.get('spearman_p'))} & "
                    f"{fmt_with_stars(rss_l15.get('spearman_r'), rss_l15.get('spearman_p'))} \\\\")
            lines.append(line)
        lines.append(r"\midrule")

    # Render Summary (Average) section
    if summary_rows:
        lines.append(
            r"\multicolumn{9}{l}{\textit{Summary (Average across all tasks)}} \\")
        lines.append(r"\midrule")
        for row in summary_rows:
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            # Determine which are the max values for bold formatting (compare all 4 metrics)
            pearson_vals = [srs_neg.get('pearson_r'), srs_pos.get('pearson_r'),
                            srs_mean.get('pearson_r'), rss_l15.get('pearson_r')]
            spearman_vals = [srs_neg.get('spearman_r'), srs_pos.get('spearman_r'),
                             srs_mean.get('spearman_r'), rss_l15.get('spearman_r')]

            max_pearson = max(
                [v for v in pearson_vals if v is not None], default=None)
            max_spearman = max(
                [v for v in spearman_vals if v is not None], default=None)

            line = (f"Average & "
                    f"{fmt_bold_stars(srs_neg.get('pearson_r'), srs_neg.get('pearson_p'), srs_neg.get('pearson_r') == max_pearson)} & "
                    f"{fmt_bold_stars(srs_pos.get('pearson_r'), srs_pos.get('pearson_p'), srs_pos.get('pearson_r') == max_pearson)} & "
                    f"{fmt_bold_stars(srs_mean.get('pearson_r'), srs_mean.get('pearson_p'), srs_mean.get('pearson_r') == max_pearson)} & "
                    f"{fmt_bold_stars(rss_l15.get('pearson_r'), rss_l15.get('pearson_p'), rss_l15.get('pearson_r') == max_pearson)} & "
                    f"{fmt_bold_stars(srs_neg.get('spearman_r'), srs_neg.get('spearman_p'), srs_neg.get('spearman_r') == max_spearman)} & "
                    f"{fmt_bold_stars(srs_pos.get('spearman_r'), srs_pos.get('spearman_p'), srs_pos.get('spearman_r') == max_spearman)} & "
                    f"{fmt_bold_stars(srs_mean.get('spearman_r'), srs_mean.get('spearman_p'), srs_mean.get('spearman_r') == max_spearman)} & "
                    f"{fmt_bold_stars(rss_l15.get('spearman_r'), rss_l15.get('spearman_p'), rss_l15.get('spearman_r') == max_spearman)} \\\\")
            lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    return "\n".join(lines)


def _render_latex_appendix_detail(ir_rows: list, correlation_type: str) -> str:
    """
    Appendix Renderer: Generates detailed table with raw r/p or ρ/p values.

    Args:
        ir_rows: Intermediate representation data
        correlation_type: 'pearson' or 'spearman'

    Returns:
        LaTeX table string with 9 columns: Task + 4 pairs of (r, p) or (ρ, p)
    """
    is_pearson = correlation_type == 'pearson'
    r_symbol = "r" if is_pearson else r"\rho"
    r_key = 'pearson_r' if is_pearson else 'spearman_r'
    p_key = 'pearson_p' if is_pearson else 'spearman_p'

    def fmt_val(val):
        if val is None:
            return "--"
        return f"{val:.4f}"

    def fmt_p(val):
        if val is None:
            return "--"
        if val < 0.0001:
            return "$<$0.0001"
        return f"{val:.4f}"

    lines = [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Detailed " + ("Pearson" if is_pearson else "Spearman") +
        r" correlation coefficients and p-values for all metrics.}",
        r"\label{tbl:" + ("pearson" if is_pearson else "spearman") +
        r"-correl-appendix}",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$\mathbf{srs}_{\text{neg}}$} & \multicolumn{2}{c}{$\mathbf{srs}_{\text{pos}}$} & \multicolumn{2}{c}{$\mathbf{srs}_{\text{mean}}$} & \multicolumn{2}{c}{$\text{RSS}_{L=15}$} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}",
        f"Task & ${r_symbol}$ & $p$ & ${r_symbol}$ & $p$ & ${r_symbol}$ & $p$ & ${r_symbol}$ & $p$ \\\\",
        r"\midrule"
    ]

    # Group rows by type
    synthetic_rows = [r for r in ir_rows if r['task_type'] == 'Synthetic']
    real_rows = [r for r in ir_rows if r['task_type'] == 'Real']
    summary_rows = [r for r in ir_rows if r['row_type'] == 'average_header']

    # Render Synthetic Tasks section
    if synthetic_rows:
        lines.append(r"\multicolumn{9}{l}{\textit{Synthetic Tasks}} \\")
        lines.append(r"\midrule")
        for row in synthetic_rows:
            task = row['task']
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            line = (f"{task} & "
                    f"{fmt_val(srs_neg.get(r_key))} & {fmt_p(srs_neg.get(p_key))} & "
                    f"{fmt_val(srs_pos.get(r_key))} & {fmt_p(srs_pos.get(p_key))} & "
                    f"{fmt_val(srs_mean.get(r_key))} & {fmt_p(srs_mean.get(p_key))} & "
                    f"{fmt_val(rss_l15.get(r_key))} & {fmt_p(rss_l15.get(p_key))} \\\\")
            lines.append(line)
        lines.append(r"\midrule")

    # Render Real Tasks section
    if real_rows:
        lines.append(r"\multicolumn{9}{l}{\textit{Real Tasks}} \\")
        lines.append(r"\midrule")
        for row in real_rows:
            task = row['task']
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            line = (f"{task} & "
                    f"{fmt_val(srs_neg.get(r_key))} & {fmt_p(srs_neg.get(p_key))} & "
                    f"{fmt_val(srs_pos.get(r_key))} & {fmt_p(srs_pos.get(p_key))} & "
                    f"{fmt_val(srs_mean.get(r_key))} & {fmt_p(srs_mean.get(p_key))} & "
                    f"{fmt_val(rss_l15.get(r_key))} & {fmt_p(rss_l15.get(p_key))} \\\\")
            lines.append(line)
        lines.append(r"\midrule")

    # Render Summary (Average) section
    if summary_rows:
        lines.append(
            r"\multicolumn{9}{l}{\textit{Summary (Average across all tasks)}} \\")
        lines.append(r"\midrule")
        for row in summary_rows:
            sub = {s['metric_key']: s for s in row['sub_rows']}

            srs_neg = sub.get('srs_neg', {})
            srs_pos = sub.get('srs_pos', {})
            srs_mean = sub.get('srs_mean', {})
            rss_l15 = sub.get('rss_l15', {})

            line = (f"Average & "
                    f"{fmt_val(srs_neg.get(r_key))} & {fmt_p(srs_neg.get(p_key))} & "
                    f"{fmt_val(srs_pos.get(r_key))} & {fmt_p(srs_pos.get(p_key))} & "
                    f"{fmt_val(srs_mean.get(r_key))} & {fmt_p(srs_mean.get(p_key))} & "
                    f"{fmt_val(rss_l15.get(r_key))} & {fmt_p(rss_l15.get(p_key))} \\\\")
            lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    return "\n".join(lines)


def generate_table_4_master_appendix_pearson(directory: str, config: dict = None) -> str:
    """
    Public API: Generate appendix table with detailed Pearson correlations.

    Args:
        directory: Root directory containing benchmark JSON files
        config: Configuration dict with optional 'exclude_models' list

    Returns:
        LaTeX table string with all Pearson r and p-values
    """
    if config is None:
        config = {'exclude_models': []}

    all_data = gather_all_data(directory)
    if not all_data:
        print("No data found.")
        return ""

    ir_rows = _process_data_into_ir(all_data, config)
    return _render_latex_appendix_detail(ir_rows, 'pearson')


def generate_table_4_master_appendix_spearman(directory: str, config: dict = None) -> str:
    """
    Public API: Generate appendix table with detailed Spearman correlations.

    Args:
        directory: Root directory containing benchmark JSON files
        config: Configuration dict with optional 'exclude_models' list

    Returns:
        LaTeX table string with all Spearman ρ and p-values
    """
    if config is None:
        config = {'exclude_models': []}

    all_data = gather_all_data(directory)
    if not all_data:
        print("No data found.")
        return ""

    ir_rows = _process_data_into_ir(all_data, config)
    return _render_latex_appendix_detail(ir_rows, 'spearman')


def generate_table_4_master_correl(directory: str, config: dict = None) -> tuple:
    """
    Public API: Orchestrates the 3-layer pipeline.

    Args:
        directory: Root directory containing benchmark JSON files
        config: Configuration dict with optional 'exclude_models' list

    Returns:
        (raw_data, latex_table, markdown_table) tuple
    """
    if config is None:
        config = {'exclude_models': []}

    # Layer 1: Load data
    all_data = gather_all_data(directory)

    if not all_data:
        print("No data found.")
        return [], "", ""

    # Layer 2: Process into IR
    ir_rows = _process_data_into_ir(all_data, config)

    # Layer 3: Render
    md_table = _render_markdown(ir_rows)
    latex_table = _render_latex(ir_rows)

    return ir_rows, latex_table, md_table

