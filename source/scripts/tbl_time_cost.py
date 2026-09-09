"""
Table 17: Benchmark Wall-Clock Time Cost Statistics.

This script is a streamlined artifact generator that:
1. Gathers total_wall_clock_elapsed from finesse (RSS/SRS) and LEMB report.txt
2. Calculates n, mean, median, std, IQR (and 1.5x-IQR outliers) per metric
3. Reports time in minutes as well as raw seconds
4. Returns LaTeX and Markdown tables

Metrics:
- RSS : benchmarks/finesse/model_eval/<model>/rss/report.txt   (L=4..16 full run)
- SRS : benchmarks/finesse/model_eval/<model>/srs/report.txt   (L=8 full run)
- LEMB: benchmarks/lemb/model_eval/<model>/report.txt          (6-task full run)

All report.txt files are JSON with evaluation.total_wall_clock_elapsed.
"""

import json
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Tuple, Any

# =============================================================================
# Configuration
# =============================================================================

METRICS = ["RSS", "SRS", "LEMB"]

DEFAULT_EXCLUDE_MODELS = ["average-synth_multilingual-e5-base"]


# =============================================================================
# Statistics Helpers (stdlib only, no numpy dependency)
# =============================================================================

def _percentile(sorted_vals: List[float], p: float) -> float:
    """Linear-interpolation percentile over an already-sorted list."""
    if not sorted_vals:
        raise ValueError("empty list")
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_vals) else f
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def _stats(vals: List[float]) -> Dict[str, Any]:
    """Compute n, mean, median, std, IQR, min/max and 1.5*IQR outliers."""
    sv = sorted(vals)
    q1 = _percentile(sv, 25.0)
    q3 = _percentile(sv, 75.0)
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return {
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "std": pstdev(vals),
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "min": min(vals),
        "max": max(vals),
        "outliers": [v for v in vals if v < lo or v > hi],
    }


def _load_wall_clock(path: Path) -> float:
    """Read wall-clock elapsed seconds from a report.txt (JSON)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data["evaluation"]["total_wall_clock_elapsed"])


# =============================================================================
# Data Processing
# =============================================================================

def _calculate_metrics(directory: str, config: dict = None) -> Dict[str, Any]:
    """
    Gather wall-clock times for RSS, SRS, LEMB across all models.

    Returns dict keyed by metric with per-metric statistics, plus:
    - 'raw_seconds': per-model raw seconds for each metric
    - 'model_names': ordered model names
    - 'ratios': mean-of-means and per-model ratio means
    - 'metadata'
    """
    config = config or {}
    exclude = set(config.get("exclude_models", DEFAULT_EXCLUDE_MODELS))

    root = Path(directory)
    fin = root / "finesse" / "model_eval" / "finesse-main"
    lemb = root / "lemb" / "model_eval" / "lemb-main" / "included"

    if not fin.exists() or not lemb.exists():
        raise FileNotFoundError(
            f"Expected directories missing under '{directory}': "
            f"finesse/model_eval={fin.exists()}, lemb/model_eval={lemb.exists()}")

    names: List[str] = []
    raw = {"RSS": [], "SRS": [], "LEMB": []}

    for d in sorted(fin.iterdir()):
        if not d.is_dir():
            continue
        model = d.name
        if model in exclude:
            print(f"  o Excluded model: {model}")
            continue

        lembtxt = lemb / model / "report.txt"
        rss_txt = d / "rss" / "report.txt"
        srs_txt = d / "srs" / "report.txt"

        if not (lembtxt.exists() and rss_txt.exists() and srs_txt.exists()):
            print(f"  x Skipping {model} (missing report.txt)")
            continue

        names.append(model)
        raw["RSS"].append(_load_wall_clock(rss_txt))
        raw["SRS"].append(_load_wall_clock(srs_txt))
        raw["LEMB"].append(_load_wall_clock(lembtxt))

    if not names:
        raise ValueError("No models with complete RSS/SRS/LEMB report.txt found.")

    stats = {m: _stats(raw[m]) for m in METRICS}

    # Ratios (of means)
    m_r, m_s, m_l = stats["RSS"]["mean"], stats["SRS"]["mean"], stats["LEMB"]["mean"]
    ratios_of_means = {
        "LEMB/RSS": m_l / m_r,
        "SRS/RSS": m_s / m_r,
        "SRS/LEMB": m_s / m_l,
        "RSS/LEMB": m_r / m_l,
    }

    # Per-model ratio means
    def _ratio_mean(a, b):
        r = [x / y for x, y in zip(raw[a], raw[b])]
        return {"mean": mean(r), "median": median(r)}

    per_model_ratios = {
        "LEMB/RSS": _ratio_mean("LEMB", "RSS"),
        "SRS/RSS": _ratio_mean("SRS", "RSS"),
        "SRS/LEMB": _ratio_mean("SRS", "LEMB"),
        "RSS/LEMB": _ratio_mean("RSS", "LEMB"),
    }

    return {
        "stats": stats,
        "raw_seconds": raw,
        "model_names": names,
        "ratios_of_means": ratios_of_means,
        "per_model_ratios": per_model_ratios,
        "metadata": {
            "n_models": len(names),
            "excluded_models": list(exclude),
            "directory": directory,
        },
    }


# =============================================================================
# LaTeX Rendering
# =============================================================================

def _fmt_min(v: float) -> str:
    return f"{v / 60.0:.2f}"


def _render_latex(raw_data: Dict[str, Any]) -> str:
    """Render wall-clock statistics as LaTeX (times in minutes)."""
    stats = raw_data["stats"]
    rows = []
    for m in METRICS:
        s = stats[m]
        rows.append(
            f"{m:4s} & {s['n']:2d} & {_fmt_min(s['mean'])} & {_fmt_min(s['median'])} "
            f"& {_fmt_min(s['std'])} & {_fmt_min(s['iqr'])} "
            f"& {_fmt_min(s['min'])} & {_fmt_min(s['max'])} \\"
        )
    rows_str = "\n".join(rows)

    return (
        "\\begin{table}[ht!]\n"
        "\\centering\n"
        "\\caption{Benchmark wall-clock time (minutes) across "
        + str(raw_data['metadata']['n_models']) + " models.}\n"
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Metric & $n$ & Mean & Median & Std & IQR & Min & Max \\\n"
        "\\midrule\n"
        + rows_str +
        "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _render_markdown(raw_data: Dict[str, Any]) -> str:
    """Render wall-clock statistics as Markdown (times in minutes)."""
    stats = raw_data["stats"]
    lines = [
        "| Metric | n | Mean | Median | Std | IQR | Min | Max |",
        "|--------|---|------|--------|-----|-----|-----|-----|",
    ]
    for m in METRICS:
        s = stats[m]
        lines.append(
            f"| {m} | {s['n']} | {_fmt_min(s['mean'])} | {_fmt_min(s['median'])} "
            f"| {_fmt_min(s['std'])} | {_fmt_min(s['iqr'])} "
            f"| {_fmt_min(s['min'])} | {_fmt_min(s['max'])} |"
        )
    lines.append("")
    lines.append(
        f"**Note:** Times in minutes, over {raw_data['metadata']['n_models']} models. "
        f"1.5x-IQR outlier counts -- RSS: {len(stats['RSS']['outliers'])}, "
        f"SRS: {len(stats['SRS']['outliers'])}, LEMB: {len(stats['LEMB']['outliers'])}.")
    return "\n".join(lines)


# =============================================================================
# Master Orchestrator
# =============================================================================

def generate_table_time_cost(
    directory: str,
    config: dict = None
) -> Tuple[Dict[str, Any], str, str]:
    """Generate Table 17: benchmark wall-clock statistics.

    Returns (raw_data, latex_string, markdown_string).
    """
    if config is None:
        config = {"exclude_models": DEFAULT_EXCLUDE_MODELS}

    print("=" * 60)
    print("TABLE 17: BENCHMARK WALL-CLOCK TIME COST STATISTICS")
    print("=" * 60)
    print(f"Scanning directory: {directory}")
    print()

    raw_data = _calculate_metrics(directory, config)

    print(f"\nProcessed {raw_data['metadata']['n_models']} models.")
    print("\nStatistics (minutes):")
    for m in METRICS:
        s = raw_data["stats"][m]
        print(f"  {m:4s}: n={s['n']}, mean={s['mean']/60:.2f}, "
              f"median={s['median']/60:.2f}, std={s['std']/60:.2f}, "
              f"IQR={s['iqr']/60:.2f}, min={s['min']/60:.2f}, max={s['max']/60:.2f}, "
              f"outliers={len(s['outliers'])}")

    print("\nRatios (of means):")
    for k, v in raw_data["ratios_of_means"].items():
        print(f"  {k:10s} = {v:.3f}")

    print("\nPer-model ratio means:")
    for k, v in raw_data["per_model_ratios"].items():
        print(f"  {k:10s} = mean {v['mean']:.3f}, median {v['median']:.3f}")

    print("\nRendering LaTeX and Markdown...")
    latex_string = _render_latex(raw_data)
    markdown_string = _render_markdown(raw_data)
    return raw_data, latex_string, markdown_string


if __name__ == "__main__":
    raw_data, latex_str, md_str = generate_table_time_cost("benchmarks")
    print("\n" + "=" * 60)
    print("GENERATED LATEX:")
    print("=" * 60)
    print(latex_str)
