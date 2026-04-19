import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path


def generate_rank_heatmap(input_parquet: str, output_dir: str) -> str:
    """
    Generate rank_pos vs rank_neg joint distribution heatmap.
    
    Args:
        input_parquet: Path to input parquet file containing rank data
        output_dir: Directory to save the output SVG
    
    Returns:
        Path to the generated SVG file
    """
    input_path = Path(input_parquet)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {input_parquet}")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine output filename
    output_file = output_path / f"{input_path.stem}_rank_heatmap.svg"
    
    print(f"=" * 60)
    print("RANK HEATMAP GENERATION")
    print(f"=" * 60)
    print(f"Loading: {input_parquet}")
    
    # Load data
    df = pd.read_parquet(input_parquet)
    total = len(df)
    print(f"  Total rows: {total:,}")
    
    # Get max rank values
    max_rank_pos = int(df["max_rank_pos"].iloc[0])
    max_rank_neg = int(df["max_rank_neg"].iloc[0])
    n_pos = max_rank_pos - 1
    n_neg = max_rank_neg - 1
    print(f"  Cluster sizes: N_pos = {n_pos}, N_neg = {n_neg}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 11))
    
    # Integer-centered bins from 1 to max_rank inclusive
    x_edges = np.arange(1, max_rank_pos + 2) - 0.5
    y_edges = np.arange(1, max_rank_neg + 2) - 0.5
    
    H, _, _ = np.histogram2d(
        df["rank_pos"].to_numpy(),
        df["rank_neg"].to_numpy(),
        bins=[x_edges, y_edges],
    )
    # H shape (n_x, n_y) -> transpose for pcolormesh (rows=y, cols=x)
    H_plot = H.T
    
    # Use viridis colormap with LogNorm
    cmap = plt.get_cmap("viridis").copy()
    mesh = ax.pcolormesh(
        x_edges, y_edges, H_plot,
        cmap=cmap, shading="flat", zorder=2,
        norm=colors.LogNorm(vmin=1, vmax=10000),
    )
    cbar = fig.colorbar(mesh, ax=ax, pad=0.015, fraction=0.04)
    
    # rank_pos == rank_neg diagonal
    diag_hi = min(max_rank_pos, max_rank_neg)
    ax.plot(
        [0.5, diag_hi + 0.5],
        [0.5, diag_hi + 0.5],
        color="white", linestyle="--", linewidth=1.2, alpha=0.8,
        zorder=3, label="rank_pos = rank_neg",
    )
    
    # Axis labels
    ax.set_xlabel(
        "negative rank",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylabel(
        "positive rank",
        fontsize=11, fontweight="bold"
    )
    ax.set_xlim(0.5, max_rank_pos + 0.5)
    ax.set_ylim(0.5, max_rank_neg + 0.5)
    ax.set_aspect("equal", adjustable="box")
    
    # ticks
    def _ticks(max_rank):
        if max_rank <= 26:
            return list(range(1, max_rank + 1))
        step = max(1, max_rank // 13)
        return list(range(1, max_rank + 1, step))
    ax.set_xticks(_ticks(max_rank_pos))
    ax.set_yticks(_ticks(max_rank_neg))
    
    plt.tight_layout()
    plt.savefig(output_file, format='svg', dpi=300, bbox_inches="tight")
    print(f"\nSaved: {output_file}")
    plt.close()
    
    return str(output_file)
