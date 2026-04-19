import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from pathlib import Path


# =============================================================================
# Loaders
# =============================================================================
def load_srs_length_results(pt_path):
    """Load SRS length results from .pt file."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    raw = data.get("raw_results", data)
    return raw.get("length_results", raw)


def extract_subtest(length_results, target_length, probe_len, probe_pos, sample_idx):
    """Pull (probe_vec, pos_vecs, neg_vecs) for one sub-test."""
    if target_length not in length_results:
        if str(target_length) in length_results:
            target_length = str(target_length)
        else:
            raise KeyError(f"L={target_length} not in length_results")

    samples = length_results[target_length].get("sample_results", [])
    if sample_idx >= len(samples):
        raise IndexError(f"sample {sample_idx} out of range (have {len(samples)})")

    sample = samples[sample_idx]
    probe_data = sample[str(probe_len)]
    probe_vec = probe_data["probe_embedding"]

    pos_embs_dict = probe_data["probe_pos_embeddings"]
    pos_data = pos_embs_dict[str(probe_pos)]

    pos_vecs = pos_data["positive_embeddings"]
    neg_vecs = pos_data["negative_embeddings"]

    return probe_vec, pos_vecs, neg_vecs


def compute_cos_stats(probe_vec, pos_vecs, neg_vecs):
    """
    Per-vector cosine similarity from probe to each cluster member,
    computed in the ORIGINAL embedding space (pre-PCA).
    """
    probe = probe_vec.unsqueeze(0).float()
    pos_t = torch.stack(pos_vecs).float()
    neg_t = torch.stack(neg_vecs).float()

    sim_pos = F.cosine_similarity(probe, pos_t, dim=1).numpy()
    sim_neg = F.cosine_similarity(probe, neg_t, dim=1).numpy()

    srs_score = (np.median(sim_pos) - np.median(sim_neg)) * 1000.0

    return {
        "sim_pos": sim_pos,
        "sim_neg": sim_neg,
        "pos_mean": float(sim_pos.mean()),
        "pos_std": float(sim_pos.std()),
        "neg_mean": float(sim_neg.mean()),
        "neg_std": float(sim_neg.std()),
        "srs_score": srs_score,
    }


# =============================================================================
# Plot
# =============================================================================
def _norm(arr):
    """Normalize array to [0, 1] range."""
    if arr.max() - arr.min() < 1e-9:
        return np.full_like(arr, 0.5)
    return (arr - arr.min()) / (arr.max() - arr.min())


def plot_twcp_3d(ax, probe_3d, pos_3d, neg_3d, cos_stats, title):
    """Plot single TWCP 3D visualization on given axis."""
    # probe — gold star
    ax.scatter(*probe_3d.T, s=550, c="gold", marker="*",
               edgecolor="black", linewidth=1.5, zorder=10, label="Probe")

    sim_pos = cos_stats["sim_pos"]
    sim_neg = cos_stats["sim_neg"]

    # Higher similarity → larger marker (closer to probe in cosine sense)
    pos_sizes = 80 + 200 * _norm(sim_pos)
    neg_sizes = 80 + 200 * _norm(sim_neg)

    # positive cluster — blue circles
    ax.scatter(*pos_3d.T, s=pos_sizes, c="dodgerblue", alpha=0.7,
               edgecolor="navy", linewidth=0.5, label=f"Positive [{len(sim_pos)}]")

    # negative cluster — red triangles
    ax.scatter(*neg_3d.T, s=neg_sizes, c="tomato", alpha=0.7, marker="^",
               edgecolor="darkred", linewidth=0.5, label=f"Negative [{len(sim_neg)}]")

    # Lines from probe to each point (faint, for context)
    for pt in pos_3d:
        ax.plot(
            [probe_3d[0, 0], pt[0]],
            [probe_3d[0, 1], pt[1]],
            [probe_3d[0, 2], pt[2]],
            color="dodgerblue", alpha=0.15, linewidth=0.5,
        )
    for pt in neg_3d:
        ax.plot(
            [probe_3d[0, 0], pt[0]],
            [probe_3d[0, 1], pt[1]],
            [probe_3d[0, 2], pt[2]],
            color="tomato", alpha=0.15, linewidth=0.5,
        )

    # Cluster centroids in the 3D view, for inline label placement
    pos_centroid = pos_3d.mean(axis=0)
    neg_centroid = neg_3d.mean(axis=0)

    # Midpoint between probe and each centroid
    pos_mid = (probe_3d[0] + pos_centroid) / 2.0
    neg_mid = (probe_3d[0] + neg_centroid) / 2.0

    pos_mid[2] += 0.04
    neg_mid[2] -= 0.04

    # Inline cosine labels (mean only)
    ax.text(
        pos_mid[0], pos_mid[1], pos_mid[2],
        f" 1 - cos = {(1 - cos_stats['pos_mean']):.3f}",
        color="navy", fontsize=10, fontweight="bold",
        ha="center", va="center", zorder=20,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="navy", alpha=0.85, linewidth=0.8),
    )
    ax.text(
        neg_mid[0], neg_mid[1], neg_mid[2],
        f" 1 - cos = {(1 - cos_stats['neg_mean']):.3f}",
        color="darkred", fontsize=10, fontweight="bold",
        ha="center", va="center", zorder=20,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="darkred", alpha=0.85, linewidth=0.8),
    )

    # Top-left info box: cosine mean ± std
    info = (
        f"cos(probe, pos): {cos_stats['pos_mean']:.4f} ± {cos_stats['pos_std']:.4f}\n"
        f"cos(probe, neg): {cos_stats['neg_mean']:.4f} ± {cos_stats['neg_std']:.4f}\n"
        f"Δ(pos - neg)   : {cos_stats['pos_mean'] - cos_stats['neg_mean']:+.4f}"
    )
    ax.text2D(
        0.02, 0.98, info,
        transform=ax.transAxes, fontsize=8, fontfamily="monospace",
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
    )

    ax.set_xlabel("PCA 1", fontsize=12)
    ax.set_ylabel("PCA 2", fontsize=12)
    ax.set_zlabel("PCA 3", fontsize=12)
    ax.legend(loc="upper right", fontsize=8)


# =============================================================================
# Main Generator
# =============================================================================
def generate_twcp_figure(config: dict, output_dir: str) -> str:
    """
    Generate TWCP (Two-Cluster Problem) theory figure.
    
    Args:
        config: Configuration dictionary with keys:
            - pt_path: Path to .pt file containing SRS embeddings
            - output_dir: Directory to save output SVG
            - target_length: Target sequence length (L)
            - probe_len: Probe length (l)
            - probe_pos: Probe position (x)
            - sample_idx: Sample index (default: 0)
    
    Returns:
        Path to the generated SVG file
    """
    # Extract configuration
    pt_path = config.get("pt_path")
    target_length = config.get("target_length")
    probe_len = config.get("probe_len")
    probe_pos = config.get("probe_pos")
    sample_idx = config.get("sample_idx", 0)
    
    # Validate required parameters
    if not all([pt_path, output_dir, target_length is not None, 
                probe_len is not None, probe_pos is not None]):
        raise ValueError("Missing required config parameters: pt_path, output_dir, "
                        "target_length, probe_len, probe_pos")
    
    # Generate output filename programmatically
    output_filename = f"twcp_3d.svg"
    output_path = Path(output_dir) / output_filename
    
    print(f"=" * 60)
    print("TWCP FIGURE GENERATION")
    print(f"=" * 60)
    print(f"Loading: {pt_path}")
    
    # Load data
    length_results = load_srs_length_results(pt_path)
    
    print(f"Extracting sub-test: L={target_length}, l={probe_len}, x={probe_pos}")
    probe_vec, pos_vecs, neg_vecs = extract_subtest(
        length_results, target_length, probe_len, probe_pos, sample_idx
    )
    
    # Compute cosine statistics
    cos_stats = compute_cos_stats(probe_vec, pos_vecs, neg_vecs)
    
    print(f"  SRS = {cos_stats['srs_score']:+.2f}")
    print(f"  cos pos = {cos_stats['pos_mean']:.4f} ± {cos_stats['pos_std']:.4f}")
    print(f"  cos neg = {cos_stats['neg_mean']:.4f} ± {cos_stats['neg_std']:.4f}")
    
    # Prepare data for PCA
    probe_np = probe_vec.numpy()
    pos_np = np.stack([v.numpy() for v in pos_vecs])
    neg_np = np.stack([v.numpy() for v in neg_vecs])
    
    # Joint PCA fit
    joint = np.vstack([
        probe_np[None, :],
        pos_np,
        neg_np,
    ])
    
    print(f"\nFitting PCA on joint matrix: {joint.shape}")
    pca = PCA(n_components=3)
    projected = pca.fit_transform(joint)
    print(f"  explained variance ratio: {pca.explained_variance_ratio_.round(4).tolist()}")
    
    # Slice projected data
    n_pos = len(pos_vecs)
    n_neg = len(neg_vecs)
    
    probe_3d = projected[0:1]
    pos_3d = projected[1:1 + n_pos]
    neg_3d = projected[1 + n_pos:1 + n_pos + n_neg]
    
    # Create figure
    fig = plt.figure(figsize=(10, 9))

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    
    title = f"x = {probe_pos}"
    plot_twcp_3d(ax, probe_3d, pos_3d, neg_3d, cos_stats, title)
    
    # Set axis limits
    mins = projected.min(axis=0)
    maxs = projected.max(axis=0)
    pad = (maxs - mins) * 0.05
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    ax.view_init(elev=25, azim=135)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as SVG
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches="tight", pad_inches=0.3)
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return output_path

