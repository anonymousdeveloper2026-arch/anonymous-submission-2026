"""
Sweep probe-rank-vs-cluster across all models, lengths, samples, and sub-tests.

For every (model, L, sample_idx, probe_len l, probe_pos x):
  - L2-normalize all vectors
  - compute the L2-renormalized centroid of the positive cluster and the
    negative cluster (independently)
  - rank the probe against the cluster members by cosine similarity to the
    centroid, where rank=1 means probe is the FARTHEST from the centroid
    (i.e. most outside the cluster) and rank=N+1 means probe is the closest
    (i.e. dead-center inside the cluster)
  - tie-breaking: < (ties count against the probe → larger rank)

Output: a single Parquet file with one row per (model, L, sample, l, x),
plus a sidecar JSON with sweep metadata.
"""

import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd





# =============================================================================
# Loaders (mirrors first script)
# =============================================================================
def gather_srs_pt_paths(config: dict):
    """Walk the benchmark directory and return {model_name: pt_path}."""
    dir_path = Path(config["benchmark_dir"])
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{config['benchmark_dir']}' does not exist.")

    out = {}
    for pt_path in dir_path.rglob("*.pt"):
        if pt_path.parent.name == "srs":
            model_name = pt_path.parent.parent.name
            out[model_name] = pt_path
    return out


def load_srs_length_results(pt_path):
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    raw = data.get("raw_results", data)
    return raw.get("length_results", raw)

# =============================================================================
# Core: rank computation
# =============================================================================
def _l2_normalize(x, eps=1e-12):
    """L2-normalize a 1-D or 2-D numpy array along the last axis."""
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norm, eps, None)


def rank_probe_in_cluster(probe_np, cluster_np):
    """
    probe_np: shape (D,)
    cluster_np: shape (N, D)

    Returns (rank, max_rank) where:
      - both probe and cluster vectors are L2-normalized
      - centroid = mean of normalized cluster vectors, then L2-renormalized
      - rank = 1  -> probe is the FARTHEST point from centroid
      - rank = N+1 -> probe is the CLOSEST point to centroid (dead center)
      - tie-breaking: < (ties count toward the probe, i.e. push rank DOWN)
    """
    probe_n = _l2_normalize(probe_np)
    cluster_n = _l2_normalize(cluster_np)

    centroid = cluster_n.mean(axis=0)
    centroid = _l2_normalize(centroid)

    # cosine sim == dot product since everything is unit-norm
    sim_probe = float(np.dot(probe_n, centroid))
    sim_members = cluster_n @ centroid  # shape (N,)

    # rank from the FAR end: count members that are FARTHER-or-equal from
    # centroid than the probe is. Those members sit "below" the probe in
    # the far-to-near ordering, so the probe ranks just above them.
    # - probe extremely far (sim_probe ≈ 0): no member is farther → rank = 1
    # - probe dead center (sim_probe ≈ 1): all members are farther → rank = N+1
    # Tie-breaking: < means ties count toward the probe (push rank up).
    n_farther_or_equal = int(np.sum(sim_members < sim_probe))

    rank = n_farther_or_equal + 1
    max_rank = len(cluster_np) + 1
    return rank, max_rank, centroid


# =============================================================================
# Sweep
# =============================================================================
def iter_sub_tests(target_length):
    """Yield (probe_len, probe_pos) pairs: l ∈ [2, L-1], x ∈ [0, L-l]."""
    for l in range(target_length - 1, 1, -1):
        for x in range(0, target_length - l + 1):
            yield l, x


def process_subtest(probe_data, probe_pos):
    """Pull (probe_vec, pos_vecs, neg_vecs) for one (probe_len, probe_pos).
    Returns None if data is missing/empty."""
    if not isinstance(probe_data, dict) or "probe_pos_embeddings" not in probe_data:
        return None
    pos_embs_dict = probe_data["probe_pos_embeddings"]
    pp_key = str(probe_pos)
    if pp_key not in pos_embs_dict:
        return None

    pos_data = pos_embs_dict[pp_key]
    pos_vecs = pos_data.get("positive_embeddings", [])
    neg_vecs = pos_data.get("negative_embeddings", [])
    if not pos_vecs or not neg_vecs:
        return None

    probe_vec = probe_data["probe_embedding"]
    probe_np = probe_vec.numpy() if hasattr(probe_vec, "numpy") else np.asarray(probe_vec)
    pos_np = np.stack([
        v.numpy() if hasattr(v, "numpy") else np.asarray(v) for v in pos_vecs
    ])
    neg_np = np.stack([
        v.numpy() if hasattr(v, "numpy") else np.asarray(v) for v in neg_vecs
    ])
    return probe_np, pos_np, neg_np


def sweep_one_model(model_name, length_results):
    """Yield row dicts for every (L, sample_idx, l, x) in this model."""
    for L_key, L_block in length_results.items():
        # length_results keys may be int or str depending on how it was saved
        try:
            L = int(L_key)
        except (ValueError, TypeError):
            continue
        if L < 3:  # need at least l=2, x=0..1
            continue

        samples = L_block.get("sample_results", [])
        for sample_idx, sample in enumerate(samples):
            if not isinstance(sample, dict):
                continue
            for l, x in iter_sub_tests(L):
                pl_key = str(l)
                if pl_key not in sample:
                    continue
                probe_data = sample[pl_key]
                pulled = process_subtest(probe_data, x)
                if pulled is None:
                    continue
                probe_np, pos_np, neg_np = pulled

                rank_pos, max_rank_pos, centroid_pos = rank_probe_in_cluster(probe_np, pos_np)
                rank_neg, max_rank_neg, centroid_neg = rank_probe_in_cluster(probe_np, neg_np)

                # Calculate cosine similarity between positive and negative cluster centroids
                sim_centroids = float(np.dot(centroid_pos, centroid_neg))

                yield {
                    "model": model_name,
                    "target_length": L,
                    "sample_idx": sample_idx,
                    "probe_len": l,
                    "probe_pos": x,
                    "n_pos": int(pos_np.shape[0]),
                    "n_neg": int(neg_np.shape[0]),

                    "rank_pos": rank_pos,
                    "max_rank_pos": max_rank_pos,
                    "rank_neg": rank_neg,
                    "max_rank_neg": max_rank_neg,
                    "sim_centroids": sim_centroids,
                }
# =============================================================================
# Main
# =============================================================================
def index_rank(config: dict):
    """Execute the probe-rank index sweep with the given configuration."""
    # Fail fast if pyarrow is missing — silent TSV fallback would be a debugging trap.
    try:
        import pyarrow  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "pyarrow is required to write Parquet output. "
            "Install with: pip install pyarrow"
        ) from e

    t_start = time.time()
    print(f"Gathering SRS .pt files from '{config['benchmark_dir']}'...")
    pt_paths = gather_srs_pt_paths(config)
    print(f"Found {len(pt_paths)} model(s) with SRS .pt files.")

    excluded_set = set(config.get("excluded_models", []))
    if excluded_set:
        print(f"Exclusion list: {sorted(excluded_set)}")

    all_rows = []
    skipped = {}  # model -> reason
    excluded_hits = []
    per_model_counts = {}

    for i, (model_name, pt_path) in enumerate(sorted(pt_paths.items()), 1):
        if model_name in excluded_set:
            print(f"\n[{i}/{len(pt_paths)}] {model_name}  -- EXCLUDED, skipping")
            excluded_hits.append(model_name)
            skipped[model_name] = "excluded_by_config"
            continue

        print(f"\n[{i}/{len(pt_paths)}] {model_name}")
        try:
            length_results = load_srs_length_results(pt_path)
        except Exception as e:
            print(f"  ✗ failed to load .pt: {e}")
            skipped[model_name] = f"load_error: {e}"
            continue

        try:
            model_rows = list(sweep_one_model(model_name, length_results))
        except Exception as e:
            print(f"  ✗ sweep error: {e}")
            skipped[model_name] = f"sweep_error: {e}"
            continue

        if not model_rows:
            print("  ! no rows produced (empty length_results?)")
            skipped[model_name] = "no_rows"
            continue

        # Quick per-model summary
        Ls = sorted({r["target_length"] for r in model_rows})
        n_samples_per_L = {
            L: len({r["sample_idx"] for r in model_rows if r["target_length"] == L})
            for L in Ls
        }
        per_model_counts[model_name] = len(model_rows)
        print(
            f"  ✓ {len(model_rows):>6d} rows  |  "
            f"L = {Ls}  |  samples/L = {n_samples_per_L}"
        )
        all_rows.extend(model_rows)

    # Sanity-check the exclusion list — warn if any configured name didn't match.
    unmatched = excluded_set - set(excluded_hits)
    if unmatched:
        print(
            f"\n!! warning: EXCLUDED_MODELS entries with no match in benchmark dir: "
            f"{sorted(unmatched)}"
        )

    if not all_rows:
        raise SystemExit("No rows collected from any model. Aborting.")

    print(f"\nTotal rows: {len(all_rows):,}")
    print("Building DataFrame...")
    df = pd.DataFrame(all_rows)

    # Tighten dtypes — these are all small ints, no need for int64.
    int_cols = [
        "target_length", "sample_idx", "probe_len", "probe_pos",
        "n_pos", "n_neg", "rank_pos", "max_rank_pos", "rank_neg", "max_rank_neg",
    ]

    for c in int_cols:
        df[c] = df[c].astype("int32")
    df["sim_centroids"] = df["sim_centroids"].astype("float32")
    df["model"] = df["model"].astype("category")

    os.makedirs(os.path.dirname(config["output_parquet"]) or ".", exist_ok=True)
    df.to_parquet(config["output_parquet"], index=False, compression="zstd")
    print(f"Wrote: {config['output_parquet']}  ({os.path.getsize(config['output_parquet']) / 1024:.1f} KiB)")

    elapsed = time.time() - t_start
    meta = {
        "benchmark_dir": str(Path(config["benchmark_dir"]).resolve()),
        "output_parquet": str(Path(config["output_parquet"]).resolve()),
        "n_models_found": len(pt_paths),
        "n_models_processed": len(per_model_counts),
        "n_models_skipped": len(skipped),
        "excluded_models_config": sorted(excluded_set),
        "excluded_models_matched": sorted(excluded_hits),
        "excluded_models_unmatched": sorted(unmatched),
        "skipped": skipped,
        "per_model_row_counts": per_model_counts,
        "total_rows": len(all_rows),
        "tie_breaking": "<  (ties push rank down, toward the probe)",
        "rank_convention": (
            "rank=1  -> probe is the FARTHEST from centroid (most outside cluster); "
            "rank=N+1 -> probe is the CLOSEST to centroid (dead-center inside cluster)"
        ),
        "normalization": "L2-normalize all vectors; centroid = mean(normed) then L2-renormalized",
        "sim_centroids": "Cosine similarity between pos and neg cluster centroids",
        "elapsed_seconds": round(elapsed, 2),
    }
    with open(config["output_meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Wrote: {config['output_meta']}")
    print(f"\nDone in {elapsed:.1f}s.")


if __name__ == "__main__":
    """Initialize default configuration and run the index rank sweep."""
    config = {
        "benchmark_dir": "benchmarks",
        "output_parquet": "source/index/probe_rank_sweep.parquet",
        "output_meta": "source/index/probe_rank_sweep_meta.json",
        "excluded_models": [
            "average-synth_multilingual-e5-base",
        ]
    }
    index_rank(config)