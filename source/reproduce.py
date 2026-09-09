import json
import os
from pathlib import Path

from packages.v_rank_index import index_rank

from scripts.tbl1_rss import generate_table_1_rss
from scripts.tbl2_lemb import generate_table_2_lemb
from scripts.tbl3_rss_lemb_correl import generate_table_3_rss_lemb_correl
from scripts.tbl4_master_correl import generate_table_4_master_correl, generate_table_4_master_appendix_pearson, generate_table_4_master_appendix_spearman
from scripts.tbl7_quadrant_analysis import generate_table_7_analysis
from scripts.tbl10_td_bu_rss_correl import generate_table_10_tdburss_analysis
from scripts.tbl11_lemb_iqr import generate_table_11_lemb_iqr
from scripts.tbl16_welch_ttest import generate_table_16_welch_ttest
from scripts.tbl_time_cost import generate_table_time_cost
from scripts.tbl_dim_ablation import generate_tbl_dim_partial_correl, MODEL_DIM_PARAMS

from scripts.fig_srs_rss import generate_srs_rss_figures
from scripts.fig_rank_heatmap import generate_rank_heatmap
from scripts.fig_twcp_theory import generate_twcp_figure
from scripts.fig_cross_corpus_rss import generate_cross_corpus_figure
from scripts.fig_lx_master_correl import generate_lx_master_figure
from scripts.fig_l15_scatter import generate_l15_lemb_scatters
from scripts.fig_lemb_statistic import generate_lemb_statistics
from scripts.fig_rss_jackknife import generate_rss_jackknife_figures
from scripts.fig_srs_jackknife import generate_srs_jackknife_figures


def save_artifacts(output_dir: str, raw_data, latex_table: str, md_table: str):
    """Save generated table artifacts to the specified directory.

    Args:
        output_dir: Directory path where artifacts will be saved
        raw_data: The raw data dictionary (will be saved as JSON)
        latex_table: LaTeX table string (will be saved as .tex)
        md_table: Markdown table string (will be saved as .md)
    """
    # Create output directory if it doesn't exist
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Save raw data as JSON
    if raw_data:
        raw_file = out_path / "raw_data.json"
        with open(raw_file, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved raw data: {raw_file}")

    # Save LaTeX table
    if latex_table:
        latex_file = out_path / "table.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"  ✓ Saved LaTeX table: {latex_file}")

    # Save Markdown table
    if md_table:
        md_file = out_path / "table.md"
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_table)
        print(f"  ✓ Saved Markdown table: {md_file}")

# ======== CONFIGURATION =========


BASELINE_MODEL_NAME = "average-synth"

BENCHMARK_EVALS_DIR = "evals/typical"
FINESSE_EVALS_DIR = "evals/typical/finesse"
LEMB_EVALS_DIR = "evals/typical/lemb"

SRS16_DIR = "evals/special/srs-16"

WIKIPEDIA_RSS15_DIR = "evals/special/rss-wikipedia"

RSS64_DIR = "evals/special/rss-64"

FINESSE_PT_FILES_DIR = "evals/typical/finesse/model_eval_pt"
FINESSE_SRS_QWEN_PATH = f"{FINESSE_PT_FILES_DIR}/finesse-pt-2/Qwen_Qwen3-Embedding-0.6B/srs/embeddings_native_mode_srs_Qwen_Qwen3-Embedding-0.6B.pt"
FINESSE_SRS_NEMOTRON_PATH = f"{FINESSE_PT_FILES_DIR}/finesse-pt-2/nvidia_llama-embed-nemotron-8b/srs/embeddings_native_mode_srs_nvidia_llama-embed-nemotron-8b.pt"


TWO_CLUSTER_AND_PROBE_DIR = "source/index/"
TWO_CLUSTER_AND_PROBE_META_PATH = f"{TWO_CLUSTER_AND_PROBE_DIR}probe_rank_sweep_meta.json"
TWO_CLUSTER_AND_PROBE_INDEX_PATH = f"{TWO_CLUSTER_AND_PROBE_DIR}probe_rank_sweep.parquet"

INDEX_CONFIG = {
    "benchmark_dir": FINESSE_PT_FILES_DIR,
    "output_parquet": TWO_CLUSTER_AND_PROBE_INDEX_PATH,
    "output_meta": TWO_CLUSTER_AND_PROBE_META_PATH,
    "excluded_models": [
        BASELINE_MODEL_NAME,
    ]
}

TBL_1_CONFIG = {
    "columns": {
        "target_lengths": [4, 8, 16],
        "avg_range": range(4, 17)  # L=4 to L=16 inclusive
    },
    "groups": [
        {
            "group_name": "Top 4 models ranked by avg. score",
            "models": [
                "nvidia_llama-embed-nemotron-8b",
                "codefuse-ai_F2LLM-v2-4B",
                "codefuse-ai_F2LLM-v2-1.7B",
                "codefuse-ai_F2LLM-v2-8B"
            ],
            "is_baseline": False
        },
        {
            "group_name": "The nomic-embed-text series",
            "models": [
                "nomic-ai_nomic-embed-text-v1",
                "nomic-ai_nomic-embed-text-v1.5"
            ],
            "is_baseline": False
        },
        {
            "group_name": "Baseline",
            "models": [BASELINE_MODEL_NAME],
            "is_baseline": True
        }
    ]
}

TBL_2_CONFIG = {
    "models": [
        "nomic-ai_nomic-embed-text-v1",
        "nomic-ai_nomic-embed-text-v1.5",
        BASELINE_MODEL_NAME
    ]
}

TBL_3_CONFIG = {
    "l_range": range(4, 17),
    "exclude_models": [BASELINE_MODEL_NAME]
}

TBL_4_CONFIG = {
    'exclude_models': [BASELINE_MODEL_NAME]
}

TBL_7_CONFIG = {
    "include_models": None,  # Use all models, or set to comma-separated string
    "target_models": [
        "nvidia_llama-embed-nemotron-8b",
        "nomic-ai_nomic-embed-text-v1.5",
        "Snowflake_snowflake-arctic-embed-l-v2.0",
        "Qwen_Qwen3-Embedding-4B",
        "bflhc_MoD-Embedding",
        "codefuse-ai_F2LLM-0.6B",
    ],
}

TBL_10_CONFIG = {
    'target_lengths': range(4, 17),
    'exclude_models': [BASELINE_MODEL_NAME],
}

TBL_11_CONFIG = {
    'exclude_models': [BASELINE_MODEL_NAME]
}

TBL_12_CONFIG = {
    "l_range": range(4, 17),
    'include_srs_neg_correlation': True,
    "exclude_models": [BASELINE_MODEL_NAME, "nomic-ai_nomic-embed-text-v1.5"]
}

TBL_13_CONFIG = {
    "l_range": range(4, 17),
    'include_srs_neg_correlation': True,
    "exclude_models": [
        BASELINE_MODEL_NAME,
        'codefuse-ai_F2LLM-0.6B',
        'codefuse-ai_F2LLM-1.7B',
        'codefuse-ai_F2LLM-4B',
        'codefuse-ai_F2LLM-v2-0.6B',
        'codefuse-ai_F2LLM-v2-1.7B',
        'codefuse-ai_F2LLM-v2-4B',
        'codefuse-ai_F2LLM-v2-8B'
    ]
}

TBL_14_CONFIG = {
    "heatmap": {"enabled": True},
    "columns": {
        "target_lengths": [4, 8, 16],
        "avg_range": range(4, 17)  # L=4 to L=16 inclusive
    },
    "groups": None
}

TBL_RSS64_CONFIG = {
    "heatmap": {"enabled": True},
    "columns": {
        "target_lengths": [8, 16, 24, 32, 40, 48, 56, 64],
        "show_avg": False,
        "avg_range": []
    },
    "groups": None
}

TBL_15_CONFIG = {
    "heatmap": {"enabled": True},
    "models": "all"
}

TBL_16_CONFIG = {
    'exclude_models': [BASELINE_MODEL_NAME],
    'alpha': 0.05,
}

TBL_DIM_CONFIG = {
    'target_lengths': list(range(4, 17)),
    'exclude_models': [BASELINE_MODEL_NAME],
    'dim_params_map': MODEL_DIM_PARAMS,
}


FIG_SRS_RSS_CONFIG = {
    "aliases": {"annamodels_LGAI-Embedding-Preview": "LGAI-Embedding-Preview", "Salesforce_SFR-Embedding-Mistral": "SFR-Embedding-Mistral", "Salesforce_SFR-Embedding-2_R": "SFR-Embedding-2_R", "Haon-Chen_speed-embedding-7b-instruct": "speed-embedding-7b-instruct", "sbintuitions_sarashina-embedding-v1-1b": "sarashina-embedding-v1-1b", "zeta-alpha-ai_Zeta-Alpha-E5-Mistral": "Zeta-Alpha-E5-Mistral", "Linq-Al-Research_Linq-Embed-Mistral": "Linq-Embed-Mistral", "jinaai_jina-embeddings-v5-text-nano": "jina-embeddings-v5-text-nano", "jinaai_jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small", "codefuse-ai_F2LLM-0.6B": "F2LLM-0.6B", "codefuse-ai_F2LLM-1.7B": "F2LLM-1.7B", "Qwen_Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B", "nomic-ai_nomic-embed-text-v1": "nomic-embed-text-v1", "BAAI_bge-m3": "bge-m3", "ibm-granite_granite-embedding-english-r2": "granite-embedding-english-r2", "ibm-granite_granite-embedding-small-english-r2": "granite-embedding-small-english-r2", "bflhc_Octen-Embedding-8B": "Octen-Embedding-8B", "Qwen_Qwen3-Embedding-4B": "Qwen3-Embedding-4B", "codefuse-ai_F2LLM-4B": "F2LLM-4B", "BAAI_bge-m3-unsupervised": "bge-m3-unsupervised", "bflhc_Octen-Embedding-0.6B": "Octen-Embedding-0.6B", "codefuse-ai_F2LLM-v2-0.6B": "F2LLM-v2-0.6B", "Alibaba-NLP_gte-modernbert-base": "gte-modernbert-base", "Qwen_Qwen3-Embedding-8B": "Qwen3-Embedding-8B", "nvidia_llama-embed-nemotron-8b": "llama-embed-nemotron-8b", "Snowflake_snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0", "nomic-ai_modernbert-embed-base": "modernbert-embed-base", "nomic-ai_nomic-embed-text-v1.5": "nomic-embed-text-v1.5", "codefuse-ai_F2LLM-v2-8B": "F2LLM-v2-8B", "codefuse-ai_F2LLM-v2-1.7B": "F2LLM-v2-1.7B", "codefuse-ai_F2LLM-v2-4B": "F2LLM-v2-4B", "bflhc_Octen-Embedding-4B": "Octen-Embedding-4B", "bflhc_MoD-Embedding": "MoD-Embedding", "ICT-TIME-and-Querit_BOOM_4B_v1": "BOOM_4B_v1"}
}

FIG_SRS_16_CONFIG = {
    "aliases": {"annamodels_LGAI-Embedding-Preview": "LGAI-Embedding-Preview", "Salesforce_SFR-Embedding-Mistral": "SFR-Embedding-Mistral", "Salesforce_SFR-Embedding-2_R": "SFR-Embedding-2_R", "Haon-Chen_speed-embedding-7b-instruct": "speed-embedding-7b-instruct", "sbintuitions_sarashina-embedding-v1-1b": "sarashina-embedding-v1-1b", "zeta-alpha-ai_Zeta-Alpha-E5-Mistral": "Zeta-Alpha-E5-Mistral", "Linq-Al-Research_Linq-Embed-Mistral": "Linq-Embed-Mistral", "jinaai_jina-embeddings-v5-text-nano": "jina-embeddings-v5-text-nano", "jinaai_jina-embeddings-v5-text-small": "jina-embeddings-v5-text-small", "codefuse-ai_F2LLM-0.6B": "F2LLM-0.6B", "codefuse-ai_F2LLM-1.7B": "F2LLM-1.7B", "Qwen_Qwen3-Embedding-0.6B": "Qwen3-Embedding-0.6B", "nomic-ai_nomic-embed-text-v1": "nomic-embed-text-v1", "BAAI_bge-m3": "bge-m3", "ibm-granite_granite-embedding-english-r2": "granite-embedding-english-r2", "ibm-granite_granite-embedding-small-english-r2": "granite-embedding-small-english-r2", "bflhc_Octen-Embedding-8B": "Octen-Embedding-8B", "Qwen_Qwen3-Embedding-4B": "Qwen3-Embedding-4B", "codefuse-ai_F2LLM-4B": "F2LLM-4B", "BAAI_bge-m3-unsupervised": "bge-m3-unsupervised", "bflhc_Octen-Embedding-0.6B": "Octen-Embedding-0.6B", "codefuse-ai_F2LLM-v2-0.6B": "F2LLM-v2-0.6B", "Alibaba-NLP_gte-modernbert-base": "gte-modernbert-base", "Qwen_Qwen3-Embedding-8B": "Qwen3-Embedding-8B", "nvidia_llama-embed-nemotron-8b": "llama-embed-nemotron-8b", "Snowflake_snowflake-arctic-embed-l-v2.0": "snowflake-arctic-embed-l-v2.0", "nomic-ai_modernbert-embed-base": "modernbert-embed-base", "nomic-ai_nomic-embed-text-v1.5": "nomic-embed-text-v1.5", "codefuse-ai_F2LLM-v2-8B": "F2LLM-v2-8B", "codefuse-ai_F2LLM-v2-1.7B": "F2LLM-v2-1.7B", "codefuse-ai_F2LLM-v2-4B": "F2LLM-v2-4B", "bflhc_Octen-Embedding-4B": "Octen-Embedding-4B", "bflhc_MoD-Embedding": "MoD-Embedding", "ICT-TIME-and-Querit_BOOM_4B_v1": "BOOM_4B_v1"},
    'p_thresh': None
}

FIG_TWCP_QWEN_CONFIG = {
    "pt_path": FINESSE_SRS_QWEN_PATH,
    "target_length": 8,
    "probe_len": 7,
    "probe_pos": 0,
    "sample_idx": 0,
}

FIG_TWCP_NEMOTRON_CONFIG = {
    "pt_path": FINESSE_SRS_NEMOTRON_PATH,
    "target_length": 8,
    "probe_len": 7,
    "probe_pos": 0,
    "sample_idx": 0,
}

FIG_CROSS_CORPUS_CONFIG = {
    "corpus_a_dir": FINESSE_EVALS_DIR,
    "corpus_b_dir": WIKIPEDIA_RSS15_DIR,
    "corpus_a_name": "CulturaX",
    "corpus_b_name": "Wikipedia",
    "exclude_models": [BASELINE_MODEL_NAME]
}

FIG_RSS_LX_MASTER_CONFIG = {
    'exclude_models': [BASELINE_MODEL_NAME]
}


FIG_RSS_L15_SCATTER_CONFIG = {
    'lemb_tasks': ['summscreen', 'narrativeqa'],
    'baseline_model': BASELINE_MODEL_NAME,
    'aliases': {
        'nvidia_llama-embed-nemotron-8b': 'llama-embed-nemotron-8b',
        'nomic-ai_nomic-embed-text-v1.5': 'nomic-embed-v1.5',
    }
}

FIG_RSS_JACKKNIFE_CONFIG = {
    'target_lengths': list(range(4, 17)),  # L=4 to L=16
    'exclude_models': [BASELINE_MODEL_NAME],
}

FIG_SRS_JACKKNIFE_CONFIG = {
    'exclude_models': [BASELINE_MODEL_NAME],
}

if __name__ == '__main__':

    # Define paths
    output_base_dir = "source/compiled"

    # Indexing
    index_source = TWO_CLUSTER_AND_PROBE_INDEX_PATH
    if not (os.path.exists(index_source)):
        index_rank(INDEX_CONFIG)

    # Tables

    # TBL 1
    tbl1_output_dir = os.path.join(output_base_dir, "tbl_rss")
    if not os.path.exists(tbl1_output_dir):
        tbl1_source = FINESSE_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 1 (RSS Scores)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_1_rss(
            tbl1_source, TBL_1_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl1_output_dir}':")
        save_artifacts(tbl1_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 2
    tbl2_output_dir = os.path.join(output_base_dir, "tbl_lemb")
    if not os.path.exists(tbl2_output_dir):
        tbl2_source = LEMB_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 2 (LEMB Scores)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_2_lemb(
            tbl2_source, TBL_2_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl2_output_dir}':")
        save_artifacts(tbl2_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 3
    tbl3_output_dir = os.path.join(output_base_dir, "tbl_rsslemb_correl")
    if not os.path.exists(tbl3_output_dir):
        tbl3_source = BENCHMARK_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 3 (RSS-LEMB Correl)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_3_rss_lemb_correl(
            tbl3_source, TBL_3_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl3_output_dir}':")
        save_artifacts(tbl3_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 4 (5, 6)
    tbl4_output_dir = os.path.join(output_base_dir, "tbl_srslemb_correl")
    if not os.path.exists(tbl4_output_dir):
        tbl4_source = BENCHMARK_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 4 (SRS - LEMB Correl)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_4_master_correl(
            tbl4_source, TBL_4_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl4_output_dir}':")
        save_artifacts(tbl4_output_dir, raw_data, latex_table, md_table)
        print("GENERATING TABLE 4.pearson (SRS - LEMB Correl Full Pearson p-value Table)")
        latex_table = generate_table_4_master_appendix_pearson(
            tbl4_source, TBL_4_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl4_output_dir}':")
        save_artifacts(os.path.join(tbl4_output_dir, "pearson_tbl"),
                       None, latex_table, None)
        print("GENERATING TABLE 4.spearman (SRS - LEMB Correl Full Spearman p-value Table)")
        latex_table = generate_table_4_master_appendix_spearman(
            tbl4_source, TBL_4_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl4_output_dir}':")
        save_artifacts(os.path.join(tbl4_output_dir, "spearman_tbl"),
                       None, latex_table, None)
        print(f"{'='*60}")

    # TBL 7 (8, 9)
    tbl7_output_dir = os.path.join(output_base_dir, "tbl_twcp")
    if not os.path.exists(tbl7_output_dir):
        tbl7_source = TWO_CLUSTER_AND_PROBE_INDEX_PATH
        print("GENERATING TABLE 7 (Two Clusters and Probe Hypo. Tables)")
        raw_data, latex_table, md_table = generate_table_7_analysis(
            tbl7_source, TBL_7_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl7_output_dir}':")
        save_artifacts(tbl7_output_dir, raw_data, latex_table, md_table)

    # TBL 10
    tbl10_output_dir = os.path.join(output_base_dir, "tbl_tdbu")
    if not os.path.exists(tbl10_output_dir):
        tbl10_source = FINESSE_PT_FILES_DIR
        print("GENERATING TABLE 10 (TDBU-RSS correl.)")
        raw_data, latex_table, md_table = generate_table_10_tdburss_analysis(
            tbl10_source, TBL_10_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl10_output_dir}':")
        save_artifacts(tbl10_output_dir, raw_data, latex_table, md_table)

    # TBL 11
    tbl11_output_dir = os.path.join(output_base_dir, "tbl_lemb_iqr")
    if not os.path.exists(tbl11_output_dir):
        tbl11_source = LEMB_EVALS_DIR
        print("GENERATING TABLE 11 (LEMB IQR.)")
        raw_data, latex_table, md_table = generate_table_11_lemb_iqr(
            tbl11_source, TBL_11_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl11_output_dir}':")
        save_artifacts(tbl11_output_dir, raw_data, latex_table, md_table)

    # TBL 12
    tbl12_output_dir = os.path.join(output_base_dir, "tbl_nomic")
    if not os.path.exists(tbl12_output_dir):
        tbl12_source = BENCHMARK_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 12 (RSS-LEMB Correl)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_3_rss_lemb_correl(
            tbl12_source, TBL_12_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl12_output_dir}':")
        save_artifacts(tbl12_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 13
    tbl13_output_dir = os.path.join(output_base_dir, "tbl_f2llm")
    if not os.path.exists(tbl13_output_dir):
        tbl13_source = BENCHMARK_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 13 (RSS-LEMB Correl)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_3_rss_lemb_correl(
            tbl13_source, TBL_13_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl13_output_dir}':")
        save_artifacts(tbl13_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 14
    tbl14_output_dir = os.path.join(output_base_dir, "tbl_rssfull")
    if not os.path.exists(tbl14_output_dir):
        tbl14_source = FINESSE_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 14 (RSS Scores)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_1_rss(
            tbl14_source, TBL_14_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl14_output_dir}':")
        save_artifacts(tbl14_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL RSS64
    tblrss64_output_dir = os.path.join(output_base_dir, "tbl_rss64")
    if not os.path.exists(tblrss64_output_dir):
        tblrss64_source = RSS64_DIR
        print("=" * 60)
        print("GENERATING TABLE rss64 (RSS Scores)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_1_rss(
            tblrss64_source, TBL_RSS64_CONFIG)
        print(f"SAVING ARTIFACTS to '{tblrss64_output_dir}':")
        save_artifacts(tblrss64_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 15
    tbl15_output_dir = os.path.join(output_base_dir, "tbl_lembfull")
    if not os.path.exists(tbl15_output_dir):
        tbl15_source = LEMB_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 15 (LEMB Scores)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_2_lemb(
            tbl15_source, TBL_15_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl15_output_dir}':")
        save_artifacts(tbl15_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL 16
    tbl16_output_dir = os.path.join(output_base_dir, "tbl_ttest")
    if not os.path.exists(tbl16_output_dir):
        tbl16_source = FINESSE_EVALS_DIR
        print("=" * 60)
        print("GENERATING TABLE 16 (Welch's T-TEST)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_16_welch_ttest(
            tbl16_source, TBL_16_CONFIG)
        print(f"SAVING ARTIFACTS to '{tbl16_output_dir}':")
        save_artifacts(tbl16_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # TBL Time
    tbltime_output_dir = os.path.join(output_base_dir, "tbl_time")
    if not os.path.exists(tbltime_output_dir):
        print("=" * 60)
        print("GENERATING TABLE Time (Time-Consuming)")
        print("=" * 60)
        raw_data, latex_table, md_table = generate_table_time_cost(
            BENCHMARK_EVALS_DIR,
            {
                'exclude_models': [BASELINE_MODEL_NAME]
            })
        print(f"SAVING ARTIFACTS to '{tbltime_output_dir}':")
        save_artifacts(tbltime_output_dir, raw_data, latex_table, md_table)
        print(f"{'='*60}")

    # Figures

    # SRS & RSS Basic Figure
    fig_srs_rss_output_dir = os.path.join(output_base_dir, "fig_srs_rss")
    if not os.path.exists(fig_srs_rss_output_dir):
        fig_srs_rss_source = FINESSE_EVALS_DIR
        print("=" * 60)
        print("GENERATING Figure SRS & RSS Basic")
        print("=" * 60)
        generate_srs_rss_figures(fig_srs_rss_source,
                                 fig_srs_rss_output_dir, FIG_SRS_RSS_CONFIG)

    # SRS Advanced(L16) Figure
    fig_srs_advanced_output_dir = os.path.join(output_base_dir, "fig_srs_16")
    if not os.path.exists(fig_srs_advanced_output_dir):
        fig_srs_advanced_source = SRS16_DIR
        print("=" * 60)
        print("GENERATING Figure SRS Advanced(L16)")
        print("=" * 60)
        generate_srs_rss_figures(fig_srs_advanced_source,
                                 fig_srs_advanced_output_dir, FIG_SRS_16_CONFIG)

    # Centroid based Rank in Cluster Heatmap Figure
    fig_crc_heatmap_output_dir = os.path.join(
        output_base_dir, "fig_crc_heatmap")
    if not os.path.exists(fig_crc_heatmap_output_dir):
        fig_crc_heatmap_source = TWO_CLUSTER_AND_PROBE_INDEX_PATH
        print("=" * 60)
        print("GENERATING Centroid based Rank in Cluster Heatmap Figure")
        print("=" * 60)
        generate_rank_heatmap(fig_crc_heatmap_source,
                              fig_crc_heatmap_output_dir)

    # Two Cluster and Probe Theory 3D Figure (Nemotron)
    fig_twcp_nemotron_output_dir = os.path.join(
        output_base_dir, "fig_twcp_nemotron")
    if not os.path.exists(fig_twcp_nemotron_output_dir):
        print("=" * 60)
        print("GENERATING Two Cluster and Probe Theory 3D Figure (Nemotron)")
        print("=" * 60)
        generate_twcp_figure(FIG_TWCP_NEMOTRON_CONFIG,
                             fig_twcp_nemotron_output_dir)

    # Two Cluster and Probe Theory 3D Figure (Qwen)
    fig_twcp_qwen_output_dir = os.path.join(output_base_dir, "fig_twcp_qwen")
    if not os.path.exists(fig_twcp_qwen_output_dir):
        print("=" * 60)
        print("GENERATING Two Cluster and Probe Theory 3D Figure (Qwen)")
        print("=" * 60)
        generate_twcp_figure(FIG_TWCP_QWEN_CONFIG,
                             fig_twcp_qwen_output_dir)

    # Cross Corpus RSS(L=15) Correl Figure
    fig_cross_corpus_output_dir = os.path.join(
        output_base_dir, "fig_cross_corpus")
    if not os.path.exists(fig_cross_corpus_output_dir):
        print("=" * 60)
        print("GENERATING Cross Corpus RSS(L=15) Correl Figure")
        print("=" * 60)
        generate_cross_corpus_figure(FIG_CROSS_CORPUS_CONFIG,
                                     fig_cross_corpus_output_dir)

    # RSS(L=x) to LEMB Correl Figure
    fig_lx_master_output_dir = os.path.join(output_base_dir, "fig_lx_master")
    if not os.path.exists(fig_lx_master_output_dir):
        print("=" * 60)
        print("GENERATING RSS(L=x) to LEMB Correl Figure")
        print("=" * 60)
        generate_lx_master_figure(BENCHMARK_EVALS_DIR, fig_lx_master_output_dir,
                                  FIG_RSS_LX_MASTER_CONFIG)

    # RSS(L=15) to LEMB Tasktype Correl Figure
    fig_l15_lembtask_output_dir = os.path.join(
        output_base_dir, "fig_l15_lembtask")
    if not os.path.exists(fig_l15_lembtask_output_dir):
        print("=" * 60)
        print("GENERATING RSS(L=15) to LEMB Tasktype Correl Figure")
        print("=" * 60)
        generate_l15_lemb_scatters(BENCHMARK_EVALS_DIR, fig_l15_lembtask_output_dir,
                                   FIG_RSS_L15_SCATTER_CONFIG)

    # LEMB Statistic Figure
    fig_lemb_statistic_output_dir = os.path.join(
        output_base_dir, "fig_lemb_statistic")
    if not os.path.exists(fig_lemb_statistic_output_dir):
        print("=" * 60)
        print("GENERATING LEMB Statistic Figure")
        print("=" * 60)
        generate_lemb_statistics(fig_lemb_statistic_output_dir,
                                 BENCHMARK_EVALS_DIR)

    # RSS Jackknife Figure
    fig_rss_jackknife_output_dir = os.path.join(
        output_base_dir, "fig_rss_jackknife")
    if not os.path.exists(fig_rss_jackknife_output_dir):
        print("=" * 60)
        print("GENERATING RSS Jackknife Figure")
        print("=" * 60)
        generate_rss_jackknife_figures(BENCHMARK_EVALS_DIR, fig_rss_jackknife_output_dir,
                                       FIG_RSS_JACKKNIFE_CONFIG)

    # SRS Jackknife Figure
    fig_srs_jackknife_output_dir = os.path.join(
        output_base_dir, "fig_srs_jackknife")
    if not os.path.exists(fig_srs_jackknife_output_dir):
        print("=" * 60)
        print("GENERATING SRS Jackknife Figure")
        print("=" * 60)
        generate_srs_jackknife_figures(BENCHMARK_EVALS_DIR, fig_srs_jackknife_output_dir,
                                       FIG_SRS_JACKKNIFE_CONFIG)

    # RSS-DIM Retrieval Partial Correlation
    tbl_rssdim_output_dir = os.path.join(
        output_base_dir, "tbl_dim_partial_correl")
    if not os.path.exists(tbl_rssdim_output_dir):
        print("=" * 60)
        print("GENERATING RSS-DIM: RSS-DIM Retrieval Partial Correlation")
        print("=" * 60)
        result = generate_tbl_dim_partial_correl(
            BENCHMARK_EVALS_DIR, tbl_rssdim_output_dir,
            TBL_DIM_CONFIG)
        print(f"  \u2713 LaTeX table: {result.get('latex_path')}")
        print("=" * 60)
    print(f"*** All artifacts saved to: {Path(output_base_dir).resolve()} ***")
