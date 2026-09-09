import os
import json
import yaml
import re
from ruamel.yaml import YAML
from typing import Optional
import typer
import torch
import numpy as np
import traceback
from importlib import resources
from pathlib import Path
import importlib.util
from .utils import get_content_hash, get_model_hash
from typing import Dict, List, Any
from .config import BenchmarkConfig, LocalModelSelector
from .evaluator import FinesseEvaluator
from .scoring import calculate_self_attestation_scores, calculate_self_attestation_scores_bottom_up, calculate_sample_latency, calculate_srs_score

app = typer.Typer(no_args_is_help=True)


@app.command("generate")
def generate_raw_data(
    config_path: str = typer.Option(..., "--config",
                                    help="Path to benchmark.yaml config file"),
    output_dir: str = typer.Option(
        "results", "--output", help="Directory to save raw embedding data"),
):
    # Load config
    if not os.path.exists(config_path):
        typer.echo(f"Error: Config file not found: {config_path}")
        raise typer.Exit(code=1)
    with open(config_path, "r") as f:
        yaml_data = yaml.safe_load(f)

    try:
        config = BenchmarkConfig.model_validate(yaml_data)
        typer.echo(f"Loaded config from {config_path}")
    except Exception as e:
        typer.echo(f"Error validating config: {e}")
        raise typer.Exit(code=1)

    # Validate that datasets list is not empty
    if not config.datasets:
        typer.echo(
            "❌ Error: The 'datasets' list in your config is empty. Please specify at least one dataset.")
        raise typer.Exit(code=1)

    # Validate sequence lengths - minimum length must be 4 for valid scoring
    sequence_length_min = config.probe_config.sequence_length.min

    if sequence_length_min < 4:
        typer.echo(
            f"❌ Error: Invalid sequence lengths minimum: {sequence_length_min}")
        typer.echo("   Minimum sequence length must be 4 for valid scoring.")
        typer.echo(
            "   For lengths < 4, the scoring system cannot properly evaluate")
        typer.echo("   contextual coherence and bottom-up coherence.")
        raise typer.Exit(code=1)

    typer.echo(f"✅ Valid sequence lengths: {sequence_length_min}")

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Helper function to load local model engines
    def load_local_engine(model_config: LocalModelSelector):
        file_path = Path(model_config.local_path)
        class_name = model_config.local_class

        if not file_path.exists():
            raise FileNotFoundError(f"Local model file not found: {file_path}")

        spec = importlib.util.spec_from_file_location(
            file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        ModelClass = getattr(module, class_name)

        config_path = str(file_path.parent)
        return ModelClass(config_path=config_path)

    # Import the engine implementations
    from .implementations import HuggingFaceEmbedder, HuggingFaceSynthesizer, NullSynthesizer  # , ByokEmbedder

    # Initialize embedder and synthesizer based on mode
    if config.mode == 'merger_mode':
        typer.echo("  Mode: merger_mode")
        merger_config = config.models.merger
        embedder_config = config.models.base_embedder

        # Load synthesizer (merger)
        if isinstance(merger_config, LocalModelSelector):
            synthesizer = load_local_engine(merger_config)
            typer.echo(
                f"  Synthesizer: Local - {merger_config.local_path}::{merger_config.local_class}")
        else:
            synthesizer = HuggingFaceSynthesizer(
                merger_config.name, merger_config.pool_type)
            typer.echo(f"  Synthesizer: {merger_config.name}")

        # Load embedder (base)
        if isinstance(embedder_config, LocalModelSelector):
            embedder = load_local_engine(embedder_config)
            typer.echo(
                f"  Embedder: Local - {embedder_config.local_path}::{embedder_config.local_class}")
        else:
            embedder = HuggingFaceEmbedder(
                embedder_config.name,
                prefix=embedder_config.prefix,
                pool_type=embedder_config.pool_type,
                max_length=embedder_config.max_context_length
            )
            typer.echo(f"  Embedder: {embedder_config.name}")

    elif config.mode == 'native_mode':
        typer.echo("  Mode: native_mode")
        embedder_config = config.models.native_embedder

        if isinstance(embedder_config, LocalModelSelector):
            embedder = load_local_engine(embedder_config)
            typer.echo(
                f"  Embedder: Local - {embedder_config.local_path}::{embedder_config.local_class}")
        else:
            embedder = HuggingFaceEmbedder(
                embedder_config.name,
                pool_type=embedder_config.pool_type,
                prefix=embedder_config.prefix,
                max_length=embedder_config.max_context_length
            )
            typer.echo(f"  Embedder: {embedder_config.name}")

        synthesizer = NullSynthesizer()
        typer.echo("  Synthesizer: pass-through")

    else:
        typer.echo(f"❌ Error: Unknown mode '{config.mode}'")
        raise typer.Exit(code=1)

    # Initialize evaluator with the engines
    evaluator = FinesseEvaluator(
        embedder_engine=embedder, synthesizer_engine=synthesizer, config=config)

    # Run raw evaluation
    typer.echo("Generating raw embeddings...")
    typer.echo(f"mode={config.metric}")
    raw_data = None
    if config.metric == "srs":
        if config.mode == "merger_mode":
            raw_data = evaluator.merger_run_srs()
            typer.echo("  Running merger_run_srs for SRS benchmark.")
        else:
            raw_data = evaluator.native_run_srs()
            typer.echo("  Running native_run_srs for SRS benchmark.")
    else:  # rss
        if config.mode == "merger_mode":
            raw_data = evaluator.merger_run()
            typer.echo("  Running merger_run for RSS benchmark.")
        else:
            raw_data = evaluator.native_run()
            typer.echo("  Running native_run for RSS benchmark.")

    # Save full raw data (config + raw_results) to .pt file
    if config.mode == "merger_mode":
        representative = config.models.merger.name
    elif config.mode == "byok_mode":
        representative = config.models.byok_embedder.name
    elif config.mode == "native_mode":
        representative = config.models.native_embedder.name

    representative = representative.replace('/', '_')

    save_path = os.path.join(
        output_dir, f"embeddings_{config.mode}_{config.metric}_{representative}.pt")
    torch.save(raw_data, save_path)

    typer.echo(f"Raw data (with config) saved to {save_path}")
    length_results = raw_data['raw_results'].get('length_results', {})
    num_lengths = len(length_results)
    typer.echo(
        f"Processed {num_lengths} sequence lengths with raw probe and synthesis embeddings.")

    return save_path


@app.command("score")
def score_embeddings(
    pt_path: str = typer.Option(..., "--pt-path",
                                help="Path to the raw .pt data file from the generate command"),
    output_dir: str = typer.Option(
        "results", "--output", help="Directory to save scored results"),
):
    if not os.path.exists(pt_path):
        typer.echo(f"Error: Input .pt file not found: {pt_path}")
        raise typer.Exit(code=1)

    raw_data = torch.load(pt_path, weights_only=False)
    config_dict = raw_data['config']

    # Assign eval_mode to config.formula for hash consistency
    eval_mode = config_dict['formula']
    typer.echo(f"Using eval_mode: {eval_mode} (assigned to config.formula)")
    metadata = raw_data.get('metadata', {})
    length_results = raw_data.get('raw_results', {}).get('length_results', {})

    if not length_results:
        typer.echo("Error: No length results found in .pt file.")
        raise typer.Exit(code=1)

    scoring_method = metadata.get('scoring_method', 'rss')
    mode = config_dict.get('mode')

    avg_score = None

    if scoring_method == 'srs':
        # Calculate SRS scores using helper
        srs_data = _calculate_srs_scores(length_results, eval_mode=eval_mode)
        length_scores = srs_data['length_scores']
        all_individual_scores = srs_data['all_individual_scores']

        # Calculate PBI
        average_pbi = _calculate_pbi(length_scores)
        typer.echo(f"Computed PBI: {average_pbi}")

        # Averages (no latencies for SRS)
        avg_srs = np.mean(
            all_individual_scores) if all_individual_scores else 0.0
        avg_srs = round(avg_srs, 6)
        avg_score = avg_srs
        # Prepare base results without hash
        base_results = {
            'config': config_dict,
            'average_srs': avg_srs,
            'average_pbi': average_pbi,
            'length_scores': length_scores,
            'metadata': metadata  # Includes device_info for hardware provenance from .pt
        }
        typer.echo("Computed SRS scores.")

    else:
        # RSS branch
        # Calculate RSS scores using helper
        rss_data = _calculate_rss_scores(
            length_results, mode, eval_mode=eval_mode)
        length_scores = rss_data['length_scores']
        all_individual_scores = rss_data['all_individual_scores']
        all_total_latencies = rss_data['all_total_latencies']
        all_synthesis_latencies = rss_data['all_synthesis_latencies']

        # Averages
        avg_rss = np.mean(
            all_individual_scores) if all_individual_scores else 0.0
        avg_rss = round(avg_rss, 6)
        avg_score = avg_rss
        avg_total_latency = np.mean(
            all_total_latencies) if all_total_latencies else 0.0
        avg_total_latency = round(avg_total_latency, 6)
        avg_synthesis_latency = np.mean(
            all_synthesis_latencies) if all_synthesis_latencies else 0.0
        avg_synthesis_latency = round(avg_synthesis_latency, 6)

        # Prepare base results without hash
        base_results = {
            'config': config_dict,
            'average_rss': avg_rss,
            'average_total_latency': avg_total_latency,
            'average_synthesis_latency': avg_synthesis_latency,
            'length_scores': length_scores,
            'metadata': metadata  # Includes device_info for hardware provenance from .pt
        }
        typer.echo("Computed RSS scores.")

    # Compute model hash for notarization (before content_hash)
    try:
        config = BenchmarkConfig.model_validate(config_dict)
        model_hash_dict = {}

        if config.mode == 'merger_mode':
            # Dual Notarization Protocol: Hash both merger and base_embedder
            merger_path = config.models.merger.name
            base_path = config.models.base_embedder.name
            model_hash_dict['merger'] = get_model_hash(merger_path)
            model_hash_dict['base_embedder'] = get_model_hash(base_path)
            typer.echo(
                f"Merger model hash computed: {model_hash_dict['merger'][:16]}...")
            typer.echo(
                f"Base embedder hash computed: {model_hash_dict['base_embedder'][:16]}...")
        elif config.mode == 'native_mode':
            native_path = config.models.native_embedder.name
            model_hash_dict['native'] = get_model_hash(native_path)
            typer.echo(
                f"Native model hash computed: {model_hash_dict['native'][:16]}...")
        elif config.mode == 'byok_mode':
            # Diplomat Passport Protocol: Hash the identity string for BYOK models
            provider = config.models.byok_embedder.provider
            name = config.models.byok_embedder.name
            hash_string = f"byok:{provider}:{name}"
            model_hash_dict['byok'] = get_content_hash(
                {'identity': hash_string})
            typer.echo(
                f"BYOK model identity hash computed: {model_hash_dict['byok'][:16]}...")

        base_results['model_hash'] = model_hash_dict
    except Exception as e:
        typer.echo(f"Warning: Could not compute model hash: {e}")
        base_results['model_hash'] = None

    # Create output dir before hashing to ensure debug path exists
    os.makedirs(output_dir, exist_ok=True)

    # Create copy for hashing with fixed frame ('content_hash': '')
    hash_data = base_results.copy()
    hash_data['content_hash'] = ''

    # Compute content hash on the fixed frame with debug
    content_hash = get_content_hash(hash_data)

    # Add the hash to final results
    results = base_results.copy()
    results['content_hash'] = content_hash

    # Save to JSON
    output_path = os.path.join(output_dir, "benchmark_results.json")
    with open(output_path, "w", encoding='utf-8', newline='') as f:
        json.dump(results, f, indent=2)

    typer.echo(f"Scored results saved to {output_path}")
    if scoring_method == 'rss':
        typer.echo(f"Average RSS: {avg_score}")
    elif scoring_method == 'srs':
        typer.echo(f"Average SRS: {avg_score}")


def _calculate_rss_scores(length_results: Dict[int, Any], mode: str, eval_mode: str = 'q1q3') -> Dict[str, Any]:
    length_scores = {}
    all_individual_scores = []
    all_total_latencies = []
    all_synthesis_latencies = []

    for target_length, raw in length_results.items():
        sample_results = raw.get('sample_results', [])
        sample_scores = []
        total_latency_list = []
        synthesis_latency_list = []

        sample_tds = []
        sample_bus = []

        for sample_dict in sample_results:
            probe_embeddings = sample_dict.get('chunk_embeddings')
            synthesis_embeddings = sample_dict.get('synthesis_embeddings')

            if probe_embeddings and synthesis_embeddings and len(probe_embeddings) >= 2:
                td_scores = calculate_self_attestation_scores(
                    probe_embeddings, synthesis_embeddings, eval_mode=eval_mode)
                bu_scores = calculate_self_attestation_scores_bottom_up(
                    probe_embeddings, synthesis_embeddings, eval_mode=eval_mode)

                avg_td = td_scores['contextual_coherence']
                avg_bu = bu_scores['bottom_up_coherence']
                final_score = avg_td + avg_bu
                sample_scores.append(final_score)
                sample_tds.append(avg_td)
                sample_bus.append(avg_bu)
            else:
                sample_scores.append(0.0)
                sample_tds.append(0.0)
                sample_bus.append(0.0)
            # Calculate latencies for this sample
            chunk_times = sample_dict.get('chunk_times', None)
            synth_times = sample_dict.get('synth_times', [])
            if mode and synth_times:
                try:
                    latency_dict = calculate_sample_latency(
                        mode, chunk_times, synth_times)
                    total_latency_list.append(latency_dict['total_latency'])
                    synthesis_latency_list.append(
                        latency_dict['synthesis_latency'])
                except Exception as e:
                    # Fallback for invalid data
                    total_latency_list.append(0.0)
                    synthesis_latency_list.append(0.0)
            else:
                total_latency_list.append(0.0)
                synthesis_latency_list.append(0.0)

        # Store scaled RSS and rounded latency scores as lists
        scaled_rss = [round(score * 500, 6) for score in sample_scores]
        scaled_total = [round(t, 6) for t in total_latency_list]
        scaled_synth = [round(s, 6) for s in synthesis_latency_list]

        length_scores[target_length] = {
            'rss_scores': scaled_rss,
            'total_latency_scores': scaled_total,  # ms, cold start
            'synthesis_latency_scores': scaled_synth,  # ms, warm start
            'raw_td': td_scores,
            'raw_bu': bu_scores
        }
        all_individual_scores.extend(scaled_rss)
        all_total_latencies.extend(scaled_total)
        all_synthesis_latencies.extend(scaled_synth)

    return {
        'length_scores': length_scores,
        'all_individual_scores': all_individual_scores,
        'all_total_latencies': all_total_latencies,
        'all_synthesis_latencies': all_synthesis_latencies
    }


def _calculate_srs_scores(length_results: Dict[int, Any], eval_mode: str = 'q1q3') -> Dict[str, Any]:
    length_scores = {}
    all_individual_scores = []
    for target_length, raw in length_results.items():
        sample_results = raw.get('sample_results', [])
        processed_samples = []

        for sample_idx, sample_dict in enumerate(sample_results):
            if not isinstance(sample_dict, dict):
                continue  # Skip invalid samples

            new_sample = {}  # {probe_len: [scores for pos0, pos1, ...]}
            all_scores_for_sample = []  # Temp for flattening

            for probe_len_str, probe_data in sample_dict.items():
                if not isinstance(probe_data, dict) or 'probe_embedding' not in probe_data:
                    continue  # Invalid probe_len data

                probe_embedding = probe_data['probe_embedding']
                pos_embeddings_dict = probe_data.get(
                    'probe_pos_embeddings', {})

                scores_for_probe_len = []  # List of SRS scores for each probe_pos

                for pos_str, pos_data in pos_embeddings_dict.items():
                    if not isinstance(pos_data, dict):
                        continue

                    pos_group = pos_data.get('positive_embeddings', [])
                    neg_group = pos_data.get('negative_embeddings', [])

                    try:
                        srs_score = round(calculate_srs_score(
                            probe_embedding, pos_group, neg_group, eval_mode=eval_mode) * 1000, 6)
                        scores_for_probe_len.append(srs_score)
                        all_scores_for_sample.append(srs_score)
                    except ValueError:
                        scores_for_probe_len.append(0.0)
                        all_scores_for_sample.append(0.0)

                new_sample[probe_len_str] = scores_for_probe_len

            processed_samples.append(new_sample)
            all_individual_scores.extend(all_scores_for_sample)
        length_scores[target_length] = {
            'sample_results': processed_samples
        }

    return {
        'length_scores': length_scores,
        'all_individual_scores': all_individual_scores
    }


def _calculate_pbi(length_scores: Dict[int, Any]) -> float:
    all_probe_std_devs = []

    for target_length_str, length_data in length_scores.items():
        target_length = int(target_length_str)
        max_probe_len = target_length // 2
        sample_results = length_data.get('sample_results', [])

        if not sample_results:
            continue

        # Aggregate scores across all samples for each probe_length and position
        # Structure: probe_len -> [list of scores per position across all samples]
        probe_len_aggregated = {}

        for sample in sample_results:
            for probe_len_str, pos_scores in sample.items():
                probe_len = int(probe_len_str)
                if probe_len > max_probe_len:
                    continue

                if probe_len not in probe_len_aggregated:
                    probe_len_aggregated[probe_len] = []

                # pos_scores is a list of scores for different positions
                # We need to collect all positions' scores
                probe_len_aggregated[probe_len].extend(pos_scores)

        # For each probe_len, compute the std dev of the aggregated position scores
        for probe_len, all_pos_scores in probe_len_aggregated.items():
            if len(all_pos_scores) > 1:
                std_dev = np.std(all_pos_scores)
                all_probe_std_devs.append(std_dev)

    if all_probe_std_devs:
        return round(np.mean(all_probe_std_devs), 6)
    return 0.0


if __name__ == "__main__":
    app()
