import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

# =============================================================================
# LEMB Task Mapping: Short keywords to full JSON keys
# =============================================================================
LEMB_TASK_MAP = {
    'needle': 'LEMBNeedleRetrieval',
    'passkey': 'LEMBPasskeyRetrieval',
    'summscreen': 'LEMBSummScreenFDRetrieval',
    'qmsum': 'LEMBQMSumRetrieval',
    'wikimqa': 'LEMBWikimQARetrieval',
    'narrativeqa': 'LEMBNarrativeQARetrieval',
}


# =============================================================================
# Data Gathering Functions
# =============================================================================
def gather_lemb_scores(directory: str) -> Dict[str, List[float]]:
    """
    Gather LEMB scores from all overall_results.json files in the given directory.
    
    Args:
        directory: Root directory to search for benchmark results
        
    Returns:
        Dictionary mapping task names to lists of scores from all models
    """
    print(f"\n{'='*60}")
    print(f"Gathering LEMB scores from '{directory}'...")
    print(f"{'='*60}")
    
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return {}
    
    # Find all overall_results.json files
    json_files = list(dir_path.rglob("overall_results.json"))
    
    if not json_files:
        print(f"No overall_results.json files found.")
        return {}
    
    print(f"Found {len(json_files)} benchmark result file(s).")
    
    # Initialize score collection dictionary
    task_scores: Dict[str, List[float]] = {
        task_key: [] for task_key in LEMB_TASK_MAP.values()
    }
    
    lemb_files_found = 0
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if this is a LEMB result file
            is_lemb = False
            for task_key in LEMB_TASK_MAP.values():
                if task_key in data:
                    is_lemb = True
                    break
            
            if not is_lemb:
                continue
            
            lemb_files_found += 1
            model_name = json_file.parent.name
            print(f"  ✓ LEMB: {json_file.name} -> {model_name}")
            
            # Extract scores for each task
            for short_key, full_key in LEMB_TASK_MAP.items():
                if full_key in data:
                    task_data = data[full_key]
                    
                    # Extract score based on task type
                    if short_key in ['needle', 'passkey']:
                        # Needle and Passkey use 'avg'
                        score = task_data.get('avg') if isinstance(task_data, dict) else task_data
                    else:
                        # Other tasks use 'ndcg@10'
                        score = task_data.get('ndcg@10') if isinstance(task_data, dict) else task_data
                    
                    if score is not None:
                        task_scores[full_key].append(float(score))
                        
        except Exception as e:
            print(f"  ✗ Error loading '{json_file.name}': {e}")
    
    print(f"\nProcessed {lemb_files_found} LEMB result file(s).")
    

    # Print summary statistics
    print(f"\n{'='*60}")
    print("Score Summary by Task:")
    print(f"{'='*60}")
    for task_name, scores in task_scores.items():
        if scores:
            mean_score = np.mean(scores)
            q3, q1 = np.percentile(scores, [75, 25])
            iqr = q3 - q1
            print(f"  {task_name:30s}: n={len(scores):3d}, mean={mean_score:.4f}, IQR={iqr:.4f}")
        else:
            print(f"  {task_name:30s}: n=0, no data")

    
    return task_scores
def create_boxplot(task_scores: Dict[str, List[float]], output_path: str):
    """
    Create a box plot visualization of LEMB task score distributions.
    
    Args:
        task_scores: Dictionary mapping task names to lists of scores
        output_path: Path to save the plot
    """
    print(f"\n{'='*60}")
    print("Creating box plot visualization...")
    print(f"{'='*60}")
    
    # Convert to long-format DataFrame for seaborn
    data_rows = []
    for task_name, scores in task_scores.items():
        # Use shorter display name
        display_name = task_name.replace('LEMB', '').replace('Retrieval', '')
        for score in scores:
            data_rows.append({'Task': display_name, 'Score': score})
    
    if not data_rows:
        print("No data to plot!")
        return
    
    df = pd.DataFrame(data_rows)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create box plot
    sns.boxplot(
        data=df,
        x='Task',
        y='Score',
        palette='Set2',
        ax=ax,
        width=0.6
    )

    # Configure plot
    ax.set_xlabel('', fontsize=12, fontweight='bold')
    ax.set_ylabel('', fontsize=12, fontweight='bold')
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=30, ha='right')
    
    plt.tight_layout()
    
    # Save plot as SVG
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    plt.close()
# =============================================================================
# Main Entry Point
# =============================================================================
def generate_lemb_statistics(output_dir: str, data_dir: str) -> str:
    """
    Analyze LEMB task score distributions and create box plot visualization.
    
    Args:
        output_dir: Directory to save the output SVG
        data_dir: Directory containing benchmark results
        
    Returns:
        Path to the generated SVG file
    """
    # Gather scores
    task_scores = gather_lemb_scores(data_dir)
    
    if not task_scores or all(len(scores) == 0 for scores in task_scores.values()):
        print("\nNo LEMB data found. Exiting.")
        raise ValueError("No LEMB data found")
    
    # Create visualization
    output_path = Path(output_dir) / "lemb_task_distribution.svg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_boxplot(task_scores, str(output_path))
    
    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}")
    
    return str(output_path)
