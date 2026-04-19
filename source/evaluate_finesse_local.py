import time
import os
import sys
import platform
from pathlib import Path
from tqdm.auto import tqdm
from datetime import datetime
from finesse import run_benchmark_from_config
import json
import traceback

# split1/upper~ split4/lower
SPLIT = ""

TARGET_FOLDER = "examples/evals/finesse" + "/" + SPLIT

# please write the format like "Alibaba-NLP_gte-modernbert-base:srs"
EXCLUDED_MODEL_EVALTYPES = [
    # "Alibaba-NLP_gte-base-en-v1.5:srs",
    # "Alibaba-NLP_gte-base-en-v1.5:rss"
]

# Optional imports - gracefully handle if not available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import accelerate
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False


def probe_system_info():
    """Probe system information."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage('.')
        return {
            "mem": {
                "total": mem.total,
                "available": mem.available,
                "percent": mem.percent,
            },
            "disk": {
                "total": disk.total,
                "available": disk.free,
            },
        }
    return {}


def probe_gpu_info():
    """Probe GPU information."""

    if not TORCH_AVAILABLE:
        print("PyTorch not available - cannot probe GPU info")
        return {}

    if torch.cuda.is_available():
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu = {
                "name": props.name,
                "total_memory": f"{props.total_memory / (1024**3):.2f} GB",
                "reserved_memory": f"{torch.cuda.memory_reserved(i) / (1024**3):.2f} GB",
                "allocated_memory": f"{torch.cuda.memory_allocated(i) / (1024**3):.2f} GB",
                "compute_cap": f"{props.major}.{props.minor}",
                "multi_procs": f"{props.multi_processor_count}",
            }
            gpus.append(gpu)

        return {
            "torch": {
                "version": str(torch.__version__),
                "avilable": torch.cuda.is_available(),
                "current_device": f"{torch.cuda.current_device()} ({torch.cuda.get_device_name(0)})"
            },
            "cuda": {
                "version-cuda": str(torch.version.cuda),
                "version-cudnn": str(torch.backends.cudnn.version()),
                "gpus": gpus
            }
        }

    return {}


def probe_python_packages():
    """Probe installed Python packages."""
    packages_to_check = [
        ("torch", TORCH_AVAILABLE),
        ("transformers", TRANSFORMERS_AVAILABLE),
        ("accelerate", ACCELERATE_AVAILABLE),
        ("psutil", PSUTIL_AVAILABLE),
    ]

    # Add more common packages
    try:
        import huggingface_hub
        packages_to_check.append(("huggingface_hub", True))
    except ImportError:
        packages_to_check.append(("huggingface_hub", False))

    try:
        import sentence_transformers
        packages_to_check.append(("sentence_transformers", True))
    except ImportError:
        packages_to_check.append(("sentence_transformers", False))

    try:
        import numpy
        packages_to_check.append(("numpy", True))
    except ImportError:
        packages_to_check.append(("numpy", False))

    try:
        import pandas
        packages_to_check.append(("pandas", True))
    except ImportError:
        packages_to_check.append(("pandas", False))

    return packages_to_check


def probe_environment_variables():
    """Probe relevant environment variables."""

    env_vars = [
        "CUDA_VISIBLE_DEVICES",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TRANSFORMERS_CACHE",
        "HF_HOME",
        "TOKENIZERS_PARALLELISM",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
    ]

    res = {}
    for var in env_vars:
        value = os.environ.get(var, "(not set)")
        res[var] = value

    return res


def generate_device_info_dict():
    """Generate a dictionary containing device information for saving."""

    return {
        "sys_info": probe_system_info(),
        "gpu_info": probe_gpu_info(),
        "pkg_info": probe_python_packages(),
        "env_info": probe_environment_variables(),
    }


def main(target_folder):
    base_path = Path(target_folder)
    status_log_path = base_path / "status.log"
    print(
        f" Starting stateful FINESSE evaluation across: {base_path.resolve()}")
    print(f"   Status Log: {status_log_path}")

    # 1. Read status.log to find completed tasks (format: model_name:eval_type)
    completed_tasks = set()
    if status_log_path.exists():
        with open(status_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    completed_tasks.add(parts[1])

    print(
        f"Found {len(completed_tasks)} previously processed tasks in status log.")

    # Find all srs.yaml and rss.yaml files recursively
    all_config_files = list(base_path.rglob('srs.yaml')) + \
        list(base_path.rglob('rss.yaml'))

    tasks_to_evaluate = []
    for config_path in all_config_files:
        model_name = config_path.parent.parent.name
        eval_type = config_path.parent.name.lower()
        task_id = f"{model_name}:{eval_type}"

        if task_id not in completed_tasks and task_id not in [x.lower() for x in EXCLUDED_MODEL_EVALTYPES]:
            tasks_to_evaluate.append(config_path)

    if not tasks_to_evaluate:
        print("✨ All FINESSE evaluations have already been completed. Nothing to do. ✨")
        return

    print(f"Found {len(tasks_to_evaluate)} new tasks to evaluate.")
    progress_bar = tqdm(tasks_to_evaluate, desc="Evaluation Progress")

    for config_path in progress_bar:
        model_name = config_path.parent.parent.name
        eval_type = config_path.parent.name.upper()
        task_id = f"{model_name}:{eval_type.lower()}"

        progress_bar.set_description(
            f" Running [{eval_type}] for [{model_name}]")
        output_dir = config_path.parent
        status = "FAIL"

        try:
            start_time = time.monotonic()
            run_benchmark_from_config(config_path=str(
                config_path), output_dir=str(output_dir))
            end_time = time.monotonic()

            report_path = os.path.join(output_dir, "report.txt")
            with open(report_path, "w", encoding="utf-8") as f:
                evaluation = {
                    "total_wall_clock_elapsed": (end_time - start_time)
                }
                env = generate_device_info_dict()
                dummy = {
                    "evaluation": evaluation,
                    "env": env
                }
                f.write(json.dumps(dummy, indent=4))

            status = "SUCCESS"
            print(f"  ✅ FINISHED: [{eval_type}] for [{model_name}]")
        except Exception as e:
            error_log_path = output_dir / 'error.log'
            error_message = f"---\nERROR during [{eval_type}] for [{model_name}]\nConfig: {config_path}\nError: {e}\n---"
            print(
                f"   FAILED: [{eval_type}] for [{model_name}]. See {error_log_path}")

            with open(error_log_path, "w") as f:
                f.write(error_message)
                f.write('\n')
                f.write(traceback.format_exception())

        finally:
            with open(status_log_path, 'a', encoding='utf-8') as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{task_id},{status}\n")
            if status == 'FAIL':
                break

    print("\n\n⏸️ FINESSE evaluations loop halted! ⏸️")


if __name__ == '__main__':
    main(TARGET_FOLDER)
